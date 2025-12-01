"""
Simplified Training Script for FPO Flow Variants

This script directly modifies the original FPO to use different flow schedules.
Instead of rewriting the entire training loop, we patch the core CFM loss computation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import jax
import jax_dataclasses as jdc
import numpy as onp
import wandb
from jax import numpy as jnp
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/home/wayneleo8/Desktop/fpo_official/playground/src')
sys.path.insert(0, '/home/wayneleo8/Desktop/fpo-humanoid-getup')

from flow_policy import fpo, rollouts
from flow_policy.networks import flow_mlp_fwd

# Import environment
from humanoid_getup import HumanoidGetup, default_config

# Import our flow schedules
from flow_schedules import FlowType, get_flow_coefficients


def patch_fpo_for_flow_variant(flow_type: FlowType, sigma_min: float = 0.01, sigma_max: float = 80.0):
    """
    Monkey-patch FPO's _compute_cfm_loss to use different flow schedules.

    This allows us to use the original FPO training loop while testing
    different flow interpolation methods.
    """
    original_compute_cfm_loss = fpo.FpoState._compute_cfm_loss

    def patched_compute_cfm_loss(self, obs_norm, action, eps, t):
        """Modified CFM loss using different flow schedules."""
        (*batch_dims, action_dim) = action.shape
        samples_dim = self.config.n_samples_per_action
        obs_dim = self.env.observation_size
        sample_shape = (*batch_dims, samples_dim)

        # Get flow coefficients for the specified schedule
        coeffs = get_flow_coefficients(
            t, flow_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        # Compute x_t using the schedule-specific interpolation
        # x_t = alpha_t * x_0 + sigma_t * eps
        x_t = coeffs.alpha_t * action[..., None, :] + coeffs.sigma_t * eps

        # Network prediction
        network_pred = (
            flow_mlp_fwd(
                self.params.policy,
                jnp.broadcast_to(obs_norm[..., None, :], (*sample_shape, obs_dim)),
                x_t,
                self.embed_timestep(t),
            )
            * self.config.policy_mlp_output_scale
        )

        # Compute loss based on flow type
        if flow_type == "ot":
            # Original OT/CFM loss (eps supervision)
            velocity_pred = network_pred
            x0_pred = x_t - t * velocity_pred
            x1_pred = x0_pred + velocity_pred
            out = jnp.mean((eps - x1_pred) ** 2, axis=-1)
        else:
            # For VP, VE, Cosine: velocity supervision
            velocity_target = (
                coeffs.d_alpha_dt * action[..., None, :]
                + coeffs.d_sigma_dt * eps
            )
            out = jnp.mean((network_pred - velocity_target) ** 2, axis=-1)

        return out

    # Apply the patch
    fpo.FpoState._compute_cfm_loss = patched_compute_cfm_loss

    return original_compute_cfm_loss  # Return original for restoration if needed


def train_fpo_variant(
    flow_type: FlowType,
    wandb_entity: str = "waynehacking8-national-taiwan-university-of-science-and-",
    wandb_project: str = "fpo-flow-variants",
    seed: int = 0,
    num_timesteps: int = 3_000_000,
    save_dir: str = "./results",
) -> dict:
    """Train FPO with a specific flow variant."""

    print(f"\n{'='*60}")
    print(f"Training FPO with {flow_type.upper()} flow schedule")
    print(f"{'='*60}\n")

    # Patch FPO to use our flow schedule
    original_fn = patch_fpo_for_flow_variant(flow_type)

    # Configure based on flow type
    if flow_type == "ve":
        # VE needs more steps and lower learning rate
        config = fpo.FpoConfig(
            num_timesteps=num_timesteps,
            num_evals=10,
            episode_length=1000,
            num_envs=2048,
            unroll_length=30,
            learning_rate=1e-4,  # Lower for VE stability
            batch_size=1024,
            num_minibatches=32,
            num_updates_per_batch=16,
            clipping_epsilon=0.1,  # Wider for VE
            discounting=0.995,
            gae_lambda=0.95,
            value_loss_coeff=0.25,
            normalize_observations=True,
        )
    else:
        # Standard config for OT, VP, Cosine
        config = fpo.FpoConfig(
            num_timesteps=num_timesteps,
            num_evals=10,
            episode_length=1000,
            num_envs=2048,
            unroll_length=30,
            learning_rate=3e-4,
            batch_size=1024,
            num_minibatches=32,
            num_updates_per_batch=16,
            clipping_epsilon=0.05,
            discounting=0.995,
            gae_lambda=0.95,
            value_loss_coeff=0.25,
            normalize_observations=True,
        )

    # Load environment
    env_config = default_config()
    env = HumanoidGetup(config=env_config)

    # Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"fpo_{flow_type}_HumanoidGetup_{timestamp}"

    wandb_run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=exp_name,
        config={
            "env_name": "HumanoidGetup",
            "flow_type": flow_type,
            "fpo_params": jdc.asdict(config),
            "seed": seed,
        },
        reinit=True,
    )

    # Initialize
    agent_state = fpo.FpoState.init(prng=jax.random.key(seed), env=env, config=config)
    rollout_state = rollouts.BatchedRolloutState.init(
        env,
        prng=jax.random.key(seed),
        num_envs=config.num_envs,
    )

    # Training loop
    outer_iters = config.num_timesteps // (config.iterations_per_env * config.num_envs)
    eval_iters = set(onp.linspace(0, outer_iters - 1, config.num_evals, dtype=int))

    eval_results = []
    times = [time.time()]

    for i in tqdm(range(outer_iters), desc=f"FPO-{flow_type.upper()}"):
        # Evaluation
        if i in eval_iters:
            eval_outputs = rollouts.eval_policy(
                agent_state,
                prng=jax.random.fold_in(agent_state.prng, i),
                num_envs=128,
                max_episode_length=config.episode_length,
            )

            s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}

            print(f"\n[{flow_type.upper()}] Eval at step {i}:")
            print(f"  Reward: mean={s_np['reward_mean']:.2f}, std={s_np['reward_std']:.2f}")

            eval_results.append({
                "step": i,
                "reward_mean": float(s_np['reward_mean']),
                "reward_std": float(s_np['reward_std']),
                "reward_min": float(s_np['reward_min']),
                "reward_max": float(s_np['reward_max']),
            })

            if not onp.isnan(s_np['reward_mean']):
                try:
                    eval_outputs.log_to_wandb(wandb_run, step=i)
                except ValueError as e:
                    print(f"  Warning: {e}")

        # Training step
        rollout_state, transitions = rollout_state.rollout(
            agent_state,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
        )
        agent_state, metrics = agent_state.training_step(transitions)

        # Log training metrics
        log_dict = {
            "train/mean_reward": onp.mean(transitions.reward),
            **{f"train/{k}": onp.mean(v) for k, v in metrics.items()},
        }
        wandb_run.log(log_dict, step=i)

        times.append(time.time())

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    results = {
        "flow_type": flow_type,
        "seed": seed,
        "num_timesteps": num_timesteps,
        "eval_results": eval_results,
        "total_time": times[-1] - times[0],
    }

    results_path = os.path.join(save_dir, f"fpo_{flow_type}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Total training time: {times[-1] - times[0]:.1f}s")

    wandb.finish()

    # Restore original function
    fpo.FpoState._compute_cfm_loss = original_fn

    return results


def main():
    parser = argparse.ArgumentParser(description="Train FPO with different flow variants")
    parser.add_argument(
        "--flow_type",
        type=str,
        choices=["ot", "vp", "ve", "cosine"],
        default="ot",
        help="Flow schedule type",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all flow variants",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=3_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    if args.all:
        all_results = {}
        for flow_type in ["ot", "vp", "ve", "cosine"]:
            results = train_fpo_variant(
                flow_type=flow_type,
                seed=args.seed,
                num_timesteps=args.num_timesteps,
                save_dir=args.save_dir,
            )
            all_results[flow_type] = results

        combined_path = os.path.join(args.save_dir, "all_variants_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to {combined_path}")
    else:
        train_fpo_variant(
            flow_type=args.flow_type,
            seed=args.seed,
            num_timesteps=args.num_timesteps,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
