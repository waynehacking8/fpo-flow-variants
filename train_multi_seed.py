"""
Multi-Seed Training Script for Statistical Analysis

Runs each flow variant with multiple random seeds to compute
mean and standard deviation for statistical significance.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from functools import partial
from typing import Literal

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

import mujoco_playground as mjp
from mujoco_playground import locomotion

# Import flow schedules
from flow_schedules import FlowType, get_flow_coefficients
from env_wrappers import wrap_env_for_fpo


def patch_fpo_for_flow_variant(flow_type: FlowType, sigma_min: float = 0.01, sigma_max: float = 80.0):
    """Monkey-patch FPO's _compute_cfm_loss for different flow schedules."""
    def patched_compute_cfm_loss(self, obs_norm, action, eps, t):
        (*batch_dims, action_dim) = action.shape
        samples_dim = self.config.n_samples_per_action
        obs_dim = self.env.observation_size
        sample_shape = (*batch_dims, samples_dim)

        coeffs = get_flow_coefficients(t, flow_type, sigma_min=sigma_min, sigma_max=sigma_max)
        x_t = coeffs.alpha_t * action[..., None, :] + coeffs.sigma_t * eps

        network_pred = (
            flow_mlp_fwd(
                self.params.policy,
                jnp.broadcast_to(obs_norm[..., None, :], (*sample_shape, obs_dim)),
                x_t,
                self.embed_timestep(t),
            )
            * self.config.policy_mlp_output_scale
        )

        if flow_type == "ot":
            velocity_pred = network_pred
            x0_pred = x_t - t * velocity_pred
            x1_pred = x0_pred + velocity_pred
            out = jnp.mean((eps - x1_pred) ** 2, axis=-1)
        else:
            velocity_target = (
                coeffs.d_alpha_dt * action[..., None, :]
                + coeffs.d_sigma_dt * eps
            )
            out = jnp.mean((network_pred - velocity_target) ** 2, axis=-1)

        return out

    original_fn = fpo.FpoState._compute_cfm_loss
    fpo.FpoState._compute_cfm_loss = patched_compute_cfm_loss
    return original_fn


def load_environment(env_name: str):
    """Load environment by name and wrap for FPO compatibility."""
    if env_name == "humanoid_getup":
        from humanoid_getup import HumanoidGetup, default_config
        return HumanoidGetup(config=default_config())
    elif env_name == "go1_getup":
        env = locomotion.go1_getup.Getup(config=locomotion.go1_getup.default_config())
        return wrap_env_for_fpo(env)
    elif env_name == "go1_joystick":
        env = locomotion.go1_joystick.Joystick(config=locomotion.go1_joystick.default_config())
        return wrap_env_for_fpo(env)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def train_single_run(
    env_name: str,
    flow_type: FlowType,
    seed: int,
    num_timesteps: int = 10_000_000,
    save_dir: str = "./results_multiseed",
) -> dict:
    """Train a single run with given seed."""

    print(f"\n{'='*60}")
    print(f"Training FPO-{flow_type.upper()} on {env_name} (seed={seed})")
    print(f"{'='*60}\n")

    # Patch FPO
    original_fn = patch_fpo_for_flow_variant(flow_type)

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
    env = load_environment(env_name)

    # Initialize with specific seed
    agent_state = fpo.FpoState.init(prng=jax.random.key(seed), env=env, config=config)
    rollout_state = rollouts.BatchedRolloutState.init(
        env,
        prng=jax.random.key(seed),
        num_envs=config.num_envs,
    )

    # Training
    outer_iters = config.num_timesteps // (config.iterations_per_env * config.num_envs)
    eval_iters = set(onp.linspace(0, outer_iters - 1, config.num_evals, dtype=int))

    eval_results = []
    times = [time.time()]

    for i in tqdm(range(outer_iters), desc=f"{env_name}-{flow_type.upper()}-seed{seed}"):
        if i in eval_iters:
            eval_outputs = rollouts.eval_policy(
                agent_state,
                prng=jax.random.fold_in(agent_state.prng, i),
                num_envs=128,
                max_episode_length=config.episode_length,
            )

            s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}

            eval_results.append({
                "step": i,
                "reward_mean": float(s_np['reward_mean']),
                "reward_std": float(s_np['reward_std']),
            })

        rollout_state, transitions = rollout_state.rollout(
            agent_state,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
        )
        agent_state, metrics = agent_state.training_step(transitions)
        times.append(time.time())

    # Save
    os.makedirs(save_dir, exist_ok=True)
    results = {
        "env_name": env_name,
        "flow_type": flow_type,
        "seed": seed,
        "eval_results": eval_results,
        "final_reward": eval_results[-1]["reward_mean"] if eval_results else None,
        "total_time": times[-1] - times[0],
    }

    results_path = os.path.join(save_dir, f"{env_name}_{flow_type}_seed{seed}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")

    fpo.FpoState._compute_cfm_loss = original_fn
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="humanoid_getup",
                       choices=["humanoid_getup", "go1_getup", "go1_joystick"])
    parser.add_argument("--flow_type", type=str, default="ot",
                       choices=["ot", "vp", "cosine"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num_timesteps", type=int, default=10_000_000)
    parser.add_argument("--save_dir", type=str, default="./results_multiseed")

    args = parser.parse_args()

    all_results = []
    for seed in args.seeds:
        try:
            results = train_single_run(
                env_name=args.env,
                flow_type=args.flow_type,
                seed=seed,
                num_timesteps=args.num_timesteps,
                save_dir=args.save_dir,
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error training seed {seed}: {e}")

    # Compute statistics
    if all_results:
        final_rewards = [r["final_reward"] for r in all_results if r["final_reward"] is not None]
        if final_rewards:
            mean_reward = onp.mean(final_rewards)
            std_reward = onp.std(final_rewards)
            print(f"\n{'='*60}")
            print(f"MULTI-SEED STATISTICS for {args.env} - {args.flow_type.upper()}")
            print(f"{'='*60}")
            print(f"Seeds: {args.seeds}")
            print(f"Final Rewards: {final_rewards}")
            print(f"Mean: {mean_reward:.4f}")
            print(f"Std:  {std_reward:.4f}")
            print(f"{'='*60}")

            # Save aggregate statistics
            stats = {
                "env_name": args.env,
                "flow_type": args.flow_type,
                "seeds": args.seeds,
                "final_rewards": final_rewards,
                "mean": float(mean_reward),
                "std": float(std_reward),
            }
            stats_path = os.path.join(args.save_dir, f"{args.env}_{args.flow_type}_stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
