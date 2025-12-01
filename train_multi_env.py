"""
Multi-Environment Training Script for FPO Flow Variants

Tests different flow schedules across multiple environments to analyze
task-dependency of flow schedule effectiveness.

Environments:
- HumanoidGetup (custom)
- Go1 (quadruped locomotion)
- Go1Getup (quadruped getup)
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
        # Quadruped getup task
        env = locomotion.go1_getup.Getup(config=locomotion.go1_getup.default_config())
        return wrap_env_for_fpo(env)
    elif env_name == "go1_joystick":
        # Quadruped locomotion with joystick control
        env = locomotion.go1_joystick.Joystick(config=locomotion.go1_joystick.default_config())
        return wrap_env_for_fpo(env)
    elif env_name == "go1_handstand":
        # Quadruped handstand task
        env = locomotion.go1_handstand.Handstand(config=locomotion.go1_handstand.default_config())
        return wrap_env_for_fpo(env)
    elif env_name == "spot_getup":
        # Spot robot getup task
        env = locomotion.spot_getup.Getup(config=locomotion.spot_getup.default_config())
        return wrap_env_for_fpo(env)
    elif env_name == "h1_walk":
        # H1 humanoid walking with gait tracking
        env = locomotion.h1_joystick_gait_tracking.JoystickGaitTracking(
            config=locomotion.h1_joystick_gait_tracking.default_config()
        )
        return wrap_env_for_fpo(env)
    else:
        raise ValueError(f"Unknown environment: {env_name}. Available: humanoid_getup, go1_getup, go1_joystick, go1_handstand, spot_getup, h1_walk")


def train_variant_on_env(
    env_name: str,
    flow_type: FlowType,
    wandb_project: str = "fpo-flow-variants-multienv",
    seed: int = 0,
    num_timesteps: int = 3_000_000,
    save_dir: str = "./results_multienv",
) -> dict:
    """Train a flow variant on a specific environment."""

    print(f"\n{'='*60}")
    print(f"Training FPO-{flow_type.upper()} on {env_name}")
    print(f"{'='*60}\n")

    # Patch FPO
    original_fn = patch_fpo_for_flow_variant(flow_type)

    # Config based on flow type
    if flow_type == "ve":
        config = fpo.FpoConfig(
            num_timesteps=num_timesteps,
            num_evals=10,
            episode_length=1000,
            num_envs=2048,
            unroll_length=30,
            learning_rate=1e-4,
            batch_size=1024,
            num_minibatches=32,
            num_updates_per_batch=16,
            clipping_epsilon=0.1,
            discounting=0.995,
            gae_lambda=0.95,
            value_loss_coeff=0.25,
            normalize_observations=True,
        )
    else:
        # Calculate outer_iters to ensure enough eval points
        # iterations_per_env = 480 (16 updates * 30 unroll), so each outer iter = 480 * 2048 = ~1M timesteps
        # For 10 eval points, need at least 10 outer iterations = 10M timesteps
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

    # Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"fpo_{flow_type}_{env_name}_{timestamp}"

    wandb_run = wandb.init(
        project=wandb_project,
        name=exp_name,
        config={
            "env_name": env_name,
            "flow_type": flow_type,
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

    # Training
    outer_iters = config.num_timesteps // (config.iterations_per_env * config.num_envs)
    eval_iters = set(onp.linspace(0, outer_iters - 1, config.num_evals, dtype=int))

    eval_results = []
    times = [time.time()]

    for i in tqdm(range(outer_iters), desc=f"{env_name}-{flow_type.upper()}"):
        if i in eval_iters:
            eval_outputs = rollouts.eval_policy(
                agent_state,
                prng=jax.random.fold_in(agent_state.prng, i),
                num_envs=128,
                max_episode_length=config.episode_length,
            )

            s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}

            print(f"\n[{env_name}-{flow_type.upper()}] Step {i}: reward={s_np['reward_mean']:.2f}")

            eval_results.append({
                "step": i,
                "reward_mean": float(s_np['reward_mean']),
                "reward_std": float(s_np['reward_std']),
            })

            if not onp.isnan(s_np['reward_mean']):
                try:
                    eval_outputs.log_to_wandb(wandb_run, step=i)
                except:
                    pass

        rollout_state, transitions = rollout_state.rollout(
            agent_state,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
        )
        agent_state, metrics = agent_state.training_step(transitions)

        log_dict = {
            "train/mean_reward": onp.mean(transitions.reward),
            **{f"train/{k}": onp.mean(v) for k, v in metrics.items()},
        }
        wandb_run.log(log_dict, step=i)

        times.append(time.time())

    # Save
    os.makedirs(save_dir, exist_ok=True)
    results = {
        "env_name": env_name,
        "flow_type": flow_type,
        "seed": seed,
        "eval_results": eval_results,
        "total_time": times[-1] - times[0],
    }

    results_path = os.path.join(save_dir, f"{env_name}_{flow_type}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")
    wandb.finish()

    fpo.FpoState._compute_cfm_loss = original_fn
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="go1_getup",
                       choices=["humanoid_getup", "go1_getup", "go1_joystick", "go1_handstand", "spot_getup", "h1_walk"])
    parser.add_argument("--flow_type", type=str, default="ot",
                       choices=["ot", "vp", "ve", "cosine"])
    parser.add_argument("--all_flows", action="store_true")
    parser.add_argument("--all_envs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_timesteps", type=int, default=3_000_000)
    parser.add_argument("--save_dir", type=str, default="./results_multienv")

    args = parser.parse_args()

    # Three diverse environments: humanoid getup, quadruped getup, quadruped locomotion
    envs = ["humanoid_getup", "go1_getup", "go1_joystick"] if args.all_envs else [args.env]
    flows = ["ot", "vp", "ve", "cosine"] if args.all_flows else [args.flow_type]

    all_results = {}
    for env_name in envs:
        all_results[env_name] = {}
        for flow_type in flows:
            try:
                results = train_variant_on_env(
                    env_name=env_name,
                    flow_type=flow_type,
                    seed=args.seed,
                    num_timesteps=args.num_timesteps,
                    save_dir=args.save_dir,
                )
                all_results[env_name][flow_type] = results
            except Exception as e:
                print(f"Error training {env_name}-{flow_type}: {e}")
                all_results[env_name][flow_type] = {"error": str(e)}

    # Save combined
    os.makedirs(args.save_dir, exist_ok=True)
    combined_path = os.path.join(args.save_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
