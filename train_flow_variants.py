"""
Training Script for FPO Flow Variants Comparison

This script trains FPO with different flow schedules (OT, VP, VE, Cosine)
on the HumanoidGetup task and compares their performance.

Usage:
    python train_flow_variants.py --flow_type ot
    python train_flow_variants.py --flow_type vp
    python train_flow_variants.py --flow_type ve
    python train_flow_variants.py --flow_type cosine
    python train_flow_variants.py --all  # Run all variants
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Literal

import jax
import jax_dataclasses as jdc
import numpy as onp
import wandb
from jax import Array
from jax import numpy as jnp
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/home/wayneleo8/Desktop/fpo_official/playground/src')
sys.path.insert(0, '/home/wayneleo8/Desktop/fpo-humanoid-getup')

from flow_policy import rollouts
from flow_policy.networks import MlpWeights, mlp_init, flow_mlp_fwd, value_mlp_fwd
from flow_policy.math_utils import RunningStats

from flow_schedules import (
    FlowType,
    get_flow_coefficients,
    compute_x_t,
    compute_velocity_target,
)

# Import environment
from humanoid_getup import HumanoidGetup, default_config


# ============================================================================
# FPO with Flow Variants - Full Implementation
# ============================================================================

@jdc.pytree_dataclass
class FpoVariantConfig:
    """Configuration for FPO with flow variant support."""

    # Flow variant parameters
    flow_type: jdc.Static[FlowType] = "ot"
    sigma_min: float = 0.01
    sigma_max: float = 80.0

    # Flow parameters
    flow_steps: jdc.Static[int] = 10
    output_mode: jdc.Static[Literal["velocity", "eps"]] = "velocity"
    timestep_embed_dim: jdc.Static[int] = 8
    n_samples_per_action: jdc.Static[int] = 8
    average_losses_before_exp: jdc.Static[bool] = True

    discretize_t_for_training: jdc.Static[bool] = True
    feather_std: float = 0.0
    policy_mlp_output_scale: float = 0.25

    clipping_epsilon: float = 0.05

    # PPO parameters
    batch_size: jdc.Static[int] = 1024
    discounting: float = 0.995
    episode_length: int = 1000
    learning_rate: float = 3e-4
    normalize_observations: jdc.Static[bool] = True
    num_envs: jdc.Static[int] = 2048
    num_evals: jdc.Static[int] = 10
    num_minibatches: jdc.Static[int] = 32
    num_timesteps: jdc.Static[int] = 3_000_000
    num_updates_per_batch: jdc.Static[int] = 16
    reward_scaling: float = 10.0
    unroll_length: jdc.Static[int] = 30

    gae_lambda: float = 0.95
    normalize_advantage: jdc.Static[bool] = True
    value_loss_coeff: float = 0.25

    def __post_init__(self) -> None:
        assert self.timestep_embed_dim % 2 == 0

    @property
    def iterations_per_env(self) -> int:
        return (
            self.num_minibatches * self.batch_size * self.unroll_length
        ) // self.num_envs


@jdc.pytree_dataclass
class FpoParams:
    policy: MlpWeights
    value: MlpWeights


@jdc.pytree_dataclass
class FpoActionInfo:
    loss_eps: Array
    loss_t: Array
    initial_cfm_loss: Array


@jdc.pytree_dataclass
class FlowSchedule:
    t_current: Array
    t_next: Array


FpoTransition = rollouts.TransitionStruct[FpoActionInfo]


@jdc.pytree_dataclass
class FpoVariantState:
    """FPO agent state with flow variant support."""

    env: jdc.Static
    config: FpoVariantConfig
    params: FpoParams
    obs_stats: RunningStats

    opt: jdc.Static
    opt_state: jdc.Static

    prng: Array
    steps: Array

    @staticmethod
    def init(prng: Array, env, config: FpoVariantConfig) -> FpoVariantState:
        import optax

        obs_size = env.observation_size
        action_size = env.action_size
        assert isinstance(obs_size, int)

        prng0, prng1, prng2 = jax.random.split(prng, num=3)

        # Policy network: obs + action + timestep_embed -> action
        actor_net = mlp_init(
            prng0,
            (
                obs_size + action_size + config.timestep_embed_dim,
                64, 64, 64, 64,
                action_size,
            ),
        )

        # Value network: obs -> scalar
        critic_net = mlp_init(
            prng1,
            (obs_size, 256, 256, 256, 256, 1),
        )

        network_params = FpoParams(actor_net, critic_net)
        opt = optax.scale_by_adam()

        return FpoVariantState(
            env=env,
            config=config,
            params=network_params,
            obs_stats=RunningStats.init((obs_size,)),
            opt=opt,
            opt_state=opt.init(network_params),
            prng=prng2,
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    def get_schedule(self) -> FlowSchedule:
        """Get flow timestep schedule from t=1 (noise) to t=0 (data)."""
        full_t_path = jnp.linspace(1.0, 0.0, self.config.flow_steps + 1)
        return FlowSchedule(
            t_current=full_t_path[:-1],
            t_next=full_t_path[1:],
        )

    def embed_timestep(self, t: Array) -> Array:
        """Sinusoidal timestep embedding."""
        assert t.shape[-1] == 1
        freqs = 2 ** jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * freqs
        return jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)

    def _compute_cfm_loss(
        self,
        obs_norm: Array,
        action: Array,
        eps: Array,
        t: Array,
    ) -> Array:
        """Compute CFM loss with flow variant support."""
        (*batch_dims, action_dim) = action.shape
        samples_dim = self.config.n_samples_per_action
        obs_dim = self.env.observation_size
        sample_shape = (*batch_dims, samples_dim)

        # Get flow coefficients for the current schedule
        coeffs = get_flow_coefficients(
            t, self.config.flow_type,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
        )

        # Compute x_t using the schedule-specific interpolation
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

        # Compute loss based on output mode and flow type
        if self.config.output_mode == "velocity":
            # Target velocity = d_alpha_dt * action + d_sigma_dt * eps
            velocity_target = (
                coeffs.d_alpha_dt * action[..., None, :]
                + coeffs.d_sigma_dt * eps
            )
            loss = jnp.mean((network_pred - velocity_target) ** 2, axis=-1)

        elif self.config.output_mode == "eps":
            # For OT: eps supervision (original FPO style)
            if self.config.flow_type == "ot":
                velocity_pred = network_pred
                x0_pred = x_t - t * velocity_pred
                x1_pred = x0_pred + velocity_pred
                loss = jnp.mean((eps - x1_pred) ** 2, axis=-1)
            else:
                # For other schedules: use velocity target
                velocity_target = (
                    coeffs.d_alpha_dt * action[..., None, :]
                    + coeffs.d_sigma_dt * eps
                )
                loss = jnp.mean((network_pred - velocity_target) ** 2, axis=-1)

        else:
            raise ValueError(f"Unknown output mode: {self.config.output_mode}")

        return loss

    def sample_action(
        self,
        obs: Array,
        prng: Array,
        deterministic: bool = False,
    ) -> tuple[Array, FpoActionInfo]:
        """Sample action using the flow model."""
        if self.config.normalize_observations:
            obs_norm = (obs - self.obs_stats.mean) / self.obs_stats.std
        else:
            obs_norm = obs

        (*batch_dims, obs_dim) = obs.shape
        action_size = self.env.action_size

        def euler_step(carry: Array, inputs: tuple[FlowSchedule, Array]) -> tuple[Array, Array]:
            x_t = carry
            schedule_t, noise = inputs

            dt = schedule_t.t_next - schedule_t.t_current

            # Get velocity prediction
            velocity = (
                flow_mlp_fwd(
                    self.params.policy,
                    obs_norm,
                    x_t,
                    jnp.broadcast_to(
                        self.embed_timestep(schedule_t.t_current[None]),
                        (*batch_dims, self.config.timestep_embed_dim),
                    ),
                )
                * self.config.policy_mlp_output_scale
            )

            # Euler step (same for all schedules)
            x_t_next = x_t + dt * velocity

            return x_t_next, x_t

        prng_sample, prng_loss, prng_feather, prng_noise = jax.random.split(prng, num=4)

        # Generate noise for each step
        noise_path = jax.random.normal(
            prng_noise,
            (self.config.flow_steps, *batch_dims, action_size),
        )

        # Run flow from t=1 (noise) to t=0 (action)
        x0, _ = jax.lax.scan(
            euler_step,
            init=jax.random.normal(prng_sample, (*batch_dims, action_size)),
            xs=(self.get_schedule(), noise_path),
        )

        if not deterministic:
            perturb = (
                jax.random.normal(prng_feather, (*batch_dims, action_size))
                * self.config.feather_std
            )
            x0 = x0 + perturb

        # Compute initial CFM loss for FPO ratio
        sample_shape = (*batch_dims, self.config.n_samples_per_action)
        prng_eps, prng_t = jax.random.split(prng_loss)
        eps = jax.random.normal(prng_eps, (*sample_shape, action_size))

        if self.config.discretize_t_for_training:
            t = self.get_schedule().t_current[
                jax.random.randint(
                    prng_t,
                    shape=(*sample_shape, 1),
                    minval=0,
                    maxval=self.config.flow_steps,
                )
            ]
        else:
            t = jax.random.uniform(prng_t, (*sample_shape, 1))

        initial_cfm_loss = self._compute_cfm_loss(obs_norm, x0, eps=eps, t=t)

        return x0, FpoActionInfo(
            loss_eps=eps,
            loss_t=t,
            initial_cfm_loss=initial_cfm_loss,
        )

    def get_value(self, obs: Array) -> Array:
        """Get value estimate for observation."""
        if self.config.normalize_observations:
            obs_norm = (obs - self.obs_stats.mean) / self.obs_stats.std
        else:
            obs_norm = obs
        return value_mlp_fwd(self.params.value, obs_norm)

    @jdc.jit
    def training_step(
        self, transitions: FpoTransition
    ) -> tuple[FpoVariantState, dict[str, Array]]:
        """One training step over transitions."""
        config = self.config
        assert transitions.reward.shape == (config.iterations_per_env, config.num_envs)

        state = self
        if config.normalize_observations:
            with jdc.copy_and_mutate(state) as state:
                state.obs_stats = state.obs_stats.update(transitions.obs)
        del self

        def step_batch(state: FpoVariantState, _):
            step_prng = jax.random.fold_in(state.prng, state.steps)
            state, metrics = jax.lax.scan(
                partial(
                    FpoVariantState._step_minibatch, prng=jax.random.fold_in(step_prng, 0)
                ),
                init=state,
                xs=transitions.prepare_minibatches(
                    step_prng, config.num_minibatches, config.batch_size
                ),
            )
            return state, metrics

        state, metrics = jax.lax.scan(
            step_batch,
            init=state,
            length=config.num_updates_per_batch,
        )

        return state, metrics

    def _step_minibatch(
        self, transitions: FpoTransition, prng: Array
    ) -> tuple[FpoVariantState, dict[str, Array]]:
        """One training step over a minibatch."""
        (loss, metrics), grads = jax.value_and_grad(
            lambda params: FpoVariantState._compute_loss(
                jdc.replace(self, params=params),
                transitions,
                prng,
            ),
            has_aux=True,
        )(self.params)

        param_update, new_opt_state = self.opt.update(grads, self.opt_state)
        param_update = jax.tree.map(
            lambda x: -self.config.learning_rate * x, param_update
        )
        with jdc.copy_and_mutate(self) as state:
            state.params = jax.tree.map(jnp.add, self.params, param_update)
            state.opt_state = new_opt_state
            state.steps = state.steps + 1
        return state, metrics

    def _compute_loss(
        self, transitions: FpoTransition, prng: Array
    ) -> tuple[Array, dict[str, Array]]:
        """Compute FPO loss."""
        del prng

        (timesteps, batch_dim) = transitions.reward.shape
        metrics = dict[str, Array]()

        if self.config.normalize_observations:
            obs_norm = (transitions.obs - self.obs_stats.mean) / self.obs_stats.std
        else:
            obs_norm = transitions.obs

        value_pred = value_mlp_fwd(self.params.value, obs_norm)

        bootstrap_obs_norm = (
            transitions.next_obs[-1:, :, :] - self.obs_stats.mean
        ) / self.obs_stats.std
        bootstrap_value = value_mlp_fwd(self.params.value, bootstrap_obs_norm)

        gae_vs, gae_advantages = jax.lax.stop_gradient(
            rollouts.compute_gae(
                truncation=transitions.truncation,
                discount=transitions.discount * self.config.discounting,
                rewards=transitions.reward * self.config.reward_scaling,
                values=value_pred,
                bootstrap_value=bootstrap_value,
                gae_lambda=self.config.gae_lambda,
            )
        )

        metrics["advantages_mean"] = jnp.mean(gae_advantages)
        metrics["advantages_std"] = jnp.std(gae_advantages)

        if self.config.normalize_advantage:
            gae_advantages = (gae_advantages - gae_advantages.mean()) / (
                gae_advantages.std() + 1e-8
            )

        # Compute CFM loss
        cfm_loss = self._compute_cfm_loss(
            obs_norm,
            transitions.action,
            eps=transitions.action_info.loss_eps,
            t=transitions.action_info.loss_t,
        )

        # Compute FPO ratio
        if self.config.average_losses_before_exp:
            rho_s = jnp.exp(
                jnp.mean(transitions.action_info.initial_cfm_loss, axis=-1, keepdims=True)
                - jnp.mean(cfm_loss, axis=-1, keepdims=True)
            )
        else:
            rho_s = jnp.exp(
                jnp.clip(
                    transitions.action_info.initial_cfm_loss - cfm_loss, -3.0, 3.0
                )
            )

        # PPO loss
        surrogate_loss1 = rho_s * gae_advantages[..., None]
        surrogate_loss2 = (
            jnp.clip(
                rho_s,
                1 - self.config.clipping_epsilon,
                1 + self.config.clipping_epsilon,
            )
            * gae_advantages[..., None]
        )

        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

        # Metrics
        metrics["policy_ratio_mean"] = jnp.mean(rho_s)
        metrics["policy_loss"] = policy_loss
        metrics["cfm_loss_mean"] = jnp.mean(cfm_loss)

        # Value loss
        v_error = (gae_vs - value_pred) * (1 - transitions.truncation)
        v_loss = jnp.mean(v_error**2) * self.config.value_loss_coeff
        metrics["v_loss"] = v_loss

        total_loss = policy_loss + v_loss
        return total_loss, metrics


# ============================================================================
# Factory Functions
# ============================================================================

def create_config(
    flow_type: FlowType = "ot",
    num_timesteps: int = 3_000_000,
    **kwargs,
) -> FpoVariantConfig:
    """Create FPO config with sensible defaults for each flow type."""
    defaults = {
        "ot": {
            "clipping_epsilon": 0.05,
            "learning_rate": 3e-4,
            "flow_steps": 10,
            "output_mode": "eps",
        },
        "vp": {
            "clipping_epsilon": 0.05,
            "learning_rate": 3e-4,
            "flow_steps": 10,
            "output_mode": "velocity",
        },
        "ve": {
            "clipping_epsilon": 0.1,
            "learning_rate": 1e-4,
            "flow_steps": 20,
            "sigma_min": 0.01,
            "sigma_max": 80.0,
            "output_mode": "velocity",
        },
        "cosine": {
            "clipping_epsilon": 0.05,
            "learning_rate": 3e-4,
            "flow_steps": 10,
            "output_mode": "velocity",
        },
    }

    config_dict = defaults.get(flow_type, defaults["ot"]).copy()
    config_dict["flow_type"] = flow_type
    config_dict["num_timesteps"] = num_timesteps
    config_dict.update(kwargs)

    return FpoVariantConfig(**config_dict)


# ============================================================================
# Training Function
# ============================================================================

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

    # Create config
    config = create_config(flow_type, num_timesteps=num_timesteps)

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
    agent_state = FpoVariantState.init(prng=jax.random.key(seed), env=env, config=config)
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
    return results


# ============================================================================
# Main
# ============================================================================

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
        "--wandb_project",
        type=str,
        default="fpo-flow-variants",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    if args.all:
        # Run all variants
        all_results = {}
        for flow_type in ["ot", "vp", "ve", "cosine"]:
            results = train_fpo_variant(
                flow_type=flow_type,
                seed=args.seed,
                num_timesteps=args.num_timesteps,
                wandb_project=args.wandb_project,
                save_dir=args.save_dir,
            )
            all_results[flow_type] = results

        # Save combined results
        combined_path = os.path.join(args.save_dir, "all_variants_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to {combined_path}")

    else:
        # Run single variant
        train_fpo_variant(
            flow_type=args.flow_type,
            seed=args.seed,
            num_timesteps=args.num_timesteps,
            wandb_project=args.wandb_project,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
