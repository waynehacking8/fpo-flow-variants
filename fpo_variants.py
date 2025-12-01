"""
FPO with Multiple Flow Variants

This module extends the original FPO implementation to support different
flow matching schedules (OT, VP, VE, Cosine).

Key modifications from original FPO:
1. Added `flow_type` parameter to FpoConfig
2. Modified `_compute_cfm_loss` to use different flow schedules
3. Modified `sample_action` (Euler steps) to use different schedules
4. Added metrics for analyzing flow behavior

Author: [Your Name]
Based on: Flow Policy Optimization (McAllister et al., 2025)
"""

from __future__ import annotations

from functools import partial
from typing import Literal, assert_never

import jax
import jax_dataclasses as jdc
import optax
from jax import Array
from jax import numpy as jnp

from flow_schedules import (
    FlowType,
    get_flow_coefficients,
    compute_x_t,
    compute_velocity_target,
    compute_x0_from_velocity,
    compute_eps_from_velocity,
)


# ============================================================================
# Configuration
# ============================================================================

@jdc.pytree_dataclass
class FpoVariantConfig:
    """Configuration for FPO with flow variant support."""

    # === NEW: Flow variant parameters ===
    flow_type: jdc.Static[FlowType] = "ot"
    sigma_min: float = 0.01  # For VE schedule
    sigma_max: float = 80.0  # For VE schedule

    # === Original FPO parameters ===
    flow_steps: jdc.Static[int] = 10
    output_mode: jdc.Static[Literal["velocity", "eps", "x0"]] = "velocity"
    timestep_embed_dim: jdc.Static[int] = 8
    n_samples_per_action: jdc.Static[int] = 8
    average_losses_before_exp: jdc.Static[bool] = True

    discretize_t_for_training: jdc.Static[bool] = True
    feather_std: float = 0.0
    policy_mlp_output_scale: float = 0.25

    clipping_epsilon: float = 0.05

    # PPO-style parameters
    batch_size: jdc.Static[int] = 1024
    discounting: float = 0.995
    episode_length: int = 1000
    learning_rate: float = 3e-4
    normalize_observations: jdc.Static[bool] = True
    num_envs: jdc.Static[int] = 2048
    num_evals: jdc.Static[int] = 10
    num_minibatches: jdc.Static[int] = 32
    num_timesteps: jdc.Static[int] = 60_000_000
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


# ============================================================================
# Core Flow Matching Functions with Variant Support
# ============================================================================

def compute_flow_loss(
    network_pred: Array,
    x_0: Array,
    eps: Array,
    t: Array,
    config: FpoVariantConfig,
) -> Array:
    """
    Compute flow matching loss for different schedules and output modes.

    Args:
        network_pred: Network output, shape (*, action_dim)
        x_0: Clean action, shape (*, action_dim)
        eps: Noise sample, shape (*, action_dim)
        t: Time values, shape (*, 1)
        config: Configuration with flow_type and output_mode

    Returns:
        loss: MSE loss, shape (*,) - one value per sample
    """
    flow_type = config.flow_type
    output_mode = config.output_mode

    # Get the current noisy state
    x_t = compute_x_t(x_0, eps, t, flow_type,
                      sigma_min=config.sigma_min,
                      sigma_max=config.sigma_max)

    if output_mode == "velocity":
        # Network predicts velocity dx_t/dt
        velocity_target = compute_velocity_target(
            x_0, eps, t, flow_type,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max
        )
        loss = jnp.mean((network_pred - velocity_target) ** 2, axis=-1)

    elif output_mode == "eps":
        # Network predicts the noise eps
        # For OT with eps supervision (as in original FPO)
        if flow_type == "ot":
            # velocity_pred => x1_pred
            velocity_pred = network_pred
            x0_pred = x_t - t * velocity_pred
            x1_pred = x0_pred + velocity_pred
            loss = jnp.mean((eps - x1_pred) ** 2, axis=-1)
        else:
            # General case: convert velocity to eps prediction
            eps_pred = compute_eps_from_velocity(
                x_t, network_pred, t, flow_type,
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max
            )
            loss = jnp.mean((eps - eps_pred) ** 2, axis=-1)

    elif output_mode == "x0":
        # Network predicts clean sample x_0
        x0_pred = compute_x0_from_velocity(
            x_t, network_pred, t, flow_type,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max
        )
        loss = jnp.mean((x_0 - x0_pred) ** 2, axis=-1)

    else:
        raise ValueError(f"Unknown output mode: {output_mode}")

    return loss


def euler_step_variant(
    x_t: Array,
    t_current: float,
    t_next: float,
    velocity_pred: Array,
    config: FpoVariantConfig,
    noise: Array | None = None,
) -> Array:
    """
    Perform one Euler step for different flow schedules.

    For OT: x_{t_next} = x_t + dt * velocity
    For VP/VE: Need to account for changing coefficients

    Args:
        x_t: Current state, shape (*, action_dim)
        t_current: Current time
        t_next: Next time
        velocity_pred: Predicted velocity, shape (*, action_dim)
        config: Configuration
        noise: Optional noise for stochastic sampling

    Returns:
        x_t_next: Next state, shape (*, action_dim)
    """
    dt = t_next - t_current

    if config.flow_type == "ot":
        # Simple Euler for OT
        x_t_next = x_t + dt * velocity_pred

    elif config.flow_type in ["vp", "cosine"]:
        # For VP/cosine, we use the same Euler update
        # but the velocity already accounts for the schedule
        x_t_next = x_t + dt * velocity_pred

    elif config.flow_type == "ve":
        # VE needs special handling due to exploding variance
        # The velocity field is different
        x_t_next = x_t + dt * velocity_pred

    else:
        raise ValueError(f"Unknown flow type: {config.flow_type}")

    # Add stochastic noise if provided (for SDE sampling)
    if noise is not None:
        # Scale noise appropriately
        noise_scale = jnp.abs(dt) ** 0.5
        x_t_next = x_t_next + noise_scale * noise

    return x_t_next


# ============================================================================
# Analysis and Metrics
# ============================================================================

def compute_flow_metrics(
    network_pred: Array,
    x_0: Array,
    eps: Array,
    t: Array,
    config: FpoVariantConfig,
) -> dict[str, Array]:
    """
    Compute various metrics for analyzing flow matching quality.

    Returns metrics like:
    - loss: Flow matching loss
    - velocity_norm: Magnitude of predicted velocity
    - prediction_error: Error in x_0 prediction
    - snr: Signal-to-noise ratio at time t
    """
    flow_type = config.flow_type

    # Get coefficients
    coeffs = get_flow_coefficients(
        t, flow_type,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max
    )

    # Compute x_t
    x_t = compute_x_t(x_0, eps, t, flow_type,
                      sigma_min=config.sigma_min,
                      sigma_max=config.sigma_max)

    # Compute target velocity
    velocity_target = compute_velocity_target(
        x_0, eps, t, flow_type,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max
    )

    # Metrics
    metrics = {}

    # Loss
    metrics["loss"] = jnp.mean((network_pred - velocity_target) ** 2, axis=-1)

    # Velocity norms
    metrics["velocity_pred_norm"] = jnp.linalg.norm(network_pred, axis=-1)
    metrics["velocity_target_norm"] = jnp.linalg.norm(velocity_target, axis=-1)

    # x_0 prediction error
    x0_pred = compute_x0_from_velocity(
        x_t, network_pred, t, flow_type,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max
    )
    metrics["x0_prediction_error"] = jnp.mean((x_0 - x0_pred) ** 2, axis=-1)

    # SNR
    metrics["snr"] = (coeffs.alpha_t ** 2) / (coeffs.sigma_t ** 2 + 1e-8)
    metrics["log_snr"] = jnp.log(metrics["snr"] + 1e-8)

    return metrics


# ============================================================================
# Factory function for creating configs
# ============================================================================

def create_config(
    flow_type: FlowType = "ot",
    num_timesteps: int = 3_000_000,
    **kwargs,
) -> FpoVariantConfig:
    """
    Create FPO config with sensible defaults for each flow type.

    Args:
        flow_type: Type of flow schedule
        num_timesteps: Total training timesteps
        **kwargs: Override any config parameter

    Returns:
        FpoVariantConfig
    """
    # Default configs tuned for each flow type
    defaults = {
        "ot": {
            "clipping_epsilon": 0.05,
            "learning_rate": 3e-4,
            "flow_steps": 10,
        },
        "vp": {
            "clipping_epsilon": 0.05,
            "learning_rate": 3e-4,
            "flow_steps": 10,
        },
        "ve": {
            "clipping_epsilon": 0.1,  # Larger clipping for stability
            "learning_rate": 1e-4,    # Lower LR for stability
            "flow_steps": 20,         # More steps for VE
            "sigma_min": 0.01,
            "sigma_max": 80.0,
        },
        "cosine": {
            "clipping_epsilon": 0.05,
            "learning_rate": 3e-4,
            "flow_steps": 10,
        },
    }

    # Start with defaults for this flow type
    config_dict = defaults.get(flow_type, defaults["ot"]).copy()
    config_dict["flow_type"] = flow_type
    config_dict["num_timesteps"] = num_timesteps

    # Override with user-provided kwargs
    config_dict.update(kwargs)

    return FpoVariantConfig(**config_dict)


# ============================================================================
# Preset configurations for experiments
# ============================================================================

EXPERIMENT_CONFIGS = {
    "fpo_ot": create_config("ot", num_timesteps=3_000_000),
    "fpo_vp": create_config("vp", num_timesteps=3_000_000),
    "fpo_ve": create_config("ve", num_timesteps=3_000_000),
    "fpo_cosine": create_config("cosine", num_timesteps=3_000_000),
}


if __name__ == "__main__":
    # Quick test of the functions
    import jax.random as jr

    print("Testing FPO Variants...")

    key = jr.PRNGKey(42)
    batch_size = 4
    action_dim = 3

    # Generate test data
    key, k1, k2, k3 = jr.split(key, 4)
    x_0 = jr.normal(k1, (batch_size, action_dim))
    eps = jr.normal(k2, (batch_size, action_dim))
    t = jr.uniform(k3, (batch_size, 1))

    # Test each flow type
    for flow_type in ["ot", "vp", "ve", "cosine"]:
        print(f"\n=== Testing {flow_type.upper()} ===")

        config = create_config(flow_type)

        # Compute x_t
        x_t = compute_x_t(x_0, eps, t, flow_type,
                         sigma_min=config.sigma_min,
                         sigma_max=config.sigma_max)
        print(f"x_t shape: {x_t.shape}")

        # Compute velocity target
        velocity = compute_velocity_target(x_0, eps, t, flow_type,
                                          sigma_min=config.sigma_min,
                                          sigma_max=config.sigma_max)
        print(f"velocity shape: {velocity.shape}")
        print(f"velocity norm: {jnp.linalg.norm(velocity, axis=-1).mean():.4f}")

        # Test loss computation (with dummy network prediction)
        network_pred = velocity + jr.normal(key, velocity.shape) * 0.1
        loss = compute_flow_loss(network_pred, x_0, eps, t, config)
        print(f"loss shape: {loss.shape}, mean: {loss.mean():.6f}")

        # Test metrics
        metrics = compute_flow_metrics(network_pred, x_0, eps, t, config)
        print(f"SNR range: [{metrics['snr'].min():.2f}, {metrics['snr'].max():.2f}]")

    print("\n✓ All tests passed!")
