"""
Flow Schedule Variants for Policy Optimization

This module implements different flow/diffusion schedules that can be used
with Flow Policy Optimization (FPO). Each schedule defines how to interpolate
between noise (t=1) and data (t=0).

Schedules implemented:
1. Optimal Transport (OT) - Linear interpolation (default in FPO)
2. Variance Preserving (VP) - Cosine schedule, similar to DDPM
3. Variance Exploding (VE) - Exponential noise schedule
4. Cosine - Improved DDPM cosine schedule

References:
- Flow Matching: Lipman et al. 2023
- DDPM: Ho et al. 2020
- Score SDE: Song et al. 2021
- Improved DDPM: Nichol & Dhariwal 2021
"""

from typing import Literal, NamedTuple
import jax.numpy as jnp
from jax import Array


class FlowCoefficients(NamedTuple):
    """Coefficients for the flow interpolation x_t = alpha_t * x_0 + sigma_t * eps"""
    alpha_t: Array  # Coefficient for clean data (x_0 / action)
    sigma_t: Array  # Coefficient for noise (eps)
    d_alpha_dt: Array  # Time derivative of alpha_t
    d_sigma_dt: Array  # Time derivative of sigma_t


FlowType = Literal["ot", "vp", "ve", "cosine"]


def get_flow_coefficients(
    t: Array,
    flow_type: FlowType,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
) -> FlowCoefficients:
    """
    Get flow interpolation coefficients for different schedule types.

    The forward process is: x_t = alpha_t * x_0 + sigma_t * eps
    where t=0 is clean data and t=1 is pure noise.

    Args:
        t: Time values in [0, 1], shape (*, 1) or (*,)
        flow_type: Type of flow schedule
        sigma_min: Minimum sigma for VE schedule
        sigma_max: Maximum sigma for VE schedule

    Returns:
        FlowCoefficients with alpha_t, sigma_t and their derivatives
    """
    # Ensure t has the right shape
    if t.ndim == 0:
        t = t[None]

    if flow_type == "ot":
        # Optimal Transport: Linear interpolation
        # x_t = (1-t) * x_0 + t * eps
        alpha_t = 1.0 - t
        sigma_t = t
        d_alpha_dt = -jnp.ones_like(t)
        d_sigma_dt = jnp.ones_like(t)

    elif flow_type == "vp":
        # Variance Preserving: Cosine schedule
        # Ensures alpha_t^2 + sigma_t^2 = 1
        # x_t = cos(t * pi/2) * x_0 + sin(t * pi/2) * eps
        alpha_t = jnp.cos(t * jnp.pi / 2)
        sigma_t = jnp.sin(t * jnp.pi / 2)
        d_alpha_dt = -jnp.pi / 2 * jnp.sin(t * jnp.pi / 2)
        d_sigma_dt = jnp.pi / 2 * jnp.cos(t * jnp.pi / 2)

    elif flow_type == "ve":
        # Variance Exploding: Exponential sigma growth
        # x_t = x_0 + sigma_t * eps, where sigma grows exponentially
        alpha_t = jnp.ones_like(t)
        sigma_t = sigma_min * (sigma_max / sigma_min) ** t
        d_alpha_dt = jnp.zeros_like(t)
        d_sigma_dt = sigma_t * jnp.log(sigma_max / sigma_min)

    elif flow_type == "cosine":
        # Improved DDPM cosine schedule
        # More gradual noise addition at the start
        s = 0.008  # Small offset to prevent singularity
        f_t = jnp.cos((t + s) / (1 + s) * jnp.pi / 2) ** 2
        f_0 = jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
        alpha_t_sq = f_t / f_0
        alpha_t = jnp.sqrt(jnp.clip(alpha_t_sq, 1e-8, 1.0))
        sigma_t = jnp.sqrt(jnp.clip(1.0 - alpha_t_sq, 1e-8, 1.0))

        # Numerical derivatives (could compute analytically but this is cleaner)
        eps = 1e-4
        t_plus = jnp.clip(t + eps, 0, 1)
        t_minus = jnp.clip(t - eps, 0, 1)

        f_plus = jnp.cos((t_plus + s) / (1 + s) * jnp.pi / 2) ** 2
        f_minus = jnp.cos((t_minus + s) / (1 + s) * jnp.pi / 2) ** 2
        alpha_plus = jnp.sqrt(jnp.clip(f_plus / f_0, 1e-8, 1.0))
        alpha_minus = jnp.sqrt(jnp.clip(f_minus / f_0, 1e-8, 1.0))

        d_alpha_dt = (alpha_plus - alpha_minus) / (2 * eps)
        d_sigma_dt = -d_alpha_dt * alpha_t / (sigma_t + 1e-8)

    else:
        raise ValueError(f"Unknown flow type: {flow_type}")

    return FlowCoefficients(
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        d_alpha_dt=d_alpha_dt,
        d_sigma_dt=d_sigma_dt,
    )


def compute_x_t(
    x_0: Array,
    eps: Array,
    t: Array,
    flow_type: FlowType,
    **kwargs,
) -> Array:
    """
    Compute noisy sample x_t given clean sample x_0 and noise eps.

    x_t = alpha_t * x_0 + sigma_t * eps

    Args:
        x_0: Clean samples (actions), shape (*, action_dim)
        eps: Noise samples, shape (*, action_dim)
        t: Time values, shape (*, 1)
        flow_type: Type of flow schedule

    Returns:
        x_t: Noisy samples, shape (*, action_dim)
    """
    coeffs = get_flow_coefficients(t, flow_type, **kwargs)
    # Handle broadcasting for different shapes
    alpha = coeffs.alpha_t
    sigma = coeffs.sigma_t

    # Ensure proper broadcasting
    if alpha.shape[-1] == 1 and x_0.shape[-1] != 1:
        pass  # Broadcasting will handle it

    return alpha * x_0 + sigma * eps


def compute_velocity_target(
    x_0: Array,
    eps: Array,
    t: Array,
    flow_type: FlowType,
    **kwargs,
) -> Array:
    """
    Compute the target velocity for flow matching training.

    The velocity is defined as dx_t/dt = d_alpha_dt * x_0 + d_sigma_dt * eps

    For OT: velocity = eps - x_0 (constant velocity pointing from x_0 to eps)
    For VP/VE/Cosine: velocity depends on time derivatives of coefficients

    Args:
        x_0: Clean samples (actions), shape (*, action_dim)
        eps: Noise samples, shape (*, action_dim)
        t: Time values, shape (*, 1)
        flow_type: Type of flow schedule

    Returns:
        velocity: Target velocity, shape (*, action_dim)
    """
    coeffs = get_flow_coefficients(t, flow_type, **kwargs)

    # velocity = d(alpha_t * x_0 + sigma_t * eps)/dt
    #          = d_alpha_dt * x_0 + d_sigma_dt * eps
    velocity = coeffs.d_alpha_dt * x_0 + coeffs.d_sigma_dt * eps

    return velocity


def compute_x0_from_velocity(
    x_t: Array,
    velocity: Array,
    t: Array,
    flow_type: FlowType,
    **kwargs,
) -> Array:
    """
    Recover x_0 prediction from x_t and predicted velocity.

    For OT: x_0 = x_t - t * velocity
    For other schedules: requires solving the ODE relationship

    Args:
        x_t: Noisy samples, shape (*, action_dim)
        velocity: Predicted velocity, shape (*, action_dim)
        t: Time values, shape (*, 1)
        flow_type: Type of flow schedule

    Returns:
        x_0_pred: Predicted clean samples, shape (*, action_dim)
    """
    coeffs = get_flow_coefficients(t, flow_type, **kwargs)

    if flow_type == "ot":
        # For OT: x_t = (1-t)*x_0 + t*eps, velocity = eps - x_0
        # So: x_0 = x_t - t * velocity
        x_0_pred = x_t - t * velocity
    else:
        # General case: need to solve for x_0
        # x_t = alpha_t * x_0 + sigma_t * eps
        # velocity = d_alpha_dt * x_0 + d_sigma_dt * eps
        #
        # From velocity: eps = (velocity - d_alpha_dt * x_0) / d_sigma_dt
        # Substitute into x_t equation:
        # x_t = alpha_t * x_0 + sigma_t * (velocity - d_alpha_dt * x_0) / d_sigma_dt
        # x_t = alpha_t * x_0 + sigma_t * velocity / d_sigma_dt - sigma_t * d_alpha_dt * x_0 / d_sigma_dt
        # x_t = x_0 * (alpha_t - sigma_t * d_alpha_dt / d_sigma_dt) + sigma_t * velocity / d_sigma_dt
        # x_0 = (x_t - sigma_t * velocity / d_sigma_dt) / (alpha_t - sigma_t * d_alpha_dt / d_sigma_dt)

        d_sigma_dt_safe = coeffs.d_sigma_dt + 1e-8 * jnp.sign(coeffs.d_sigma_dt + 1e-10)
        denominator = coeffs.alpha_t - coeffs.sigma_t * coeffs.d_alpha_dt / d_sigma_dt_safe
        denominator_safe = denominator + 1e-8 * jnp.sign(denominator + 1e-10)

        x_0_pred = (x_t - coeffs.sigma_t * velocity / d_sigma_dt_safe) / denominator_safe

    return x_0_pred


def compute_eps_from_velocity(
    x_t: Array,
    velocity: Array,
    t: Array,
    flow_type: FlowType,
    **kwargs,
) -> Array:
    """
    Recover eps prediction from x_t and predicted velocity.

    Args:
        x_t: Noisy samples, shape (*, action_dim)
        velocity: Predicted velocity, shape (*, action_dim)
        t: Time values, shape (*, 1)
        flow_type: Type of flow schedule

    Returns:
        eps_pred: Predicted noise, shape (*, action_dim)
    """
    coeffs = get_flow_coefficients(t, flow_type, **kwargs)

    if flow_type == "ot":
        # For OT: velocity = eps - x_0, x_t = (1-t)*x_0 + t*eps
        # x_0 = x_t - t * velocity
        # eps = velocity + x_0 = velocity + x_t - t * velocity = x_t + (1-t) * velocity
        x_0_pred = x_t - t * velocity
        eps_pred = velocity + x_0_pred
    else:
        # General: eps = (velocity - d_alpha_dt * x_0) / d_sigma_dt
        x_0_pred = compute_x0_from_velocity(x_t, velocity, t, flow_type, **kwargs)
        d_sigma_dt_safe = coeffs.d_sigma_dt + 1e-8 * jnp.sign(coeffs.d_sigma_dt + 1e-10)
        eps_pred = (velocity - coeffs.d_alpha_dt * x_0_pred) / d_sigma_dt_safe

    return eps_pred


# ============================================================================
# Utility functions for analysis and visualization
# ============================================================================

def get_snr(t: Array, flow_type: FlowType, **kwargs) -> Array:
    """
    Compute Signal-to-Noise Ratio at time t.

    SNR = alpha_t^2 / sigma_t^2

    This is useful for understanding the noise level at different timesteps.
    """
    coeffs = get_flow_coefficients(t, flow_type, **kwargs)
    return (coeffs.alpha_t ** 2) / (coeffs.sigma_t ** 2 + 1e-8)


def get_log_snr(t: Array, flow_type: FlowType, **kwargs) -> Array:
    """Compute log Signal-to-Noise Ratio."""
    return jnp.log(get_snr(t, flow_type, **kwargs) + 1e-8)


def visualize_schedules():
    """
    Generate visualization data for different schedules.
    Returns dict with t values and corresponding coefficients for each schedule.
    """
    t_values = jnp.linspace(0, 1, 100)[:, None]

    results = {}
    for flow_type in ["ot", "vp", "ve", "cosine"]:
        coeffs = get_flow_coefficients(t_values, flow_type)
        results[flow_type] = {
            "t": t_values.squeeze(),
            "alpha": coeffs.alpha_t.squeeze(),
            "sigma": coeffs.sigma_t.squeeze(),
            "d_alpha_dt": coeffs.d_alpha_dt.squeeze(),
            "d_sigma_dt": coeffs.d_sigma_dt.squeeze(),
            "snr": get_snr(t_values, flow_type).squeeze(),
        }

    return results


if __name__ == "__main__":
    # Test the schedules
    import matplotlib.pyplot as plt

    results = visualize_schedules()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot alpha_t
    ax = axes[0, 0]
    for name, data in results.items():
        ax.plot(data["t"], data["alpha"], label=name)
    ax.set_xlabel("t")
    ax.set_ylabel("alpha_t")
    ax.set_title("Signal Coefficient")
    ax.legend()
    ax.grid(True)

    # Plot sigma_t
    ax = axes[0, 1]
    for name, data in results.items():
        ax.plot(data["t"], data["sigma"], label=name)
    ax.set_xlabel("t")
    ax.set_ylabel("sigma_t")
    ax.set_title("Noise Coefficient")
    ax.legend()
    ax.grid(True)

    # Plot SNR
    ax = axes[1, 0]
    for name, data in results.items():
        if name != "ve":  # VE has very different scale
            ax.plot(data["t"], jnp.log10(data["snr"] + 1e-8), label=name)
    ax.set_xlabel("t")
    ax.set_ylabel("log10(SNR)")
    ax.set_title("Signal-to-Noise Ratio (log scale)")
    ax.legend()
    ax.grid(True)

    # Plot velocity magnitude for unit vectors
    ax = axes[1, 1]
    t_test = jnp.linspace(0.01, 0.99, 100)[:, None]
    x_0 = jnp.ones((100, 1))
    eps = jnp.ones((100, 1))
    for name in ["ot", "vp", "cosine"]:
        vel = compute_velocity_target(x_0, eps, t_test, name)
        ax.plot(t_test.squeeze(), jnp.abs(vel.squeeze()), label=name)
    ax.set_xlabel("t")
    ax.set_ylabel("|velocity|")
    ax.set_title("Velocity Magnitude (unit x_0, eps)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("flow_schedules_comparison.png", dpi=150)
    plt.show()
    print("Saved flow_schedules_comparison.png")
