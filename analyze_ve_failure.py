"""
VE (Variance Exploding) Failure Analysis

This script analyzes why VE flow schedule fails in FPO training,
generating diagnostic plots and explanations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create plots directory
os.makedirs("./plots_analysis", exist_ok=True)

# Flow schedule parameters
sigma_min = 0.01
sigma_max = 80.0
t_values = np.linspace(0, 1, 100)

# Compute coefficients for each flow schedule
def compute_ot(t):
    alpha = 1 - t
    sigma = t
    d_alpha = -np.ones_like(t)
    d_sigma = np.ones_like(t)
    return alpha, sigma, d_alpha, d_sigma

def compute_vp(t):
    alpha = np.cos(np.pi * t / 2)
    sigma = np.sin(np.pi * t / 2)
    d_alpha = -np.pi / 2 * np.sin(np.pi * t / 2)
    d_sigma = np.pi / 2 * np.cos(np.pi * t / 2)
    return alpha, sigma, d_alpha, d_sigma

def compute_ve(t):
    alpha = np.ones_like(t)
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    d_alpha = np.zeros_like(t)
    d_sigma = sigma * np.log(sigma_max / sigma_min)
    return alpha, sigma, d_alpha, d_sigma

def compute_cosine(t):
    # Cosine schedule from improved DDPM
    s = 0.008
    f_t = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f_t / f_0
    alpha_bar = np.clip(alpha_bar, 0.001, 0.999)
    alpha = np.sqrt(alpha_bar)
    sigma = np.sqrt(1 - alpha_bar)
    # Numerical derivatives
    dt = t[1] - t[0]
    d_alpha = np.gradient(alpha, dt)
    d_sigma = np.gradient(sigma, dt)
    return alpha, sigma, d_alpha, d_sigma

# Compute all schedules
ot_alpha, ot_sigma, ot_da, ot_ds = compute_ot(t_values)
vp_alpha, vp_sigma, vp_da, vp_ds = compute_vp(t_values)
ve_alpha, ve_sigma, ve_da, ve_ds = compute_ve(t_values)
cos_alpha, cos_sigma, cos_da, cos_ds = compute_cosine(t_values)

# Figure 1: Sigma comparison (log scale)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(t_values, ot_sigma, 'b-', label='OT', linewidth=2)
ax.plot(t_values, vp_sigma, 'purple', label='VP', linewidth=2)
ax.plot(t_values, ve_sigma, 'r-', label='VE', linewidth=2)
ax.plot(t_values, cos_sigma, 'orange', label='Cosine', linewidth=2)
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('sigma_t', fontsize=12)
ax.set_title('Noise Scale (sigma) Comparison', fontsize=14)
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='sigma=1')

ax = axes[1]
ax.plot(t_values, ot_sigma, 'b-', label='OT', linewidth=2)
ax.plot(t_values, vp_sigma, 'purple', label='VP', linewidth=2)
ax.plot(t_values, ve_sigma, 'r-', label='VE', linewidth=2)
ax.plot(t_values, cos_sigma, 'orange', label='Cosine', linewidth=2)
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('sigma_t', fontsize=12)
ax.set_title('Noise Scale (Linear)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('VE Failure Analysis: Sigma Explosion', fontsize=16)
plt.tight_layout()
plt.savefig('./plots_analysis/ve_sigma_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ./plots_analysis/ve_sigma_comparison.png")

# Figure 2: Velocity target magnitudes
def compute_velocity_target_magnitude(alpha, sigma, d_alpha, d_sigma):
    """Compute |v_target| = |d_alpha/dt * x_0 + d_sigma/dt * eps| for unit x_0 and eps."""
    # Assuming ||x_0|| = 1 and ||eps|| = 1, worst case magnitude
    return np.abs(d_alpha) + np.abs(d_sigma)

ot_vmag = compute_velocity_target_magnitude(ot_alpha, ot_sigma, ot_da, ot_ds)
vp_vmag = compute_velocity_target_magnitude(vp_alpha, vp_sigma, vp_da, vp_ds)
ve_vmag = compute_velocity_target_magnitude(ve_alpha, ve_sigma, ve_da, ve_ds)
cos_vmag = compute_velocity_target_magnitude(cos_alpha, cos_sigma, cos_da, cos_ds)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_values, ot_vmag, 'b-', label='OT', linewidth=2)
ax.plot(t_values, vp_vmag, 'purple', label='VP', linewidth=2)
ax.plot(t_values, ve_vmag, 'r-', label='VE', linewidth=2)
ax.plot(t_values, cos_vmag, 'orange', label='Cosine', linewidth=2)
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('|velocity target| (upper bound)', fontsize=12)
ax.set_title('Velocity Target Magnitude Comparison\n(Explains gradient explosion in VE)', fontsize=14)
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
ax.text(0.5, 15, 'Gradient stability threshold', fontsize=10, alpha=0.7)

plt.tight_layout()
plt.savefig('./plots_analysis/ve_velocity_magnitude.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ./plots_analysis/ve_velocity_magnitude.png")

# Figure 3: Signal-to-Noise Ratio
snr_ot = ot_alpha / (ot_sigma + 1e-8)
snr_vp = vp_alpha / (vp_sigma + 1e-8)
snr_ve = ve_alpha / (ve_sigma + 1e-8)
snr_cos = cos_alpha / (cos_sigma + 1e-8)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_values, snr_ot, 'b-', label='OT', linewidth=2)
ax.plot(t_values, snr_vp, 'purple', label='VP', linewidth=2)
ax.plot(t_values, snr_ve, 'r-', label='VE', linewidth=2)
ax.plot(t_values, snr_cos, 'orange', label='Cosine', linewidth=2)
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('Signal-to-Noise Ratio (alpha/sigma)', fontsize=12)
ax.set_title('SNR Comparison Across Flow Schedules', fontsize=14)
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./plots_analysis/flow_snr_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ./plots_analysis/flow_snr_comparison.png")

# Figure 4: Why VE fails - comprehensive view
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Alpha comparison
ax = axes[0, 0]
ax.plot(t_values, ot_alpha, 'b-', label='OT', linewidth=2)
ax.plot(t_values, vp_alpha, 'purple', label='VP', linewidth=2)
ax.plot(t_values, ve_alpha, 'r-', label='VE (constant=1)', linewidth=2)
ax.plot(t_values, cos_alpha, 'orange', label='Cosine', linewidth=2)
ax.set_xlabel('t')
ax.set_ylabel('alpha_t')
ax.set_title('Data Coefficient (alpha)')
ax.legend()
ax.grid(True, alpha=0.3)

# Sigma comparison
ax = axes[0, 1]
ax.plot(t_values, ot_sigma, 'b-', label='OT', linewidth=2)
ax.plot(t_values, vp_sigma, 'purple', label='VP', linewidth=2)
ax.plot(t_values, ve_sigma, 'r-', label='VE (EXPLODES!)', linewidth=2)
ax.plot(t_values, cos_sigma, 'orange', label='Cosine', linewidth=2)
ax.set_xlabel('t')
ax.set_ylabel('sigma_t')
ax.set_title('Noise Coefficient (sigma) - VE Problem')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# d_sigma/dt comparison
ax = axes[1, 0]
ax.plot(t_values, np.abs(ot_ds), 'b-', label='OT', linewidth=2)
ax.plot(t_values, np.abs(vp_ds), 'purple', label='VP', linewidth=2)
ax.plot(t_values, np.abs(ve_ds), 'r-', label='VE', linewidth=2)
ax.plot(t_values, np.abs(cos_ds), 'orange', label='Cosine', linewidth=2)
ax.set_xlabel('t')
ax.set_ylabel('|d_sigma/dt|')
ax.set_title('Noise Derivative (causes gradient explosion)')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Total variance
total_var_ot = ot_alpha**2 + ot_sigma**2
total_var_vp = vp_alpha**2 + vp_sigma**2
total_var_ve = ve_alpha**2 + ve_sigma**2
total_var_cos = cos_alpha**2 + cos_sigma**2

ax = axes[1, 1]
ax.plot(t_values, total_var_ot, 'b-', label='OT', linewidth=2)
ax.plot(t_values, total_var_vp, 'purple', label='VP (=1)', linewidth=2)
ax.plot(t_values, total_var_ve, 'r-', label='VE (EXPLODES!)', linewidth=2)
ax.plot(t_values, total_var_cos, 'orange', label='Cosine', linewidth=2)
ax.set_xlabel('t')
ax.set_ylabel('alpha^2 + sigma^2')
ax.set_title('Total Variance (VP preserves, VE explodes)')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.suptitle('Why VE (Variance Exploding) Fails in FPO', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./plots_analysis/ve_failure_comprehensive.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ./plots_analysis/ve_failure_comprehensive.png")

# Print summary
print("\n" + "="*60)
print("VE FAILURE ANALYSIS SUMMARY")
print("="*60)
print(f"\nAt t=1.0:")
print(f"  OT sigma:  {ot_sigma[-1]:.4f}")
print(f"  VP sigma:  {vp_sigma[-1]:.4f}")
print(f"  VE sigma:  {ve_sigma[-1]:.4f}  <-- 80x larger!")
print(f"  Cosine sigma: {cos_sigma[-1]:.4f}")
print(f"\nVE d_sigma/dt at t=1.0: {ve_ds[-1]:.4f}")
print(f"This causes gradients to explode during backpropagation.")
print("\n" + "="*60)
print("CONCLUSION: VE is unsuitable for FPO because:")
print("1. sigma grows exponentially (up to sigma_max=80)")
print("2. d_sigma/dt grows proportionally, causing gradient explosion")
print("3. The network cannot learn stable velocity predictions")
print("="*60)
