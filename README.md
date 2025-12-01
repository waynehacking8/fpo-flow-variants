# FPO Flow Variants: Comparing Flow Schedules for Policy Optimization

This project extends Flow Policy Optimization (FPO) to support multiple flow/diffusion schedules, enabling systematic comparison of different flow formulations for reinforcement learning.

## Research Question

**How do different flow schedules affect the performance of Flow Policy Optimization in continuous control tasks?**

The original FPO uses Optimal Transport (OT) / Conditional Flow Matching (CFM) with linear interpolation. We extend this to support:

1. **Optimal Transport (OT)** - Linear interpolation (original FPO)
2. **Variance Preserving (VP)** - Cosine schedule, similar to DDPM
3. **Variance Exploding (VE)** - Exponential noise schedule
4. **Cosine** - Improved DDPM cosine schedule

## Mathematical Background

### Flow Interpolation
All schedules define a path from noise (t=1) to data (t=0):

```
x_t = α_t · x_0 + σ_t · ε
```

where:
- `x_0` is the clean action (from the policy)
- `ε` is standard Gaussian noise
- `α_t` and `σ_t` are schedule-dependent coefficients

### Schedule Definitions

| Schedule | α_t | σ_t | Properties |
|----------|-----|-----|------------|
| OT | 1-t | t | Linear, constant velocity |
| VP | cos(πt/2) | sin(πt/2) | α² + σ² = 1 (variance preserved) |
| VE | 1 | σ_min·(σ_max/σ_min)^t | Signal preserved, noise grows |
| Cosine | √(f(t)/f(0)) | √(1-f(t)/f(0)) | Gradual noise at start |

### Velocity Target
The flow matching objective supervises velocity prediction:

```
v_target = dα_t/dt · x_0 + dσ_t/dt · ε
```

## Project Structure

```
fpo-flow-variants/
├── flow_schedules.py      # Core flow schedule implementations
├── fpo_variants.py        # FPO configuration and helper functions
├── fpo_full.py           # Complete FPO implementation
├── train_flow_variants.py # Training script
├── analyze_results.py     # Analysis and visualization
└── README.md
```

## Usage

### Train a Single Variant
```bash
python train_flow_variants.py --flow_type ot
python train_flow_variants.py --flow_type vp
python train_flow_variants.py --flow_type ve
python train_flow_variants.py --flow_type cosine
```

### Train All Variants
```bash
python train_flow_variants.py --all --num_timesteps 3000000
```

### Analyze Results
```bash
python analyze_results.py --results_dir ./results --output_dir ./plots
```

## Key Modifications from Original FPO

1. **`flow_schedules.py`**: Implements `get_flow_coefficients()` returning α_t, σ_t, and their derivatives for each schedule type.

2. **`_compute_cfm_loss()`**: Modified to use schedule-specific interpolation:
   ```python
   coeffs = get_flow_coefficients(t, self.config.flow_type, ...)
   x_t = coeffs.alpha_t * action + coeffs.sigma_t * eps
   velocity_target = coeffs.d_alpha_dt * action + coeffs.d_sigma_dt * eps
   ```

3. **`sample_action()`**: Euler integration uses the same velocity field for all schedules.

## Hypotheses

1. **OT should have stable gradients** due to constant velocity magnitude
2. **VP may improve sample efficiency** with variance-preserving property
3. **VE requires careful tuning** due to exploding variance at t→1
4. **Cosine may be more forgiving** with gradual noise schedule

## Expected Results

We expect to observe:
- Different convergence speeds across schedules
- Varying sensitivity to hyperparameters
- Trade-offs between sample efficiency and final performance

## Requirements

- JAX
- jax_dataclasses
- mujoco_playground
- optax
- wandb
- matplotlib
- numpy

## References

- [Flow Policy Optimization](https://arxiv.org/abs/2507.21053) - McAllister et al., 2025
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - Lipman et al., 2023
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- [Score-Based Generative Modeling](https://arxiv.org/abs/2011.13456) - Song et al., 2021
