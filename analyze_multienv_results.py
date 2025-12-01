"""
Multi-Environment Flow Variants Analysis

Generates comparison plots for FPO flow variants across multiple environments:
- HumanoidGetup (humanoid)
- Go1 Getup (quadruped getup)
- Go1 Joystick (quadruped locomotion)
- Go1 Handstand (quadruped multimodal task)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load all results from ./results_multienv/
humanoid_results = {}
go1_getup_results = {}
go1_joystick_results = {}
go1_handstand_results = {}

multienv_dir = "./results_multienv"
for filename in os.listdir(multienv_dir):
    if filename.endswith("_results.json") and filename != "all_results.json":
        with open(os.path.join(multienv_dir, filename)) as f:
            data = json.load(f)
            env_name = data["env_name"]
            flow_type = data["flow_type"]
            if env_name == "humanoid_getup":
                humanoid_results[flow_type] = data
            elif env_name == "go1_getup":
                go1_getup_results[flow_type] = data
            elif env_name == "go1_joystick":
                go1_joystick_results[flow_type] = data
            elif env_name == "go1_handstand":
                go1_handstand_results[flow_type] = data

print("Loaded results:")
print(f"  HumanoidGetup: {list(humanoid_results.keys())}")
print(f"  Go1 Getup: {list(go1_getup_results.keys())}")
print(f"  Go1 Joystick: {list(go1_joystick_results.keys())}")
print(f"  Go1 Handstand: {list(go1_handstand_results.keys())}")

# Collect final performance data
def get_final_reward(results):
    if not results.get("eval_results"):
        return None
    final = results["eval_results"][-1]
    return final["reward_mean"]

def get_final_std(results):
    if not results.get("eval_results"):
        return None
    final = results["eval_results"][-1]
    return final.get("reward_std", 0)

# Create plots directory
os.makedirs("./plots_multienv", exist_ok=True)

# Colors for each flow type
colors = {
    "ot": "#1f77b4",
    "vp": "#9467bd",
    "ve": "#d62728",
    "cosine": "#ff7f0e"
}
flow_labels = {
    "ot": "Optimal Transport (OT)",
    "vp": "Variance Preserving (VP)",
    "ve": "Variance Exploding (VE)",
    "cosine": "Cosine Schedule"
}

# Figure 1: Final Performance Comparison (Bar Chart)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

envs = [
    ("HumanoidGetup", humanoid_results),
    ("Go1 Getup", go1_getup_results),
    ("Go1 Joystick", go1_joystick_results),
    ("Go1 Handstand", go1_handstand_results)
]
flow_types = ["ot", "vp", "cosine"]  # Exclude VE as it fails

for idx, (env_name, results) in enumerate(envs):
    ax = axes[idx]
    rewards = []
    stds = []
    labels = []
    bar_colors = []

    for ft in flow_types:
        if ft in results:
            r = get_final_reward(results[ft])
            s = get_final_std(results[ft])
            if r is not None and not np.isnan(r):
                rewards.append(r)
                stds.append(s if s else 0)
                labels.append(flow_labels[ft])
                bar_colors.append(colors[ft])

    if rewards:
        x = np.arange(len(rewards))
        bars = ax.bar(x, rewards, yerr=stds, capsize=5, color=bar_colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Final Episode Return")
        ax.set_title(env_name)

        # Add value labels on bars
        for bar, r in zip(bars, rewards):
            height = bar.get_height()
            ax.annotate(f'{r:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

plt.suptitle("Flow Schedule Performance Comparison Across Environments", fontsize=14)
plt.tight_layout()
plt.savefig("./plots_multienv/final_performance_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ./plots_multienv/final_performance_comparison.png")

# Figure 2: Training Curves (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (env_name, results) in enumerate(envs):
    ax = axes[idx]

    for ft in flow_types:
        if ft in results:
            eval_results = results[ft].get("eval_results", [])
            steps = [e["step"] for e in eval_results]
            rewards = [e["reward_mean"] for e in eval_results]
            stds = [e.get("reward_std", 0) for e in eval_results]

            if rewards and not any(np.isnan(rewards)):
                rewards = np.array(rewards)
                stds = np.array(stds)
                ax.plot(steps, rewards, 'o-', label=flow_labels[ft], color=colors[ft])
                ax.fill_between(steps, rewards - stds, rewards + stds, alpha=0.2, color=colors[ft])

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Episode Return")
    ax.set_title(env_name)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Training Curves: Flow Variants Across Environments", fontsize=14)
plt.tight_layout()
plt.savefig("./plots_multienv/training_curves_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ./plots_multienv/training_curves_comparison.png")

# Figure 3: Normalized Performance (relative to OT baseline)
fig, ax = plt.subplots(figsize=(12, 6))

env_labels = ["HumanoidGetup", "Go1 Getup", "Go1 Joystick", "Go1 Handstand"]
x = np.arange(len(env_labels))
width = 0.25

for i, ft in enumerate(["ot", "vp", "cosine"]):
    normalized_rewards = []

    for env_name, results in envs:
        if "ot" in results and ft in results:
            ot_reward = get_final_reward(results["ot"])
            ft_reward = get_final_reward(results[ft])
            if ot_reward and ft_reward and ot_reward > 0:
                normalized_rewards.append(ft_reward / ot_reward * 100)
            else:
                normalized_rewards.append(100 if ft == "ot" else 0)
        else:
            normalized_rewards.append(0)

    bars = ax.bar(x + (i - 1) * width, normalized_rewards, width,
                  label=flow_labels[ft], color=colors[ft], alpha=0.8)

ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='OT Baseline')
ax.set_ylabel("Performance Relative to OT (%)")
ax.set_xlabel("Environment")
ax.set_title("Normalized Performance Comparison (OT = 100%)")
ax.set_xticks(x)
ax.set_xticklabels(env_labels)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("./plots_multienv/normalized_performance.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ./plots_multienv/normalized_performance.png")

# Figure 4: Summary Heatmap
fig, ax = plt.subplots(figsize=(10, 6))

env_names_short = ["HumanoidGetup", "Go1 Getup", "Go1 Joystick", "Go1 Handstand"]
flow_names = ["OT", "VP", "Cosine"]

# Create data matrix (normalized to max in each env)
data = np.zeros((len(env_names_short), len(flow_names)))
for i, (env_name, results) in enumerate(envs):
    max_reward = 0
    for j, ft in enumerate(["ot", "vp", "cosine"]):
        if ft in results:
            r = get_final_reward(results[ft])
            if r and not np.isnan(r):
                data[i, j] = r
                max_reward = max(max_reward, r)
    # Normalize
    if max_reward > 0:
        data[i, :] = data[i, :] / max_reward * 100

im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
ax.set_xticks(np.arange(len(flow_names)))
ax.set_yticks(np.arange(len(env_names_short)))
ax.set_xticklabels(flow_names)
ax.set_yticklabels(env_names_short)

# Add value annotations
for i in range(len(env_names_short)):
    for j in range(len(flow_names)):
        text = ax.text(j, i, f'{data[i, j]:.0f}%', ha="center", va="center",
                      color="white" if data[i, j] > 50 else "black")

ax.set_title("Flow Schedule Effectiveness by Environment\n(Normalized to Best in Each Environment)")
cbar = plt.colorbar(im)
cbar.set_label("Relative Performance (%)")

plt.tight_layout()
plt.savefig("./plots_multienv/performance_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ./plots_multienv/performance_heatmap.png")

# Print summary
print("\n" + "="*60)
print("MULTI-ENVIRONMENT ANALYSIS SUMMARY")
print("="*60)

for env_name, results in envs:
    print(f"\n{env_name}:")
    for ft in ["ot", "vp", "cosine"]:
        if ft in results:
            r = get_final_reward(results[ft])
            s = get_final_std(results[ft])
            if r is not None and not np.isnan(r):
                print(f"  {ft.upper():8s}: {r:8.4f} +/- {s:.4f}")

print("\n" + "="*60)
print("KEY FINDINGS:")
print("="*60)

# Find best flow for each environment
for env_name, results in envs:
    best_ft = None
    best_reward = -float('inf')
    for ft in ["ot", "vp", "cosine"]:
        if ft in results:
            r = get_final_reward(results[ft])
            if r and not np.isnan(r) and r > best_reward:
                best_reward = r
                best_ft = ft
    if best_ft:
        print(f"  {env_name}: Best = {best_ft.upper()} ({best_reward:.4f})")
