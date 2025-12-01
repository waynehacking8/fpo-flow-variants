"""
Update all plots in the report to include Go1 Handstand results.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directories
os.makedirs("plots_multienv", exist_ok=True)
os.makedirs("plots_analysis", exist_ok=True)

# ========== Data ==========
# All environments with their results
env_data = {
    "HumanoidGetup": {
        "OT": 4201.94,
        "VP": 4105.79,
        "Cosine": 4116.02,
    },
    "Go1 Getup": {
        "OT": 18.29,
        "VP": 8.67,
        "Cosine": 10.03,
    },
    "Go1 Joystick": {
        "OT": 4.39,
        "VP": 4.00,
        "Cosine": 3.51,
    },
    "Go1 Handstand": {
        "OT": 3.34,
        "VP": 1.18,
        "Cosine": 1.37,
    },
}

# Training curves (steps for each environment)
training_data = {
    "HumanoidGetup": {
        "OT": [(0, 3183), (1, 3393), (2, 3629)],
        "VP": [(0, 3096), (1, 3340), (2, 3495)],
        "Cosine": [(0, 3120), (1, 3380), (2, 3520)],
    },
    "Go1 Getup": {
        "OT": [(0, 5.61), (1, 12.35), (2, 18.29)],
        "VP": [(0, 4.89), (1, 6.12), (2, 8.67)],
        "Cosine": [(0, 5.02), (1, 7.45), (2, 10.03)],
    },
    "Go1 Joystick": {
        "OT": [(0, 2.51), (1, 3.45), (2, 4.39)],
        "VP": [(0, 2.38), (1, 3.21), (2, 4.00)],
        "Cosine": [(0, 2.25), (1, 2.88), (2, 3.51)],
    },
    "Go1 Handstand": {
        "OT": [(0, 0.53), (1, 1.51), (2, 3.34)],
        "VP": [(0, 0.54), (1, 0.69), (2, 1.18)],
        "Cosine": [(0, 0.53), (1, 0.71), (2, 1.37)],
    },
}

colors = {'OT': '#2ecc71', 'VP': '#3498db', 'Cosine': '#9b59b6'}

# ========== Plot 1: Final Performance Comparison ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (env_name, data) in enumerate(env_data.items()):
    ax = axes[idx]
    methods = list(data.keys())
    rewards = list(data.values())

    bars = ax.bar(methods, rewards, color=[colors[m] for m in methods],
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, reward in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rewards)*0.02,
                f'{reward:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Final Reward', fontsize=11)
    ax.set_title(env_name, fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(rewards) * 1.15)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Final Performance Comparison Across All Environments\n(OT vs VP vs Cosine)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots_multienv/final_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots_multienv/final_performance_comparison.png")


# ========== Plot 2: Training Curves Comparison ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (env_name, data) in enumerate(training_data.items()):
    ax = axes[idx]

    for flow_type, curve in data.items():
        steps = [p[0] for p in curve]
        rewards = [p[1] for p in curve]
        ax.plot(steps, rewards, 'o-', color=colors[flow_type], label=flow_type,
                linewidth=2, markersize=8)

    ax.set_xlabel('Training Step (eval index)', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title(env_name, fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

plt.suptitle('Training Curves: Flow Schedule Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots_multienv/training_curves_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots_multienv/training_curves_comparison.png")


# ========== Plot 3: Normalized Performance Heatmap ==========
fig, ax = plt.subplots(figsize=(10, 6))

envs = list(env_data.keys())
schedules = ['OT', 'VP', 'Cosine']

# Normalize each environment to its max
normalized_data = []
for env_name in envs:
    max_val = max(env_data[env_name].values())
    row = [env_data[env_name][s] / max_val * 100 for s in schedules]
    normalized_data.append(row)

normalized_data = np.array(normalized_data)

im = ax.imshow(normalized_data, cmap='RdYlGn', aspect='auto', vmin=30, vmax=100)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Normalized Performance (%)', rotation=-90, va="bottom", fontsize=11)

# Show all ticks and label them
ax.set_xticks(np.arange(len(schedules)))
ax.set_yticks(np.arange(len(envs)))
ax.set_xticklabels(schedules, fontsize=11)
ax.set_yticklabels(envs, fontsize=11)

# Add text annotations
for i in range(len(envs)):
    for j in range(len(schedules)):
        val = normalized_data[i, j]
        text_color = 'white' if val < 60 else 'black'
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                color=text_color, fontsize=11, fontweight='bold')

ax.set_title('Normalized Performance Heatmap\n(Each environment normalized to its best = 100%)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plots_multienv/performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots_multienv/performance_heatmap.png")


# ========== Plot 4: OT Advantage Chart ==========
fig, ax = plt.subplots(figsize=(12, 6))

envs = list(env_data.keys())
ot_vs_vp = []
ot_vs_cosine = []

for env in envs:
    ot = env_data[env]['OT']
    vp = env_data[env]['VP']
    cosine = env_data[env]['Cosine']
    ot_vs_vp.append((ot - vp) / vp * 100)
    ot_vs_cosine.append((ot - cosine) / cosine * 100)

x = np.arange(len(envs))
width = 0.35

bars1 = ax.bar(x - width/2, ot_vs_vp, width, label='OT vs VP', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, ot_vs_cosine, width, label='OT vs Cosine', color='#9b59b6', edgecolor='black')

ax.axhline(0, color='black', linewidth=1)
ax.set_ylabel('OT Improvement (%)', fontsize=12)
ax.set_xlabel('Environment', fontsize=12)
ax.set_title('OT Schedule Advantage Over VP and Cosine', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(envs, fontsize=10)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height > 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('plots_multienv/ot_advantage_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots_multienv/ot_advantage_comparison.png")


# ========== Plot 5: Summary Bar Chart with PPO ==========
fig, ax = plt.subplots(figsize=(14, 7))

# FPO vs PPO data
fpo_ppo_data = {
    "HumanoidGetup": {"FPO-OT": 3629, "PPO": 2910},
    "Go1 Getup": {"FPO-OT": 18.29, "PPO": 12.50},
    "Go1 Joystick": {"FPO-OT": 4.39, "PPO": 16.94},
    "Go1 Handstand": {"FPO-OT": 3.34, "PPO": 1.78},
}

envs = list(fpo_ppo_data.keys())
fpo_rewards = [fpo_ppo_data[e]["FPO-OT"] for e in envs]
ppo_rewards = [fpo_ppo_data[e]["PPO"] for e in envs]

x = np.arange(len(envs))
width = 0.35

bars1 = ax.bar(x - width/2, fpo_rewards, width, label='FPO (OT)', color='#2ecc71', edgecolor='black')
bars2 = ax.bar(x + width/2, ppo_rewards, width, label='PPO', color='#e74c3c', edgecolor='black')

# Calculate and show improvements
for i, (f, p) in enumerate(zip(fpo_rewards, ppo_rewards)):
    imp = (f - p) / p * 100
    color = 'green' if imp > 0 else 'red'
    text = f'+{imp:.0f}%' if imp > 0 else f'{imp:.0f}%'
    y_pos = max(f, p) * 1.05
    ax.text(x[i], y_pos, text, ha='center', va='bottom', fontsize=11,
            color=color, fontweight='bold')

ax.set_ylabel('Final Reward', fontsize=12)
ax.set_xlabel('Environment', fontsize=12)
ax.set_title('FPO (OT) vs PPO: Performance Comparison Across All Environments',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(envs, fontsize=11)
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add task type regions
ax.axvline(2.5, color='gray', linestyle='--', alpha=0.5)
ax.text(0.75, ax.get_ylim()[1] * 0.85, 'Goal-Oriented Tasks\n(FPO advantage)',
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(2, ax.get_ylim()[1] * 0.65, 'Continuous\nControl',
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
ax.text(3, ax.get_ylim()[1] * 0.85, 'Multimodal\n(FPO best)',
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('plots_multienv/fpo_vs_ppo_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots_multienv/fpo_vs_ppo_summary.png")


# ========== Plot 6: Flow Schedule Performance by Task Type ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Goal-oriented tasks (Getup + Handstand)
ax1 = axes[0]
goal_envs = ["HumanoidGetup", "Go1 Getup", "Go1 Handstand"]
for i, env in enumerate(goal_envs):
    data = env_data[env]
    x_offset = i * 0.25
    for j, (schedule, reward) in enumerate(data.items()):
        ax1.bar(j + x_offset, reward / max(data.values()) * 100,
                width=0.2, color=colors[schedule], alpha=0.7 + i*0.1,
                label=f'{env}' if j == 0 else None)

ax1.set_xticks([0.25, 1.25, 2.25])
ax1.set_xticklabels(['OT', 'VP', 'Cosine'])
ax1.set_ylabel('Normalized Performance (%)')
ax1.set_title('Goal-Oriented Tasks\n(HumanoidGetup, Go1 Getup, Go1 Handstand)')
ax1.set_ylim(0, 110)
ax1.grid(axis='y', alpha=0.3)

# Right: Continuous control
ax2 = axes[1]
cont_env = "Go1 Joystick"
data = env_data[cont_env]
schedules = list(data.keys())
rewards = list(data.values())
ax2.bar(schedules, [r/max(rewards)*100 for r in rewards],
        color=[colors[s] for s in schedules], edgecolor='black')
ax2.set_ylabel('Normalized Performance (%)')
ax2.set_title('Continuous Control Task\n(Go1 Joystick)')
ax2.set_ylim(0, 110)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Flow Schedule Performance by Task Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots_multienv/performance_by_task_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots_multienv/performance_by_task_type.png")


print("\n=== All plots updated with Go1 Handstand results! ===")
print("Total environments: 4 (HumanoidGetup, Go1 Getup, Go1 Joystick, Go1 Handstand)")
