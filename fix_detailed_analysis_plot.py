"""
Fix the fpo_vs_ppo_detailed_analysis.png plot - move Performance Summary Table title above the table.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Create output directory
os.makedirs("plots_analysis", exist_ok=True)

# Data from original plot
fpo_data = {
    'steps': [0, 1, 2],
    'rewards': [3183.5, 3350, 3628.8],
    'stds': [100, 140, 88.3],
    'min': [3050, 3100, 3400],
    'max': [3550, 3700, 3826],
}

ppo_data = {
    'steps': [0, 1, 2],
    'rewards': [2981.9, 3180, 2909.6],
    'stds': [100, 250, 134.2],
    'min': [2780, 2780, 2800],
    'max': [3200, 3500, 3330],
}

# Create figure with GridSpec for better control
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1.2], hspace=0.35, wspace=0.3)

# Colors
fpo_color = '#1f77b4'
ppo_color = '#d62728'

# ============ Top subplot: Learning Curves with Standard Deviation ============
ax1 = fig.add_subplot(gs[0, :])

# FPO
ax1.errorbar(fpo_data['steps'], fpo_data['rewards'], yerr=fpo_data['stds'],
             fmt='o-', color=fpo_color, label='FPO (mean ± std)',
             capsize=5, linewidth=2, markersize=8)
# PPO
ax1.errorbar(ppo_data['steps'], ppo_data['rewards'], yerr=ppo_data['stds'],
             fmt='s-', color=ppo_color, label='PPO (mean ± std)',
             capsize=5, linewidth=2, markersize=8)

# Add improvement annotations
ax1.annotate('+14.0%', xy=(2, 3628.8), xytext=(2.1, 3750),
            fontsize=11, color='green', fontweight='bold')
ax1.annotate('-2.4%', xy=(2, 2909.6), xytext=(2.1, 2800),
            fontsize=11, color='red', fontweight='bold')

ax1.set_xlabel('Evaluation Step', fontsize=11)
ax1.set_ylabel('Episode Return', fontsize=11)
ax1.set_title('Learning Curves with Standard Deviation', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([0, 1, 2])

# ============ Bottom Left: Min-Max Range per Evaluation ============
ax2 = fig.add_subplot(gs[1, 0])

for i, step in enumerate(fpo_data['steps']):
    ax2.plot([step-0.1, step-0.1], [fpo_data['min'][i], fpo_data['max'][i]],
             color=fpo_color, linewidth=2)
    ax2.scatter([step-0.1]*2, [fpo_data['min'][i], fpo_data['max'][i]],
               color=fpo_color, s=50, marker='o')

    ax2.plot([step+0.1, step+0.1], [ppo_data['min'][i], ppo_data['max'][i]],
             color=ppo_color, linewidth=2)
    ax2.scatter([step+0.1]*2, [ppo_data['min'][i], ppo_data['max'][i]],
               color=ppo_color, s=50, marker='s')

ax2.set_xlabel('Evaluation Step', fontsize=11)
ax2.set_ylabel('Reward Range', fontsize=11)
ax2.set_title('Min-Max Range per Evaluation', fontsize=12)
ax2.legend(handles=[
    mpatches.Patch(color=fpo_color, label='FPO'),
    mpatches.Patch(color=ppo_color, label='PPO')
], loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks([0, 1, 2])

# ============ Bottom Right: Policy Stability ============
ax3 = fig.add_subplot(gs[1, 1])

ax3.plot(fpo_data['steps'], fpo_data['stds'], 'o-', color=fpo_color,
         label='FPO', linewidth=2, markersize=8)
ax3.plot(ppo_data['steps'], ppo_data['stds'], 's-', color=ppo_color,
         label='PPO', linewidth=2, markersize=8)

# Add final std annotations
ax3.annotate(f'{fpo_data["stds"][-1]:.1f}', xy=(2, fpo_data['stds'][-1]),
            xytext=(2.15, fpo_data['stds'][-1]), fontsize=10, color=fpo_color)
ax3.annotate(f'{ppo_data["stds"][-1]:.1f}', xy=(2, ppo_data['stds'][-1]),
            xytext=(2.15, ppo_data['stds'][-1]), fontsize=10, color=ppo_color)

ax3.set_xlabel('Evaluation Step', fontsize=11)
ax3.set_ylabel('Standard Deviation', fontsize=11)
ax3.set_title('Policy Stability (Lower is More Stable)', fontsize=12)
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks([0, 1, 2])

# ============ Table at bottom ============
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

# Table data
table_data = [
    ['Initial Mean', '3183.5', '2981.9', 'FPO (+6.8%)'],
    ['Final Mean', '3628.8', '2909.6', 'FPO (+24.7%)'],
    ['Improvement', '+14.0%', '-2.4%', 'FPO'],
    ['Final Stability (std)', '88.3', '134.2', 'FPO (-34.2%)'],
    ['Max Achieved', '3825.9', '3330.0', 'FPO (+14.9%)'],
]
col_labels = ['Metric', 'FPO', 'PPO', 'Winner']

# Create table with proper positioning
table = ax4.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
    colColours=['#e6e6e6']*4,
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header row
for j in range(4):
    table[(0, j)].set_text_props(fontweight='bold')

# Color the winner column
for i in range(1, 6):
    table[(i, 3)].set_text_props(color='#2ca02c')

# Color FPO and PPO columns
for i in range(1, 6):
    table[(i, 1)].set_text_props(color=fpo_color)
    table[(i, 2)].set_text_props(color=ppo_color)

# Add table title ABOVE the table (not inside)
ax4.text(0.5, 0.95, 'Performance Summary Table', transform=ax4.transAxes,
         ha='center', va='bottom', fontsize=13, fontweight='bold')

# ============ Key Findings box at the very bottom ============
findings_text = """Key Findings:
• FPO achieves 14.0% improvement vs PPO's -2.4% degradation
• FPO shows better stability (lower std) in final policy
• FPO consistently outperforms PPO across all metrics
• Training time comparable (~62s for FPO, ~62s for PPO per step)"""

fig.text(0.02, 0.02, findings_text, fontsize=9,
         verticalalignment='bottom', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='#e8f4e8', alpha=0.8))

# Main title
fig.suptitle('FPO vs PPO: Detailed Training Analysis on HumanoidGetup Task',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('plots_analysis/fpo_vs_ppo_detailed_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: plots_analysis/fpo_vs_ppo_detailed_analysis.png")
print("Fixed: 'Performance Summary Table' title now appears above the table, not overlapping with it.")
