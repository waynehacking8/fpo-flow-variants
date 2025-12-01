"""
Analysis and Visualization for FPO Flow Variants Comparison

This script generates publication-quality plots comparing different flow schedules.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})


# Flow type display names and colors
FLOW_DISPLAY_NAMES = {
    "ot": "Optimal Transport (OT)",
    "vp": "Variance Preserving (VP)",
    "ve": "Variance Exploding (VE)",
    "cosine": "Cosine Schedule",
}

FLOW_COLORS = {
    "ot": "#2E86AB",     # Blue
    "vp": "#A23B72",     # Magenta
    "ve": "#F18F01",     # Orange
    "cosine": "#C73E1D", # Red
}

FLOW_MARKERS = {
    "ot": "o",
    "vp": "s",
    "ve": "^",
    "cosine": "D",
}


def load_results(results_dir: str) -> dict:
    """Load all results from directory."""
    results = {}
    results_path = Path(results_dir)

    for flow_type in ["ot", "vp", "ve", "cosine"]:
        file_path = results_path / f"fpo_{flow_type}_results.json"
        if file_path.exists():
            with open(file_path) as f:
                results[flow_type] = json.load(f)

    # Also try combined results
    combined_path = results_path / "all_variants_results.json"
    if combined_path.exists() and not results:
        with open(combined_path) as f:
            results = json.load(f)

    return results


def plot_training_curves(results: dict, save_path: Optional[str] = None):
    """Plot training curves for all flow variants."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for flow_type, data in results.items():
        if "eval_results" not in data:
            continue

        eval_data = data["eval_results"]
        steps = [e["step"] for e in eval_data]
        rewards = [e["reward_mean"] for e in eval_data]
        stds = [e["reward_std"] for e in eval_data]

        color = FLOW_COLORS.get(flow_type, "#333333")
        marker = FLOW_MARKERS.get(flow_type, "o")
        label = FLOW_DISPLAY_NAMES.get(flow_type, flow_type.upper())

        # Plot mean with confidence interval
        rewards = np.array(rewards)
        stds = np.array(stds)

        ax.plot(steps, rewards, color=color, marker=marker,
                label=label, linewidth=2, markersize=8)
        ax.fill_between(steps, rewards - stds, rewards + stds,
                       color=color, alpha=0.15)

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Episode Return")
    ax.set_title("FPO Flow Variants: Training Curves on HumanoidGetup")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_final_performance_bar(results: dict, save_path: Optional[str] = None):
    """Bar chart comparing final performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    flow_types = []
    final_rewards = []
    final_stds = []
    colors = []

    for flow_type in ["ot", "vp", "ve", "cosine"]:
        if flow_type not in results:
            continue

        data = results[flow_type]
        if "eval_results" not in data or not data["eval_results"]:
            continue

        flow_types.append(FLOW_DISPLAY_NAMES.get(flow_type, flow_type.upper()))
        final_rewards.append(data["eval_results"][-1]["reward_mean"])
        final_stds.append(data["eval_results"][-1]["reward_std"])
        colors.append(FLOW_COLORS.get(flow_type, "#333333"))

    x = np.arange(len(flow_types))
    bars = ax.bar(x, final_rewards, yerr=final_stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel("Flow Schedule")
    ax.set_ylabel("Final Episode Return")
    ax.set_title("Final Performance Comparison (HumanoidGetup)")
    ax.set_xticks(x)
    ax.set_xticklabels(flow_types, rotation=15, ha='right')
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, reward in zip(bars, final_rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
               f'{reward:.0f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_improvement_analysis(results: dict, save_path: Optional[str] = None):
    """Analyze improvement rate for each variant."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Improvement over training
    ax1 = axes[0]
    for flow_type, data in results.items():
        if "eval_results" not in data:
            continue

        eval_data = data["eval_results"]
        if len(eval_data) < 2:
            continue

        steps = [e["step"] for e in eval_data]
        rewards = [e["reward_mean"] for e in eval_data]

        # Compute relative improvement from first eval
        initial = rewards[0]
        improvements = [(r - initial) / abs(initial) * 100 for r in rewards]

        color = FLOW_COLORS.get(flow_type, "#333333")
        label = FLOW_DISPLAY_NAMES.get(flow_type, flow_type.upper())

        ax1.plot(steps, improvements, color=color,
                label=label, linewidth=2, marker='o', markersize=6)

    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Relative Improvement (%)")
    ax1.set_title("Learning Progress: Relative Improvement")
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Right: Sample efficiency (reward per training time)
    ax2 = axes[1]

    flow_types = []
    efficiencies = []
    colors_list = []

    for flow_type in ["ot", "vp", "ve", "cosine"]:
        if flow_type not in results:
            continue

        data = results[flow_type]
        if "eval_results" not in data or not data["eval_results"]:
            continue
        if "total_time" not in data:
            continue

        final_reward = data["eval_results"][-1]["reward_mean"]
        initial_reward = data["eval_results"][0]["reward_mean"]
        total_time = data["total_time"]

        # Efficiency = improvement per second
        efficiency = (final_reward - initial_reward) / total_time * 60  # per minute

        flow_types.append(FLOW_DISPLAY_NAMES.get(flow_type, flow_type.upper()))
        efficiencies.append(efficiency)
        colors_list.append(FLOW_COLORS.get(flow_type, "#333333"))

    if efficiencies:
        x = np.arange(len(flow_types))
        ax2.bar(x, efficiencies, color=colors_list, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel("Flow Schedule")
        ax2.set_ylabel("Improvement per Minute")
        ax2.set_title("Sample Efficiency")
        ax2.set_xticks(x)
        ax2.set_xticklabels(flow_types, rotation=15, ha='right')
        ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def generate_summary_table(results: dict) -> str:
    """Generate a markdown summary table."""
    lines = [
        "# FPO Flow Variants Comparison Summary",
        "",
        "| Flow Schedule | Final Reward | Improvement (%) | Training Time (s) |",
        "|--------------|--------------|-----------------|-------------------|",
    ]

    for flow_type in ["ot", "vp", "ve", "cosine"]:
        if flow_type not in results:
            continue

        data = results[flow_type]
        if "eval_results" not in data or not data["eval_results"]:
            continue

        name = FLOW_DISPLAY_NAMES.get(flow_type, flow_type.upper())
        final = data["eval_results"][-1]["reward_mean"]
        initial = data["eval_results"][0]["reward_mean"]
        improvement = (final - initial) / abs(initial) * 100
        time_s = data.get("total_time", "N/A")

        if isinstance(time_s, (int, float)):
            time_str = f"{time_s:.1f}"
        else:
            time_str = str(time_s)

        lines.append(f"| {name} | {final:.1f} | {improvement:+.1f}% | {time_str} |")

    return "\n".join(lines)


def create_all_plots(results_dir: str = "./results", output_dir: str = "./plots"):
    """Create all analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Loaded results for: {list(results.keys())}")

    # Generate plots
    plot_training_curves(
        results,
        save_path=os.path.join(output_dir, "training_curves.png")
    )

    plot_final_performance_bar(
        results,
        save_path=os.path.join(output_dir, "final_performance.png")
    )

    plot_improvement_analysis(
        results,
        save_path=os.path.join(output_dir, "improvement_analysis.png")
    )

    # Generate summary
    summary = generate_summary_table(results)
    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Saved summary: {summary_path}")
    print("\n" + summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze FPO flow variants results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory containing results JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plots",
        help="Directory to save plots",
    )

    args = parser.parse_args()
    create_all_plots(args.results_dir, args.output_dir)
