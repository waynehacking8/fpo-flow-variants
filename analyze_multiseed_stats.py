"""
Analyze multi-seed results and perform statistical tests.
"""

import json
import numpy as np
from scipy import stats
import os

def load_results():
    """Load all multi-seed results."""
    results_dir = "./results_multiseed"

    all_results = {}

    for flow_type in ["ot", "vp", "cosine"]:
        rewards = []
        for seed in [0, 1, 2]:
            path = os.path.join(results_dir, f"humanoid_getup_{flow_type}_seed{seed}_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                    final_reward = data.get("final_reward")
                    # Filter out NaN values
                    if final_reward is not None and not np.isnan(final_reward):
                        rewards.append(final_reward)
                    else:
                        # Use the last valid reward if final is NaN
                        eval_results = data.get("eval_results", [])
                        for r in reversed(eval_results):
                            if not np.isnan(r["reward_mean"]):
                                rewards.append(r["reward_mean"])
                                print(f"Using last valid reward for {flow_type} seed {seed}: {r['reward_mean']:.2f}")
                                break

        if rewards:
            all_results[flow_type] = {
                "rewards": rewards,
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "n": len(rewards)
            }

    return all_results


def perform_ttest(results):
    """Perform pairwise t-tests between flow types."""
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS - HumanoidGetup Multi-Seed Results")
    print("="*70)

    # Print summary
    print("\n" + "-"*50)
    print("Summary Statistics:")
    print("-"*50)
    print(f"{'Flow Type':<12} {'Mean':>12} {'Std':>10} {'N':>5}")
    print("-"*50)

    for flow_type, data in results.items():
        print(f"{flow_type.upper():<12} {data['mean']:>12.2f} {data['std']:>10.2f} {data['n']:>5}")

    # Perform pairwise t-tests
    print("\n" + "-"*50)
    print("Pairwise Independent t-tests:")
    print("-"*50)

    flow_types = list(results.keys())

    for i in range(len(flow_types)):
        for j in range(i+1, len(flow_types)):
            ft1, ft2 = flow_types[i], flow_types[j]
            r1, r2 = results[ft1]["rewards"], results[ft2]["rewards"]

            # Independent two-sample t-test
            t_stat, p_value = stats.ttest_ind(r1, r2)

            # Cohen's d effect size
            pooled_std = np.sqrt(((len(r1)-1)*np.std(r1, ddof=1)**2 +
                                  (len(r2)-1)*np.std(r2, ddof=1)**2) /
                                 (len(r1) + len(r2) - 2))
            cohens_d = (np.mean(r1) - np.mean(r2)) / pooled_std if pooled_std > 0 else 0

            significance = ""
            if p_value < 0.01:
                significance = "***"
            elif p_value < 0.05:
                significance = "**"
            elif p_value < 0.1:
                significance = "*"
            else:
                significance = "n.s."

            print(f"\n{ft1.upper()} vs {ft2.upper()}:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value:     {p_value:.4f} {significance}")
            print(f"  Cohen's d:   {cohens_d:.4f}")
            print(f"  Mean diff:   {np.mean(r1) - np.mean(r2):.2f}")

    # Ranking
    print("\n" + "-"*50)
    print("Performance Ranking (by mean reward):")
    print("-"*50)

    sorted_types = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    for rank, (ft, data) in enumerate(sorted_types, 1):
        print(f"  {rank}. {ft.upper()}: {data['mean']:.2f} ± {data['std']:.2f}")

    return results


def save_corrected_stats(results):
    """Save corrected statistics."""
    output = {
        ft: {
            "rewards": data["rewards"],
            "mean": float(data["mean"]),
            "std": float(data["std"]),
            "n": data["n"]
        }
        for ft, data in results.items()
    }

    with open("./results_multiseed/humanoid_getup_corrected_stats.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nCorrected statistics saved to ./results_multiseed/humanoid_getup_corrected_stats.json")


if __name__ == "__main__":
    results = load_results()
    perform_ttest(results)
    save_corrected_stats(results)
