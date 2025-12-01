"""
Train PPO baseline on multiple environments for comparison with FPO.
"""
import argparse
import datetime
import json
import os
import sys
import time

import jax
import numpy as onp
import wandb
from jax import numpy as jnp
from tqdm import tqdm

# Add fpo playground to path
sys.path.insert(0, '/home/wayneleo8/Desktop/fpo_official/playground/src')
sys.path.insert(0, '/home/wayneleo8/Desktop/fpo-humanoid-getup')

from flow_policy import ppo, rollouts

import mujoco_playground as mjp
from mujoco_playground import locomotion

from env_wrappers import wrap_env_for_fpo


def load_environment(env_name: str):
    """Load environment by name and wrap for PPO compatibility."""
    if env_name == "humanoid_getup":
        from humanoid_getup import HumanoidGetup, default_config
        return HumanoidGetup(config=default_config())
    elif env_name == "go1_getup":
        # Quadruped getup task
        env = locomotion.go1_getup.Getup(config=locomotion.go1_getup.default_config())
        return wrap_env_for_fpo(env)
    elif env_name == "go1_joystick":
        # Quadruped locomotion with joystick control
        env = locomotion.go1_joystick.Joystick(config=locomotion.go1_joystick.default_config())
        return wrap_env_for_fpo(env)
    elif env_name == "go1_handstand":
        # Quadruped handstand task
        env = locomotion.go1_handstand.Handstand(config=locomotion.go1_handstand.default_config())
        return wrap_env_for_fpo(env)
    elif env_name == "spot_getup":
        # Spot robot getup task
        env = locomotion.spot_getup.Getup(config=locomotion.spot_getup.default_config())
        return wrap_env_for_fpo(env)
    else:
        raise ValueError(f"Unknown environment: {env_name}. Available: humanoid_getup, go1_getup, go1_joystick, go1_handstand, spot_getup")


def train_ppo_on_env(
    env_name: str,
    wandb_project: str = "fpo-flow-variants-ppo",
    seed: int = 0,
    num_timesteps: int = 3_000_000,
    save_dir: str = "./results_ppo",
) -> dict:
    """Train PPO on a specific environment."""

    print(f"\n{'='*60}")
    print(f"Training PPO on {env_name}")
    print(f"{'='*60}\n")

    # PPO config
    ppo_params = {
        "num_timesteps": num_timesteps,
        "num_evals": 10,
        "episode_length": 1000,
        "num_envs": 2048,
        "unroll_length": 30,
        "learning_rate": 1e-4,
        "batch_size": 1024,
        "num_minibatches": 32,
        "num_updates_per_batch": 8,
        "clipping_epsilon": 0.3,
        "discounting": 0.95,
        "gae_lambda": 0.95,
        "entropy_cost": 1e-2,
        "value_loss_coeff": 0.25,
        "reward_scaling": 10.0,
        "action_repeat": 1,
        "normalize_observations": True,
        "normalize_advantage": True,
    }

    # Load environment
    env = load_environment(env_name)

    # Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"ppo_{env_name}_{timestamp}"

    wandb_run = wandb.init(
        project=wandb_project,
        name=exp_name,
        config={
            "env_name": env_name,
            "algorithm": "ppo",
            "seed": seed,
            "ppo_params": ppo_params,
        },
        reinit=True,
    )

    # Initialize
    config = ppo.PpoConfig(**ppo_params)
    agent_state = ppo.PpoState.init(prng=jax.random.key(seed), env=env, config=config)
    rollout_state = rollouts.BatchedRolloutState.init(
        env,
        prng=jax.random.key(seed),
        num_envs=config.num_envs,
    )

    # Training
    outer_iters = config.num_timesteps // (config.iterations_per_env * config.num_envs)
    eval_iters = set(onp.linspace(0, outer_iters - 1, config.num_evals, dtype=int))

    eval_results = []
    times = [time.time()]

    for i in tqdm(range(outer_iters), desc=f"PPO-{env_name}"):
        if i in eval_iters:
            eval_outputs = rollouts.eval_policy(
                agent_state,
                prng=jax.random.fold_in(agent_state.prng, i),
                num_envs=128,
                max_episode_length=config.episode_length,
            )

            s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}

            print(f"\n[PPO-{env_name}] Step {i}: reward={s_np['reward_mean']:.2f}")

            eval_results.append({
                "step": i,
                "reward_mean": float(s_np['reward_mean']),
                "reward_std": float(s_np['reward_std']),
            })

            if not onp.isnan(s_np['reward_mean']):
                try:
                    eval_outputs.log_to_wandb(wandb_run, step=i)
                except:
                    pass

        rollout_state, transitions = rollout_state.rollout(
            agent_state,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
        )
        agent_state, metrics = agent_state.training_step(transitions)

        log_dict = {
            "train/mean_reward": onp.mean(transitions.reward),
            **{f"train/{k}": onp.mean(v) for k, v in metrics.items()},
        }
        wandb_run.log(log_dict, step=i)

        times.append(time.time())

    # Get final reward
    final_reward = eval_results[-1]["reward_mean"] if eval_results else float('nan')

    # Save
    os.makedirs(save_dir, exist_ok=True)
    results = {
        "env_name": env_name,
        "algorithm": "ppo",
        "seed": seed,
        "eval_results": eval_results,
        "final_reward": final_reward,
        "total_time": times[-1] - times[0],
    }

    results_path = os.path.join(save_dir, f"{env_name}_ppo_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")
    print(f"Final reward: {final_reward:.2f}")
    wandb.finish()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="go1_getup",
                       choices=["humanoid_getup", "go1_getup", "go1_joystick", "go1_handstand", "spot_getup"])
    parser.add_argument("--all_envs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_timesteps", type=int, default=3_000_000)
    parser.add_argument("--save_dir", type=str, default="./results_ppo")

    args = parser.parse_args()

    envs = ["go1_getup", "go1_joystick"] if args.all_envs else [args.env]

    all_results = {}
    for env_name in envs:
        try:
            results = train_ppo_on_env(
                env_name=env_name,
                seed=args.seed,
                num_timesteps=args.num_timesteps,
                save_dir=args.save_dir,
            )
            all_results[env_name] = results
        except Exception as e:
            print(f"Error training PPO on {env_name}: {e}")
            all_results[env_name] = {"error": str(e)}

    # Save combined
    os.makedirs(args.save_dir, exist_ok=True)
    combined_path = os.path.join(args.save_dir, "ppo_all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
