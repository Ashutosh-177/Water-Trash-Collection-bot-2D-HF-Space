#!/usr/bin/env python3
"""
train.py — Train a PPO agent to collect trash using stable-baselines3.

The agent learns from the 6-dimensional observation vector which simulates
what a camera + perception pipeline would provide:
  - Robot position (x, y, theta)
  - Nearest trash distance and relative angle (from camera detection)
  - Remaining trash count

Usage:
    python train.py                          # Train on medium difficulty
    python train.py --task hard --steps 200000  # Train on hard
    python train.py --eval                   # Evaluate a saved model
    python train.py --eval --render          # Watch the trained agent play
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def make_env(task_level="medium", render_mode=None):
    """Create wrapped environment."""
    from gym_wrapper import WaterTrashGymEnv
    return WaterTrashGymEnv(task_level=task_level, render_mode=render_mode)


def train(args):
    """Train PPO agent."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    print("=" * 60)
    print("  Water Trash Collector — RL Training (PPO)")
    print("=" * 60)
    print(f"  Task Level     : {args.task}")
    print(f"  Total Steps    : {args.steps:,}")
    print(f"  Learning Rate  : {args.lr}")
    print(f"  Batch Size     : {args.batch_size}")
    print(f"  Num Envs       : {args.n_envs}")
    print("=" * 60)

    # Create vectorized environments for parallel training
    env = make_vec_env(
        lambda: make_env(task_level=args.task),
        n_envs=args.n_envs,
        vec_env_cls=DummyVecEnv,
    )

    # Eval env (single instance)
    eval_env = make_vec_env(
        lambda: make_env(task_level=args.task),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )

    # Model save directory
    model_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    log_dir = os.path.join(os.path.dirname(__file__), "training_logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.steps // 10, 1000),
        save_path=model_dir,
        name_prefix="ppo_water_trash",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(args.steps // 20, 500),
        n_eval_episodes=5,
        deterministic=True,
    )

    # Create or load the PPO model
    model_path = os.path.join(model_dir, "best_model.zip")
    if args.resume and os.path.exists(model_path):
        print(f"\n  Resuming from: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,       # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            # tensorboard_log=log_dir,  # Enable if tensorboard is installed
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
            ),
        )

    print(f"\n  Starting training for {args.steps:,} steps...")
    start_time = time.time()

    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save final model
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"  Final model saved to: {final_path}.zip")

    env.close()
    eval_env.close()

    # Run quick evaluation
    print("\n  Running post-training evaluation...")
    evaluate(args, model_path=final_path)


def evaluate(args, model_path=None):
    """Evaluate a trained model."""
    from stable_baselines3 import PPO

    model_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    if model_path is None:
        # Try best_model first, then final_model
        best = os.path.join(model_dir, "best_model.zip")
        final = os.path.join(model_dir, "final_model.zip")
        if os.path.exists(best):
            model_path = best
        elif os.path.exists(final):
            model_path = final
        else:
            print("ERROR: No trained model found. Run training first:")
            print("  python train.py --steps 100000")
            sys.exit(1)

    print(f"\n  Loading model: {model_path}")
    model = PPO.load(model_path)

    render_mode = "human" if args.render else None
    env = make_env(task_level=args.task, render_mode=render_mode)

    n_episodes = args.eval_episodes
    rewards = []
    collected_counts = []
    step_counts = []

    print(f"  Evaluating {n_episodes} episodes on task={args.task}...")
    print("-" * 50)

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if render_mode:
                env.render()

        collected = info.get("collected", "?")
        total_trash = info.get("total_trash", "?")
        rewards.append(total_reward)
        collected_counts.append(collected if isinstance(collected, int) else 0)
        step_counts.append(steps)

        print(f"  Episode {ep+1:3d} | Steps: {steps:3d} | "
              f"Reward: {total_reward:.4f} | "
              f"Collected: {collected}/{total_trash}")

    print("-" * 50)
    print(f"  Mean Reward   : {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"  Mean Collected: {np.mean(collected_counts):.1f}")
    print(f"  Mean Steps    : {np.mean(step_counts):.1f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train & evaluate a PPO agent for Water Trash Collector"
    )
    parser.add_argument("--task", type=str, default="medium",
                        choices=["easy", "medium", "hard"],
                        help="Difficulty level (default: medium)")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Total training timesteps (default: 100000)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size (default: 64)")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel envs (default: 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from best_model.zip")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate instead of training")
    parser.add_argument("--render", action="store_true",
                        help="Render during evaluation")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")

    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
