"""
Scripted expert + demo collector for Gymnasium CarRacing.

The expert follows the track centerline with pure pursuit and controls speed
based on the instruction. Each episode is saved as an .npz file
(frames, actions, instruction), and index.csv lists all episodes.

Setup (Mac):
    brew install swig
    pip install "gymnasium[box2d]" numpy

Quick visual check:
    python collect_demos.py --episodes 1 --render
Collect data:
    python collect_demos.py --episodes 50 --out data
"""
import argparse
import csv
import math
import os

import gymnasium as gym
import numpy as np

# Target speeds per instruction. Tune after watching a few episodes.
INSTRUCTIONS = {"go slow": 25.0, "go fast": 55.0}

STEER_SIGN = 1.0     # flip to -1.0 if the car steers away from the track
SKIP_STEPS = 50      # CarRacing zooms in at the start; don't record those frames


def expert_action(env, target_speed, lookahead=6, steer_gain=1.5):
    """Pure pursuit on the centerline plus a simple speed controller."""
    u = env.unwrapped
    hull = u.car.hull
    pos = np.array(tuple(hull.position))
    track = np.array(u.track)[:, 2:4]          # entries are (alpha, beta, x, y)

    i = int(np.argmin(np.linalg.norm(track - pos, axis=1)))
    target = track[(i + lookahead) % len(track)]
    d = target - pos

    fwd = np.array(tuple(hull.GetWorldVector((0, 1))))
    right = np.array(tuple(hull.GetWorldVector((1, 0))))
    angle_err = math.atan2(d @ right, d @ fwd)
    steer = float(np.clip(STEER_SIGN * steer_gain * angle_err, -1.0, 1.0))

    speed = float(np.linalg.norm(tuple(hull.linearVelocity)))
    if speed < target_speed:
        gas = float(np.clip(0.1 * (target_speed - speed), 0.0, 1.0))
        gas *= 1.0 - 0.5 * abs(steer)          # ease off in sharp turns to avoid skids
        brake = 0.0
    else:
        gas = 0.0
        brake = float(np.clip(0.05 * (speed - target_speed), 0.0, 0.8))

    return np.array([steer, gas, brake], dtype=np.float32)


def run_episode(env, instruction, seed, noise_std, noise_prob, max_steps, rng):
    obs, _ = env.reset(seed=seed)
    frames, actions = [], []
    target_speed = INSTRUCTIONS[instruction]

    for t in range(max_steps):
        clean = expert_action(env, target_speed)

        # Execute a noisy action sometimes, but always label with the clean one.
        # This shows the model how to recover from small mistakes.
        executed = clean.copy()
        if t >= SKIP_STEPS and rng.random() < noise_prob:
            executed[0] = np.clip(executed[0] + rng.normal(0, noise_std), -1, 1)

        if t >= SKIP_STEPS:
            frames.append(obs)
            actions.append(clean)

        obs, _, terminated, truncated, _ = env.step(executed)
        if terminated or truncated:
            break

    return np.array(frames, dtype=np.uint8), np.array(actions, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50, help="episodes per instruction")
    p.add_argument("--out", default="data")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--noise-std", type=float, default=0.3)
    p.add_argument("--noise-prob", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = gym.make("CarRacing-v3", continuous=True,
                   render_mode="human" if args.render else None)
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(args.out, "index.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "instruction", "seed", "steps", "file"])
        ep = 0
        for instruction in INSTRUCTIONS:
            for k in range(args.episodes):
                seed = args.seed + k   # same tracks across instructions, cleaner comparison
                frames, actions = run_episode(env, instruction, seed, args.noise_std,
                                              args.noise_prob, args.max_steps, rng)
                fname = f"ep_{ep:05d}.npz"
                np.savez_compressed(os.path.join(args.out, fname), frames=frames,
                                    actions=actions, instruction=instruction)
                writer.writerow([ep, instruction, seed, len(frames), fname])
                print(f"[{ep}] {instruction!r} seed={seed} steps={len(frames)}")
                ep += 1

    env.close()


if __name__ == "__main__":
    main()