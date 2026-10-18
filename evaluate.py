"""
GymDriveVLA: run the trained model in CarRacing (closed loop).

Watch it drive:
    python evaluate.py --instruction "go fast" --episodes 1 --render
Evaluate both instructions on unseen tracks:
    python evaluate.py --episodes 20

Needs train.py and collect_demos.py in the same folder.
"""
import argparse

import gymnasium as gym
import numpy as np
import torch

from collect_demos import INSTRUCTIONS, expert_action
from train import TinyDriveVLA, preprocess

SKIP_STEPS = 50   # zoom-in frames; the expert drives these, same as during data collection


def run_episode(env, model, text, seed, device, warmup_speed, max_steps=1000):
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    speeds, off_track, reason = [], 0, "timeout"

    for t in range(max_steps):
        if t < SKIP_STEPS:
            # Expert warm-up so the model starts from the same kind of state it saw in training
            action = expert_action(env, warmup_speed)
        else:
            img = preprocess(torch.from_numpy(obs.copy()).unsqueeze(0), device)
            with torch.no_grad():
                action = model(img, text)[0].cpu().numpy()

            speeds.append(float(np.linalg.norm(tuple(u.car.hull.linearVelocity))))
            if all(len(w.tiles) == 0 for w in u.car.wheels):
                off_track += 1

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            reason = "lap" if u.new_lap else ("timeout" if truncated else "off_map")
            break

    steps = max(len(speeds), 1)
    return {
        "reason": reason,
        "track_covered": u.tile_visited_count / len(u.track),
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "off_track_pct": 100.0 * off_track / steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/drive_vla.pt")
    p.add_argument("--instruction", default=None, help="default: all trained instructions")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=1000, help="use seeds not seen in training")
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    instructions, text_emb = ckpt["instructions"], ckpt["text_emb"]

    model = TinyDriveVLA(text_dim=text_emb.shape[1])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    to_run = [args.instruction] if args.instruction else instructions
    for instr in to_run:
        if instr not in instructions:
            raise ValueError(f"{instr!r} not in trained instructions {instructions}")

    env = gym.make("CarRacing-v3", continuous=True,
                   render_mode="human" if args.render else None)

    for instr in to_run:
        text = text_emb[instructions.index(instr)].unsqueeze(0).to(device)
        warmup_speed = INSTRUCTIONS[instr]
        results = []
        for k in range(args.episodes):
            seed = args.seed_start + k
            r = run_episode(env, model, text, seed, device, warmup_speed)
            results.append(r)
            print(f"{instr!r} seed={seed}  {r['reason']:8s} "
                  f"covered {r['track_covered']:.0%}  speed {r['mean_speed']:.1f}  "
                  f"off-track {r['off_track_pct']:.1f}%")

        laps = np.mean([r["reason"] == "lap" for r in results])
        print(f"== {instr!r}: lap rate {laps:.0%}, "
              f"mean speed {np.mean([r['mean_speed'] for r in results]):.1f}, "
              f"off-track {np.mean([r['off_track_pct'] for r in results]):.1f}%\n")

    env.close()


if __name__ == "__main__":
    main()