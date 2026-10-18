"""
GymDriveVLA: record a GIF of the trained model driving.

Side by side, same unseen track, one panel per instruction:
    python record_gif.py --out gymdrivevla.gif
Single instruction:
    python record_gif.py --instructions "go fast" --out go_fast.gif

Needs train.py and collect_demos.py in the same folder.
"""
import argparse

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from collect_demos import INSTRUCTIONS, expert_action
from train import TinyDriveVLA, preprocess

SKIP_STEPS = 50   # zoom-in; expert drives, frames not recorded


def record(env, model, text, warmup_speed, seed, device, steps, every, width):
    obs, _ = env.reset(seed=seed)
    frames = []
    for t in range(SKIP_STEPS + steps):
        if t < SKIP_STEPS:
            action = expert_action(env, warmup_speed)
        else:
            img = preprocess(torch.from_numpy(obs.copy()).unsqueeze(0), device)
            with torch.no_grad():
                action = model(img, text)[0].cpu().numpy()
            if (t - SKIP_STEPS) % every == 0:
                frame = Image.fromarray(env.render())          # full-res 600x400 view
                h = int(frame.height * width / frame.width)
                frames.append(frame.resize((width, h), Image.LANCZOS))

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return frames


def label(frame, text):
    draw = ImageDraw.Draw(frame)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:                     # older Pillow
        font = ImageFont.load_default()
    draw.rectangle([0, 0, frame.width, 28], fill=(0, 0, 0))
    draw.text((8, 4), text, fill=(255, 255, 255), font=font)
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/drive_vla.pt")
    p.add_argument("--instructions", nargs="+", default=["go slow", "go fast"])
    p.add_argument("--seed", type=int, default=1000, help="an unseen track")
    p.add_argument("--steps", type=int, default=450, help="model steps to record")
    p.add_argument("--every", type=int, default=3, help="keep every Nth frame")
    p.add_argument("--width", type=int, default=320, help="width of each panel")
    p.add_argument("--out", default="gymdrivevla.gif")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    instructions, text_emb = ckpt["instructions"], ckpt["text_emb"]
    model = TinyDriveVLA(text_dim=text_emb.shape[1])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")

    panels = []
    for instr in args.instructions:
        text = text_emb[instructions.index(instr)].unsqueeze(0).to(device)
        frames = record(env, model, text, INSTRUCTIONS[instr], args.seed, device,
                        args.steps, args.every, args.width)
        panels.append([label(f, f'"{instr}"') for f in frames])
        print(f"{instr!r}: {len(frames)} frames")
    env.close()

    # Pad shorter panels by holding their last frame, then place side by side
    n = max(len(p) for p in panels)
    panels = [p + [p[-1]] * (n - len(p)) for p in panels]
    w, h = panels[0][0].size
    gif_frames = []
    for i in range(n):
        canvas = Image.new("RGB", (w * len(panels), h))
        for j, p in enumerate(panels):
            canvas.paste(p[i], (j * w, 0))
        gif_frames.append(canvas.convert("P", palette=Image.ADAPTIVE))

    duration_ms = int(1000 / 50 * args.every)      # CarRacing runs at 50 steps/sec
    gif_frames[0].save(args.out, save_all=True, append_images=gif_frames[1:],
                       duration=duration_ms, loop=0, optimize=True)
    print(f"saved {args.out} ({n} frames)")


if __name__ == "__main__":
    main()