"""
GymDriveVLA: train a tiny language-conditioned driving policy.

Model:  camera frame -> ResNet-18 (pretrained) -> 512 features
        instruction -> frozen MiniLM sentence encoder -> 384 features
        concatenate -> MLP -> [steer (-1..1), gas (0..1), brake (0..1)]

Setup (Python 3.11 venv):
    pip install torch torchvision transformers numpy

Run:
    python train.py --data data --epochs 5
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel, AutoTokenizer

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, ~22M params, frozen
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- language
@torch.no_grad()
def embed_instructions(instructions):
    """Embed each unique instruction once with a frozen sentence encoder."""
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    enc = AutoModel.from_pretrained(TEXT_MODEL).eval()
    batch = tok(instructions, padding=True, return_tensors="pt")
    hidden = enc(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    emb = (hidden * mask).sum(1) / mask.sum(1)            # mean pooling
    return F.normalize(emb, dim=-1)                        # [K, 384]


# ---------------------------------------------------------------- model
class TinyDriveVLA(nn.Module):
    """Image features and instruction embedding are concatenated, then an MLP
    predicts the action."""
    def __init__(self, text_dim=384):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.DEFAULT)
        r.fc = nn.Identity()                      # ResNet now outputs 512 numbers
        self.vision = r
        self.head = nn.Sequential(nn.Linear(512 + text_dim, 256), nn.ReLU(),
                                  nn.Linear(256, 3))

    def forward(self, img, text):
        img_feat = self.vision(img)               # [B, 512]
        x = torch.cat([img_feat, text], dim=1)    # [B, 512 + 384]
        out = self.head(x)                        # [B, 3] raw numbers

        steer = torch.tanh(out[:, 0:1])           # -1 (full left) to +1 (full right)
        gas = torch.sigmoid(out[:, 1:2])          # 0 (off) to 1 (full)
        brake = torch.sigmoid(out[:, 2:3])        # 0 (off) to 1 (full)
        return torch.cat([steer, gas, brake], dim=1)


def preprocess(frames_uint8, device):
    """[B, 96, 96, 3] uint8 -> normalized [B, 3, 96, 96] float on device."""
    x = frames_uint8.to(device).permute(0, 3, 1, 2).float() / 255.0
    return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


# ---------------------------------------------------------------- data
def load_dataset(data_dir, val_frac):
    with open(os.path.join(data_dir, "index.csv")) as f:
        rows = list(csv.DictReader(f))

    instructions = sorted({r["instruction"] for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    n_val = max(1, int(len(seeds) * val_frac))
    val_seeds = set(seeds[-n_val:])          # hold out whole tracks, not random frames

    total = sum(int(r["steps"]) for r in rows)
    frames = np.empty((total, 96, 96, 3), dtype=np.uint8)   # preallocate, no 2x memory spike
    actions = np.empty((total, 3), dtype=np.float32)
    instr_ids = np.empty(total, dtype=np.int64)
    is_val = np.empty(total, dtype=bool)

    i = 0
    for r in rows:
        d = np.load(os.path.join(data_dir, r["file"]))
        n = len(d["actions"])
        frames[i:i + n] = d["frames"]
        actions[i:i + n] = d["actions"]
        instr_ids[i:i + n] = instructions.index(r["instruction"])
        is_val[i:i + n] = int(r["seed"]) in val_seeds
        i += n

    print(f"{total} frames, {len(instructions)} instructions, val tracks: {sorted(val_seeds)}")
    return (torch.from_numpy(frames), torch.from_numpy(actions),
            torch.from_numpy(instr_ids), torch.from_numpy(is_val), instructions)


# ---------------------------------------------------------------- train
def run_epoch(model, idx, frames, actions, instr_ids, text_emb, device,
              batch_size, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        idx = idx[torch.randperm(len(idx))]

    total_loss, abs_err, n = 0.0, torch.zeros(3), 0
    with torch.set_grad_enabled(training):
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            img = preprocess(frames[b], device)
            text = text_emb[instr_ids[b]].to(device)
            target = actions[b].to(device)

            pred = model(img, text)
            loss = F.mse_loss(pred, target)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(b)
            abs_err += (pred - target).abs().sum(0).detach().cpu()
            n += len(b)

    mae = abs_err / n
    return total_loss / n, mae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--out", default="checkpoints/drive_vla.pt")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    frames, actions, instr_ids, is_val, instructions = load_dataset(args.data, args.val_frac)
    text_emb = embed_instructions(instructions)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)

    model = TinyDriveVLA(text_dim=text_emb.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {n_params:.1f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, train_idx, frames, actions, instr_ids, text_emb,
                               device, args.batch_size, optimizer)
        va_loss, va_mae = run_epoch(model, val_idx, frames, actions, instr_ids, text_emb,
                                    device, args.batch_size)
        print(f"epoch {epoch}  train {tr_loss:.4f}  val {va_loss:.4f}  "
              f"val MAE steer/gas/brake {va_mae[0]:.3f}/{va_mae[1]:.3f}/{va_mae[2]:.3f}  "
              f"({time.time() - t0:.0f}s)")

        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "instructions": instructions,
                        "text_emb": text_emb, "text_model": TEXT_MODEL}, args.out)
            print(f"  saved {args.out}")


if __name__ == "__main__":
    main()