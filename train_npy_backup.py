import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model import SKYNETModel

ROOT = Path(__file__).resolve().parents[1]

def load_config(path):
    return json.loads(Path(path).read_text())

def get_batch(data, batch_size, block_size, device):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[int(s):int(s)+block_size].astype(np.int64))
        for s in starts
    ])
    y = torch.stack([
        torch.from_numpy(data[int(s)+1:int(s)+block_size+1].astype(np.int64))
        for s in starts
    ])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(cfg["eval_steps"]):
            x, y = get_batch(data, cfg["batch_size"], cfg["block_size"], device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT/"configs/small.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_data = np.load(ROOT/"data"/"train.npy", mmap_mode="r")
    val_data = np.load(ROOT/"data"/"val.npy", mmap_mode="r")

    model = SKYNETModel(
        cfg["vocab_size"], cfg["block_size"], cfg["n_layer"],
        cfg["n_head"], cfg["n_embd"], cfg["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )

    ckpt_dir = ROOT/"checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    latest = ckpt_dir/"latest.pt"
    start = 0

    if latest.exists():
        checkpoint = torch.load(latest, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["step"] + 1
        print("Resumed from step", start)

    model.train()
    pbar = tqdm(range(start, cfg["max_steps"]), initial=start, total=cfg["max_steps"])

    for step in pbar:
        x, y = get_batch(model and train_data, cfg["batch_size"], cfg["block_size"], device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            print(f"\nstep {step}: train={losses['train']:.4f} val={losses['val']:.4f}")

        if step % cfg["save_interval"] == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg
            }, latest)

    torch.save({
        "step": cfg["max_steps"] - 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg
    }, latest)

    print("Training complete:", latest)

if __name__ == "__main__":
    main()
