import argparse
import json
from pathlib import Path

import torch

from model import SKYNETModel
from tokenizer import encode, decode

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT/"checkpoints/binary_latest.pt"))
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    model = SKYNETModel(
        cfg["vocab_size"], cfg["block_size"], cfg["n_layer"],
        cfg["n_head"], cfg["n_embd"], cfg["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, args.tokens, args.temperature)
    print(decode(out[0].tolist()))

if __name__ == "__main__":
    main()
