from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

files = [p for p in RAW.rglob("*") if p.is_file()]
if not files:
    print("No files found in data/raw/")
    print("Put .txt, .py, .md, .kv, .json, .html, .js etc. there and run again.")
    sys.exit(1)

chunks = []
for p in files:
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            chunks.append(f"\n--- FILE: {p.name} ---\n{text}\n")
    except Exception as e:
        print("Skipping", p, e)

all_text = "\n".join(chunks).encode("utf-8")
if len(all_text) < 5000:
    print("Warning: dataset is very small. Add substantially more clean training text.")

arr = np.frombuffer(all_text, dtype=np.uint8).copy()
split = max(1, int(len(arr) * 0.9))

np.save(ROOT / "data" / "train.npy", arr[:split])
np.save(ROOT / "data" / "val.npy", arr[split:])

print(f"Total bytes: {len(arr):,}")
print(f"Train bytes: {split:,}")
print(f"Val bytes: {len(arr)-split:,}")
