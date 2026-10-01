# v0.1 uses a byte tokenizer.
# Every UTF-8 byte maps directly to token IDs 0..255.
# This keeps the first training system dependency-light and language-agnostic.

def encode(text):
    return list(text.encode("utf-8"))

def decode(ids):
    return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")

def vocab_size():
    return 256
