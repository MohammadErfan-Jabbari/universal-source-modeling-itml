#!/usr/bin/env python3
"""Classical compression baselines for Activity B."""
from __future__ import annotations
import argparse, bz2, json, lzma, time, zlib
from pathlib import Path
def bpc(num_bytes: int, num_chars: int) -> float: return 8.0 * num_bytes / num_chars
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", required=True)
    parser.add_argument("--max-chars", type=int, default=1_000_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raw = Path(args.text_path).read_bytes()[: args.max_chars]
    if not raw: raise SystemExit(f"No bytes read from {args.text_path}")
    n = len(raw)
    results: dict[str, object] = {"text_path": args.text_path, "chars": n, "raw_bits_per_character": 8.0, "methods": {}}
    methods = {"zlib_9": lambda x: zlib.compress(x, level=9), "bz2_9": lambda x: bz2.compress(x, compresslevel=9), "lzma_9": lambda x: lzma.compress(x, preset=9)}
    for name, fn in methods.items():
        t0 = time.perf_counter(); comp = fn(raw); elapsed = time.perf_counter() - t0
        results["methods"][name] = {"compressed_bytes": len(comp), "bits_per_character": bpc(len(comp), n), "elapsed_seconds": elapsed}
        print(f"{name}: {bpc(len(comp), n):.6f} bpc ({len(comp)} bytes, {elapsed:.3f}s)")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")
if __name__ == "__main__": main()
