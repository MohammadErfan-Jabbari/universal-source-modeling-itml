#!/usr/bin/env python3
"""Download and extract the text8 corpus."""
from __future__ import annotations
import argparse, io, urllib.request, zipfile
from pathlib import Path
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=TEXT8_URL)
    parser.add_argument("--out", default="activity_b_llmzip/data/text8.txt")
    args = parser.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.url} ...")
    with urllib.request.urlopen(args.url, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        if "text8" not in zf.namelist():
            raise RuntimeError(f"Expected text8 in zip, found {zf.namelist()}")
        data = zf.read("text8")
    out.write_bytes(data)
    print(f"Wrote {out} ({len(data)} bytes/chars)")
if __name__ == "__main__": main()
