"""Minimal SHA256 checksum verifier for released datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from competition.release_test import _sha256_file


def verify_hashes(directory: str | Path) -> tuple[bool, list[str]]:
    """Verify files listed in `sha256.txt` under a directory."""

    base = Path(directory)
    sha_path = base / "sha256.txt"
    if not sha_path.exists():
        return False, [f"Missing checksum file: {sha_path}"]

    messages: list[str] = []
    ok = True

    for line_no, raw_line in enumerate(sha_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            ok = False
            messages.append(f"Line {line_no}: malformed checksum entry: {raw_line!r}")
            continue
        expected_hash, filename = parts
        file_path = base / filename.strip()
        if not file_path.exists():
            ok = False
            messages.append(f"FAIL {filename}: missing file")
            continue
        actual_hash = _sha256_file(file_path)
        if actual_hash != expected_hash:
            ok = False
            messages.append(f"FAIL {filename}: hash mismatch")
        else:
            messages.append(f"PASS {filename}")

    if not messages:
        return False, [f"No checksum entries found in {sha_path}"]
    return ok, messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify files listed in sha256.txt.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing sha256.txt")
    args = parser.parse_args()

    ok, messages = verify_hashes(args.dir)
    print(f"Checksum verification for {args.dir}")
    for msg in messages:
        print(f"  {msg}")
    print("STATUS: PASS" if ok else "STATUS: FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

