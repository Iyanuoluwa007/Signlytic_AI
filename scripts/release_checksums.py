#!/usr/bin/env python3
"""
Write or check the SHA256SUMS.txt that is attached to every release.

    python scripts/release_checksums.py FILE [FILE ...]      # write SHA256SUMS.txt beside the first file
    python scripts/release_checksums.py --out sums.txt FILE [FILE ...]
    python scripts/release_checksums.py --check SHA256SUMS.txt

The file uses the same format as the `sha256sum` and `shasum -a 256` tools
("<hash>  <name>", one line per file), so anyone can check a download with
those, with `Get-FileHash` on Windows, or with --check here.

Names are written the way GitHub serves them. GitHub replaces each space in an
asset name with a dot, so "Signlytic AI Setup 0.3.11.exe" is listed as
"Signlytic.AI.Setup.0.3.11.exe", the name a user actually downloads.

Needs only Python 3.8 or newer.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def served_name(path: Path) -> str:
    return path.name.replace(" ", ".")


def write(files: list[Path], out: Path | None) -> int:
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        sys.exit("Not found: " + ", ".join(missing))
    names = [served_name(f) for f in files]
    if len(set(names)) != len(names):
        sys.exit("Two files would be listed under the same name.")
    out = out or files[0].parent / "SHA256SUMS.txt"
    lines = [f"{sha256_of(f)}  {name}" for f, name in sorted(zip(files, names), key=lambda p: p[1])]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {out}")
    for line in lines:
        print(line)
    return 0


def check(sums: Path) -> int:
    if not sums.is_file():
        sys.exit(f"{sums} does not exist.")
    failures = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, _, name = line.partition(" ")
        # sha256sum marks binary mode with a leading asterisk.
        name = name.strip().lstrip("*")
        target = sums.parent / name
        if not target.is_file():
            # The file may still have the spaces it was built with.
            spaced = [p for p in sums.parent.iterdir() if p.is_file() and served_name(p) == name]
            target = spaced[0] if spaced else target
        if not target.is_file():
            print(f"{name}: MISSING")
            failures += 1
        elif sha256_of(target) == expected.lower():
            print(f"{name}: OK")
        else:
            print(f"{name}: FAILED")
            failures += 1
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or check a release's SHA256SUMS.txt.")
    parser.add_argument("files", nargs="*", type=Path, help="release files to list")
    parser.add_argument("--out", type=Path, help="where to write (default: SHA256SUMS.txt beside the first file)")
    parser.add_argument("--check", type=Path, metavar="SUMS", help="check the files listed in this SHA256SUMS.txt")
    args = parser.parse_args()
    if args.check:
        return check(args.check)
    if not args.files:
        parser.error("give at least one file, or --check SHA256SUMS.txt")
    return write(args.files, args.out)


if __name__ == "__main__":
    sys.exit(main())
