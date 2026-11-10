#!/usr/bin/env python3
"""
Build the Chrome extension zip the same way every time, or check a downloaded
zip against the source it claims to come from.

    python scripts/build_extension_zip.py                    # build from HEAD
    python scripts/build_extension_zip.py --ref <commit>     # build from a commit or tag
    python scripts/build_extension_zip.py --verify signlytic-extension.zip --ref <commit>

Build. The zip holds exactly the files git has under signlytic-extension/ at
the chosen commit, except the folder's README. They are read from git's own
store, not from the working folder, so uncommitted edits, stray files and
Windows line-ending conversion cannot leak in. Entries are sorted, carry a
fixed date and fixed permissions, and are compressed at a fixed level. Two
builds of one commit with the same Python therefore give the same bytes and the
same SHA256.

Verify. Compression is the one thing that can differ between machines: a
different zlib gives different bytes for the same files. So a release is
checked by content, not by the zip's own checksum. --verify unpacks nothing; it
hashes every file inside the zip and compares it with the same file at the
commit. Exit code 0 means the zip contains exactly that commit's extension and
nothing else.

Zips published before this script existed (extension v0.4.4 and earlier) were
made by hand from a Windows working folder, so a few text files in them have
Windows line endings where git stores Unix ones. Add --ignore-line-endings to
check those: such files are listed, and count as matching.

Needs only Python 3.8 or newer and git. No packages to install.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = "signlytic-extension"
# In the repository for people reading the code, not part of the extension.
EXCLUDED = {"README.md"}
# The earliest date a zip entry can carry. Any constant would do.
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
FILE_MODE = 0o100644


def git(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, input=data, capture_output=True, check=False
    )
    if result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def source_files(ref: str) -> dict[str, bytes]:
    """Every extension file at `ref`, as {path inside the zip: exact bytes}."""
    listing = git("ls-tree", "-r", "-z", ref, "--", SOURCE_DIR)
    blobs: list[tuple[str, str]] = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        _mode, kind, sha = meta.decode().split()
        if kind != "blob":
            continue
        name = path.decode("utf-8")[len(SOURCE_DIR) + 1:]
        if name in EXCLUDED:
            continue
        blobs.append((name, sha))
    if not blobs:
        sys.exit(f"No files found under {SOURCE_DIR}/ at {ref}.")

    # One git process for all files. Each reply is "<sha> blob <size>\n<bytes>\n".
    raw = git("cat-file", "--batch", data="".join(sha + "\n" for _, sha in blobs).encode())
    files: dict[str, bytes] = {}
    pos = 0
    for name, sha in blobs:
        end = raw.index(b"\n", pos)
        got_sha, kind, size = raw[pos:end].decode().split()
        if got_sha != sha or kind != "blob":
            sys.exit(f"Unexpected reply from git for {name}.")
        start = end + 1
        files[name] = raw[start:start + int(size)]
        pos = start + int(size) + 1
    return files


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_digest(files: dict[str, bytes]) -> str:
    """One hash for a set of files that ignores how they were compressed."""
    lines = "".join(f"{sha256(files[name])}  {name}\n" for name in sorted(files))
    return sha256(lines.encode("utf-8"))


def build(ref: str, out: Path) -> int:
    files = source_files(ref)
    commit = git("rev-parse", f"{ref}^{{commit}}").decode().strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=FIXED_DATE)
            info.create_system = 3  # Unix, so the permissions below are read as such
            info.external_attr = FILE_MODE << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, files[name], compresslevel=9)
    print(f"Built    {out}")
    print(f"Commit   {commit}")
    print(f"Files    {len(files)}")
    print(f"Contents {content_digest(files)}")
    print(f"SHA256   {sha256(out.read_bytes())}")
    return 0


def same_text(a: bytes, b: bytes) -> bool:
    return a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")


def verify(zip_path: Path, ref: str, ignore_line_endings: bool = False) -> int:
    if not zip_path.is_file():
        sys.exit(f"{zip_path} does not exist.")
    expected = source_files(ref)
    with zipfile.ZipFile(zip_path) as archive:
        actual = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if not info.is_dir()
        }

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(n for n in set(expected) & set(actual) if expected[n] != actual[n])
    line_endings_only: list[str] = []
    if ignore_line_endings:
        line_endings_only = [n for n in changed if same_text(expected[n], actual[n])]
        changed = [n for n in changed if n not in line_endings_only]

    commit = git("rev-parse", f"{ref}^{{commit}}").decode().strip()
    print(f"Zip      {zip_path}")
    print(f"SHA256   {sha256(zip_path.read_bytes())}")
    print(f"Commit   {commit}")
    print(f"Files    {len(actual)} in the zip, {len(expected)} in the source")
    for label, names in (("Missing from the zip", missing),
                         ("In the zip but not in the source", extra),
                         ("Different from the source", changed)):
        for name in names:
            print(f"  {label}: {name}")
    for name in line_endings_only:
        print(f"  Same text, Windows line endings: {name}")
    if missing or extra or changed:
        print("MISMATCH: the zip is not this commit's extension.")
        return 1
    if line_endings_only:
        print(f"OK: every file matches the source and nothing is extra. "
              f"{len(line_endings_only)} text file(s) differ only in line endings.")
        return 0
    print(f"Contents {content_digest(actual)}")
    print("OK: every file in the zip is identical to the source, and nothing is extra.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--ref", default="HEAD", help="commit or tag to build from or check against (default: HEAD)")
    parser.add_argument("--out", default="dist/signlytic-extension.zip", help="where to write the zip (default: dist/signlytic-extension.zip)")
    parser.add_argument("--verify", metavar="ZIP", help="check this zip against --ref and build nothing")
    parser.add_argument("--ignore-line-endings", action="store_true",
                        help="with --verify: accept text files that differ only in Windows or Unix line endings")
    args = parser.parse_args()
    if args.verify:
        return verify(Path(args.verify), args.ref, args.ignore_line_endings)
    return build(args.ref, Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
