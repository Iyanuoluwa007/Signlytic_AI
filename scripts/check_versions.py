#!/usr/bin/env python3
"""
Check that every hardcoded version agrees with its source of truth.

Three product streams move independently, and each one's version is written out
by hand in several places across two repositories. A missed one does not break a
build; it ships a download link pointing at a release that does not exist, which
is only discovered by a user clicking it.

    python scripts/check_versions.py            # local files only
    python scripts/check_versions.py --remote   # also the Signlytic-Overlay
                                                # READMEs and the live assets

Exit code is 0 when everything agrees and 1 otherwise, so this can gate a
release.

Two kinds of check run:

  URLs      Every GitHub release URL found anywhere in the tracked text is
            parsed, matched to the stream that owns it by repository and tag
            shape, and its version compared with that stream's source of truth.
            Asset filenames carry the version separately from the tag, so both
            are checked. This needs no maintenance when a new link is added.

  Anchors   The handful of places where a version appears as prose or a button
            label rather than a URL. These have to be listed by hand, so each
            one is also asserted to still be present: a check that silently
            stops checking, because a file was reworded, is worse than no check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OWNER = "Iyanuoluwa007"
MAIN_REPO = "Signlytic_AI"
OVERLAY_REPO = "Signlytic-Overlay"

SEMVER = r"\d+\.\d+\.\d+"


# --------------------------------------------------------------------------
# Streams
# --------------------------------------------------------------------------

def _json_version(rel_path: str) -> str:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))["version"]


class Stream:
    def __init__(self, key, label, source_desc, version, tag_prefix, repo):
        self.key = key
        self.label = label
        self.source_desc = source_desc
        self.version = version
        self.tag_prefix = tag_prefix   # "v" or "desktop-v"
        self.repo = repo

    @property
    def tag(self) -> str:
        return f"{self.tag_prefix}{self.version}"


def load_streams() -> dict:
    return {
        "extension": Stream(
            "extension", "Chrome extension",
            "signlytic-extension/manifest.json",
            _json_version("signlytic-extension/manifest.json"),
            "v", OVERLAY_REPO,
        ),
        "desktop": Stream(
            "desktop", "Desktop app",
            "signlytic-desktop/package.json",
            _json_version("signlytic-desktop/package.json"),
            "desktop-v", OVERLAY_REPO,
        ),
        "local": Stream(
            "local", "Local Python system",
            "signlytic-ai-website/app/demo/page.tsx (LOCAL_VERSION)",
            "",  # filled in below: this stream has no manifest to read
            "v", MAIN_REPO,
        ),
    }


LOCAL_VERSION_RE = re.compile(r'LOCAL_VERSION\s*=\s*"v(' + SEMVER + r')"')


# --------------------------------------------------------------------------
# Which files are searched
# --------------------------------------------------------------------------

SEARCH_FILES = [
    "README.md",
    "signlytic-extension/README.md",
    "signlytic-desktop/README.md",
    "signlytic-ai-website/app/page.tsx",
    "signlytic-ai-website/app/extension/page.tsx",
    "signlytic-ai-website/app/demo/page.tsx",
]

# Fetched only with --remote. The overlay repo is not checked out here.
REMOTE_FILES = [
    ("README.md", f"https://raw.githubusercontent.com/{OWNER}/{OVERLAY_REPO}/main/README.md"),
    ("Extension/README.md", f"https://raw.githubusercontent.com/{OWNER}/{OVERLAY_REPO}/main/Extension/README.md"),
    ("Software App/README.md", f"https://raw.githubusercontent.com/{OWNER}/{OVERLAY_REPO}/main/Software%20App/README.md"),
]


# --------------------------------------------------------------------------
# Anchors: versions that appear as prose or a label, not inside a URL
# --------------------------------------------------------------------------
# (file, stream key, compiled regex capturing the bare version, description)

ANCHORS = [
    ("signlytic-extension/README.md", "extension",
     re.compile(r"the extension source\. Version (" + SEMVER + r")"),
     "intro line"),
    ("signlytic-ai-website/app/extension/page.tsx", "extension",
     re.compile(r"Download Beta - v(" + SEMVER + r")"),
     "download button label"),
    ("signlytic-ai-website/app/demo/page.tsx", "local",
     LOCAL_VERSION_RE,
     "LOCAL_VERSION constant"),
]

REMOTE_ANCHORS = [
    ("Extension/README.md", "extension",
     re.compile(r"\*\*Version:\*\* (" + SEMVER + r")"),
     "version header"),
    ("README.md", "extension",
     re.compile(r"\| Chrome Extension \| Chrome \| v(" + SEMVER + r") beta"),
     "downloads table"),
    ("README.md", "desktop",
     re.compile(r"\| Desktop App \| Windows 11 \| v(" + SEMVER + r") beta"),
     "downloads table, Windows"),
    ("README.md", "desktop",
     re.compile(r"\| Desktop App \| macOS 13\+ \| v(" + SEMVER + r") beta"),
     "downloads table, macOS"),
    ("Software App/README.md", "desktop",
     re.compile(r"\| Windows 11 \| v(" + SEMVER + r") beta"),
     "download table, Windows"),
    ("Software App/README.md", "desktop",
     re.compile(r"\| macOS 13 or later \| v(" + SEMVER + r") beta"),
     "download table, macOS"),
]


# --------------------------------------------------------------------------
# URL parsing
# --------------------------------------------------------------------------

URL_RE = re.compile(
    r"https://github\.com/" + re.escape(OWNER) + r"/(?P<repo>[A-Za-z0-9_.-]+)"
    r"/releases/(?:download|tag)/(?P<tag>[A-Za-z0-9_.-]+)"
    r"(?:/(?P<asset>[^)\s\"'<>]+))?"
)


def stream_for(repo: str, tag: str, streams: dict):
    """Work out which stream a release URL belongs to, by repo and tag shape."""
    if repo == OVERLAY_REPO:
        if tag.startswith("desktop-v"):
            return streams["desktop"]
        if tag.startswith("v"):
            return streams["extension"]
        return None
    if repo == MAIN_REPO and tag.startswith("v"):
        return streams["local"]
    return None


def check_urls(label: str, text: str, streams: dict, problems: list, seen_urls: set):
    for m in URL_RE.finditer(text):
        repo, tag, asset = m.group("repo"), m.group("tag"), m.group("asset")
        line = text[: m.start()].count("\n") + 1
        st = stream_for(repo, tag, streams)
        if st is None:
            problems.append(f"{label}:{line}  unrecognised release URL, repo={repo} tag={tag}")
            continue

        seen_urls.add(m.group(0))

        if tag != st.tag:
            problems.append(
                f"{label}:{line}  {st.label}: tag is {tag}, expected {st.tag}"
                f"  (source of truth: {st.source_desc})")

        # The asset filename carries the version independently of the tag, so a
        # correct tag with a stale filename is a live 404. Check it separately.
        if asset:
            found = re.findall(SEMVER, asset)
            for v in found:
                if v != st.version:
                    problems.append(
                        f"{label}:{line}  {st.label}: asset filename says {v}, "
                        f"expected {st.version}  ({asset})")


def check_anchors(anchors, read, streams: dict, problems: list):
    for rel, key, pattern, desc in anchors:
        text = read(rel)
        if text is None:
            problems.append(f"{rel}  missing, but an anchor is defined for it")
            continue
        st = streams[key]
        found = pattern.findall(text)
        if not found:
            problems.append(
                f"{rel}  anchor no longer matches: {desc}. The file was probably "
                f"reworded. Update ANCHORS in scripts/check_versions.py, or this "
                f"location stops being checked.")
            continue
        for v in found:
            if v != st.version:
                problems.append(
                    f"{rel}  {st.label} {desc}: says {v}, expected {st.version}")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", action="store_true",
                    help="also check the Signlytic-Overlay READMEs and that every "
                         "release asset URL actually resolves")
    args = ap.parse_args()

    streams = load_streams()

    # The local Python system has no manifest; its own constant is the source.
    demo = (ROOT / "signlytic-ai-website/app/demo/page.tsx").read_text(encoding="utf-8")
    m = LOCAL_VERSION_RE.search(demo)
    if not m:
        print("  could not read LOCAL_VERSION from the demo page", file=sys.stderr)
        return 1
    streams["local"].version = m.group(1)

    print("Source of truth")
    for st in streams.values():
        print(f"  {st.label:<22} {st.tag:<16} {st.source_desc}")
    print()

    problems: list = []
    seen_urls: set = set()

    def read_local(rel):
        p = ROOT / rel
        return p.read_text(encoding="utf-8") if p.exists() else None

    for rel in SEARCH_FILES:
        text = read_local(rel)
        if text is None:
            problems.append(f"{rel}  listed in SEARCH_FILES but not found")
            continue
        check_urls(rel, text, streams, problems, seen_urls)

    check_anchors(ANCHORS, read_local, streams, problems)

    if args.remote:
        import urllib.request

        def fetch(url):
            req = urllib.request.Request(url, headers={"User-Agent": "signlytic-version-check"})
            return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")

        remote_text = {}
        print(f"Fetching {OVERLAY_REPO} docs")
        for rel, url in REMOTE_FILES:
            try:
                remote_text[rel] = fetch(url)
                print(f"  ok    {rel}")
            except Exception as e:
                remote_text[rel] = None
                problems.append(f"{OVERLAY_REPO}/{rel}  could not fetch: {e}")
        print()

        for rel, text in remote_text.items():
            if text:
                check_urls(f"{OVERLAY_REPO}/{rel}", text, streams, problems, seen_urls)
        check_anchors(REMOTE_ANCHORS, lambda r: remote_text.get(r), streams, problems)

        # A version can be internally consistent everywhere and still point at a
        # release that was never published, so confirm the assets exist.
        assets = sorted(u for u in seen_urls if "/releases/download/" in u)
        print(f"Checking {len(assets)} release assets resolve")
        for url in assets:
            try:
                req = urllib.request.Request(url, method="HEAD",
                                             headers={"User-Agent": "signlytic-version-check"})
                code = urllib.request.urlopen(req, timeout=45).status
            except Exception as e:
                code = getattr(e, "code", None) or str(e)
            if code == 200:
                print(f"  200   {url.rsplit('/', 1)[-1]}")
            else:
                problems.append(f"release asset does not resolve ({code}): {url}")
        print()

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1

    scope = "local files and the overlay repo" if args.remote else "local files"
    print(f"All versions agree across {scope}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
