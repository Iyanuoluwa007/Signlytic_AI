#!/usr/bin/env python3
"""Report sign coverage for ranked English words using the app's BSL glosses.

Each word is converted with the repository's simple ``TextToGloss`` rules.
Words that convert to ``<unk>`` are reported separately and excluded from the
coverage denominator. Repeated glosses are queried once with an HTTP HEAD
request, so no sign-frame response body is downloaded or read. Requests are at
least 0.5 seconds apart, and processing stops immediately on HTTP 429.

By default the script converts the first 500 entries in
``common_english_words.txt`` and writes ``sign_coverage_report.md``. Use
``--limit 50`` for the issue's smaller manual check.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_WORDS = SCRIPT_DIR / "common_english_words.txt"
DEFAULT_REPORT = SCRIPT_DIR / "sign_coverage_report.md"
API_BASE = "https://signlytic-ai-website.vercel.app/api/signs"
DEFAULT_INTERVAL = 0.5
UNSIGNED_TOKENS = {"<unk>", "<pad>", "<sos>", "<eos>"}


def default_ssl_context() -> ssl.SSLContext:
    """Use a system CA bundle when Python has no configured bundle of its own."""
    paths = ssl.get_default_verify_paths()
    if paths.cafile and Path(paths.cafile).is_file():
        return ssl.create_default_context(cafile=paths.cafile)
    for candidate in (
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
    ):
        if Path(candidate).is_file():
            return ssl.create_default_context(cafile=candidate)
    return ssl.create_default_context()


SSL_CONTEXT = default_ssl_context()


@dataclass(frozen=True)
class WordGloss:
    """One frequency-ranked English word and the glosses the app looks up."""

    word: str
    glosses: tuple[str, ...]


def read_words(path: Path) -> list[str]:
    """Read unique English words in order, ignoring blank and comment lines."""
    words: list[str] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        word = raw.strip()
        if not word or word.startswith("#"):
            continue
        if not word.isascii() or not word.isalpha():
            raise ValueError(f"{path}:{line_number}: expected one ASCII word per line")
        normalized = word.lower()
        if normalized in seen:
            raise ValueError(f"{path}:{line_number}: duplicate word {word!r}")
        words.append(normalized)
        seen.add(normalized)
    return words


def convert_words(words: list[str], converter: object) -> list[WordGloss]:
    """Map words through the app converter and preserve frequency order."""
    mappings: list[WordGloss] = []
    for word in words:
        converted = converter.convert(word)
        if not converted or any(token.lower() in UNSIGNED_TOKENS for token in converted):
            glosses: tuple[str, ...] = ()
        else:
            glosses = tuple(dict.fromkeys(token.upper() for token in converted))
        mappings.append(WordGloss(word, glosses))
    return mappings


def load_converter():
    """Load the same rule-based converter the app uses, without GPU modules."""
    inference_dir = str(REPO_ROOT / "src" / "inference")
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    from speech_to_bsl import TextToGloss

    return TextToGloss(mode="simple")


def fetch_status(gloss: str, timeout: float) -> int | None:
    """Return a HEAD response status only; never fetch or read sign frames."""
    request = Request(
        f"{API_BASE}/{quote(gloss, safe='')}",
        headers={"User-Agent": "Signlytic-sign-coverage/1.0"},
        method="HEAD",
    )
    try:
        response = urlopen(request, timeout=timeout, context=SSL_CONTEXT)
    except HTTPError as error:
        status = error.code
        error.close()
        return status
    except (URLError, TimeoutError, OSError):
        return None

    try:
        return response.status
    finally:
        response.close()


def _word_status(mapping: WordGloss, statuses: dict[str, int | None]) -> str:
    if not mapping.glosses:
        return "not signed (<unk>)"
    gloss_statuses = [statuses.get(gloss) for gloss in mapping.glosses]
    if any(status == 404 for status in gloss_statuses):
        return "missing"
    if all(status == 200 for status in gloss_statuses):
        return "present"
    return "unknown"


def write_report(
    path: Path,
    mappings: list[WordGloss],
    statuses: dict[str, int | None],
    *,
    interval: float,
    rate_limited: bool,
) -> None:
    """Write word/gloss/status metadata only, never API response bodies."""
    words_with_glosses = [mapping for mapping in mappings if mapping.glosses]
    known_words = [
        mapping
        for mapping in words_with_glosses
        if _word_status(mapping, statuses) in {"present", "missing"}
    ]
    present = [mapping for mapping in known_words if _word_status(mapping, statuses) == "present"]
    missing = [mapping for mapping in mappings if _word_status(mapping, statuses) == "missing"]
    no_gloss = [mapping for mapping in mappings if not mapping.glosses]
    unknown = [mapping for mapping in words_with_glosses if _word_status(mapping, statuses) == "unknown"]
    coverage = f"{len(present) / len(known_words) * 100:.1f}%" if known_words else "not available"
    unique_glosses = list(dict.fromkeys(gloss for mapping in mappings for gloss in mapping.glosses))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Sign coverage report",
        "",
        f"Generated: {generated}",
        "",
        "Conversion: repository `TextToGloss(mode=\"simple\")`; each English word is checked as the glosses the app actually looks up.",
        "Words converting to `<unk>` are listed separately and excluded from coverage.",
        "The API was queried with HEAD requests. Only status codes were recorded; response bodies and sign frames were never downloaded or read.",
        "",
        "Word list: first 500 entries from `common_english_words.txt`, in frequency order.",
        "Source: [yungng-lab/top10k-words](https://github.com/yungng-lab/top10k-words), MIT-licensed; see `WORDLIST_LICENSE.txt`.",
        f"Distinct glosses requested: {len(statuses)} of {len(unique_glosses)}; minimum interval: {interval:.1f} seconds.",
        f"Word-level coverage: {len(present)}/{len(known_words)} convertible words with all signs present ({coverage}).",
        f"HTTP 429 received: {'yes, stopped immediately' if rate_limited else 'no'}.",
        "",
        "## Words with at least one missing sign (frequency order)",
        "",
    ]
    lines.extend(f"- {mapping.word}: {', '.join(gloss for gloss in mapping.glosses if statuses.get(gloss) == 404)}" for mapping in missing)
    if not missing:
        lines.append("- None in the completed requests.")
    lines.extend(["", "## Words not converted to a sign gloss", ""])
    lines.extend(f"- {mapping.word}" for mapping in no_gloss)
    if not no_gloss:
        lines.append("- None.")
    lines.extend(["", "## Words with unknown sign status", ""])
    lines.extend(f"- {mapping.word}: {', '.join(gloss for gloss in mapping.glosses if statuses.get(gloss) not in (200, 404))}" for mapping in unknown)
    if not unknown:
        lines.append("- None.")
    lines.extend(["", "## Word to gloss results", "", "| Rank | English word | App glosses | Result |", "|---:|---|---|---|"])
    for rank, mapping in enumerate(mappings, 1):
        glosses = ", ".join(mapping.glosses) if mapping.glosses else "<unk>"
        lines.append(f"| {rank} | {mapping.word} | {glosses} | {_word_status(mapping, statuses)} |")
    lines.extend(["", "## Unique gloss lookup statuses", "", "| Gloss | HTTP status |", "|---|---:|"])
    for gloss in unique_glosses:
        status = statuses.get(gloss)
        lines.append(f"| {gloss} | {status if status is not None else 'unavailable'} |")
    if rate_limited:
        lines.extend(["", "This report is partial because the service returned HTTP 429; the script stopped without sending further requests."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS, help="one word per line, in frequency order")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT, help="Markdown report path")
    parser.add_argument("--limit", type=int, help="convert and check only the first N words")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="minimum seconds between unique gloss requests (at least 0.5)")
    parser.add_argument("--timeout", type=float, default=15.0, help="per-request timeout in seconds")
    args = parser.parse_args(argv)

    if args.interval < DEFAULT_INTERVAL:
        parser.error("--interval must be at least 0.5 seconds")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    try:
        words = read_words(args.words)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.limit is not None:
        words = words[: args.limit]
    if not words:
        parser.error("the selected word list is empty")

    mappings = convert_words(words, load_converter())
    glosses = list(dict.fromkeys(gloss for mapping in mappings for gloss in mapping.glosses))
    statuses: dict[str, int | None] = {}
    rate_limited = False
    for index, gloss in enumerate(glosses):
        if index:
            time.sleep(args.interval)
        status = fetch_status(gloss, args.timeout)
        if status is None:
            # Retry a dropped request once; transport failures are not missing signs.
            time.sleep(args.interval)
            status = fetch_status(gloss, args.timeout)
        statuses[gloss] = status
        if status is None:
            print(f"{index + 1}/{len(glosses)} {gloss}: unavailable", file=sys.stderr)
        else:
            print(f"{index + 1}/{len(glosses)} {gloss}: HTTP {status}", file=sys.stderr)
        if status == 429:
            rate_limited = True
            break

    write_report(args.output, mappings, statuses, interval=args.interval, rate_limited=rate_limited)
    print(f"Wrote status-only report to {args.output}")
    return 1 if rate_limited or any(status is None for status in statuses.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
