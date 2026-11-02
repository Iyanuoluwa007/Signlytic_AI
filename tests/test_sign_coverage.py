from io import BytesIO
from pathlib import Path
import sys
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sign_coverage


class FakeConverter:
    def convert(self, word: str) -> list[str]:
        return {
            "went": ["GO"],
            "goes": ["GO"],
            "the": ["<unk>"],
            "thank-you": ["THANK", "YOU"],
        }[word]


def test_read_words_normalizes_and_preserves_frequency_order(tmp_path: Path) -> None:
    word_file = tmp_path / "words.txt"
    word_file.write_text("# rank 1\nWent\nGOES\n", encoding="utf-8")

    assert sign_coverage.read_words(word_file) == ["went", "goes"]


def test_read_words_rejects_duplicates_and_non_ascii(tmp_path: Path) -> None:
    word_file = tmp_path / "words.txt"
    word_file.write_text("hello\nHELLO\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        sign_coverage.read_words(word_file)

    word_file.write_text("café\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ASCII"):
        sign_coverage.read_words(word_file)


def test_converter_maps_inflections_and_separates_unknowns() -> None:
    mappings = sign_coverage.convert_words(["went", "goes", "the", "thank-you"], FakeConverter())

    assert [mapping.glosses for mapping in mappings] == [
        ("GO",), ("GO",), (), ("THANK", "YOU")
    ]


def test_fetch_status_uses_head_and_closes_without_reading_body(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status = 200

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def read(self, *_args: object, **_kwargs: object) -> bytes:
            raise AssertionError("sign response body must not be read")

    response = Response()
    seen = []

    def fake_urlopen(request, **_kwargs):
        seen.append((request.get_method(), request.full_url))
        return response

    monkeypatch.setattr(sign_coverage, "urlopen", fake_urlopen)

    assert sign_coverage.fetch_status("GO", timeout=1) == 200
    assert seen == [("HEAD", f"{sign_coverage.API_BASE}/GO")]
    assert response.closed


def test_fetch_status_closes_429_without_reading_body(monkeypatch: pytest.MonkeyPatch) -> None:
    body = BytesIO(b"response body must stay unread")
    error = HTTPError(f"{sign_coverage.API_BASE}/GO", 429, "rate limited", {}, body)

    def raise_429(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(sign_coverage, "urlopen", raise_429)

    assert sign_coverage.fetch_status("GO", timeout=1) == 429
    assert body.closed


def test_report_separates_missing_signs_and_unknown_words(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    mappings = sign_coverage.convert_words(["went", "goes", "the", "thank-you"], FakeConverter())
    sign_coverage.write_report(
        report,
        mappings,
        {"GO": 404, "THANK": 200, "YOU": None},
        interval=0.5,
        rate_limited=False,
    )

    content = report.read_text(encoding="utf-8")
    assert "Word-level coverage: 0/2" in content
    assert "- went: GO" in content
    assert "- goes: GO" in content
    assert "- the" in content
    assert "| 4 | thank-you | THANK, YOU | unknown |" in content
    assert "| GO | 404 |" in content
    assert "| YOU | unavailable |" in content


def test_main_stops_after_http_429(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    word_file = tmp_path / "words.txt"
    report = tmp_path / "report.md"
    word_file.write_text("first\nsecond\nthird\n", encoding="utf-8")
    requests: list[str] = []

    class OneGlossConverter:
        def convert(self, word: str) -> list[str]:
            return {"first": ["GO"], "second": ["THANK"], "third": ["YOU"]}[word]

    def fake_status(gloss: str, _timeout: float) -> int:
        requests.append(gloss)
        return 429 if gloss == "THANK" else 200

    monkeypatch.setattr(sign_coverage, "load_converter", lambda: OneGlossConverter())
    monkeypatch.setattr(sign_coverage, "fetch_status", fake_status)
    monkeypatch.setattr(sign_coverage.time, "sleep", lambda _seconds: None)

    result = sign_coverage.main(["--words", str(word_file), "--output", str(report)])

    assert result == 1
    assert requests == ["GO", "THANK"]
    assert "HTTP 429 received: yes" in report.read_text(encoding="utf-8")
