"""
Gloss to English conversion: provider order, the rate-limit cooldown, the
rule-based fallback and the rule against inventing words.

Everything here runs with no network, no API keys, no GPU and no torch:

- The Cerebras call is plain HTTP, so `requests.post` is replaced with a fake.
- The Groq fallback goes through its SDK, so a fake client stands in for it.
- The clock is a fake, so the cooldown is tested without waiting.
- A fixture makes any real network connection fail the test.

The modules are loaded straight from their files. Importing them as
`src.inference.<name>` would run `src/inference/__init__.py`, which imports
torch.

The gloss lists are ordinary words typed by hand. No sign data is used.

    pip install -r requirements-dev.txt
    pytest tests/ -q
"""
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"


def _load(name, relative_path):
    """Load one source file as a module, keeping torch out of the process."""
    saved = sys.modules.get("torch", "absent")
    # A None entry makes `import torch` raise ImportError, which each module
    # already treats as "torch is not installed".
    sys.modules["torch"] = None
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved == "absent":
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved
    return module


g2t = _load("signlytic_gloss_to_text", "src/inference/gloss_to_text.py")
b2s = _load("signlytic_bsl_to_speech", "src/inference/bsl_to_speech.py")
s2b = _load("signlytic_speech_to_bsl", "src/inference/speech_to_bsl.py")
GlossToText = g2t.GlossToText


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeClock:
    def __init__(self, now=1_000.0):
        self.now = now

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeResponse:
    def __init__(self, status_code=200, content="A sentence.", headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeCerebras:
    """Stands in for requests.post. Replies are used in order; the last repeats."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        reply = self.replies[0] if len(self.replies) == 1 else self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeGroq:
    """Stands in for the Groq SDK client: client.chat.completions.create(...)."""

    def __init__(self, content="Groq sentence.", error=None):
        self.content = content
        self.error = error
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = type("Message", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No keys from the developer's own shell, no vocabulary file, no network."""
    for key in ("CEREBRAS_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    # False means "looked and found no vocabulary file", which is what a fresh
    # checkout has. Tests that need a vocabulary set their own.
    monkeypatch.setattr(GlossToText, "_VOCABULARY", False)

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a real network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(g2t, "time", fake)
    return fake


def hosted(monkeypatch, cerebras, groq=None, cerebras_key="test-cerebras-key"):
    """A GlossToText in hosted ('groq') mode wired to the fakes."""
    if cerebras_key:
        monkeypatch.setenv("CEREBRAS_API_KEY", cerebras_key)
    monkeypatch.setattr(requests, "post", cerebras)
    if groq is None:
        # No Groq key and no SDK: the fallback is simply unavailable.
        monkeypatch.setitem(sys.modules, "groq", None)
        return GlossToText(mode="groq")
    # The real SDK is not installed in the test environment, so a stand-in
    # module hands back the fake client.
    sdk = type("FakeGroqSdk", (), {"Groq": staticmethod(lambda api_key: groq)})
    monkeypatch.setitem(sys.modules, "groq", sdk)
    return GlossToText(mode="groq", groq_api_key="test-groq-key")


# --------------------------------------------------------------------------
# mode="simple": the rule-based converter
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "glosses, english",
    [
        ("THANK YOU MUCH", "Thank you much."),
        ("MY NAME SARAH", "My name sarah."),
        # A question word anywhere makes it a question.
        ("WHERE YOU LIVE", "Where you live?"),
        # A leading time marker is set off with a comma, and "meeting" gets an article.
        ("TOMORROW MEETING WHAT TIME", "Tomorrow, the meeting what time?"),
        # No second article after a possessive.
        ("MY DOCTOR HELP ME", "My doctor help me."),
    ],
)
def test_simple_mode_sentences(glosses, english):
    assert GlossToText(mode="simple").convert(glosses.split()) == english


def test_simple_mode_drops_special_tokens():
    converter = GlossToText(mode="simple")
    assert converter.convert(["<sos>", "HELLO", "<unk>", "<pad>", "<eos>"]) == "Hello."
    assert converter.convert(["<pad>", "<unk>"]) == ""


def test_empty_input_gives_empty_text():
    converter = GlossToText(mode="simple")
    assert converter.convert([]) == ""
    assert converter.convert(["", "   "]) == ""


def test_simple_mode_passes_unknown_gloss_through():
    english = GlossToText(mode="simple").convert(["ME", "WANT", "ZZZQQQ"])
    assert english == "Me want zzzqqq."
    assert "sleep" not in english.lower()


def test_simple_mode_makes_no_request(monkeypatch):
    cerebras = FakeCerebras(FakeResponse())
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-cerebras-key")
    monkeypatch.setattr(requests, "post", cerebras)
    GlossToText(mode="simple").convert(["HELLO", "YOU"])
    assert cerebras.calls == []


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError):
        GlossToText(mode="telepathy").convert(["HELLO"])


def test_convert_batch_converts_each_sequence():
    converter = GlossToText(mode="simple")
    assert converter.convert_batch([["THANK", "YOU"], [], ["WHERE", "YOU", "LIVE"]]) == [
        "Thank you.",
        "",
        "Where you live?",
    ]


# --------------------------------------------------------------------------
# Hosted mode: provider order
# --------------------------------------------------------------------------

def test_cerebras_is_tried_first_and_groq_is_left_alone(monkeypatch, clock):
    cerebras = FakeCerebras(FakeResponse(content="I went to the shop yesterday."))
    groq = FakeGroq()
    converter = hosted(monkeypatch, cerebras, groq)

    assert converter.convert(["YESTERDAY", "ME", "GO", "SHOP"]) == "I went to the shop yesterday."
    assert len(cerebras.calls) == 1
    assert groq.calls == []


def test_cerebras_request_is_what_the_provider_expects(monkeypatch, clock):
    cerebras = FakeCerebras(FakeResponse())
    converter = hosted(monkeypatch, cerebras, FakeGroq())
    converter.convert(["yesterday", "me", "go", "<unk>", "shop"])

    call = cerebras.calls[0]
    assert call["url"] == CEREBRAS_URL
    assert call["headers"]["Authorization"] == "Bearer test-cerebras-key"
    assert call["timeout"] == converter.cerebras_timeout
    body = call["json"]
    assert body["model"] == "gpt-oss-120b"
    # gpt-oss reasons before it answers; without this the answer can be crowded out.
    assert body["reasoning_effort"] == "low"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    # Glosses go out upper-cased, with special tokens removed.
    assert body["messages"][1]["content"] == "BSL Glosses: YESTERDAY ME GO SHOP"


@pytest.mark.parametrize(
    "failure",
    [
        FakeResponse(status_code=500),
        FakeResponse(status_code=401),
        requests.Timeout("timed out"),
        requests.ConnectionError("no route"),
        FakeResponse(content=""),
        FakeResponse(content="   "),
        FakeResponse(content=None),
    ],
    ids=["http-500", "http-401", "timeout", "connection-error", "empty", "blank", "null"],
)
def test_groq_serves_when_cerebras_fails(monkeypatch, clock, failure):
    cerebras = FakeCerebras(failure)
    groq = FakeGroq(content="I went to the shop yesterday.")
    converter = hosted(monkeypatch, cerebras, groq)

    assert converter.convert(["YESTERDAY", "ME", "GO", "SHOP"]) == "I went to the shop yesterday."
    assert len(cerebras.calls) == 1
    assert len(groq.calls) == 1
    sent = groq.calls[0]
    assert sent["model"] == "openai/gpt-oss-120b"
    assert sent["reasoning_effort"] == "low"
    assert sent["messages"][1]["content"] == "BSL Glosses: YESTERDAY ME GO SHOP"


def test_rule_based_output_when_both_providers_fail(monkeypatch, clock):
    cerebras = FakeCerebras(FakeResponse(status_code=503))
    groq = FakeGroq(error=RuntimeError("groq is down"))
    converter = hosted(monkeypatch, cerebras, groq)

    assert converter.convert(["THANK", "YOU", "MUCH"]) == "Thank you much."
    assert len(cerebras.calls) == 1 and len(groq.calls) == 1


def test_rule_based_output_when_cerebras_fails_and_there_is_no_groq(monkeypatch, clock):
    cerebras = FakeCerebras(requests.Timeout("timed out"))
    converter = hosted(monkeypatch, cerebras, groq=None)

    assert converter.groq_client is None
    assert converter.convert(["THANK", "YOU", "MUCH"]) == "Thank you much."


def test_groq_alone_serves_every_request_without_a_cerebras_key(monkeypatch, clock):
    cerebras = FakeCerebras(FakeResponse())
    groq = FakeGroq(content="Thank you very much.")
    converter = hosted(monkeypatch, cerebras, groq, cerebras_key=None)

    assert converter.convert(["THANK", "YOU", "MUCH"]) == "Thank you very much."
    assert cerebras.calls == []
    assert len(groq.calls) == 1


def test_hosted_mode_refuses_to_start_with_no_provider(monkeypatch):
    monkeypatch.setitem(sys.modules, "groq", None)
    with pytest.raises(ValueError, match="No hosted provider"):
        GlossToText(mode="groq")


def test_a_groq_key_without_the_sdk_is_not_a_provider(monkeypatch):
    monkeypatch.setitem(sys.modules, "groq", None)
    with pytest.raises(ValueError, match="No hosted provider"):
        GlossToText(mode="groq", groq_api_key="test-groq-key")


def test_cerebras_works_without_the_groq_sdk_installed(monkeypatch, clock):
    cerebras = FakeCerebras(FakeResponse(content="Hello there."))
    converter = hosted(monkeypatch, cerebras, groq=None)
    assert converter.convert(["HELLO"]) == "Hello there."


def test_groq_client_is_built_on_demand_from_the_sdk(monkeypatch, clock):
    built = []

    class FakeSdk:
        @staticmethod
        def Groq(api_key):
            client = FakeGroq(content="Built on demand.")
            built.append((api_key, client))
            return client

    monkeypatch.setitem(sys.modules, "groq", FakeSdk)
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-cerebras-key")
    monkeypatch.setattr(requests, "post", FakeCerebras(FakeResponse(status_code=500)))
    converter = GlossToText(mode="groq", groq_api_key="test-groq-key")

    assert converter.convert(["HELLO", "YOU"]) == "Built on demand."
    assert converter.convert(["HELLO", "YOU"]) == "Built on demand."
    # One client, reused, made with the key that was passed in.
    assert len(built) == 1 and built[0][0] == "test-groq-key"


def test_only_special_tokens_never_reaches_a_provider(monkeypatch, clock):
    cerebras = FakeCerebras(FakeResponse())
    groq = FakeGroq()
    converter = hosted(monkeypatch, cerebras, groq)

    assert converter.convert(["<unk>", "<pad>"]) == ""
    assert cerebras.calls == [] and groq.calls == []


@pytest.mark.parametrize(
    "raw, cleaned",
    [
        ("Hello there", "Hello there."),
        ('"Hello there."', "Hello there."),
        ("  'What time is it?'  ", "What time is it?"),
        ("Stop!", "Stop!"),
    ],
)
def test_provider_text_is_tidied(monkeypatch, clock, raw, cleaned):
    converter = hosted(monkeypatch, FakeCerebras(FakeResponse(content=raw)), FakeGroq())
    assert converter.convert(["HELLO"]) == cleaned


# --------------------------------------------------------------------------
# Hosted mode: the cooldown after a 429
# --------------------------------------------------------------------------

def test_429_starts_a_cooldown_and_cerebras_is_skipped_until_it_ends(monkeypatch, clock):
    cerebras = FakeCerebras(
        FakeResponse(status_code=429),
        FakeResponse(content="Cerebras is back."),
    )
    groq = FakeGroq(content="Groq covered it.")
    converter = hosted(monkeypatch, cerebras, groq)

    # The 429 itself falls back to Groq.
    assert converter.convert(["HELLO", "YOU"]) == "Groq covered it."
    assert len(cerebras.calls) == 1 and len(groq.calls) == 1

    # Inside the cooldown Cerebras is not called at all.
    clock.advance(30)
    assert converter.convert(["HELLO", "YOU"]) == "Groq covered it."
    clock.advance(29.9)
    assert converter.convert(["HELLO", "YOU"]) == "Groq covered it."
    assert len(cerebras.calls) == 1 and len(groq.calls) == 3

    # Once it has passed, Cerebras is primary again.
    clock.advance(0.2)
    assert converter.convert(["HELLO", "YOU"]) == "Cerebras is back."
    assert len(cerebras.calls) == 2 and len(groq.calls) == 3


def test_cooldown_default_is_the_per_minute_window(monkeypatch, clock):
    converter = hosted(monkeypatch, FakeCerebras(FakeResponse(status_code=429)), FakeGroq())
    converter.convert(["HELLO", "YOU"])
    assert converter._cerebras_cooldown_until == clock.now + GlossToText.CEREBRAS_COOLDOWN_SECONDS
    assert GlossToText.CEREBRAS_COOLDOWN_SECONDS == 60.0


@pytest.mark.parametrize(
    "retry_after, expected_wait",
    [("7", 7.0), ("0", 1.0), ("0.2", 1.0), ("soon", 60.0)],
    ids=["server-figure", "zero-floors-at-1s", "fraction-floors-at-1s", "unreadable-uses-default"],
)
def test_cooldown_prefers_the_servers_retry_after(monkeypatch, clock, retry_after, expected_wait):
    limited = FakeResponse(status_code=429, headers={"retry-after": retry_after})
    cerebras = FakeCerebras(limited)
    converter = hosted(monkeypatch, cerebras, FakeGroq())

    converter.convert(["HELLO", "YOU"])
    assert converter._cerebras_cooldown_until == clock.now + expected_wait

    clock.advance(expected_wait - 0.1)
    converter.convert(["HELLO", "YOU"])
    assert len(cerebras.calls) == 1
    clock.advance(0.2)
    converter.convert(["HELLO", "YOU"])
    assert len(cerebras.calls) == 2


def test_cooldown_with_no_groq_falls_back_to_rule_based(monkeypatch, clock):
    cerebras = FakeCerebras(FakeResponse(status_code=429))
    converter = hosted(monkeypatch, cerebras, groq=None)

    assert converter.convert(["THANK", "YOU", "MUCH"]) == "Thank you much."
    assert converter.convert(["THANK", "YOU", "MUCH"]) == "Thank you much."
    assert len(cerebras.calls) == 1


def test_other_errors_do_not_start_a_cooldown(monkeypatch, clock):
    cerebras = FakeCerebras(FakeResponse(status_code=500))
    converter = hosted(monkeypatch, cerebras, FakeGroq())
    converter.convert(["HELLO", "YOU"])
    converter.convert(["HELLO", "YOU"])
    assert len(cerebras.calls) == 2


# --------------------------------------------------------------------------
# Unknown glosses are passed through, never replaced
# --------------------------------------------------------------------------

def test_unknown_gloss_reaches_the_provider_unchanged(monkeypatch, clock):
    monkeypatch.setattr(GlossToText, "_VOCABULARY", {"ME", "WANT"})
    cerebras = FakeCerebras(FakeResponse(content="I want Zzzqqq."))
    converter = hosted(monkeypatch, cerebras, FakeGroq())

    english = converter.convert(["ME", "WANT", "ZZZQQQ"])

    messages = cerebras.calls[0]["json"]["messages"]
    # Not filtered out, although it is not in the vocabulary.
    assert messages[1]["content"] == "BSL Glosses: ME WANT ZZZQQQ"
    # The instruction that stops the model swapping in a word of its own.
    assert "Never substitute a word you invented" in messages[0]["content"]
    assert "keep it in the sentence as it is" in messages[0]["content"]
    assert english == "I want Zzzqqq."
    assert "sleep" not in english.lower()
    assert converter.last_unknown_glosses == ["ZZZQQQ"]


def test_the_same_rule_is_sent_to_the_groq_fallback(monkeypatch, clock):
    groq = FakeGroq(content="I want Zzzqqq.")
    converter = hosted(monkeypatch, FakeCerebras(FakeResponse(status_code=500)), groq)
    converter.convert(["ME", "WANT", "ZZZQQQ"])

    messages = groq.calls[0]["messages"]
    assert "Never substitute a word you invented" in messages[0]["content"]
    assert messages[1]["content"] == "BSL Glosses: ME WANT ZZZQQQ"


def test_unknown_gloss_survives_when_every_provider_is_down(monkeypatch, clock):
    converter = hosted(
        monkeypatch,
        FakeCerebras(requests.ConnectionError("offline")),
        FakeGroq(error=RuntimeError("offline")),
    )
    assert converter.convert(["ME", "WANT", "ZZZQQQ"]) == "Me want zzzqqq."


# --------------------------------------------------------------------------
# last_unknown_glosses
# --------------------------------------------------------------------------

def test_last_unknown_glosses_lists_glosses_with_no_sign(monkeypatch):
    monkeypatch.setattr(GlossToText, "_VOCABULARY", {"ME", "WANT", "GO"})
    converter = GlossToText(mode="simple")

    converter.convert(["me", "want", "ZZZQQQ", "go", "Sarah", "zzzqqq", "<unk>"])
    # In order, upper-cased, without repeats, and never a special token.
    assert converter.last_unknown_glosses == ["ZZZQQQ", "SARAH"]


def test_last_unknown_glosses_is_reset_by_each_call(monkeypatch):
    monkeypatch.setattr(GlossToText, "_VOCABULARY", {"ME", "WANT"})
    converter = GlossToText(mode="simple")

    converter.convert(["ME", "WANT", "ZZZQQQ"])
    assert converter.last_unknown_glosses == ["ZZZQQQ"]
    converter.convert(["ME", "WANT"])
    assert converter.last_unknown_glosses == []
    converter.convert(["ZZZQQQ"])
    assert converter.last_unknown_glosses == ["ZZZQQQ"]
    converter.convert([])
    assert converter.last_unknown_glosses == []


def test_no_vocabulary_file_reports_nothing_rather_than_everything():
    # The fixture leaves the vocabulary as "not found", as on a fresh checkout.
    converter = GlossToText(mode="simple")
    converter.convert(["ME", "WANT", "ZZZQQQ"])
    assert converter.last_unknown_glosses == []


def test_vocabulary_is_read_from_the_first_source_that_exists(monkeypatch, tmp_path):
    # Hand-written stand-ins for the real files, which are not in the repository.
    signs_dir = tmp_path / "signs"
    signs_dir.mkdir()
    for name in ("hello.json", "thank.json", "notes.txt"):
        (signs_dir / name).write_text("{}", encoding="utf-8")
    classes = tmp_path / "vocabulary.json"
    classes.write_text(json.dumps({"only": 0, "these": 1}), encoding="utf-8")

    monkeypatch.setattr(GlossToText, "_VOCABULARY", None)
    monkeypatch.setattr(
        GlossToText,
        "VOCABULARY_SOURCES",
        (
            ("json_values", str(tmp_path / "missing.json")),
            ("dir_names", str(signs_dir)),
            ("json_keys", str(classes)),
        ),
    )
    converter = GlossToText(mode="simple")
    converter.convert(["HELLO", "THANK", "ONLY"])
    assert GlossToText._VOCABULARY == {"HELLO", "THANK"}
    assert converter.last_unknown_glosses == ["ONLY"]


def test_a_broken_vocabulary_file_is_skipped(monkeypatch, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"0": "hello", "1": "thank"}), encoding="utf-8")

    monkeypatch.setattr(GlossToText, "_VOCABULARY", None)
    monkeypatch.setattr(
        GlossToText,
        "VOCABULARY_SOURCES",
        (("json_values", str(broken)), ("json_values", str(good))),
    )
    assert GlossToText(mode="simple").unknown_glosses(["HELLO", "ZZZQQQ"]) == ["ZZZQQQ"]


# --------------------------------------------------------------------------
# GlossToTextWithGroq (src/inference/bsl_to_speech.py)
# --------------------------------------------------------------------------

def test_with_groq_no_key_uses_the_plain_fallback():
    converter = b2s.GlossToTextWithGroq()
    assert converter.process(["ME", "WANT", "ZZZQQQ"]) == "Me want zzzqqq."
    assert converter.process([]) == ""


def test_with_groq_sends_cleaned_glosses_and_keeps_one_sentence(monkeypatch):
    client = FakeGroq(content='"I want Zzzqqq. And then a story nobody signed."')

    class FakeSdk:
        @staticmethod
        def Groq(api_key):
            return client

    monkeypatch.setitem(sys.modules, "groq", FakeSdk)
    converter = b2s.GlossToTextWithGroq(api_key="test-groq-key")

    assert converter.process(["ME", "ME", "WANT", "ZZZQQQ"]) == "I want Zzzqqq."
    sent = client.calls[0]
    # Repeats collapsed; the unknown gloss still goes to the model.
    assert sent["messages"][1]["content"] == "Glosses: me want zzzqqq"
    assert "Do NOT add details" in sent["messages"][0]["content"]
    assert sent["temperature"] == 0.0


def test_with_groq_falls_back_when_the_call_fails_or_the_sdk_is_missing(monkeypatch):
    class BrokenSdk:
        @staticmethod
        def Groq(api_key):
            return FakeGroq(error=RuntimeError("groq is down"))

    monkeypatch.setitem(sys.modules, "groq", BrokenSdk)
    converter = b2s.GlossToTextWithGroq(api_key="test-groq-key")
    assert converter.process(["THANK", "YOU", "MUCH"]) == "Thank you much."

    monkeypatch.setitem(sys.modules, "groq", None)
    assert converter.process(["THANK", "YOU", "MUCH"]) == "Thank you much."


def test_with_groq_cleaning_rules():
    converter = b2s.GlossToTextWithGroq(api_key="test-groq-key")
    # Consecutive repeats, filler glosses and non-word tokens are dropped.
    assert converter._clean_glosses(
        ["HELLO", "HELLO", "good", "MY", "name", "123", "SARAH"], 12
    ) == ["hello", "my", "name", "sarah"]
    # If that leaves fewer than three, the filler glosses are kept after all.
    assert converter._clean_glosses(["GOOD", "LOVE", "HAPPY"], 12) == ["good", "love", "happy"]
    assert converter._clean_glosses(["A", "B", "C", "D"], 2) == ["a", "b"]


# --------------------------------------------------------------------------
# TextToGloss, rule-based (src/inference/speech_to_bsl.py)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "english, glosses",
    [
        # Time first; articles and "is" dropped; "meeting" reduced to its root.
        ("What time is the meeting tomorrow?", ["TOMORROW", "WHAT", "TIME", "MEET"]),
        ("I don't understand.", ["I", "NOT", "UNDERSTAND"]),
        ("I have 2 cats", ["I", "TWO", "CAT"]),
        ("", []),
        # Nothing signable left.
        ("the a an", ["<unk>"]),
    ],
)
def test_text_to_gloss_simple_mode(english, glosses):
    assert s2b.TextToGloss(mode="simple").convert(english) == glosses


def test_text_to_gloss_keeps_a_word_it_does_not_know():
    assert s2b.TextToGloss(mode="simple").convert("I want zzzqqq") == ["I", "WANT", "ZZZQQQ"]
