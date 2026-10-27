"""
Mock inference mode: the server must run, and every endpoint must answer, on a
machine with no GPU, no torch, no model files, no API keys and no network.

Runs offline: sign lookups are pointed at a closed local port, so every sign
comes back missing and is fingerspelled, which exercises the same path a gloss
with no sign takes. No sign data is fetched, stored or committed.

    pip install -r requirements-dev.txt
    pytest tests/ -q
"""
import base64
import importlib
import io
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    os.environ["SIGNLYTIC_MOCK_INFERENCE"] = "1"
    os.environ["SIGNLYTIC_RATE_LIMIT"] = "0"
    # A closed local port: lookups fail at once, so the tests never touch the network.
    os.environ["SIGNLYTIC_SIGNS_API"] = "http://127.0.0.1:9"
    for key in ("CEREBRAS_API_KEY", "GROQ_API_KEY"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(ROOT))
    import app_server
    importlib.reload(app_server)
    from fastapi.testclient import TestClient
    with TestClient(app_server.app) as c:
        yield c


def _upload(name="clip.mp4", data=b"not really a video"):
    return {"file": (name, io.BytesIO(data), "application/octet-stream")}


def test_torch_is_never_imported(client):
    client.get("/api/health")
    client.post("/api/d1/glosses", data={"glosses": "HELLO MY NAME"})
    assert "torch" not in sys.modules


def test_health_reports_mock_mode(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["mock_inference"] is True
    assert body["recognizer_ready"] is True


def test_dashboard_shows_mock_banner(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Mock inference mode" in r.text


def test_video_recognition_returns_marked_mock_result(client):
    r = client.post("/api/d1/video", files=_upload())
    assert r.status_code == 200
    body = r.json()
    assert body["mock"] is True
    assert body["glosses"] and body["english"]
    assert base64.b64decode(body["audio_b64"])[:4] == b"RIFF"


def test_glosses_to_english_uses_rule_based_converter(client):
    r = client.post("/api/d1/glosses", data={"glosses": "ME NAME SARAH"})
    assert r.status_code == 200
    body = r.json()
    assert body["glosses"] == ["ME", "NAME", "SARAH"]
    assert isinstance(body["english"], str) and body["english"]
    assert body["mock"] is True and body["voice_available"] is True


def test_english_to_bsl_text_is_real(client):
    r = client.post("/api/d2/text", data={"text": "What is your name?"})
    assert r.status_code == 200
    body = r.json()
    assert body["glosses"]
    # Signs cannot be fetched offline, so each gloss gets a fingerspelling run.
    assert body["frame_count"] == len(body["frames"])


def test_speech_returns_mock_transcript(client):
    r = client.post("/api/d2/audio", files=_upload("speech.wav"))
    assert r.status_code == 200
    body = r.json()
    assert body["mock"] is True and body["text"] and body["glosses"]


def test_live_frames_cycle_and_assemble(client):
    seen = [client.post("/api/live/frame", files=_upload("f.jpg")).json()["results"][0]["gloss"]
            for _ in range(4)]
    assert len(set(seen)) > 1
    r = client.post("/api/live/assemble", json={"glosses": seen})
    assert r.status_code == 200
    assert r.json()["sentence"]


def test_shutdown_still_needs_admin_token(client):
    assert client.post("/api/shutdown").status_code == 403
