# Deploying the Signlytic demo server

This puts the browser-facing half of Signlytic on a small always-on machine, so
anyone on a phone, tablet or laptop can try BSL signing without installing
anything. The half that needs a GPU stays a download.

## What is and is not hosted

| Feature | Hosted demo | Why |
| --- | --- | --- |
| English or captions to BSL signing | Yes, in full | The avatar renders in the visitor's own browser, so the server only serves JSON. Ten simultaneous users cost almost nothing. |
| Gloss to English translation | Yes | A call to a hosted language model, not local compute. |
| BSL video recognition | No | Video-SWIN-T needs a GPU. On a free CPU box it takes tens of seconds per clip and queues. |
| Speech transcription | No | Whisper, same reason. |
| Cloned voice output | No | Coqui XTTS, same reason. |

The three unavailable features return HTTP 503 with a message and a link to the
release, rather than timing out and leaving the impression the project is
broken. That behaviour is `SIGNLYTIC_DEMO_MODE=1`; unset it and everything runs,
which is what you want on your own GPU machine.

## The box carries no sign data

Locally the server reads pose frames from `data/poses`, which is 2.6 GB, with
another 1.4 GB of sign JSON beside it. None of that is in the repository, so a
fresh clone has none of it and a naive deployment comes up serving empty
animations.

Rather than shipping several gigabytes to the box, `SIGNLYTIC_SIGNS_API` points
sign lookups at the website endpoint that already serves this data and already
caches it. Signs are immutable, so the box keeps the ones it has fetched in
memory: in testing, a four gloss sentence took 12 seconds cold and 40 ms once
warm. Playback timing is unchanged, because the remote path reuses the same
20 fps and 0.9 second per gloss defaults as the local renderer.

The consequence worth knowing: the demo box depends on the website being up. If
Vercel is down, the box serves the UI but no signing.

## Suitable hosts

Any small Linux box works. Oracle Cloud's Always Free tier is the strongest
genuinely free option: up to 4 Arm cores and 24 GB RAM, always on, free
indefinitely rather than as a trial. It has no GPU, which is exactly why demo
mode exists. A card is required for identity verification even though the free
shapes stay free, and Arm capacity can be scarce in busy regions.

## Setup

```bash
git clone https://github.com/Iyanuoluwa007/Signlytic_AI.git
sudo bash Signlytic_AI/deploy/setup.sh
```

That installs the packages, creates an unprivileged `signlytic` user, builds a
virtualenv with only the dependencies the demo paths need, generates an admin
token, installs the systemd unit and installs Caddy. It deliberately does not
start anything, because your API keys are not in place yet. It prints the three
remaining steps.

## Why HTTPS is not optional

The dashboard calls `getUserMedia` and `MediaRecorder`. Browsers only allow
those in a secure context. `http://localhost` counts, but `http://<an-ip>` does
not, so over plain HTTP on a phone the microphone and camera controls appear and
then silently do nothing. Caddy obtains and renews a Let's Encrypt certificate
by itself, which is the reason it is in this setup rather than exposing uvicorn
directly.

You need a hostname for that. A free dynamic DNS name is fine.

## How the pieces fit

```
browser ──HTTPS──> Caddy (443) ──HTTP──> uvicorn (127.0.0.1:8000)
                     │                      │
                     │                      └─ app_server.py, running as
                     │                         the signlytic user
                     └─ certificates, compression, cache headers,
                        and it returns 404 for /api/shutdown
```

The app binds to loopback only, so it is never directly reachable. Caddy is the
only thing listening publicly.

## Safety

Three layers, none of which existed before this was exposed:

- **`/api/shutdown` requires an admin token.** Unprotected it is a single
  unauthenticated POST that kills the process. `setup.sh` generates a token; the
  Caddyfile also returns 404 for that path, so it is not reachable at all from
  outside unless you remove that block.
- **Compute-heavy endpoints are rate limited**, 20 requests a minute per client
  by default, with a `Retry-After` header. Behind a proxy the caller is
  identified by `X-Forwarded-For`, which is client controlled, so treat this as
  throttling to stop one user monopolising the box rather than as a security
  boundary.
- **The server refuses to bind to a non-loopback address without a token**, and
  says how to fix it, because that mistake is silent until somebody finds it.

The environment file holds API keys, so `setup.sh` sets it to `root:root` and
mode 600. Never commit it.

## Operating it

```bash
sudo systemctl status signlytic     # is it up
journalctl -u signlytic -f          # follow the log
sudo systemctl restart signlytic    # after changing the env file
curl -s localhost:8000/api/health   # readiness
```

Startup prints its own posture: bind address, whether an admin token is set, and
the rate limit. Read that line after any change rather than assuming.

## Cost

Nothing beyond the machine, which on Oracle's Always Free tier is nothing. The
language model providers both have free tiers. Cerebras allows 5 requests a
minute, so on a shared demo most traffic will fall through to the Groq fallback:
set both keys.

## What this is not

One machine, no GPU, running your code. It suits a public demo and a portfolio
link. It is not a service other people should depend on, and if the box is busy
or down, it is down.
