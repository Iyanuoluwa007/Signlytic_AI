# Deploying the demo to a Hugging Face Space

Why this instead of a VM: Hugging Face gives 2 vCPU and 16 GB RAM on the free
tier with no card, no capacity lottery, and a real HTTPS URL. That last point
matters, because the dashboard calls `getUserMedia` and `MediaRecorder` and
browsers refuse those outside a secure context. There is no systemd, no Caddy,
no firewall and no DNS to configure.

The trade is that a free Space sleeps after a period with no visitors and takes
a few seconds to wake on the next request.

## 1. Create the Space

At https://huggingface.co/new-space:

- **Owner**: your account
- **Space name**: `signlytic-ai`
- **License**: MIT
- **SDK**: **Docker**, then **Blank** as the template
- **Hardware**: CPU basic, free
- **Visibility**: Public

## 2. Add the two files

The Space is a git repository, and its web editor is the quickest route.
Use **Files** then **Add file** then **Create a new file**, once each for:

- `Dockerfile` — copy from `deploy/huggingface/Dockerfile` in this repo
- `README.md` — copy from `deploy/huggingface/README.md`, replacing the one the
  Space was created with

The `README.md` front matter is not decoration: `sdk: docker` and
`app_port: 7860` are what tell the Space how to run and where to route traffic.
They must match the `EXPOSE` and `--port` in the Dockerfile.

## 3. Set the secrets

**Settings** then **Variables and secrets**. Add these as **Secrets**, not
variables, so they are hidden and are not baked into the public image:

| Name | Why |
| --- | --- |
| `SIGNLYTIC_ADMIN_TOKEN` | Required. The server refuses to bind beyond loopback without it. Generate one with `openssl rand -hex 32`. |
| `CEREBRAS_API_KEY` | Optional. Primary provider for gloss to English. |
| `GROQ_API_KEY` | Optional. Fallback provider, and the one that carries most traffic because Cerebras allows only 5 requests a minute. |

Everything else is already set in the Dockerfile, because none of it is
sensitive.

Without the two provider keys the Space still works: text to signing is
rule-based and needs no model. Only gloss to English translation degrades.

## 4. Watch the build

The **Logs** tab shows the Docker build, then the container. A healthy start
prints the server's own summary of its posture:

```
bind: 0.0.0.0  (network exposed)
admin token: set
rate limit: 20/min per client
[Server] demo mode: skipping recognizer warmup
```

If instead you see `Refusing to start`, the admin token secret is missing. That
is the guard doing its job, not a bug.

## 5. Check it

Replace `USER` and `SPACE` with yours:

```
https://USER-SPACE.hf.space/api/health
```

Then open the Space itself and type a sentence. On a phone, the signing should
animate.

## Updating

The Dockerfile clones from GitHub at build time, so the Space tracks `main`.
To pick up new commits, use **Settings** then **Factory rebuild**. A plain
restart reuses the cached image and will not pull anything new.

## What this does not give you

One free CPU Space, shared. It suits a public demo and a portfolio link. It is
not a service anyone should depend on, and video recognition and voice remain a
local download.
