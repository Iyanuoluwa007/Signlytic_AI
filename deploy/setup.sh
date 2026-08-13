#!/usr/bin/env bash
# Prepare a fresh Ubuntu host to run the Signlytic demo server.
#
#   sudo bash deploy/setup.sh
#
# Idempotent: safe to run again after a change. It does not start the service,
# because the environment file needs your API keys first. It prints what to do
# next.

set -euo pipefail

APP_DIR=/opt/signlytic/app
VENV_DIR=/opt/signlytic/venv
ENV_DIR=/etc/signlytic
ENV_FILE="$ENV_DIR/signlytic.env"
SERVICE_USER=signlytic
REPO_URL="${REPO_URL:-https://github.com/Iyanuoluwa007/Signlytic_AI.git}"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo." >&2
  exit 1
fi

say() { printf '\n== %s ==\n' "$1"; }

say "Packages"
apt-get update -qq
# ffmpeg is used to normalise uploaded audio; git and python for the app.
apt-get install -y -qq python3-venv python3-pip git ffmpeg curl debian-keyring \
  debian-archive-keyring apt-transport-https

say "Service user"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  # No login shell and no home: this account exists only to own the process.
  useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
  echo "created $SERVICE_USER"
else
  echo "$SERVICE_USER already exists"
fi

say "Application"
mkdir -p /opt/signlytic
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone --depth 1 "$REPO_URL" "$APP_DIR"
fi

say "Python environment"
python3 -m venv "$VENV_DIR" 2>/dev/null || true
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
# Only what the demo paths need. The full requirements.txt pulls torch, whisper
# and coqui-tts, which are large and pointless on a CPU box in demo mode.
"$VENV_DIR/bin/pip" install --quiet fastapi "uvicorn[standard]" python-multipart requests groq numpy

say "Environment file"
mkdir -p "$ENV_DIR"
if [ ! -f "$ENV_FILE" ]; then
  cp "$APP_DIR/deploy/signlytic.env.example" "$ENV_FILE"
  # Generate the admin token now so it is never left blank by accident.
  TOKEN="$(openssl rand -hex 32)"
  sed -i "s|^SIGNLYTIC_ADMIN_TOKEN=.*|SIGNLYTIC_ADMIN_TOKEN=$TOKEN|" "$ENV_FILE"
  echo "created $ENV_FILE with a generated admin token"
else
  echo "$ENV_FILE already exists, left alone"
fi
chown root:root "$ENV_FILE"
chmod 600 "$ENV_FILE"

chown -R "$SERVICE_USER:$SERVICE_USER" /opt/signlytic

say "systemd"
cp "$APP_DIR/deploy/signlytic.service" /etc/systemd/system/signlytic.service
systemctl daemon-reload
systemctl enable signlytic >/dev/null
echo "enabled (not started yet)"

say "Caddy"
if ! command -v caddy >/dev/null 2>&1; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  apt-get update -qq
  apt-get install -y -qq caddy
  echo "installed"
else
  echo "already installed"
fi
mkdir -p /var/log/caddy && chown caddy:caddy /var/log/caddy

cat <<NEXT

== Done. Three things left, in this order ==

1. Put your API keys in $ENV_FILE
     sudo nano $ENV_FILE
   The admin token is already generated. Set CEREBRAS_API_KEY and GROQ_API_KEY.
   Set both: Cerebras allows only 5 requests a minute on the free tier, so a
   shared demo will lean on the Groq fallback.

2. Point Caddy at your hostname
     sudo cp $APP_DIR/deploy/Caddyfile /etc/caddy/Caddyfile
     sudo nano /etc/caddy/Caddyfile      # replace demo.example.com
     sudo systemctl reload caddy
   Your DNS A record must already point at this machine, and ports 80 and 443
   must be open, or the certificate cannot be issued. On Oracle Cloud that
   means both the security list and the instance firewall.

3. Start it
     sudo systemctl start signlytic
     sudo systemctl status signlytic
     journalctl -u signlytic -f

Check it is healthy:
     curl -s localhost:8000/api/health

Confirm the kill switch is not reachable from outside (expect 404 from Caddy):
     curl -si https://YOUR-HOST/api/shutdown -X POST | head -1

NEXT
