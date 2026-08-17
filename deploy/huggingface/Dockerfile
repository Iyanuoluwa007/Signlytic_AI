# Signlytic AI demo, for a Hugging Face Docker Space.
#
# The Space itself stays two small files: this and README.md. The application
# is cloned from GitHub at build time, so a rebuild picks up the current main
# and nothing has to be kept in sync by hand.
#
# Hugging Face terminates TLS and gives the Space a real HTTPS URL, which is
# why this route is better than the earlier plain-HTTP box: the dashboard calls
# getUserMedia and MediaRecorder, and browsers refuse those outside a secure
# context.

FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Spaces expect the app to run as uid 1000 rather than root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

ARG REPO=https://github.com/Iyanuoluwa007/Signlytic_AI.git
ARG REF=main
RUN git clone --depth 1 --branch $REF $REPO .

# Only what the demo paths actually need. Deliberately no torch, no whisper and
# no coqui-tts: demo mode never loads them, and they would add gigabytes to the
# image and minutes to every build for code that cannot run without a GPU.
RUN pip install --no-cache-dir --user \
      fastapi "uvicorn[standard]" python-multipart requests groq numpy

# Non-secret configuration. Anything sensitive goes in the Space's secrets, not
# here, because this file is public.
ENV SIGNLYTIC_DEMO_MODE=1 \
    SIGNLYTIC_SIGNS_API=https://signlytic-ai-website.vercel.app/api/signs \
    SIGNLYTIC_FULL_APP_URL=https://github.com/Iyanuoluwa007/Signlytic_AI/releases/latest \
    SIGNLYTIC_RATE_LIMIT=20 \
    SIGNLYTIC_SIGN_CACHE=256

# Must match app_port in README.md.
EXPOSE 7860

# Binding beyond loopback requires SIGNLYTIC_ADMIN_TOKEN, set as a Space
# secret. Without it the server refuses to start and says so, which is the
# intended behaviour rather than a misconfiguration.
CMD ["python", "-u", "app_server.py", "--host", "0.0.0.0", "--port", "7860"]
