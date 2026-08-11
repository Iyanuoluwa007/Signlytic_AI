// Desktop renderer.
//
// Caption text (from Windows Live Captions or the manual box) is turned into
// glosses, then sign frames, then played on whichever engine is selected.
// Everything here is platform-neutral; only the caption source differs by OS.

const statusEl = document.getElementById("status");
const captionEl = document.getElementById("caption-log");
const glossStrip = document.getElementById("gloss-strip");

function setStatus(msg) {
  statusEl.textContent = msg;
  console.log("[desktop] " + msg);
}

// ── Engines ─────────────────────────────────────────────────────────────────
// 2D is the default: it needs no model download and its output is currently
// the more accurate of the two.
const canvas2d = document.getElementById("canvas-2d");
const canvas3d = document.getElementById("canvas-3d");

let mode = "2d";
let engine2d = null;
let engine3d = null;
let avatarLoading = false;

function activeEngine() {
  return mode === "3d" ? engine3d : engine2d;
}

function ensure2d() {
  if (!engine2d) engine2d = new SkeletonRenderer2D(canvas2d, {});
  return engine2d;
}

async function ensure3d() {
  if (engine3d || avatarLoading) return engine3d;
  avatarLoading = true;
  try {
    setStatus("Loading 3D avatar model...");
    const a = new ThreeAvatarRenderer(canvas3d, { gender: "male" });
    a.initScene();
    await a.load();
    if (!a.ready) throw new Error("avatar failed to load");
    engine3d = a;
    setStatus("3D avatar ready (" + Object.keys(a.bones).length + " bones)");
    window.signlytic.notifyAvatarReady();
    return a;
  } catch (err) {
    setStatus("3D avatar failed: " + (err && err.message ? err.message : err));
    return null;
  } finally {
    avatarLoading = false;
  }
}

function setMode(next) {
  if (next === mode) return;
  const prev = activeEngine();
  if (prev) prev.stopQueue();
  mode = next;
  canvas2d.classList.toggle("hidden", next !== "2d");
  canvas3d.classList.toggle("hidden", next !== "3d");
  document.getElementById("mode-2d").classList.toggle("active", next === "2d");
  document.getElementById("mode-3d").classList.toggle("active", next === "3d");
  if (next === "3d") ensure3d(); else ensure2d();
}

document.getElementById("mode-2d").addEventListener("click", () => setMode("2d"));
document.getElementById("mode-3d").addEventListener("click", () => setMode("3d"));

// ── Playback ────────────────────────────────────────────────────────────────
// Sentences can arrive faster than they can be signed, so queue them rather
// than cutting the current one off mid-sign.
const pending = [];
let playing = false;

function renderGlosses(glosses, activeIdx) {
  glossStrip.innerHTML = "";
  glosses.forEach((g, i) => {
    const el = document.createElement("span");
    el.className = "gloss" + (i === activeIdx ? " active" : "");
    el.textContent = g;
    glossStrip.appendChild(el);
  });
}

async function playNext() {
  if (playing || !pending.length) return;
  const text = pending.shift();
  playing = true;

  try {
    setStatus("Translating: " + text);
    const { glosses, queue } = await window.signlyticSigns.buildQueue(text);
    if (!glosses.length) { playing = false; return playNext(); }
    renderGlosses(glosses, -1);

    const engine = mode === "3d" ? (engine3d || await ensure3d()) : ensure2d();
    if (!engine) { playing = false; return playNext(); }

    setStatus("Signing " + glosses.length + " gloss" + (glosses.length > 1 ? "es" : ""));
    engine.playQueue(
      queue,
      (idx) => renderGlosses(glosses, idx),
      () => {
        renderGlosses(glosses, -1);
        setStatus("Ready");
        playing = false;
        playNext();
      }
    );
  } catch (err) {
    setStatus("Could not sign that: " + (err && err.message ? err.message : err));
    playing = false;
    playNext();
  }
}

function enqueue(text) {
  const t = String(text || "").trim();
  if (!t) return;
  pending.push(t);
  playNext();
}

// ── Caption input ───────────────────────────────────────────────────────────
window.signlytic.onCaptionText((payload) => {
  console.log("[desktop] caption in (" + payload.source + "): " + payload.text);
  const line = document.createElement("div");
  line.textContent = "[" + payload.source + "] " + payload.text;
  captionEl.prepend(line);
  enqueue(payload.text);
});

const input = document.getElementById("manual-input");
const sendBtn = document.getElementById("manual-send");

function submitManualText() {
  const text = input.value.trim();
  if (!text) return;
  window.signlytic.sendManualText(text);
  input.value = "";
}
sendBtn.addEventListener("click", submitManualText);
input.addEventListener("keydown", (e) => { if (e.key === "Enter") submitManualText(); });

// ── Live Captions control (Windows only; the button reports why elsewhere) ──
const capBtn = document.getElementById("captions-toggle");
const capLaunch = document.getElementById("captions-launch");
const capState = document.getElementById("captions-state");
let capOn = false;

function setCaptionsUi(on) {
  capOn = on;
  capBtn.textContent = on ? "Stop Live Captions" : "Start Live Captions";
  capBtn.classList.toggle("on", on);
}

capBtn.addEventListener("click", async () => {
  if (capOn) {
    await window.signlytic.captions.stop();
    setCaptionsUi(false);
    capState.textContent = "captions off";
    return;
  }
  capState.textContent = "starting...";
  const res = await window.signlytic.captions.start();
  if (res && res.ok) setCaptionsUi(true);
  else capState.textContent = (res && res.reason) || "could not start captions";
});

capLaunch.addEventListener("click", async () => {
  const res = await window.signlytic.captions.launchApp();
  capState.textContent = res && res.ok ? "opened Live Captions" : (res && res.reason) || "could not open Live Captions";
});

window.signlytic.captions.onStatus((s) => {
  capState.textContent = s.state === "waiting"
    ? "waiting for Live Captions window"
    : s.state + (s.detail ? " - " + s.detail : "");
  if (s.state === "stopped" || s.state === "error") setCaptionsUi(false);
});

// Disable the caption controls where there is no system caption source.
window.signlytic.captions.capabilities().then((c) => {
  if (!c.supported) {
    capBtn.disabled = true;
    capLaunch.disabled = true;
    capState.textContent = c.reason || "system captions not available";
  }
});

// ── Boot ────────────────────────────────────────────────────────────────────
ensure2d();
setStatus("Ready. Type something, or start Live Captions.");
