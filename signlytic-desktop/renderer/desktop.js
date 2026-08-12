// Desktop renderer.
//
// Caption text (from Windows Live Captions or the text box) is turned into
// glosses, then sign frames, then played on whichever engine is selected.
// Everything here is platform-neutral; only the caption source differs by OS.
//
// The window is transparent and frameless, so both renderers draw on a clear
// background and the panel chrome lives in index.html.

const statusEl = document.getElementById("status");
const glossStrip = document.getElementById("glosses");

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
  if (!engine2d) engine2d = new SkeletonRenderer2D(canvas2d, { transparent: true });
  return engine2d;
}

async function ensure3d() {
  if (engine3d || avatarLoading) return engine3d;
  avatarLoading = true;
  try {
    setStatus("Loading 3D avatar model...");
    const a = new ThreeAvatarRenderer(canvas3d, { gender: "male", transparent: true });
    a.initScene();
    await a.load();
    if (!a.ready) throw new Error("avatar failed to load");
    engine3d = a;
    setStatus("3D avatar ready");
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

// ── Window placement ────────────────────────────────────────────────────────
const POS_BUTTONS = {
  "top-left": "pos-top-left",
  "top-right": "pos-top-right",
  "bottom-left": "pos-bottom-left",
  "bottom-right": "pos-bottom-right",
  float: "pos-float",
};

function markPosition(modeName) {
  for (const [name, id] of Object.entries(POS_BUTTONS)) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("active", name === modeName);
  }
  // Canvases are sized from their CSS box, so re-measure after the reshape.
  setTimeout(() => {
    if (engine2d && engine2d.resize) engine2d.resize();
    if (engine3d && engine3d.resize) {
      const s = document.getElementById("stage").getBoundingClientRect();
      engine3d.resize(Math.floor(s.width), Math.floor(s.height));
    }
  }, 60);
}

for (const [name, id] of Object.entries(POS_BUTTONS)) {
  document.getElementById(id).addEventListener("click", async () => {
    const res = await window.signlytic.window.setPosition(name);
    if (res && res.ok) markPosition(res.mode);
  });
}

window.signlytic.window.onPosition((p) => markPosition(p.mode));

document.getElementById("win-min").addEventListener("click", () => window.signlytic.window.minimise());
document.getElementById("win-close").addEventListener("click", () => window.signlytic.window.close());

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
    setStatus(text);
    const { glosses, queue } = await window.signlyticSigns.buildQueue(text);
    if (!glosses.length) { playing = false; return playNext(); }
    renderGlosses(glosses, -1);

    const engine = mode === "3d" ? (engine3d || await ensure3d()) : ensure2d();
    if (!engine) { playing = false; return playNext(); }

    engine.playQueue(
      queue,
      (idx) => renderGlosses(glosses, idx),
      () => {
        renderGlosses(glosses, -1);
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
  enqueue(payload.text);
});

const input = document.getElementById("manual-input");
function submitManualText() {
  const text = input.value.trim();
  if (!text) return;
  window.signlytic.sendManualText(text);
  input.value = "";
}
input.addEventListener("keydown", (e) => { if (e.key === "Enter") submitManualText(); });

// ── Live Captions control (Windows only; the button reports why elsewhere) ──
const capBtn = document.getElementById("captions-toggle");
let capOn = false;

function setCaptionsUi(on) {
  capOn = on;
  capBtn.textContent = on ? "Stop Captions" : "Start Live Captions";
  capBtn.classList.toggle("on", on);
}

capBtn.addEventListener("click", async () => {
  if (capOn) {
    await window.signlytic.captions.stop();
    setCaptionsUi(false);
    setStatus("Captions off");
    return;
  }
  setStatus("Starting captions...");
  const res = await window.signlytic.captions.start();
  if (res && res.ok) setCaptionsUi(true);
  else setStatus((res && res.reason) || "Could not start captions");
});

window.signlytic.captions.onStatus((s) => {
  if (s.state === "waiting") setStatus("Waiting for Live Captions...");
  else if (s.state === "attached") setStatus("Listening");
  else if (s.state === "idle") setStatus("Listening (no speech yet)");
  else if (s.detail) setStatus(s.detail);
  if (s.state === "stopped" || s.state === "error") setCaptionsUi(false);
});

window.signlytic.captions.capabilities().then((c) => {
  if (!c.supported) {
    capBtn.disabled = true;
    capBtn.title = c.reason || "System captions not available";
  }
});

// Dev-only: report where the main blocks actually sit, so the signer being
// above the controls can be checked rather than assumed. The renderer console
// is mirrored to the app's stdout.
function reportLayout(tag) {
  const rect = (id) => {
    const el = document.getElementById(id);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { top: Math.round(r.top), bottom: Math.round(r.bottom), h: Math.round(r.height) };
  };
  const stage = rect("stage");
  const controls = rect("controls");
  const input = rect("manual-input");
  console.log(
    "[layout] " + tag +
    " stage=" + JSON.stringify(stage) +
    " controls=" + JSON.stringify(controls) +
    " signerAboveInput=" + (stage && input ? stage.bottom <= input.top : "n/a")
  );
}

// ── Boot ────────────────────────────────────────────────────────────────────
ensure2d();
if (window.signlytic.layoutDebug) {
  setTimeout(() => reportLayout("boot"), 800);
  window.addEventListener("resize", () => reportLayout("resize"));
}
window.addEventListener("resize", () => {
  if (engine2d && engine2d.resize) engine2d.resize();
  if (engine3d && engine3d.resize) {
    const s = document.getElementById("stage").getBoundingClientRect();
    engine3d.resize(Math.floor(s.width), Math.floor(s.height));
  }
});
setStatus("Ready. Type something, or start Live Captions.");
