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

// ── Signing speed ───────────────────────────────────────────────────────────
// Both engines recompute their frame interval from .speed on every frame, so
// assigning it takes effect on the next frame rather than only on the next
// sentence. Held here as well because the 3D engine is built lazily and must
// pick up the current setting when it appears.
let speed = 1;

function applySpeed(value) {
  speed = value;
  if (engine2d) engine2d.speed = value;
  if (engine3d) engine3d.speed = value;
}

// Both canvases are stretched to the stage by CSS, so their drawing buffers
// have to match the stage box or the picture comes out stretched. Called
// whenever the stage can have changed size.
function fitEngines() {
  const s = document.getElementById("stage").getBoundingClientRect();
  const w = Math.max(1, Math.floor(s.width));
  const h = Math.max(1, Math.floor(s.height));
  if (engine2d && engine2d.resize) engine2d.resize();
  if (engine3d && engine3d.resize) engine3d.resize(w, h);
}

function ensure2d() {
  if (!engine2d) engine2d = new SkeletonRenderer2D(canvas2d, { transparent: true, speed });
  return engine2d;
}

async function ensure3d() {
  if (engine3d || avatarLoading) return engine3d;
  avatarLoading = true;
  try {
    setStatus("Loading 3D avatar model...");
    const a = new ThreeAvatarRenderer(canvas3d, { gender: "male", transparent: true, speed });
    a.initScene();
    await a.load();
    if (!a.ready) throw new Error("avatar failed to load");
    engine3d = a;
    // initScene measured the stage before the canvas was visible, so size it
    // to the real box now that it is.
    fitEngines();
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
  setTimeout(fitEngines, 0);
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
  // The window changes shape, so re-measure once the new bounds have applied.
  setTimeout(fitEngines, 60);
}

for (const [name, id] of Object.entries(POS_BUTTONS)) {
  document.getElementById(id).addEventListener("click", async () => {
    const res = await window.signlytic.window.setPosition(name);
    if (res && res.ok) markPosition(res.mode);
  });
}

window.signlytic.window.onPosition((p) => markPosition(p.mode));

// ── Speed control ───────────────────────────────────────────────────────────
const speedEl = document.getElementById("speed");

speedEl.addEventListener("change", async () => {
  const res = await window.signlytic.speed.set(speedEl.value);
  // Main clamps to the allowed set and returns what it kept, so the menu can
  // never drift from what is actually being applied.
  const next = (res && res.speed) || 1;
  applySpeed(next);
  speedEl.value = String(next);
  setStatus("Signing speed " + next + "x");
});

// ── Settings panel ──────────────────────────────────────────────────────────
// Renderer and position used to sit in the title bar, which left it crowded
// and unreadable at the snapped width of 420px. They live behind the gear now,
// so the title bar carries the title and the window buttons and nothing else.
const settingsBtn = document.getElementById("settings-btn");
const settingsPanel = document.getElementById("settings");

function setSettingsOpen(open) {
  settingsPanel.classList.toggle("open", open);
  settingsBtn.classList.toggle("open", open);
  settingsBtn.setAttribute("aria-expanded", String(open));
}

settingsBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  setSettingsOpen(!settingsPanel.classList.contains("open"));
});

// Clicking anywhere else closes it, the way a menu is expected to behave.
// Clicks inside the panel must not, or choosing a setting would shut it.
settingsPanel.addEventListener("click", (e) => e.stopPropagation());
document.addEventListener("click", () => setSettingsOpen(false));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") setSettingsOpen(false);
});

// ── Microphone audio option ─────────────────────────────────────────────────
const micCheck = document.getElementById("mic-audio");

micCheck.addEventListener("change", async () => {
  const res = await window.signlytic.captions.setMicAudio(micCheck.checked);
  const on = res ? res.enabled : micCheck.checked;
  micCheck.checked = on;
  setStatus(
    on
      ? "Microphone audio will be switched on with captions"
      : "Microphone audio left as Live Captions has it"
  );
});

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

// Held so the label can be restored after "Stop Captions" without assuming
// which platform's wording it started with.
let capStartLabel = "Start Live Captions";

function setCaptionsUi(on) {
  capOn = on;
  capBtn.textContent = on ? "Stop Captions" : capStartLabel;
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

// Set from capabilities() below. The caption reader emits the same states on
// both platforms, but "Live Captions" is the name of a Windows app, so the
// wording for them cannot be.
let captionSourceName = "Live Captions";

window.signlytic.captions.onStatus((s) => {
  if (s.state === "waiting") setStatus("Waiting for " + captionSourceName + "...");
  else if (s.state === "attached") setStatus("Listening");
  else if (s.state === "idle") setStatus("Listening (no speech yet)");
  else if (s.detail) setStatus(s.detail);
  if (s.state === "stopped" || s.state === "error") setCaptionsUi(false);
});

// ── macOS audio source ──────────────────────────────────────────────────────
// On macOS the caption text is produced by recognising speech, so the app has
// to be told what to listen to. Windows never sees these controls: Live
// Captions picks its own audio and the checkbox above is what matters there.
const AUDIO_SOURCE_NOTES = {
  mic: "Listens to the microphone, so speech in the room is signed. Needs microphone and speech recognition permission.",
  system: "Listens to what this Mac is playing, so calls and video are signed. Needs screen recording permission, which is how macOS allows system audio to be captured. Video with copy protection will refuse to be captured.",
  captions: "Reads the caption window macOS itself produces, so the wording is Apple's rather than ours. Usually the most accurate. Needs Accessibility permission, and Live Captions has to be switched on in System Settings, Accessibility, and left running.",
};

const AUDIO_SOURCE_BUTTONS = {
  mic: "audio-mic",
  system: "audio-system",
  captions: "audio-captions",
};

const audioSourceRow = document.getElementById("audio-source-row");
const audioSourceNote = document.getElementById("audio-source-note");

function markAudioSource(source) {
  const chosen = AUDIO_SOURCE_NOTES[source] ? source : "mic";
  for (const [name, id] of Object.entries(AUDIO_SOURCE_BUTTONS)) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("active", name === chosen);
  }
  audioSourceNote.textContent = AUDIO_SOURCE_NOTES[chosen];
}

async function chooseAudioSource(source) {
  const res = await window.signlytic.captions.setAudioSource(source);
  const applied = (res && res.source) || source;
  markAudioSource(applied);
  // The helper is told its source when it is spawned, so a change only takes
  // effect on the next start. Saying so beats looking like nothing happened.
  const WILL_LISTEN = {
    mic: "Captions will listen to the microphone",
    system: "Captions will listen to system audio",
    captions: "Captions will read the macOS Live Captions window",
  };
  setStatus(
    capOn
      ? "Caption source changes when you stop and start captions again"
      : WILL_LISTEN[applied] || WILL_LISTEN.mic
  );
}

for (const [name, id] of Object.entries(AUDIO_SOURCE_BUTTONS)) {
  const el = document.getElementById(id);
  if (el) el.addEventListener("click", () => chooseAudioSource(name));
}

window.signlytic.captions.capabilities().then((c) => {
  if (!c.supported) {
    capBtn.disabled = true;
    capBtn.title = c.reason || "System captions not available";
    // The microphone option only means anything through Live Captions, so on a
    // platform without it the control is shown greyed rather than offered.
    micCheck.disabled = true;
    micCheck.closest(".set-check").classList.add("disabled");
    micCheck.closest(".set-check").title = c.reason || "System captions not available";
    return;
  }

  if (c.source === "macos-speech-recognition") {
    // "Live Captions" is the name of a Windows app, not of what happens here.
    capStartLabel = "Start Captions";
    captionSourceName = "speech recognition";
    capBtn.textContent = capStartLabel;
    audioSourceRow.classList.remove("set-gone");
    audioSourceNote.classList.remove("set-gone");
    // Driving another app's settings menu is a Windows-only affair, so the
    // checkbox that does it has nothing to act on here.
    micCheck.closest(".set-check").classList.add("set-gone");
    const micNote = document.getElementById("mic-note");
    if (micNote) micNote.classList.add("set-gone");
    window.signlytic.captions.getAudioSource().then((a) => markAudioSource(a && a.source));
  }
});

window.signlytic.captions.getMicAudio().then((m) => {
  micCheck.checked = m ? m.enabled !== false : true;
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
  // A canvas that escapes the stage sits on top of the controls and swallows
  // their clicks, so check containment rather than only vertical order.
  const canvases = ["canvas-2d", "canvas-3d"].map((id) => {
    const el = document.getElementById(id);
    if (!el) return id + "=missing";
    const r = el.getBoundingClientRect();
    const inside = stage ? Math.round(r.bottom) <= stage.bottom + 1 : false;
    const overlapsControls = controls ? Math.round(r.bottom) > controls.top : false;
    return id + "={h:" + Math.round(r.height) + ",insideStage:" + inside + ",coversControls:" + overlapsControls + "}";
  });
  console.log(
    "[layout] " + tag +
    " stage=" + JSON.stringify(stage) +
    " controls=" + JSON.stringify(controls) +
    " signerAboveInput=" + (stage && input ? stage.bottom <= input.top : "n/a") +
    " " + canvases.join(" ")
  );
}

// ── Boot ────────────────────────────────────────────────────────────────────
// Restore the saved speed. This resolves after ensure2d() below has already
// built the 2D engine, which is why applySpeed writes to live engines as well
// as to the held value.
window.signlytic.speed.get().then((s) => {
  const saved = (s && s.speed) || 1;
  applySpeed(saved);
  speedEl.value = String(saved);
  console.log("[desktop] signing speed " + saved + "x");
});

ensure2d();
if (window.signlytic.startMode === "3d") setMode("3d");
if (window.signlytic.openSettings) setSettingsOpen(true);
if (window.signlytic.layoutDebug) {
  setTimeout(() => reportLayout("boot"), 800);
  setTimeout(() => reportLayout("after-load"), 12000);
  window.addEventListener("resize", () => reportLayout("resize"));
}
window.addEventListener("resize", fitEngines);
setStatus("Ready. Type something, or start Live Captions.");
