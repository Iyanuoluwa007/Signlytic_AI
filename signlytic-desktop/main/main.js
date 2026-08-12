// Signlytic AI Desktop - Electron main process (Session 1 shell)
//
// Session 2 will add: system tray, always-on-top overlay window mode, and
// the Windows UI Automation caption sidecar (see main/captions/README.md).
const { app, BrowserWindow, ipcMain, screen } = require("electron");
const path = require("path");
const fs = require("fs");
const { CaptionStream } = require("./captions/caption-stream");

let mainWindow = null;
let captionStream = null;

// ── Window placement ─────────────────────────────────────────────────────────
// Corners rather than full-width strips: a signing avatar needs a roughly
// portrait area to read properly, and stretching it across the screen leaves
// it small and lost. Each corner snaps a compact panel against two edges,
// with float left free for anywhere else.
const SNAP_SIZE = { width: 420, height: 440 };
const SNAP_MARGIN = 12;
const FLOAT_SIZE = { width: 520, height: 460 };
const CORNERS = ["top-left", "top-right", "bottom-right", "bottom-left"];
const POSITIONS = [...CORNERS, "float"];

// Remembered between runs so the app reopens where it was left.
const PREFS_FILE = () => path.join(app.getPath("userData"), "window-prefs.json");

function readPrefs() {
  try {
    // Strip a byte order mark before parsing. Anything that rewrites this file
    // with a Windows text editor or PowerShell can leave one, and JSON.parse
    // throws on it, which would silently reset the user's placement.
    const raw = fs.readFileSync(PREFS_FILE(), "utf8").replace(/^﻿/, "");
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function writePrefs(patch) {
  try {
    fs.writeFileSync(PREFS_FILE(), JSON.stringify({ ...readPrefs(), ...patch }, null, 2));
  } catch {
    // A read-only profile should not stop the app working
  }
}

function applyPosition(mode) {
  if (!mainWindow || mainWindow.isDestroyed()) return null;
  if (!POSITIONS.includes(mode)) mode = "float";

  // Use the display the window is on, not always the primary one, so this
  // behaves on multi-monitor setups.
  const display = screen.getDisplayMatching(mainWindow.getBounds());
  const area = display.workArea;

  if (CORNERS.includes(mode)) {
    // Never let the panel exceed the display, so this still behaves on small
    // or scaled screens.
    const w = Math.min(SNAP_SIZE.width, area.width - SNAP_MARGIN * 2);
    const h = Math.min(SNAP_SIZE.height, area.height - SNAP_MARGIN * 2);
    const left = area.x + SNAP_MARGIN;
    const right = area.x + area.width - w - SNAP_MARGIN;
    const top = area.y + SNAP_MARGIN;
    const bottom = area.y + area.height - h - SNAP_MARGIN;
    const x = mode.endsWith("left") ? left : right;
    const y = mode.startsWith("top") ? top : bottom;
    mainWindow.setBounds({ x, y, width: w, height: h });
  } else {
    const saved = readPrefs().floatBounds;
    const w = (saved && saved.width) || FLOAT_SIZE.width;
    const h = (saved && saved.height) || FLOAT_SIZE.height;
    const x = saved && Number.isInteger(saved.x) ? saved.x : Math.round(area.x + (area.width - w) / 2);
    const y = saved && Number.isInteger(saved.y) ? saved.y : Math.round(area.y + (area.height - h) * 0.66);
    mainWindow.setBounds({ x, y, width: w, height: h });
  }

  writePrefs({ position: mode });
  return mode;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: FLOAT_SIZE.width,
    height: FLOAT_SIZE.height,
    // Below this the signer has no usable room and the controls start
    // crowding it, so do not allow the window to be dragged smaller.
    minWidth: 320,
    minHeight: 320,
    title: "Signlytic AI Desktop",
    // Transparent and frameless so it overlays the desktop the way Live
    // Captions does. The panel chrome is drawn in the renderer instead, which
    // is also what makes the rounded translucent look possible.
    transparent: true,
    frame: false,
    backgroundColor: "#00000000",
    hasShadow: false,
    alwaysOnTop: true,
    // Not skipTaskbar: the window has no frame, so the taskbar entry is the
    // only way back to it if it ends up behind something.
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "..", "renderer", "index.html"));

  // Mirror renderer console to stdout so shell runs are verifiable headlessly
  mainWindow.webContents.on("console-message", (_e, _level, message) => {
    console.log("[renderer] " + message);
  });

  // Float mode is the only movable/resizable one, so remember where the user
  // put it. Docked modes are derived from the display and need no memory.
  const rememberFloat = () => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    if (readPrefs().position !== "float") return;
    writePrefs({ floatBounds: mainWindow.getBounds() });
  };
  mainWindow.on("moved", rememberFloat);
  mainWindow.on("resized", rememberFloat);

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  // Restore the last placement once the page is up, so the renderer's own
  // controls can show the right mode as selected.
  mainWindow.webContents.on("did-finish-load", () => {
    const mode = readPrefs().position || "float";
    applyPosition(mode);
    mainWindow.webContents.send("window-position", { mode });
  });
}

// Single entry point for caption text reaching the avatar.
// Session 1: fed only by the renderer's manual input box (round-trips
// through main so the full IPC path is exercised).
// Session 2: the UIA sidecar calls this with Windows Live Captions text.
function sendCaptionText(text, source) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("caption-text", { text: text, source: source });
  }
}

ipcMain.on("manual-text", (_event, text) => {
  sendCaptionText(String(text || ""), "manual");
});

// ── Windows Live Captions ────────────────────────────────────────────────────
function sendCaptionStatus(payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("caption-status", payload);
  }
}

function ensureCaptionStream() {
  if (captionStream) return captionStream;
  captionStream = new CaptionStream();
  captionStream.on("sentence", (text) => {
    console.log("[captions] " + text);
    sendCaptionText(text, "livecaptions");
  });
  captionStream.on("status", (s) => {
    console.log("[captions] status: " + s.state + " - " + s.detail);
    sendCaptionStatus({ running: captionStream.running, ...s });
  });
  captionStream.on("error-text", (m) => {
    console.log("[captions] error: " + m);
    sendCaptionStatus({ running: false, state: "error", detail: m });
  });
  return captionStream;
}

// ── Window controls (the frame is gone, so the renderer drives these) ────────
ipcMain.handle("window-set-position", (_e, mode) => {
  const applied = applyPosition(String(mode || ""));
  return { ok: !!applied, mode: applied };
});

ipcMain.handle("window-get-position", () => ({ mode: readPrefs().position || "float" }));

// Playback speed, remembered between runs alongside the window placement.
// Clamped here rather than trusted from the renderer, so a bad value cannot be
// written to the prefs file and stall or race the signing on the next launch.
const SPEEDS = [1, 1.25, 1.5, 1.75, 2];

function normaliseSpeed(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return SPEEDS.includes(n) ? n : null;
}

ipcMain.handle("speed-get", () => ({ speed: normaliseSpeed(readPrefs().speed) || 1 }));

ipcMain.handle("speed-set", (_e, value) => {
  const speed = normaliseSpeed(value);
  if (!speed) return { ok: false, speed: normaliseSpeed(readPrefs().speed) || 1 };
  writePrefs({ speed });
  return { ok: true, speed };
});

ipcMain.handle("window-minimise", () => {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.minimize();
  return { ok: true };
});

ipcMain.handle("window-close", () => {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.close();
  return { ok: true };
});

ipcMain.handle("captions-capabilities", () => CaptionStream.capabilities());

ipcMain.handle("captions-start", async () => {
  const caps = CaptionStream.capabilities();
  if (!caps.supported) return { ok: false, reason: caps.reason };

  // "Start Live Captions" should do exactly that. Previously it only attached
  // the reader and then sat waiting for a window the user had to open
  // themselves from a second button, which is not what the label promises.
  const launched = await CaptionStream.ensureLiveCaptionsRunning();

  const stream = ensureCaptionStream();
  stream.start();
  return { ok: true, launched };
});

ipcMain.handle("captions-stop", () => {
  if (captionStream) captionStream.stop();
  return { ok: true };
});

// Opens the Windows Live Captions app. Its first launch asks the user to
// accept terms and download a speech model, which we cannot do for them.
ipcMain.handle("captions-launch-app", () => {
  const caps = CaptionStream.capabilities();
  if (!caps.supported) return { ok: false, reason: caps.reason };
  return { ok: CaptionStream.launchLiveCaptions() };
});

app.on("before-quit", () => {
  if (captionStream) captionStream.stop();
});

// Scaffold verification: when SIGNLYTIC_SHOT is set to a file path, capture
// a screenshot once the avatar reports ready, then exit. Dev-only helper.
ipcMain.on("avatar-ready", () => {
  console.log("[main] avatar-ready received");
  const shotPath = process.env.SIGNLYTIC_SHOT;
  if (!shotPath || !mainWindow) return;
  setTimeout(() => {
    mainWindow.webContents.capturePage().then((img) => {
      fs.writeFileSync(shotPath, img.toPNG());
      console.log("[main] screenshot written: " + shotPath);
      app.quit();
    });
  }, 2000);
});

app.whenReady().then(() => {
  createWindow();
  // Dev-only: start the caption reader without a click so the pipeline can be
  // exercised headlessly. Set SIGNLYTIC_CAPTIONS_AUTOSTART=1.
  if (process.env.SIGNLYTIC_CAPTIONS_AUTOSTART === "1") {
    console.log("[captions] autostart enabled");
    ensureCaptionStream().start();
  }
  // Dev-only: push a sentence through the same path a caption takes, so the
  // text -> glosses -> signs -> playback chain can be exercised headlessly.
  if (process.env.SIGNLYTIC_TEST_TEXT) {
    setTimeout(() => {
      console.log("[test] injecting: " + process.env.SIGNLYTIC_TEST_TEXT);
      sendCaptionText(process.env.SIGNLYTIC_TEST_TEXT, "test");
    }, 4000);
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
