// Signlytic AI Desktop - Electron main process (Session 1 shell)
//
// Session 2 will add: system tray, always-on-top overlay window mode, and
// the Windows UI Automation caption sidecar (see main/captions/README.md).
const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { CaptionStream } = require("./captions/caption-stream");

let mainWindow = null;
let captionStream = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 960,
    height: 720,
    title: "Signlytic AI Desktop",
    backgroundColor: "#06080f",
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

  mainWindow.on("closed", () => {
    mainWindow = null;
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

ipcMain.handle("captions-capabilities", () => CaptionStream.capabilities());

ipcMain.handle("captions-start", () => {
  const caps = CaptionStream.capabilities();
  if (!caps.supported) return { ok: false, reason: caps.reason };
  const stream = ensureCaptionStream();
  stream.start();
  return { ok: true };
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
