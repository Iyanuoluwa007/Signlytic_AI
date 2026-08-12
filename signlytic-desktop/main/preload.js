// Context bridge between the sandboxed renderer and the main process.
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("signlytic", {
  // Dev switch: set SIGNLYTIC_LAYOUT_DEBUG=1 to have the renderer report
  // where its blocks sit, which is how the layout order gets verified.
  layoutDebug: process.env.SIGNLYTIC_LAYOUT_DEBUG === "1",
  // Dev switch: SIGNLYTIC_START_MODE=3d boots straight into the 3D renderer,
  // so its sizing can be checked without driving the UI.
  startMode: process.env.SIGNLYTIC_START_MODE || "",
  // Dev switch: SIGNLYTIC_OPEN_SETTINGS=1 opens the settings panel at boot, so
  // its layout can be captured and checked at each window size.
  openSettings: process.env.SIGNLYTIC_OPEN_SETTINGS === "1",

  // Manual caption fallback: renderer -> main -> back via caption-text
  sendManualText: (text) => ipcRenderer.send("manual-text", text),
  // Subscribe to caption text from any source (manual or Windows Live Captions)
  onCaptionText: (cb) => ipcRenderer.on("caption-text", (_e, payload) => cb(payload)),
  // Scaffold verification signal
  notifyAvatarReady: () => ipcRenderer.send("avatar-ready"),

  // Window placement and chrome. The window is frameless, so the renderer
  // provides its own controls.
  window: {
    setPosition: (mode) => ipcRenderer.invoke("window-set-position", mode),
    getPosition: () => ipcRenderer.invoke("window-get-position"),
    minimise: () => ipcRenderer.invoke("window-minimise"),
    close: () => ipcRenderer.invoke("window-close"),
    onPosition: (cb) => ipcRenderer.on("window-position", (_e, payload) => cb(payload)),
  },

  // Signing speed, remembered between runs.
  speed: {
    get: () => ipcRenderer.invoke("speed-get"),
    set: (value) => ipcRenderer.invoke("speed-set", value),
  },

  // Windows Live Captions
  captions: {
    // Whether starting captions should also switch on "Include microphone
    // audio" in Live Captions.
    getMicAudio: () => ipcRenderer.invoke("mic-audio-get"),
    setMicAudio: (value) => ipcRenderer.invoke("mic-audio-set", value),
    capabilities: () => ipcRenderer.invoke("captions-capabilities"),
    start: () => ipcRenderer.invoke("captions-start"),
    stop: () => ipcRenderer.invoke("captions-stop"),
    launchApp: () => ipcRenderer.invoke("captions-launch-app"),
    onStatus: (cb) => ipcRenderer.on("caption-status", (_e, payload) => cb(payload)),
  },
});
