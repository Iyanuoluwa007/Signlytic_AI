// Context bridge between the sandboxed renderer and the main process.
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("signlytic", {
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

  // Windows Live Captions
  captions: {
    capabilities: () => ipcRenderer.invoke("captions-capabilities"),
    start: () => ipcRenderer.invoke("captions-start"),
    stop: () => ipcRenderer.invoke("captions-stop"),
    launchApp: () => ipcRenderer.invoke("captions-launch-app"),
    onStatus: (cb) => ipcRenderer.on("caption-status", (_e, payload) => cb(payload)),
  },
});
