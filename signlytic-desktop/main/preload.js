// Context bridge between the sandboxed renderer and the main process.
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("signlytic", {
  // Manual caption fallback: renderer -> main -> back via caption-text
  sendManualText: (text) => ipcRenderer.send("manual-text", text),
  // Subscribe to caption text from any source (manual or Windows Live Captions)
  onCaptionText: (cb) => ipcRenderer.on("caption-text", (_e, payload) => cb(payload)),
  // Scaffold verification signal
  notifyAvatarReady: () => ipcRenderer.send("avatar-ready"),

  // Windows Live Captions
  captions: {
    capabilities: () => ipcRenderer.invoke("captions-capabilities"),
    start: () => ipcRenderer.invoke("captions-start"),
    stop: () => ipcRenderer.invoke("captions-stop"),
    launchApp: () => ipcRenderer.invoke("captions-launch-app"),
    onStatus: (cb) => ipcRenderer.on("caption-status", (_e, payload) => cb(payload)),
  },
});
