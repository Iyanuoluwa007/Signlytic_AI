// Runs the Live Captions sidecar and turns its output into finalised
// sentences.
//
// The sidecar is a PowerShell script rather than a compiled binary: Windows
// already ships the UI Automation assemblies, so this needs no .NET SDK, no
// Rust toolchain, no native addon and no build step.

const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const { EventEmitter } = require("events");
const { CaptionAssembler } = require("./caption-assembler");

const SIDECAR = path.join(__dirname, "live-captions.ps1");
const LIVE_CAPTIONS_EXE = path.join(
  process.env.SystemRoot || "C:\\Windows",
  "System32",
  "LiveCaptions.exe"
);

class CaptionStream extends EventEmitter {
  // Enough to ride out a transient failure, few enough that a genuinely
  // broken setup reports itself instead of respawning forever.
  static MAX_RESTARTS = 5;

  constructor(options = {}) {
    super();
    this.assembler = new CaptionAssembler(options);
    this.proc = null;
    this.buf = "";
    this.state = "stopped";
    this._settleTimer = null;
    // Whether the user wants captions on. Distinguishes "we stopped it" from
    // "it died", which decides whether to restart.
    this._wanted = false;
    this._restarts = 0;
    this._restartTimer = null;
    this._lastStderr = "";
  }

  // Whether this OS has a system caption source we can read.
  //
  // Windows: yes, via UI Automation against the Live Captions window.
  // macOS: not yet. macOS has Live Captions from Ventura, but it is read
  //   through the Accessibility API rather than UI Automation, needs a signed
  //   helper and explicit Accessibility permission, and is a separate native
  //   implementation. The rest of the app is platform-neutral, so only this
  //   source needs adding; until then macOS uses manual text entry.
  static capabilities() {
    if (process.platform === "win32") {
      return CaptionStream.isLiveCaptionsInstalled()
        ? { supported: true, source: "windows-live-captions" }
        : { supported: false, reason: "Live Captions is not available on this version of Windows" };
    }
    if (process.platform === "darwin") {
      return { supported: false, reason: "System captions on macOS are not wired up yet; use the text box" };
    }
    return { supported: false, reason: "System captions are not available on this platform" };
  }

  static isLiveCaptionsInstalled() {
    return process.platform === "win32" && fs.existsSync(LIVE_CAPTIONS_EXE);
  }

  // Live Captions has to be running for there to be anything to read. It is a
  // normal Windows app, so we can start it, but the first ever launch shows a
  // consent and language-download flow that the user must complete themselves.
  static launchLiveCaptions() {
    if (!CaptionStream.isLiveCaptionsInstalled()) return false;
    spawn(LIVE_CAPTIONS_EXE, { detached: true, stdio: "ignore" }).unref();
    return true;
  }

  start() {
    if (this.proc) return;
    // Never spawn the PowerShell sidecar off Windows.
    const caps = CaptionStream.capabilities();
    if (!caps.supported) {
      this._setState("error", caps.reason);
      return;
    }
    this._wanted = true;
    this._restarts = 0;
    this._spawn();
  }

  _spawn() {
    if (this.proc) return;
    this.assembler.reset();
    this.buf = "";

    this.proc = spawn(
      "powershell.exe",
      [
        "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", SIDECAR,
        // So the reader stops itself if this app is force-killed rather than
        // shut down, instead of lingering as an orphan.
        "-ParentPid", String(process.pid),
      ],
      { windowsHide: true }
    );

    this.proc.stdout.setEncoding("utf8");
    this.proc.stdout.on("data", (chunk) => this._onData(chunk));
    this.proc.stderr.setEncoding("utf8");
    this.proc.stderr.on("data", (d) => {
      const msg = String(d).trim();
      if (!msg) return;
      // Keep the last stderr so an unexplained exit can be reported with the
      // reason attached, rather than just a bare exit code.
      this._lastStderr = msg.slice(-500);
      this.emit("error-text", msg);
    });
    this.proc.on("exit", (code) => {
      this.proc = null;
      if (!this._wanted) {
        this._setState("stopped", "caption reader stopped");
        return;
      }
      // Died while captions were still meant to be running. The reader talks
      // to another process over COM, so an occasional death is possible;
      // restart rather than leaving the user with a dead panel.
      const why = this._describeExit(code);
      if (this._restarts >= CaptionStream.MAX_RESTARTS) {
        this._setState("error", "caption reader keeps stopping (" + why + ")");
        this._wanted = false;
        return;
      }
      this._restarts += 1;
      const delay = 500 * this._restarts;
      this._setState("waiting", "caption reader restarting after " + why);
      this._restartTimer = setTimeout(() => {
        if (this._wanted) this._spawn();
      }, delay);
    });

    this._setState("starting", "launching caption reader");

    // The assembler holds the newest sentence back until it stops changing, so
    // without a nudge a final sentence would sit unreleased once speech stops.
    if (!this._settleTimer) {
      this._settleTimer = setInterval(() => {
        for (const s of this.assembler.push(this.assembler.lastBuffer, Date.now())) {
          this.emit("sentence", s);
        }
      }, 600);
    }
  }

  // Windows exit codes from a terminated PowerShell host are not obvious, so
  // spell them out rather than surfacing a bare number to the user.
  _describeExit(code) {
    const parts = [];
    if (code === null || code === undefined) parts.push("terminated");
    else if (code === 0) parts.push("clean exit");
    else {
      const unsigned = code < 0 ? code >>> 0 : code;
      parts.push("code " + code);
      if (unsigned === 0xfffd0000) parts.push("PowerShell host ended unexpectedly");
      else if (unsigned === 0xc000013a) parts.push("interrupted");
      else if (code === 1) parts.push("script error");
      else if (code === 2) parts.push("Live Captions window not found");
    }
    if (this._lastStderr) parts.push(this._lastStderr.split("\n")[0].slice(0, 120));
    return parts.join(": ");
  }

  stop() {
    this._wanted = false;
    if (this._restartTimer) { clearTimeout(this._restartTimer); this._restartTimer = null; }
    if (this._settleTimer) { clearInterval(this._settleTimer); this._settleTimer = null; }
    if (this.proc) {
      this.proc.kill();
      this.proc = null;
    }
    this._setState("stopped", "caption reader stopped");
  }

  get running() {
    return !!this.proc;
  }

  _onData(chunk) {
    this.buf += chunk;
    let idx;
    while ((idx = this.buf.indexOf("\n")) >= 0) {
      const line = this.buf.slice(0, idx).trim();
      this.buf = this.buf.slice(idx + 1);
      if (!line) continue;
      let rec;
      try { rec = JSON.parse(line); } catch { continue; }
      this._onRecord(rec);
    }
  }

  _onRecord(rec) {
    if (rec.type === "caption") {
      for (const s of this.assembler.push(rec.text, Date.now())) {
        this.emit("sentence", s);
      }
    } else if (rec.type === "status") {
      this._setState(rec.state, rec.detail);
    } else if (rec.type === "error") {
      this.emit("error-text", rec.message);
    }
  }

  _setState(state, detail) {
    // Reaching "attached" means the reader is healthy again, so a later death
    // is a fresh problem rather than a continuation of the last one.
    if (state === "attached") this._restarts = 0;
    this.state = state;
    this.emit("status", { state, detail });
  }
}

module.exports = { CaptionStream };
