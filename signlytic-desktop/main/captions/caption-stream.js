// Runs the platform's caption helper and turns its output into finalised
// sentences.
//
// Both helpers print the same thing: JSON lines carrying the whole caption
// buffer as it stands. Everything after that point is shared, so the assembler
// and the sentence logic have no platform branches in them at all.
//
// Windows: a PowerShell script rather than a compiled binary, because Windows
// already ships the UI Automation assemblies, so it needs no .NET SDK, no Rust
// toolchain, no native addon and no build step.
//
// macOS: a compiled Swift binary that recognises speech, rather than reading
// the system Live Captions window. There is no supported API for reading that
// window's output, and speech recognition is a public API that runs on device
// and also covers speech in the room. See main/captions/README.md, and
// mac/tools/ax-probe.swift for the tool that tests the Accessibility route.

const { spawn, execFile } = require("child_process");
const path = require("path");
const fs = require("fs");
const { EventEmitter } = require("events");
const { CaptionAssembler } = require("./caption-assembler");

// In a packaged build this file lives inside app.asar. Electron can read that
// archive, but PowerShell cannot: it is not a real path on disk, so
// `powershell -File ...\app.asar\main\captions\live-captions.ps1` fails with
// an argument error and the reader dies immediately. The script is therefore
// unpacked at build time (see asarUnpack in package.json) and we point at the
// unpacked copy. Harmless in development, where no asar exists.
const unpacked = (name) =>
  path
    .join(__dirname, name)
    .replace(`app.asar${path.sep}`, `app.asar.unpacked${path.sep}`);

const SIDECAR = unpacked("live-captions.ps1");
const MIC_SCRIPT = unpacked("enable-microphone.ps1");
// Same asar problem as the PowerShell scripts, and worse: an executable cannot
// be run from inside the archive at all. It is unpacked at build time and the
// path rewritten here to match.
const MAC_HELPER = unpacked(path.join("mac", "signlytic-captions"));
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
    // macOS only: microphone or system audio. Ignored on Windows, where Live
    // Captions decides for itself what it listens to.
    this._audioSource = options.audioSource === "system" ? "system" : "mic";
  }

  // Whether this OS has a caption source we can read.
  //
  // Windows: yes, via UI Automation against the Live Captions window.
  // macOS: yes, but by recognising speech rather than by reading the system
  //   Live Captions window. Apple publishes no API for reading that window,
  //   whereas speech recognition is public, runs on device, and also covers
  //   speech in the room, which reading a caption window never could.
  static capabilities() {
    if (process.platform === "win32") {
      return CaptionStream.isLiveCaptionsInstalled()
        ? { supported: true, source: "windows-live-captions" }
        : { supported: false, reason: "Live Captions is not available on this version of Windows" };
    }
    if (process.platform === "darwin") {
      // A source build that has not run the helper build step would otherwise
      // fail at spawn time with nothing useful to show the user.
      return fs.existsSync(MAC_HELPER)
        ? { supported: true, source: "macos-speech-recognition" }
        : { supported: false, reason: "The caption helper has not been built; run npm run build:mac-helper" };
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

  // True when LiveCaptions.exe already has a process.
  static isLiveCaptionsRunning() {
    if (process.platform !== "win32") return Promise.resolve(false);
    return new Promise((resolve) => {
      execFile(
        "tasklist",
        ["/FI", "IMAGENAME eq LiveCaptions.exe", "/NH"],
        { windowsHide: true },
        (err, stdout) => resolve(!err && /LiveCaptions\.exe/i.test(String(stdout)))
      );
    });
  }

  // Start Live Captions only if it is not already up. Launching it when it is
  // already running would toggle or re-focus it, which is not what someone
  // pressing "start" expects.
  static async ensureLiveCaptionsRunning() {
    if (await CaptionStream.isLiveCaptionsRunning()) return false;
    return CaptionStream.launchLiveCaptions();
  }

  // Switch on "Include microphone audio" in Live Captions.
  //
  // Best effort on purpose. Live Captions keeps this preference to itself, with
  // no documented API or registry key, so the only way in is UI Automation
  // against its settings menu. A future Windows build could rename or move the
  // item, so this must never be allowed to block or break the reader: it runs
  // as its own short-lived process, is capped by a timeout, and every failure
  // resolves rather than throws.
  static enableMicrophoneAudio(timeoutMs = 12000) {
    if (process.platform !== "win32") {
      return Promise.resolve({ ok: false, detail: "not Windows" });
    }
    return new Promise((resolve) => {
      let settled = false;
      const done = (result) => {
        if (settled) return;
        settled = true;
        resolve(result);
      };

      const proc = spawn(
        "powershell.exe",
        [
          "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
          "-File", MIC_SCRIPT,
        ],
        { windowsHide: true }
      );

      // The script drives another app's menus, so a hang is conceivable. Cap it
      // and move on rather than leaving the user's click unanswered.
      const timer = setTimeout(() => {
        try { proc.kill(); } catch { }
        done({ ok: false, detail: "timed out" });
      }, timeoutMs);

      let out = "";
      proc.stdout.setEncoding("utf8");
      proc.stdout.on("data", (d) => { out += d; });
      proc.on("error", () => {
        clearTimeout(timer);
        done({ ok: false, detail: "could not run the helper" });
      });
      proc.on("exit", () => {
        clearTimeout(timer);
        let rec = null;
        for (const line of out.split("\n")) {
          const t = line.trim();
          if (!t) continue;
          try { rec = JSON.parse(t); } catch { }
        }
        if (rec && rec.status === "ok") done({ ok: true, detail: rec.detail });
        else done({ ok: false, detail: (rec && rec.detail) || "could not set the option" });
      });
    });
  }

  start(options = {}) {
    if (this.proc) return;
    // Never spawn a helper the platform has no source for.
    const caps = CaptionStream.capabilities();
    if (!caps.supported) {
      this._setState("error", caps.reason);
      return;
    }
    // Taken at start rather than held from construction, so changing the
    // setting and pressing start again actually switches source.
    if (options.audioSource) {
      this._audioSource = options.audioSource === "system" ? "system" : "mic";
    }
    this._wanted = true;
    this._restarts = 0;
    this._spawn();
  }

  _spawn() {
    if (this.proc) return;
    this.assembler.reset();
    this.buf = "";

    this.proc = process.platform === "darwin"
      ? CaptionStream._spawnMacHelper(this._audioSource)
      : CaptionStream._spawnWindowsSidecar();

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
      // Restarting cannot fix a refused permission; it just buries the reason
      // under five retries before the user is finally told.
      if (process.platform === "darwin" && code >= 3 && code <= 6) {
        this._setState("error", why);
        this._wanted = false;
        return;
      }
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

  static _spawnWindowsSidecar() {
    return spawn(
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
  }

  static _spawnMacHelper(audioSource) {
    return spawn(
      MAC_HELPER,
      [
        "--source", audioSource,
        // British English, because the whole point is British Sign Language.
        "--locale", "en-GB",
        // Same orphan precaution as on Windows, and it matters more here: an
        // abandoned helper would hold the microphone open.
        "--parent-pid", String(process.pid),
      ]
    );
  }

  // Windows exit codes from a terminated PowerShell host are not obvious, and
  // the macOS helper's codes say which permission was refused, so spell both
  // out rather than surfacing a bare number to the user.
  _describeExit(code) {
    const parts = [];
    if (code === null || code === undefined) parts.push("terminated");
    else if (code === 0) parts.push("clean exit");
    else {
      const unsigned = code < 0 ? code >>> 0 : code;
      parts.push("code " + code);
      if (process.platform === "darwin") {
        // Kept in step with the exit code constants in caption-source.swift.
        // A refused permission is the likeliest failure by far, and it is not
        // something a restart can fix, so say which one plainly.
        if (code === 1) parts.push("helper error");
        else if (code === 3) parts.push("speech recognition permission was refused");
        else if (code === 4) parts.push("microphone permission was refused");
        else if (code === 5) parts.push("screen recording permission was refused");
        else if (code === 6) parts.push("speech recognition is unavailable for British English");
      } else {
        if (unsigned === 0xfffd0000) parts.push("PowerShell host ended unexpectedly");
        else if (unsigned === 0xc000013a) parts.push("interrupted");
        else if (code === 1) parts.push("script error");
        else if (code === 2) parts.push("Live Captions window not found");
      }
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
