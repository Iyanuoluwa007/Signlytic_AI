"use client";

// Shared BSL translation panel. One instance per page (architecture decision:
// a single panel that swaps content, not an avatar per sentence).
//
// Two render engines behind one playback interface:
//   2D skeleton  - default. No WebGL, no 3D model download, and it is the
//                  renderer whose output is currently correct.
//   3D avatar    - opt in. Lazy-loads Three.js plus a ~33 MB model, so it is
//                  never fetched unless the visitor asks for it.
// Both expose playQueue/stopQueue/pause/resume/speed, so switching is just
// swapping which object the controls talk to.
import { useCallback, useEffect, useRef, useState } from "react";
import { BSL_SIGN_REQUEST_EVENT, BSL_ENABLED_KEY } from "./Signable";

interface PoseFrame {
  body?: number[][];
  lh?: number[][];
  rh?: number[][];
}
interface SignClip {
  gloss: string;
  frames: (PoseFrame | null)[];
}

interface PlaybackEngine {
  speed: number;
  ready?: boolean;
  playQueue(
    queue: SignClip[],
    onGlossChange?: (idx: number) => void,
    onDone?: () => void
  ): void;
  stopQueue(): void;
  pause(): void;
  resume(): void;
  resize?(w?: number, h?: number): void;
}

interface AvatarEngine extends PlaybackEngine {
  ready: boolean;
  initScene(): void;
  load(onProgress?: (pct: number) => void): Promise<void>;
}

declare global {
  interface Window {
    ThreeAvatarRenderer?: new (
      canvas: HTMLCanvasElement,
      options?: { gender?: string; speed?: number; modelUrl?: string }
    ) => AvatarEngine;
    SkeletonRenderer2D?: new (
      canvas: HTMLCanvasElement,
      options?: { speed?: number }
    ) => PlaybackEngine;
  }
}

// avatar3d.js also defines PoseNormaliser, which skeleton2d.js needs, so it
// loads first even in 2D mode. It does not touch Three.js at load time.
const CORE_SCRIPTS = ["/bsl/avatar3d.js", "/bsl/skeleton2d.js"];
const THREE_SCRIPTS = ["/bsl/three.min.js", "/bsl/GLTFLoader.js"];
// Served by /api/avatar, which proxies the model out of the private asset
// repo. Overridable so local dev can point at an on-disk copy instead, since
// a dev machine has no GitHub token.
const AVATAR_MODEL = process.env.NEXT_PUBLIC_AVATAR_MALE_URL || "/api/avatar/male";

const GLOSS_CACHE_KEY = "signlytic-bsl-gloss-cache-v1";
const MODE_KEY = "signlytic-bsl-mode";
// Same-origin in production. Local dev has no Upstash/GitHub credentials, so
// .env.local can point sign fetches at the deployed API, which sends CORS.
const SIGNS_API_BASE = process.env.NEXT_PUBLIC_SIGNS_API_BASE || "";
// Glosses with no pose data are fingerspelled for this many frames
const FINGERSPELL_HOLD = 20;

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector('script[src="' + src + '"]')) return resolve();
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load " + src));
    document.head.appendChild(s);
  });
}

function cacheKeyFor(text: string) {
  return text.trim().toLowerCase().replace(/\s+/g, " ");
}
function getCachedGlosses(text: string): string | null {
  try {
    return JSON.parse(localStorage.getItem(GLOSS_CACHE_KEY) || "{}")[cacheKeyFor(text)] || null;
  } catch {
    return null;
  }
}
function setCachedGlosses(text: string, glosses: string) {
  try {
    const map = JSON.parse(localStorage.getItem(GLOSS_CACHE_KEY) || "{}");
    map[cacheKeyFor(text)] = glosses;
    localStorage.setItem(GLOSS_CACHE_KEY, JSON.stringify(map));
  } catch {
    // best effort only
  }
}

export default function BslSignPanel() {
  const [visible, setVisible] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [mode, setMode] = useState<"2d" | "3d">("2d");
  const [status, setStatus] = useState("");
  const [sentence, setSentence] = useState("");
  const [glossList, setGlossList] = useState<string[]>([]);
  const [activeGloss, setActiveGloss] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [avatarLoading, setAvatarLoading] = useState(false);

  const canvas2dRef = useRef<HTMLCanvasElement>(null);
  const canvas3dRef = useRef<HTMLCanvasElement>(null);
  const engine2dRef = useRef<PlaybackEngine | null>(null);
  const engine3dRef = useRef<AvatarEngine | null>(null);
  const queueRef = useRef<SignClip[]>([]);
  const busyRef = useRef(false);
  const modeRef = useRef<"2d" | "3d">("2d");
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const lastFocusRef = useRef<Element | null>(null);

  const activeEngine = useCallback(
    () => (modeRef.current === "3d" ? engine3dRef.current : engine2dRef.current),
    []
  );

  // Restore the visitor's last chosen mode
  useEffect(() => {
    try {
      const saved = localStorage.getItem(MODE_KEY);
      if (saved === "3d" || saved === "2d") {
        setMode(saved);
        modeRef.current = saved;
      }
    } catch {
      // ignore
    }
  }, []);

  const ensure2d = useCallback(async () => {
    if (engine2dRef.current) return engine2dRef.current;
    setStatus("Loading renderer...");
    for (const src of CORE_SCRIPTS) await loadScript(src);
    if (!window.SkeletonRenderer2D || !canvas2dRef.current) {
      throw new Error("2D renderer unavailable");
    }
    engine2dRef.current = new window.SkeletonRenderer2D(canvas2dRef.current, { speed });
    return engine2dRef.current;
  }, [speed]);

  const ensure3d = useCallback(async () => {
    if (engine3dRef.current) return engine3dRef.current;
    setAvatarLoading(true);
    try {
      setStatus("Loading 3D engine...");
      for (const src of CORE_SCRIPTS) await loadScript(src);
      for (const src of THREE_SCRIPTS) await loadScript(src);
      if (!window.ThreeAvatarRenderer || !canvas3dRef.current) {
        throw new Error("3D engine unavailable");
      }
      const avatar = new window.ThreeAvatarRenderer(canvas3dRef.current, {
        gender: "male",
        modelUrl: AVATAR_MODEL,
      });
      avatar.initScene();
      setStatus("Loading avatar model (this is a large download)...");
      await avatar.load();
      if (!avatar.ready) throw new Error("Avatar model failed to load");
      engine3dRef.current = avatar;
      return avatar;
    } finally {
      setAvatarLoading(false);
    }
  }, []);

  const startPlayback = useCallback((queue: SignClip[]) => {
    const engine = activeEngine();
    if (!engine || !queue.length) return;
    engine.speed = speed;
    setPlaying(true);
    setPaused(false);
    setStatus("Signing...");
    engine.playQueue(
      queue,
      (idx) => setActiveGloss(idx),
      () => {
        setPlaying(false);
        setActiveGloss(-1);
        setStatus("Finished. Use Replay to watch again.");
      }
    );
  }, [activeEngine, speed]);

  const handleRequest = useCallback(async (text: string) => {
    if (busyRef.current) return;
    busyRef.current = true;
    lastFocusRef.current = document.activeElement;
    setVisible(true);
    setSentence(text);
    setGlossList([]);
    setActiveGloss(-1);
    setPlaying(false);
    setPaused(false);

    try {
      if (modeRef.current === "3d") await ensure3d();
      else await ensure2d();

      // 1. English -> glosses (cached per sentence, so Groq is hit once)
      let glosses = getCachedGlosses(text);
      if (!glosses) {
        setStatus("Translating to BSL glosses...");
        const res = await fetch("/api/english-to-glosses", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error("Translation unavailable (" + res.status + ")");
        const data = await res.json();
        if (!data.glosses) throw new Error("Empty translation");
        glosses = data.glosses as string;
        setCachedGlosses(text, glosses);
      }
      const glossArr = glosses.split(/\s+/).filter(Boolean);
      setGlossList(glossArr);

      // 2. Pose frames per gloss. Anything without data is fingerspelled
      // rather than dropped, so a cold cache still produces a usable result.
      setStatus("Fetching sign data...");
      const clips = await Promise.all(
        glossArr.map(async (gloss): Promise<SignClip> => {
          try {
            const r = await fetch(SIGNS_API_BASE + "/api/signs/" + encodeURIComponent(gloss));
            if (r.ok) {
              const frames = await r.json();
              if (Array.isArray(frames) && frames.length) return { gloss, frames };
            }
          } catch {
            // fall through to fingerspelling
          }
          return { gloss, frames: Array(FINGERSPELL_HOLD).fill(null) };
        })
      );

      queueRef.current = clips;
      startPlayback(clips);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Something went wrong");
      setPlaying(false);
    } finally {
      busyRef.current = false;
    }
  }, [ensure2d, ensure3d, startPlayback]);

  useEffect(() => {
    const listener = (e: Event) => {
      const detail = (e as CustomEvent<{ text: string }>).detail;
      if (detail && detail.text) void handleRequest(detail.text);
    };
    document.addEventListener(BSL_SIGN_REQUEST_EVENT, listener);
    return () => document.removeEventListener(BSL_SIGN_REQUEST_EVENT, listener);
  }, [handleRequest]);

  const close = useCallback(() => {
    engine2dRef.current?.stopQueue();
    engine3dRef.current?.stopQueue();
    setPlaying(false);
    setPaused(false);
    setVisible(false);
    setExpanded(false);
    // Return focus to whatever opened the panel
    const prev = lastFocusRef.current as HTMLElement | null;
    if (prev && typeof prev.focus === "function") prev.focus();
  }, []);

  // Focus the panel when it opens, and close on Escape
  useEffect(() => {
    if (!visible) return;
    closeBtnRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [visible, close]);

  const togglePause = () => {
    const engine = activeEngine();
    if (!engine || !playing) return;
    if (paused) { engine.resume(); setPaused(false); }
    else { engine.pause(); setPaused(true); }
  };

  const replay = () => {
    if (queueRef.current.length) startPlayback(queueRef.current);
  };

  const changeSpeed = (v: number) => {
    setSpeed(v);
    const engine = activeEngine();
    if (engine) engine.speed = v;
  };

  const switchMode = async (next: "2d" | "3d") => {
    if (next === mode || busyRef.current) return;
    activeEngine()?.stopQueue();
    setPlaying(false);
    setPaused(false);
    setActiveGloss(-1);
    setMode(next);
    modeRef.current = next;
    try { localStorage.setItem(MODE_KEY, next); } catch { /* ignore */ }

    try {
      if (next === "3d") await ensure3d();
      else await ensure2d();
      if (queueRef.current.length) startPlayback(queueRef.current);
      else setStatus("");
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Could not switch renderer");
    }
  };

  if (!visible) return null;

  const frameH = expanded ? "h-[420px] sm:h-[460px]" : "h-[190px] sm:h-[210px]";

  return (
    <div
      role="dialog"
      aria-modal="false"
      aria-label="British Sign Language translation player"
      className={
        "fixed z-50 bg-[#0b0d13] border border-white/10 shadow-2xl shadow-black/60 overflow-hidden " +
        "inset-x-0 bottom-0 rounded-t-xl sm:inset-x-auto sm:bottom-4 sm:right-4 sm:rounded-xl " +
        (expanded ? "sm:w-[480px]" : "sm:w-[320px]")
      }
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06] gap-2">
        <span className="text-[11px] font-semibold text-[#5eead4]/90 whitespace-nowrap">BSL Translation</span>

        <div className="flex items-center gap-1 ml-auto" role="group" aria-label="Renderer">
          <button
            type="button"
            onClick={() => switchMode("2d")}
            aria-pressed={mode === "2d"}
            className={
              "px-2 py-0.5 rounded text-[10px] font-semibold transition-colors " +
              (mode === "2d" ? "bg-[#0e7c6b] text-white" : "bg-white/[0.06] text-white/50 hover:text-white/80")
            }
          >
            2D
          </button>
          <button
            type="button"
            onClick={() => switchMode("3d")}
            aria-pressed={mode === "3d"}
            disabled={avatarLoading}
            title="3D avatar (downloads a large model on first use)"
            className={
              "px-2 py-0.5 rounded text-[10px] font-semibold transition-colors disabled:opacity-40 " +
              (mode === "3d" ? "bg-[#0e7c6b] text-white" : "bg-white/[0.06] text-white/50 hover:text-white/80")
            }
          >
            3D
          </button>
        </div>

        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          aria-label={expanded ? "Shrink player" : "Expand player"}
          className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-white/[0.06] text-white/50 hover:text-white/80"
        >
          {expanded ? "Shrink" : "Expand"}
        </button>
        <button
          ref={closeBtnRef}
          type="button"
          onClick={close}
          aria-label="Close BSL player"
          className="text-white/40 hover:text-white/80 text-[15px] leading-none px-1"
        >
          &#215;
        </button>
      </div>

      <div className={"w-full bg-[#06080f] " + frameH}>
        <canvas
          ref={canvas2dRef}
          className={"w-full h-full block " + (mode === "2d" ? "" : "hidden")}
          aria-label="BSL skeleton animation"
        />
        <canvas
          ref={canvas3dRef}
          className={"w-full h-full block " + (mode === "3d" ? "" : "hidden")}
          aria-label="BSL avatar animation"
        />
      </div>

      <div className="px-3 py-2 space-y-2">
        <p className="text-[10px] text-white/40 leading-snug line-clamp-2">{sentence}</p>

        {glossList.length > 0 && (
          <div className="flex flex-wrap gap-1" aria-label="BSL glosses">
            {glossList.map((g, i) => (
              <span
                key={g + i}
                className={
                  "px-1.5 py-px rounded text-[9px] font-semibold tracking-wide " +
                  (i === activeGloss ? "bg-[#0e7c6b]/70 text-white" : "bg-white/[0.05] text-white/40")
                }
              >
                {g}
              </span>
            ))}
          </div>
        )}

        <p aria-live="polite" className="text-[10px] text-[#5eead4]/60 min-h-[14px]">{status}</p>

        <div className="flex items-center gap-2 flex-wrap">
          <button
            type="button"
            onClick={togglePause}
            disabled={!playing}
            className="px-2.5 py-1 rounded-md bg-[#0e7c6b] text-white text-[11px] font-semibold disabled:opacity-30"
          >
            {paused ? "Resume" : "Pause"}
          </button>
          <button
            type="button"
            onClick={replay}
            disabled={!queueRef.current.length}
            className="px-2.5 py-1 rounded-md bg-white/[0.06] text-white/70 text-[11px] font-semibold disabled:opacity-30"
          >
            Replay
          </button>
          <label className="ml-auto flex items-center gap-1 text-[10px] text-white/40">
            Speed
            <select
              value={speed}
              onChange={(e) => changeSpeed(Number(e.target.value))}
              aria-label="Signing speed"
              className="bg-white/[0.06] text-white/70 text-[10px] rounded px-1 py-0.5 border border-white/10"
            >
              <option value={0.5}>0.5x</option>
              <option value={0.75}>0.75x</option>
              <option value={1}>1x</option>
              <option value={1.5}>1.5x</option>
            </select>
          </label>
        </div>
      </div>
    </div>
  );
}

export { BSL_ENABLED_KEY };
