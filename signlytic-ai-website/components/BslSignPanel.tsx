"use client";

// Shared BSL avatar panel (architecture decision: ONE panel per page, not
// per-sentence avatars). Mounted once; hidden until the first Signable
// trigger fires. Lazily loads the Three.js engine from /public/bsl on first
// use, translates the sentence via /api/english-to-glosses (cached in
// localStorage), fetches pose frames per gloss from /api/signs/[gloss],
// and plays them on the shared ThreeAvatarRenderer instance.
//
// Engine note: this component is the seam for the pluggable-engine
// requirement. prepare/play/pause/replay below only touch the avatar via a
// small surface, so a pre-rendered <video> engine can replace it later
// without changing Signable or page code.
import { useEffect, useRef, useState } from "react";
import { BSL_SIGN_REQUEST_EVENT } from "./Signable";

declare global {
  interface Window {
    ThreeAvatarRenderer?: new (
      canvas: HTMLCanvasElement,
      options?: { gender?: string; speed?: number; modelUrl?: string }
    ) => AvatarEngine;
  }
}

interface AvatarEngine {
  ready: boolean;
  speed: number;
  initScene(): void;
  load(onProgress?: (pct: number) => void): Promise<void>;
  playQueue(
    queue: SignClip[],
    onGlossChange?: (idx: number) => void,
    onDone?: () => void
  ): void;
  stopQueue(): void;
  pause(): void;
  resume(): void;
}

interface SignClip {
  gloss: string;
  frames: unknown[];
}

const ENGINE_SCRIPTS = ["/bsl/three.min.js", "/bsl/GLTFLoader.js", "/bsl/avatar3d.js"];
const GLOSS_CACHE_KEY = "signlytic-bsl-gloss-cache-v1";
// Same-origin in production. Local dev has no Upstash/GitHub secrets, so
// .env.local can point sign-data fetches at the deployed API (CORS-enabled).
const SIGNS_API_BASE = process.env.NEXT_PUBLIC_SIGNS_API_BASE || "";

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector('script[src="' + src + '"]')) {
      resolve();
      return;
    }
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load " + src));
    document.head.appendChild(s);
  });
}

function cacheKeyFor(text: string): string {
  return text.trim().toLowerCase().replace(/\s+/g, " ");
}

function getCachedGlosses(text: string): string | null {
  try {
    const map = JSON.parse(localStorage.getItem(GLOSS_CACHE_KEY) || "{}");
    return map[cacheKeyFor(text)] || null;
  } catch {
    return null;
  }
}

function setCachedGlosses(text: string, glosses: string): void {
  try {
    const map = JSON.parse(localStorage.getItem(GLOSS_CACHE_KEY) || "{}");
    map[cacheKeyFor(text)] = glosses;
    localStorage.setItem(GLOSS_CACHE_KEY, JSON.stringify(map));
  } catch {
    // localStorage unavailable; cache is best-effort only
  }
}

export default function BslSignPanel() {
  const [visible, setVisible] = useState(false);
  const [status, setStatus] = useState("");
  const [sentence, setSentence] = useState("");
  const [glossList, setGlossList] = useState<string[]>([]);
  const [activeGloss, setActiveGloss] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1.0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const avatarRef = useRef<AvatarEngine | null>(null);
  const queueRef = useRef<SignClip[]>([]);
  const busyRef = useRef(false);

  const startPlayback = (queue: SignClip[]) => {
    const avatar = avatarRef.current;
    if (!avatar || !queue.length) return;
    setPlaying(true);
    setPaused(false);
    setStatus("Signing...");
    avatar.playQueue(
      queue,
      (idx) => setActiveGloss(idx),
      () => {
        setPlaying(false);
        setActiveGloss(-1);
        setStatus("Done. Replay to watch again.");
      }
    );
  };

  const handleRequest = async (text: string) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setVisible(true);
    setSentence(text);
    setGlossList([]);
    setActiveGloss(-1);
    setPlaying(false);
    setPaused(false);

    try {
      // 1. Engine (lazy, once)
      if (!avatarRef.current) {
        setStatus("Loading 3D engine...");
        for (const src of ENGINE_SCRIPTS) await loadScript(src);
        if (!window.ThreeAvatarRenderer || !canvasRef.current) {
          throw new Error("Avatar engine unavailable");
        }
        const avatar = new window.ThreeAvatarRenderer(canvasRef.current, {
          gender: "male",
          // Served same-origin: browsers cannot fetch GitHub release assets
          modelUrl: "/bsl/Male.glb",
        });
        avatar.initScene();
        setStatus("Loading avatar model...");
        await avatar.load();
        if (!avatar.ready) throw new Error("Avatar model failed to load");
        avatarRef.current = avatar;
      }

      // 2. English -> glosses (localStorage cache, then API)
      let glosses = getCachedGlosses(text);
      if (!glosses) {
        setStatus("Translating to BSL glosses...");
        const res = await fetch("/api/english-to-glosses", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error("Translation failed (" + res.status + ")");
        const data = await res.json();
        if (!data.glosses) throw new Error("Empty translation");
        glosses = data.glosses as string;
        setCachedGlosses(text, glosses);
      }
      const glossArr = glosses.split(/\s+/).filter(Boolean);
      setGlossList(glossArr);

      // 3. Pose frames per gloss (signs API is CDN/Redis cached server-side)
      setStatus("Fetching sign data (0/" + glossArr.length + ")...");
      let fetched = 0;
      const clips = await Promise.all(
        glossArr.map(async (gloss): Promise<SignClip | null> => {
          try {
            const r = await fetch(SIGNS_API_BASE + "/api/signs/" + encodeURIComponent(gloss));
            fetched++;
            setStatus("Fetching sign data (" + fetched + "/" + glossArr.length + ")...");
            if (!r.ok) return null;
            const frames = await r.json();
            return Array.isArray(frames) && frames.length ? { gloss, frames } : null;
          } catch {
            return null;
          }
        })
      );
      const queue = clips.filter((c): c is SignClip => c !== null);
      if (!queue.length) {
        throw new Error("No sign data available for this sentence");
      }

      // 4. Play
      queueRef.current = queue;
      avatarRef.current.speed = speed;
      startPlayback(queue);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Something went wrong");
      setPlaying(false);
    } finally {
      busyRef.current = false;
    }
  };

  useEffect(() => {
    const listener = (e: Event) => {
      const detail = (e as CustomEvent<{ text: string }>).detail;
      if (detail && detail.text) void handleRequest(detail.text);
    };
    document.addEventListener(BSL_SIGN_REQUEST_EVENT, listener);
    return () => document.removeEventListener(BSL_SIGN_REQUEST_EVENT, listener);
    // handleRequest is stable enough for the scaffold: state it closes over
    // is accessed via refs where staleness would matter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const togglePause = () => {
    const avatar = avatarRef.current;
    if (!avatar || !playing) return;
    if (paused) {
      avatar.resume();
      setPaused(false);
    } else {
      avatar.pause();
      setPaused(true);
    }
  };

  const replay = () => {
    if (queueRef.current.length) startPlayback(queueRef.current);
  };

  const close = () => {
    avatarRef.current?.stopQueue();
    setPlaying(false);
    setPaused(false);
    setVisible(false);
  };

  const changeSpeed = (v: number) => {
    setSpeed(v);
    if (avatarRef.current) avatarRef.current.speed = v;
  };

  if (!visible) return null;

  return (
    <div
      role="dialog"
      aria-label="BSL translation player"
      className="fixed bottom-4 right-4 z-50 w-[300px] bg-[#0b0d13] border border-white/10 rounded-xl shadow-2xl shadow-black/60 overflow-hidden"
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.06]">
        <span className="text-[11px] font-semibold text-[#5eead4]/90">BSL Translation</span>
        <button
          type="button"
          onClick={close}
          aria-label="Close BSL panel"
          className="text-white/30 hover:text-white/70 text-[13px] leading-none px-1"
        >
          &#215;
        </button>
      </div>

      <div className="w-full h-[210px] bg-[#06080f]">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>

      <div className="px-3 py-2 space-y-2">
        <p className="text-[10px] text-white/35 leading-snug line-clamp-2">{sentence}</p>

        {glossList.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {glossList.map((g, i) => (
              <span
                key={g + i}
                className={
                  "px-1.5 py-px rounded text-[9px] font-semibold tracking-wide " +
                  (i === activeGloss
                    ? "bg-[#0e7c6b]/60 text-white"
                    : "bg-white/[0.05] text-white/40")
                }
              >
                {g}
              </span>
            ))}
          </div>
        )}

        <p aria-live="polite" className="text-[10px] text-[#5eead4]/60 min-h-[14px]">{status}</p>

        <div className="flex items-center gap-2">
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
            disabled={!queueRef.current.length || busyRef.current}
            className="px-2.5 py-1 rounded-md bg-white/[0.06] text-white/70 text-[11px] font-semibold disabled:opacity-30"
          >
            Replay
          </button>
          <label className="ml-auto flex items-center gap-1 text-[10px] text-white/35">
            Speed
            <select
              value={speed}
              onChange={(e) => changeSpeed(Number(e.target.value))}
              className="bg-white/[0.06] text-white/70 text-[10px] rounded px-1 py-0.5 border border-white/10"
            >
              <option value={0.5}>0.5x</option>
              <option value={1}>1x</option>
              <option value={1.5}>1.5x</option>
            </select>
          </label>
        </div>
      </div>
    </div>
  );
}
