// Turns a sentence into a playable sign queue.
//
// Entirely platform-neutral: this is the same on Windows and macOS. Only the
// caption *source* is OS-specific, not what happens to the text afterwards.
//
// English -> BSL glosses -> pose frames per gloss -> queue for the renderer.
// Glosses with no pose data become a fingerspell placeholder (a run of null
// frames) rather than being dropped, so a sentence still plays through when
// part of its vocabulary is missing.

(function () {
  const API_BASE = "https://signlytic-ai-website.vercel.app";
  const FINGERSPELL_HOLD = 20; // frames, about 0.8s at 25fps

  const frameCache = new Map();  // GLOSS -> frames | null
  const glossCache = new Map();  // sentence -> gloss string

  // Prefer the hosted converter: it uses an LLM and gets BSL word order right
  // (time markers first, negation after the verb). Fall back to the local
  // dictionary from the extension so the app still works offline.
  async function toGlosses(text) {
    const key = text.trim().toLowerCase();
    if (glossCache.has(key)) return glossCache.get(key);

    let glosses = "";
    try {
      const res = await fetch(API_BASE + "/api/english-to-glosses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data && data.glosses) glosses = data.glosses;
      }
    } catch {
      // offline or the API is unreachable; fall through to the local converter
    }

    if (!glosses && typeof window.signlyticTextToGloss === "function") {
      try { glosses = window.signlyticTextToGloss(text) || ""; } catch { /* ignore */ }
    }
    if (!glosses) {
      // Last resort: strip punctuation and upper-case, so at least the words
      // are fingerspelled rather than nothing happening at all.
      glosses = text.toUpperCase().replace(/[^A-Z0-9'\s]/g, " ").replace(/\s+/g, " ").trim();
    }

    glossCache.set(key, glosses);
    return glosses;
  }

  async function loadFrames(gloss) {
    const key = gloss.toUpperCase();
    if (frameCache.has(key)) return frameCache.get(key);
    let frames = null;
    try {
      const res = await fetch(API_BASE + "/api/signs/" + encodeURIComponent(key));
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length) frames = data;
      }
    } catch {
      // leave frames null so the caller fingerspells
    }
    frameCache.set(key, frames);
    return frames;
  }

  // Returns { glosses: string[], queue: [{ gloss, frames }] }
  async function buildQueue(text) {
    const glossStr = await toGlosses(text);
    const glosses = glossStr.split(/\s+/).filter(Boolean);
    const queue = await Promise.all(
      glosses.map(async (gloss) => {
        const frames = await loadFrames(gloss);
        return {
          gloss,
          frames: frames && frames.length ? frames : Array(FINGERSPELL_HOLD).fill(null),
        };
      })
    );
    return { glosses, queue };
  }

  window.signlyticSigns = { toGlosses, loadFrames, buildQueue };
})();
