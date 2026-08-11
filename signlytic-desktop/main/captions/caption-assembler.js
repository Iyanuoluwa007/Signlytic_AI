// Turns the Live Captions buffer into finalised sentences.
//
// The buffer is not a stream. It is a rolling window that Windows rewrites in
// place: text is appended, earlier words are corrected, punctuation changes
// after the fact ("yesterday. Thank you" became "yesterday, thank you" in
// testing), and once it is long enough the oldest lines scroll away. It also
// still holds whatever was said before we attached.
//
// So this class does three things:
//   1. On attach, swallows the existing buffer so old speech is not replayed.
//   2. Only releases a sentence once it is settled - either more text has
//      arrived after it, or it has stopped changing for a moment.
//   3. Matches on normalised text, so a late punctuation or casing fix does
//      not make an already-emitted sentence look new.
//
// Kept free of Electron and Node APIs so it can be tested directly.

const SENTENCE_END = /[.!?]+["')\]]*\s*/;

// Compare on words only: Live Captions revises punctuation and casing after
// the fact, and those revisions must not read as a new sentence.
function normalise(s) {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Split into sentences, keeping any trailing fragment separate. The fragment
// is whatever is still being spoken and is never emitted as-is.
function splitSentences(text) {
  const out = [];
  let rest = text.replace(/\s+/g, " ").trim();
  for (;;) {
    const m = rest.match(SENTENCE_END);
    if (!m) break;
    const end = m.index + m[0].length;
    const sentence = rest.slice(0, end).trim();
    if (sentence) out.push(sentence);
    rest = rest.slice(end);
  }
  return { sentences: out, fragment: rest.trim() };
}

class CaptionAssembler {
  // stableMs: how long a trailing sentence must stop changing before it is
  //   treated as finished. Live Captions keeps refining the newest sentence,
  //   so releasing it immediately produces half-corrected text.
  // minWords: drop noise like a stray "I." that the recogniser emits mid-phrase.
  constructor({ stableMs = 1200, minWords = 2 } = {}) {
    this.stableMs = stableMs;
    this.minWords = minWords;
    this.reset();
  }

  reset() {
    this.primed = false;
    this.lastEmittedNorm = null;
    this.pendingSentence = null;
    this.pendingSince = 0;
    this.lastBuffer = "";
    // Every sentence we have already seen, normalised. Seeded on attach with
    // whatever was already on screen so prior speech is never replayed.
    this.seen = new Set();
    this.seenOrder = [];
  }

  _remember(norm) {
    if (this.seen.has(norm)) return;
    this.seen.add(norm);
    this.seenOrder.push(norm);
    // Bound the history; the caption window itself only holds a few lines.
    if (this.seenOrder.length > 200) {
      this.seen.delete(this.seenOrder.shift());
    }
  }

  // Feed a raw buffer. Returns an array of newly finalised sentences.
  push(buffer, now = Date.now()) {
    if (typeof buffer !== "string" || !buffer.trim()) return [];
    if (buffer === this.lastBuffer) return this._checkPendingTimeout(now);
    this.lastBuffer = buffer;

    const { sentences, fragment } = splitSentences(buffer);

    // First buffer after attaching: treat everything already there as history.
    if (!this.primed) {
      this.primed = true;
      for (const s of sentences) this._remember(normalise(s));
      // A fragment in flight when we attached is mid-sentence; let it complete.
      return [];
    }

    if (!sentences.length) return [];

    const candidates = sentences;
    const out = [];

    // Anything with text after it is settled: Live Captions has moved on.
    const settled = fragment ? candidates : candidates.slice(0, -1);
    for (const s of settled) {
      if (this._accept(s)) out.push(s);
    }

    // The final sentence with nothing after it may still be revised, so hold
    // it until it stops changing.
    if (!fragment && candidates.length) {
      const tail = candidates[candidates.length - 1];
      if (this.pendingSentence !== tail) {
        this.pendingSentence = tail;
        this.pendingSince = now;
      }
    } else {
      this.pendingSentence = null;
    }

    return out.concat(this._checkPendingTimeout(now));
  }

  _checkPendingTimeout(now) {
    if (!this.pendingSentence) return [];
    if (now - this.pendingSince < this.stableMs) return [];
    const s = this.pendingSentence;
    this.pendingSentence = null;
    return this._accept(s) ? [s] : [];
  }

  _accept(sentence) {
    const norm = normalise(sentence);
    if (!norm) return false;
    if (norm.split(" ").length < this.minWords) return false;
    if (this.seen.has(norm)) return false;

    // Live Captions sometimes merges an already-released sentence into a
    // longer one when it revises punctuation, so a sentence that simply
    // extends something we have seen is a rewrite, not new speech.
    for (const prev of this.seen) {
      if (norm.startsWith(prev + " ")) {
        this._remember(norm);
        return false;
      }
    }

    this._remember(norm);
    this.lastEmittedNorm = norm;
    return true;
  }
}

module.exports = { CaptionAssembler, splitSentences, normalise };
