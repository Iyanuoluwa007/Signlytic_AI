// gloss/converter.js — English → BSL Gloss Converter (Rule-Based)
// Exported as ES module for dynamic import() in overlay.js
//
// BSL gloss rules applied (in order):
//   1. Lowercase + punctuation strip
//   2. Number → word expansion
//   3. Contraction expansion  (I'm → I AM, don't → DO NOT, etc.)
//   4. Negation handling       (don't → NOT, isn't → IS NOT, etc.)
//   5. Tense markers           (-ed past → FINISH prefix, will/shall → WILL prefix)
//   6. -ing removal            (running → RUN)
//   7. Function word removal   (a, the, is, are, of, to, ...)
//   8. Lemmatisation           (-s, -es plurals, simple -ed)
//   9. Uppercase output

// ─── BSL function words (removed in gloss output) ────────────────────────────
const FUNCTION_WORDS = new Set([
  'a', 'an', 'the',
  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
  'have', 'has', 'had', 'do', 'does', 'did',
  'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
  'up', 'about', 'into', 'through', 'during', 'before', 'after',
  'above', 'below', 'between', 'each', 'more', 'most',
  'other', 'some', 'such', 'than', 'too', 'very',
  'just', 'that', 'this', 'these', 'those',
  'it', 'its', 'itself',
  'which', 'who', 'whom',
  'as', 'if', 'or', 'but', 'so', 'yet',
  'both', 'either', 'neither', 'whether',
  'also', 'even', 'still', 'already', 'always',
]);

// ─── Contractions table ───────────────────────────────────────────────────────
const CONTRACTIONS = {
  "i'm":      ["i", "am"],
  "i've":     ["i", "have"],
  "i'll":     ["i", "will"],
  "i'd":      ["i", "would"],
  "you're":   ["you", "are"],
  "you've":   ["you", "have"],
  "you'll":   ["you", "will"],
  "you'd":    ["you", "would"],
  "he's":     ["he", "is"],
  "he'll":    ["he", "will"],
  "he'd":     ["he", "would"],
  "she's":    ["she", "is"],
  "she'll":   ["she", "will"],
  "she'd":    ["she", "would"],
  "it's":     ["it", "is"],
  "it'll":    ["it", "will"],
  "we're":    ["we", "are"],
  "we've":    ["we", "have"],
  "we'll":    ["we", "will"],
  "we'd":     ["we", "would"],
  "they're":  ["they", "are"],
  "they've":  ["they", "have"],
  "they'll":  ["they", "will"],
  "they'd":   ["they", "would"],
  "don't":    ["not"],
  "doesn't":  ["not"],
  "didn't":   ["not"],
  "can't":    ["cannot"],
  "cannot":   ["cannot"],
  "couldn't": ["could", "not"],
  "won't":    ["will", "not"],
  "wouldn't": ["would", "not"],
  "shouldn't":["should", "not"],
  "isn't":    ["not"],
  "aren't":   ["not"],
  "wasn't":   ["not"],
  "weren't":  ["not"],
  "haven't":  ["not"],
  "hasn't":   ["not"],
  "hadn't":   ["not"],
  "that's":   ["that"],
  "what's":   ["what"],
  "there's":  ["there"],
  "let's":    ["let", "us"],
  "it'd":     ["it", "would"],
};

// ─── Numbers (0-20 + tens) ────────────────────────────────────────────────────
const NUM_WORDS = {
  '0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five',
  '6':'six','7':'seven','8':'eight','9':'nine','10':'ten',
  '11':'eleven','12':'twelve','13':'thirteen','14':'fourteen','15':'fifteen',
  '16':'sixteen','17':'seventeen','18':'eighteen','19':'nineteen','20':'twenty',
  '30':'thirty','40':'forty','50':'fifty','60':'sixty','70':'seventy',
  '80':'eighty','90':'ninety','100':'hundred',
};

function expandNumbers(token) {
  // Return array of replacement tokens
  if (/^\d+$/.test(token)) {
    const n = parseInt(token, 10);
    if (NUM_WORDS[token]) return [NUM_WORDS[token]];
    if (n > 0 && n < 100) {
      const tens = Math.floor(n / 10) * 10;
      const ones = n % 10;
      const parts = [];
      if (NUM_WORDS[String(tens)]) parts.push(NUM_WORDS[String(tens)]);
      if (ones > 0 && NUM_WORDS[String(ones)]) parts.push(NUM_WORDS[String(ones)]);
      return parts.length ? parts : [token];
    }
    // Large numbers: fingerspell digit by digit
    return token.split('').map(d => NUM_WORDS[d] || d);
  }
  return [token];
}

// ─── Tense detection ──────────────────────────────────────────────────────────
// Returns { hasPast, hasFuture } for the token array
function detectTense(tokens) {
  const futureWords = new Set(['will', 'shall', 'going', 'would', 'could', 'might', 'may']);
  const pastWords   = new Set(['was', 'were', 'had', 'did']);
  let hasFuture = tokens.some(t => futureWords.has(t));
  let hasPast   = false;
  tokens.forEach(t => {
    if (pastWords.has(t)) hasPast = true;
    if (/ed$/.test(t) && t.length > 3) hasPast = true; // walked, played, etc.
  });
  return { hasPast, hasFuture };
}

// ─── Lemmatiser ───────────────────────────────────────────────────────────────
function lemmatise(word) {
  // Remove common inflectional suffixes
  if (word.length <= 3) return word;

  // -ing → stem (running→run, playing→play)
  if (word.endsWith('ing') && word.length > 5) {
    const stem = word.slice(0, -3);
    // double-consonant: running → run
    if (stem.length >= 3 && stem[stem.length-1] === stem[stem.length-2]) {
      return stem.slice(0, -1);
    }
    // e was dropped: making → make
    if (/[aeiou]/.test(stem[stem.length-2])) return stem + 'e';
    return stem;
  }

  // -ed → stem (walked→walk, played→play, loved→love)
  if (word.endsWith('ed') && word.length > 4) {
    const stem = word.slice(0, -2);
    if (stem.length >= 3) {
      // loved → love (don't strip the e)
      if (stem.endsWith('e')) return stem;
      return stem;
    }
  }

  // -es plural (watches→watch, boxes→box)
  if (word.endsWith('es') && word.length > 4) {
    const stem = word.slice(0, -2);
    if (/[sxz]$/.test(stem) || /[cs]h$/.test(stem)) return stem;
  }

  // -s plural (dogs→dog, cats→cat) — be conservative
  if (word.endsWith('s') && !word.endsWith('ss') && word.length > 3) {
    const stem = word.slice(0, -1);
    if (!/[aeiou]$/.test(stem)) return stem; // only strip if stem ends consonant
  }

  return word;
}

// ─── Main export ──────────────────────────────────────────────────────────────
/**
 * Convert an English sentence to an array of BSL gloss tokens.
 * @param {string} text - Input English text
 * @returns {string[]} - Array of uppercase BSL gloss tokens
 */
export function textToGloss(text) {
  if (!text || typeof text !== 'string') return [];

  // 1. Lowercase, strip punctuation (keep apostrophes for contractions)
  let clean = text
    .toLowerCase()
    .replace(/[^a-z0-9'\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  // 2. Tokenise
  let tokens = clean.split(' ').filter(Boolean);

  // 3. Expand contractions
  const expanded = [];
  tokens.forEach(t => {
    if (CONTRACTIONS[t]) {
      expanded.push(...CONTRACTIONS[t]);
    } else {
      expanded.push(t);
    }
  });
  tokens = expanded;

  // 4. Expand numbers
  const numExpanded = [];
  tokens.forEach(t => numExpanded.push(...expandNumbers(t)));
  tokens = numExpanded;

  // 5. Detect tense before removing markers
  const { hasPast, hasFuture } = detectTense(tokens);

  // 6. Remove apostrophes now (for clean words after contraction handling)
  tokens = tokens.map(t => t.replace(/'/g, ''));

  // 7. Lemmatise (strip -ing, -ed, -s)
  tokens = tokens.map(lemmatise);

  // 8. Remove function words
  tokens = tokens.filter(t => !FUNCTION_WORDS.has(t) && t.length > 0);

  // 9. Add tense markers (BSL convention: FINISH for past, WILL for future)
  const glosses = [];
  if (hasPast)   glosses.push('finish');
  if (hasFuture) glosses.push('will');
  glosses.push(...tokens);

  // 10. Deduplicate consecutive identical glosses
  const deduped = [];
  glosses.forEach((g, i) => {
    if (i === 0 || g !== glosses[i - 1]) deduped.push(g);
  });

  // 11. Uppercase
  return deduped.map(g => g.toUpperCase()).filter(Boolean);
}

// ─── Named export for testing ─────────────────────────────────────────────────
export { FUNCTION_WORDS, CONTRACTIONS, lemmatise };
