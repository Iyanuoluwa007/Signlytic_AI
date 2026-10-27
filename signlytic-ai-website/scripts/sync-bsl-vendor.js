// Copies the shared BSL avatar engine from the browser extension into
// public/bsl, and the rule-based English to gloss converter into gloss/.
// The extension is the single source of truth for both; never edit the copies
// here directly. Run: npm run sync-bsl
const fs = require("fs");
const path = require("path");

const SRC = path.join(__dirname, "..", "..", "signlytic-extension", "overlay");
const DST = path.join(__dirname, "..", "public", "bsl");

// avatar3d.js also carries PoseNormaliser, which skeleton2d.js depends on,
// so avatar3d.js must load first. three.min.js and GLTFLoader.js are only
// needed for 3D mode and are loaded on demand, not up front.
const FILES = ["three.min.js", "GLTFLoader.js", "avatar3d.js", "skeleton2d.js"];

fs.mkdirSync(DST, { recursive: true });
for (const f of FILES) {
  fs.copyFileSync(path.join(SRC, f), path.join(DST, f));
  console.log("synced " + f);
}

// The converter is imported by /api/english-to-glosses, as the fallback when
// no GROQ_API_KEY is set in development, so it is a module rather than a
// public file. Vercel builds only this folder, hence a copy, not an import.
const GLOSS_DST = path.join(__dirname, "..", "gloss");
fs.mkdirSync(GLOSS_DST, { recursive: true });
fs.copyFileSync(
  path.join(SRC, "..", "gloss", "converter.js"),
  path.join(GLOSS_DST, "converter.js")
);
console.log("synced gloss/converter.js");
