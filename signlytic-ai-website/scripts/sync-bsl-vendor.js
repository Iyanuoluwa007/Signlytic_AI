// Copies the shared BSL avatar engine from the browser extension into
// public/bsl. The extension's overlay folder is the single source of truth
// for avatar3d.js and the Three.js vendor files; never edit the copies in
// public/bsl directly. Run: npm run sync-bsl
const fs = require("fs");
const path = require("path");

const SRC = path.join(__dirname, "..", "..", "signlytic-extension", "overlay");
const DST = path.join(__dirname, "..", "public", "bsl");

const FILES = ["three.min.js", "GLTFLoader.js", "avatar3d.js"];

fs.mkdirSync(DST, { recursive: true });
for (const f of FILES) {
  fs.copyFileSync(path.join(SRC, f), path.join(DST, f));
  console.log("synced " + f);
}
