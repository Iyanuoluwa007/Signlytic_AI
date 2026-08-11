// Copies the shared BSL avatar engine from the browser extension into
// public/bsl. The extension's overlay folder is the single source of truth
// for avatar3d.js and the Three.js vendor files; never edit the copies in
// public/bsl directly. Run: npm run sync-bsl
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
