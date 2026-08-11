// Copies the shared renderer assets from the browser extension into
// renderer/vendor. The extension's overlay folder stays the single source
// of truth for the avatar engine and the Three.js vendor files; never edit
// the copies in renderer/vendor directly.
const fs = require("fs");
const path = require("path");

const EXT = path.join(__dirname, "..", "..", "signlytic-extension");
const DST = path.join(__dirname, "..", "renderer", "vendor");

// [source path relative to the extension root, destination file name]
const FILES = [
  ["overlay/three.min.js", "three.min.js"],
  ["overlay/GLTFLoader.js", "GLTFLoader.js"],
  ["overlay/avatar3d.js", "avatar3d.js"],
  ["overlay/skeleton2d.js", "skeleton2d.js"],
  ["gloss/converter.js", "converter.js"],
];

fs.mkdirSync(DST, { recursive: true });
for (const [src, name] of FILES) {
  fs.copyFileSync(path.join(EXT, src), path.join(DST, name));
  console.log("synced " + name);
}
