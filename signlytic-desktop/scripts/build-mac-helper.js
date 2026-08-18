// Compiles the macOS caption helper.
//
// The Windows sidecar deliberately needs no build step: Windows ships the UI
// Automation assemblies and PowerShell can load them. macOS has no equivalent,
// so this is the one place the port cannot avoid a compiler. It uses the Swift
// toolchain in the Command Line Tools, which is a much smaller ask than a full
// Xcode install, and it was verified that Speech, ScreenCaptureKit and
// AVFoundation all build with the Command Line Tools SDK alone.
//
// Produces a universal binary so one dmg runs on Apple Silicon and Intel.
//
//   node scripts/build-mac-helper.js
//
// Runs automatically before `npm run dist` on macOS, and is a no-op elsewhere
// so a Windows build is never blocked by it.

const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const MAC_DIR = path.join(__dirname, "..", "main", "captions", "mac");
const SOURCE = path.join(MAC_DIR, "caption-source.swift");
const PLIST = path.join(MAC_DIR, "Info.plist");
const OUTPUT = path.join(MAC_DIR, "signlytic-captions");
const ARCHES = ["arm64", "x86_64"];

function run(cmd, args) {
  return execFileSync(cmd, args, { stdio: ["ignore", "pipe", "pipe"] }).toString();
}

function main() {
  if (process.platform !== "darwin") {
    console.log("not macOS, nothing to build");
    return;
  }

  try {
    run("xcrun", ["--find", "swiftc"]);
  } catch {
    console.error("Swift is not available. Install the Command Line Tools with:");
    console.error("  xcode-select --install");
    process.exit(1);
  }

  const slices = [];
  for (const arch of ARCHES) {
    const slice = path.join(MAC_DIR, `.slice-${arch}`);
    console.log(`compiling ${arch}`);
    run("xcrun", [
      "swiftc", "-O",
      "-target", `${arch}-apple-macos13.0`,
      // Without this the binary has no Info.plist, macOS finds no usage
      // description, and TCC kills the process instead of prompting. See the
      // comment in Info.plist; this was observed rather than guessed.
      "-Xlinker", "-sectcreate",
      "-Xlinker", "__TEXT",
      "-Xlinker", "__info_plist",
      "-Xlinker", PLIST,
      SOURCE,
      "-o", slice,
    ]);
    slices.push(slice);
  }

  console.log("joining into a universal binary");
  run("xcrun", ["lipo", "-create", ...slices, "-output", OUTPUT]);
  for (const slice of slices) fs.unlinkSync(slice);

  // Ad-hoc signing gives the helper a stable code identity. Without one, macOS
  // treats each rebuild as a different program and the permissions the user
  // granted have to be granted again. A real Developer ID replaces this at
  // release time; electron-builder re-signs everything inside the bundle then.
  console.log("ad-hoc signing");
  run("codesign", ["--force", "--sign", "-", "--timestamp=none", OUTPUT]);

  fs.chmodSync(OUTPUT, 0o755);
  console.log(`built ${path.relative(process.cwd(), OUTPUT)}`);
  console.log(run("lipo", ["-archs", OUTPUT]).trim());
}

main();
