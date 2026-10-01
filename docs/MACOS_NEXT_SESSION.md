# macOS: what to do next

Read this first, then `MACOS_PORT_HANDOVER.md` for how the port works and
`MACOS_PORT_PROMPT.md` for the original brief. This file is only about the work
that is still outstanding, in the order it is worth doing.

You are on a Mac. Everything here needs a Mac, which is why it is not done.

**State as of 1 October 2026.** `main` is pushed, so `git pull` gets you
everything, including both 3D avatar fixes. Windows is released at 0.3.9; the
Mac is still on 0.3.7 and pinned there. `python scripts/check_versions.py
--remote` exits 0 on what is pushed, so if it fails after your changes, it is
your changes.

The Windows machine also has some uncommitted Mac un-pin edits that were written
for 0.3.8. They were never pushed, so you will not see them, and they are out of
date. Do the un-pinning fresh from the steps below.

---

## Context in one paragraph

Signlytic AI turns speech into British Sign Language and shows it as a signing
avatar. There are four pieces: a local Python system, a website on Vercel, a
Chrome extension, and an Electron desktop app. The desktop app is the part that
matters here. It reads what the computer is saying and signs it: on Windows by
reading the Live Captions window, on macOS by recognising speech itself, because
macOS has no equivalent window to read. Everything after the text arrives is
shared between both platforms.

---

## Task 1: rebuild and release the Mac app at 0.3.9

**This is the priority.** The Windows build is already released at 0.3.9. The
Mac download on the website still points at 0.3.7 and is deliberately pinned
there, because a Mac build cannot be produced off a Mac. There is no Mac 0.3.8
and there will not be one: build 0.3.9, which carries both fixes below.

### Why the rebuild is needed

The Mac is two fixes behind, both in the 3D avatar and both in the one file.

**0.3.8 fixed where the hands go.** The avatar was placing them near the hips
and barely moving them regardless of the sign. Three faults, all in the 3D
renderer, none in the sign data:

- the forearm was positioned against the wrong reference orientation and came
  out 72 degrees off whenever the upper arm moved
- arms were aimed along captured directions rather than solved to a position, so
  the hand could only reach as far as the avatar's own arm allowed
- two different definitions of where the torso ends were being mixed

It also fixes the female avatar, which had never been posed in any release: her
rig uses a different bone-name prefix and not one bone was being mapped, so she
loaded, reported herself ready, and stood still through every sign.

Measured against the 2D renderer over 818,904 solves from 500 signs, median
error went from 0.434 to 0.003 of torso length.

**0.3.9 fixed the hands themselves.** With the arms right, hands still came out
mangled: a hand collapsing into a twisted strand, open hands clawed with the
fingers crossed. The hand bone was never turned, so the knuckles absorbed the
difference between the palm and the captured fingers, up to folding them back
through the hand; each finger bone could roll freely, wringing the mesh; there
were no joint limits; and the hand landmarks were read as square when the source
video is 4:3. The palm is now matched to the capture, fingers bend within human
limits without rolling, and the forearm takes half the palm's turn so the wrist
does not wring. Over 150 signs on both avatars, the palm facing the wrong way
went from a median 89 degrees to 0, fingers folded past 150 degrees from 3.3% to
none, and finger direction on screen against the 2D from 9 degrees to 3.

Both fixes are in `signlytic-extension/overlay/avatar3d.js`, which is the source
of truth, and is copied into the desktop app by `sync-vendor` at build time.
**You do not need to touch it. Just build.**

### Build

    git checkout main
    git pull
    cd signlytic-desktop
    npm install
    node node_modules/electron/install.js     # npm 11 skips install scripts

`npm run dist` is the normal build, but it writes its output next to the source,
and gotcha 2 in the handover says not to build onto the exFAT drive. Run the
same three steps by hand instead, sending the output elsewhere:

    npm run sync-vendor
    node scripts/build-mac-helper.js
    npx electron-builder --mac dmg --universal --config.directories.output=~/signlytic-build

**Do not shorten that to the `electron-builder` line on its own.** The two steps
above it are exactly what `npm run dist` exists to run first.
`renderer/vendor/` is gitignored, so a fresh clone contains no `avatar3d.js` at
all: it is copied in from `signlytic-extension/overlay/` by `sync-vendor` at
build time, and `build-mac-helper.js` compiles the Swift caption helper. Skip
them and the dmg ships with no avatar renderer and no captions, which is the
same class of silent trap as gotcha 4.

Confirm the built app actually contains the fix rather than assuming
`sync-vendor` ran. From inside the app bundle:

    npx asar extract "Signlytic AI.app/Contents/Resources/app.asar" /tmp/asar-check
    grep -o "detectBonePrefix\|_solveArm\|_captureHandRest\|_shareTwist" \
      /tmp/asar-check/renderer/vendor/avatar3d.js | sort -u

All four markers must be present: the first two are 0.3.8, the last two 0.3.9.
Better still, `cmp` the extracted file against
`signlytic-extension/overlay/avatar3d.js`; on Windows they were byte-identical. Gotcha 4 in the handover is exactly this trap: the
renderer loads from inside `app.asar`, so copying files into
`app.asar.unpacked` changes nothing and silently runs the old code.

### Check it before releasing

Launch it and sign something. "Do you take the bus to work?" is a good test:
WORK and BUS were among the worst signs before. The three things to look at,
because they are what changed:

1. Switch to the 3D avatar. The hands should move through the signing space in
   front of the chest, not hang near the hips. The desktop app still offers 3D
   the way it always did. The website was separately put back on 2D by default
   in `245c081` pending a visual sign-off, and that hold is **website only**,
   a single flag in `signlytic-ai-website/components/BslSignPanel.tsx`. Do not
   copy it into the desktop app or the extension.
2. Look at the hands. Open hands should be flat with the fingers in order, not
   clawed or crossed; curved hands (BUS) should curve the way fingers bend. No
   finger should fold back through the hand and no wrist should twist like a
   sweet wrapper.
3. Switch the avatar to female. She should move at all. If she is still, the
   prefix detection has not reached this build.

### Release

Attach the dmg to the existing `desktop-v0.3.9` release in the
**Signlytic-Overlay** repo, which already holds the Windows installer:

    gh release upload desktop-v0.3.9 "path/to/Signlytic AI-0.3.9-universal.dmg" \
      --repo Iyanuoluwa007/Signlytic-Overlay

GitHub turns the space into a dot, so the asset is served as
`Signlytic.AI-0.3.9-universal.dmg`, which is the name every link must use.
Attaching an asset does not move the Latest badge, so leave that alone. The
release is currently titled and described as Windows only: edit its title and
notes to cover both platforms, and keep the attribution section.

Then un-pin the Mac download. In `scripts/check_versions.py` there is a `PINNED`
entry holding the Mac URL at 0.3.7 with a reason. Remove it, update the Mac URLs
to 0.3.9 in these files, and the checker will confirm you got them all:

- `signlytic-ai-website/app/page.tsx`
- `signlytic-ai-website/app/extension/page.tsx`
- `README.md`
- `signlytic-desktop/README.md`
- in the Signlytic-Overlay repo: `README.md` and `Software App/README.md`

Then:

    python scripts/check_versions.py --remote

It must exit 0. It parses every release URL, checks asset filenames as well as
tags, and confirms each asset actually resolves.

The checker does not read prose that says the platforms differ, so fix those by
hand once both are on 0.3.9: the paragraph starting "The two platforms are not
on the same version" in `README.md`, the "Both free. Windows is on 0.3.9" line
in `signlytic-ai-website/app/extension/page.tsx`, the same paragraph in the
Overlay `README.md`, and the **Status** line in Overlay `Software App/README.md`.
Pushing `main` deploys the website, so push only after the dmg is uploaded and
the checker is green.

---

## Task 2: finish the Live Captions source, or retire it

There is a third macOS caption source on the branch
`macos-live-captions-unverified`. It reads the system Live Captions window over
the Accessibility API, which is the closer analogue to what Windows does. It is
on its own branch because half of it has never been seen working.

**Proven:** the reading approach. `mac/tools/ax-probe.swift` pulled real text out
of that window, growing as speech continued. The failure path is proven too:
without Accessibility permission the helper exits 7, the app shows the helper's
own explanation rather than a bare code, and it does not sit in a restart loop.

**Not proven:** the success path inside the packaged app. Accessibility
permission was never granted before that session ended, so `LiveCaptionsSource`
has never been observed delivering a sentence through to the avatar.

To finish it:

1. System Settings, Accessibility, switch on Live Captions
2. Grant Accessibility permission to Signlytic AI
3. In the app, choose Live Captions under "Listen to"
4. Press Start Captions

If sentences appear, merge the branch. If they do not, the fault is inside
`LiveCaptionsSource` in `caption-source.swift`, not in the approach, which the
probe already validated. If it turns out to be more work than it is worth,
deleting the branch is a legitimate outcome: microphone and system audio both
work and are what the shipped app defaults to.

Note that the 0.3.7 dmg currently on the release page was built **including**
this unverified source. A fresh build from `main` will not contain it, which is
the correct state for a release.

---

## Task 3: signing and notarisation, if you want it

Neither build is code-signed. On Windows that means a SmartScreen warning. On
macOS it is worse: the app will refuse to open normally and users have to
right-click and choose Open. This is the single biggest barrier to anyone
actually trying the Mac app.

This needs a paid Apple Developer account, so it is a spending decision rather
than a technical one. `packaging/entitlements.mac.plist` already exists and the
helper is ad-hoc signed, so the groundwork is there.

---

## House rules for this repository

These are not stylistic preferences, they are requirements:

- **No em-dashes** (the character U+2014) anywhere: code, comments, docs, commit
  messages. Use commas, parentheses, or separate sentences.
- **No emoji or icons** in code files.
- **No "Co-Authored-By" or any AI attribution** in commit messages.
- **Never `git add -A`.** Add explicit paths. The exFAT drive litters `._`
  AppleDouble files and they must not be committed.
- Commits are authored as `Oke Iyanuoluwa E. <oke.iyanuoluwa12@gmail.com>`.
- **Do not force-push.** The repository is public and its history has already
  been rewritten once.
- Commit messages explain **why**, in prose, and state what was measured. Look
  at `git log` for the register.

---

## Things already established, so you do not re-derive them

- **Do not re-extract the sign capture data.** An earlier session concluded the
  hands-too-low problem was bad capture data and proposed re-extraction. That
  was wrong. The 2D renderer plots the same landmarks and looks correct, which
  proves the data is adequate. The faults were all in the 3D path and are fixed.
- **The browser pane blocks WebGL when hidden**, so browser-based measurement of
  the avatar stalls at `initScene`. Measure headless instead: the vendored
  `three.min.js` is a UMD build and `require()`s straight into Node as real
  THREE r128. Probes that rebuild the rig from the GLB's JSON chunk and drive
  the real renderer methods were written into `scratchpad/`, which is **not in
  the repository** and stayed on the Windows machine, so a fresh clone will not
  have them. Nothing in Task 1 needs them: both fixes are already in the source
  of truth and measured. They only matter if the avatar has to be retuned, and
  they are quicker to rewrite than to chase.
- **Not every avatar file on disk is the one that ships.** The live male has the
  skeleton of `data/avatars/Male.glb` (prefix `mixamorig`); the live female has
  that of `extension-data/avatars/Female.glb` (`mixamorig8`). The
  `extension-data/avatars/Male.glb` copy (`mixamorig9`) is not shipped. If you
  measure anything, fetch the live ones from `/api/avatar/male` and
  `/api/avatar/female`.
- **Fetch with `curl`, not Python `urllib`,** when checking anything served by
  Vercel or behind Cloudflare. `urllib` gets a stripped response and makes
  working things look broken.
- **`app.asar` is readable by Electron but not by external processes.** Anything
  a spawned helper must read has to be in `asarUnpack`.

---

## Where things are

    signlytic-desktop/                  the Electron app
      main/captions/caption-stream.js   picks a caption source per platform
      main/captions/mac/                the Swift helper and its probe tools
      renderer/vendor/                  generated by sync-vendor, do not edit
    signlytic-extension/overlay/        source of truth for both renderers
      avatar3d.js                       3D avatar, the file 0.3.8 and 0.3.9 fix
      skeleton2d.js                     2D skeleton
    scripts/check_versions.py           gates a release, run with --remote
    docs/MACOS_PORT_HANDOVER.md         how the port works, and its gotchas

Releases come from the **Signlytic-Overlay** repo, not the main one. The desktop
app is tagged `desktop-vX.Y.Z` and the extension `vX.Y.Z`. The newest tag always
holds the Latest badge, whichever product it belongs to.
