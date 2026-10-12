# macOS: what to do next

Read this first, then `MACOS_PORT_HANDOVER.md` for how the port works and
`MACOS_PORT_PROMPT.md` for the original brief. This file is only about the work
that is still outstanding, in the order it is worth doing.

You are on a Mac. Everything here needs a Mac, which is why it is not done.

**State as of 12 October 2026.** `main` is pushed. Both platforms are released
at 0.3.11 from the same `desktop-v0.3.11` tag; the Mac build was made on the
Mac on 12 October, ad-hoc signed, and is no longer pinned in
`scripts/check_versions.py`. `python scripts/check_versions.py --remote` exits
0 on what is pushed, so if it fails after your changes, it is your changes.
Task 1 is done until the renderer changes again; Tasks 2 and 3 are still
open.

The Mac has no GitHub access. The last round worked like this, and should again:
the Mac builds, verifies and checks the dmg, leaves it in
`signlytic-desktop/dist/` on the shared drive with its SHA256, and the Windows
machine uploads it and does the link and checker changes.

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

## Task 1: rebuild and release the Mac app at 0.3.11 (DONE)

**Done on 12 October 2026.** The Mac is released at 0.3.11 alongside Windows
and un-pinned. There was no Mac 0.3.10. Keep this section as the recipe for the
next time the Windows build moves ahead: re-pin the Mac in
`scripts/check_versions.py` with the reason, then follow the steps below.

### What the Mac build carries

Four rounds of 3D avatar fixes, all in the one file. The Mac already has the
first two in 0.3.9; 0.3.10 and 0.3.11 add the other two.

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

**0.3.10 made the fingers move like a person's.** The capture's finger tracking
shakes: a fifth of its joint movements reverse direction from one frame to the
next. A hand with no tracking data, which is half of all frames, froze in its
last shape, often a claw. Middle and end finger joints bent backwards, curled
fingers fanned apart, and resting hands sank into the avatar's waist. Each
finger joint is now smoothed with a speed-adaptive filter, a lost hand eases
into a relaxed pose with the thumb alongside the index, fingers draw together
as they curl, and the torso's shape is measured from the mesh at load so hands
are kept outside it. Shaking halved, backward bends went from 20% of joints to
none, and hands inside the body from 12% of frames to about 1%.

**0.3.11 put each hand on the right arm.** The capture runs a body tracker and
a separate hand tracker, and the hand tracker often labels left and right the
wrong way round (NO and YES in every frame), so a hand's fingers drove the other
arm and the arms stayed where the body tracker put them. The shared pose
clean-up now re-labels such hands, judged by which elbow they sit one forearm
from, and moves each wrist onto the hand tracker's. Wrist and hand disagreeing
went from 14.6% of hand-frames to 2.2%.

All four are in `signlytic-extension/overlay/avatar3d.js`, which is the source
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
    grep -o "detectBonePrefix\|_solveArm\|_captureHandRest\|_shareTwist\|_relaxHand\|_measureTorso\|_handsSwapped" \
      /tmp/asar-check/renderer/vendor/avatar3d.js | sort -u

All seven markers must be present: the first two are 0.3.8, the next two 0.3.9,
the next two 0.3.10, and the last 0.3.11.
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
   the way it always did. The website opens on 3D by default too since 0.3.11
   (`THREE_D_APPROVED_AS_DEFAULT` in
   `signlytic-ai-website/components/BslSignPanel.tsx`).
2. Look at the hands. Open hands should be flat with the fingers in order, not
   clawed or crossed; curved hands (BUS) should curve the way fingers bend. No
   finger should fold back through the hand and no wrist should twist like a
   sweet wrapper.
3. Switch the avatar to female. She should move at all. If she is still, the
   prefix detection has not reached this build.
4. Watch the hands between signs and at rest: they should hang relaxed with the
   fingers softly curled, not frozen in a claw, and a hand at the waist should
   rest on the hip, not disappear into the body.

### Release

On the Mac: leave the dmg in `signlytic-desktop/dist/` on the shared drive and
record its SHA256 and size. On Windows:

1. Check the SHA256 with `Get-FileHash` before anything else; the file crossed
   the exFAT drive.
2. Attach it to the existing `desktop-v0.3.11` release in the
   **Signlytic-Overlay** repo, which already holds the Windows installer:

       gh release upload desktop-v0.3.11 "signlytic-desktop/dist/Signlytic AI-0.3.11-universal.dmg" --repo Iyanuoluwa007/Signlytic-Overlay

   GitHub turns the space into a dot, so it is served as
   `Signlytic.AI-0.3.11-universal.dmg`, which is the name every link must use.
   Attaching an asset does not move the Latest badge; leave it alone. Edit the
   release's title and notes to cover both platforms, keeping the Attribution
   section exactly as it is.
3. Un-pin the Mac. In `scripts/check_versions.py`, replace the whole `PINNED`
   entry with `PINNED = {}`, **and** delete the trailing
   `("desktop", "universal.dmg")` from the two anchors that name it, so they end
   at their label. The checker fails with "anchor names pin ... which is no
   longer in PINNED" if you skip the second part.
4. Update the Mac links to 0.3.11 in `signlytic-ai-website/app/page.tsx`,
   `signlytic-ai-website/app/extension/page.tsx`, `README.md`,
   `signlytic-desktop/README.md`, and in the Signlytic-Overlay repo `README.md`
   and `Software App/README.md`.
5. Fix the prose the checker does not read, since both platforms will be on
   0.3.11: the paragraph starting "The two platforms are not on the same
   version" in `README.md` and in the Overlay `README.md`, the "Both free.
   Windows is on 0.3.11" line in `signlytic-ai-website/app/extension/page.tsx`,
   and the **Status** line in Overlay `Software App/README.md`.
6. Run `python scripts/check_versions.py --remote` until it exits 0, without
   loosening it. Push the Overlay repo first and `main` last, because pushing
   `main` deploys the website.

**The Mac install instruction is Open Anyway, not right-click.** Confirmed on
macOS 26 with a quarantined copy of the 0.3.9 dmg: right-click and Open no
longer gets past Gatekeeper (Apple removed that bypass in macOS 15). What works
is System Settings, Privacy and Security, Security, Open Anyway, then
authenticate. Every user-facing instruction was reworded for 0.3.10; keep it
that way.

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

Neither build is signed with a trusted certificate. On Windows that means a
SmartScreen warning. On macOS it is worse: the Mac build is ad-hoc signed but
not notarised, so macOS refuses to open it the first time and users have to
find Open Anyway in System Settings, Privacy and Security (right-click and Open
no longer works on macOS 15 and later). This is the single biggest barrier to
anyone actually trying the Mac app.

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
      avatar3d.js                       3D avatar, the file 0.3.8 to 0.3.11 fix
      skeleton2d.js                     2D skeleton
    scripts/check_versions.py           gates a release, run with --remote
    docs/MACOS_PORT_HANDOVER.md         how the port works, and its gotchas

Releases come from the **Signlytic-Overlay** repo, not the main one. The desktop
app is tagged `desktop-vX.Y.Z` and the extension `vX.Y.Z`. The newest tag always
holds the Latest badge, whichever product it belongs to.
