# Building and checking releases

How each released file is built, how its checksum is published, and how anyone
can check that a download matches this source code.

The released files live in
[Signlytic-Overlay](https://github.com/Iyanuoluwa007/Signlytic-Overlay/releases).
The source they are built from lives here.

| File | Built from | Rebuilds to the same checksum? |
|------|------------|--------------------------------|
| `signlytic-extension.zip` | `signlytic-extension/` | Yes, from the next release on. See below for v0.4.4 and earlier. |
| `Signlytic.AI.Setup.<version>.exe` | `signlytic-desktop/`, on Windows | No. The contents can be compared, the installer cannot. |
| `Signlytic.AI-<version>-universal.dmg` | `signlytic-desktop/`, on macOS | No, for the same reason. |

None of the builds is code-signed with a trusted certificate yet. A checksum
shows a file was not changed after it was published. It does not show who
built it.

## Checksums

Every release carries a `SHA256SUMS.txt`. Write it with:

```bash
python scripts/release_checksums.py path/to/file1 path/to/file2
```

It writes `SHA256SUMS.txt` beside the first file, in the format `sha256sum` and
`shasum -a 256` read. Names are listed the way GitHub serves them, with each
space turned into a dot, so they match what a user downloads. Upload it with
the other files:

```bash
gh release upload <tag> SHA256SUMS.txt --repo Iyanuoluwa007/Signlytic-Overlay --clobber
```

To check a folder of downloads against it:

```bash
python scripts/release_checksums.py --check SHA256SUMS.txt
```

The Overlay README tells users how to do the same check with the tools already
on Windows, macOS and Linux.

## The Chrome extension

Needs Python 3.8 or newer and git. Nothing to install.

```bash
git checkout <the release commit>
python scripts/build_extension_zip.py
```

This writes `dist/signlytic-extension.zip` and prints its SHA256. The zip holds
exactly the files git has under `signlytic-extension/` at that commit, except
the folder's README. Files are read from git itself, not from the working
folder, so uncommitted edits and Windows line-ending conversion cannot leak in.
Entries are sorted and carry a fixed date, so the same commit gives the same
bytes every time.

Tested on Windows 11 with Python 3.11, 3.12 and 3.13 (all zlib 1.3.1): the
three builds of one commit were byte for byte identical.

One thing can still differ between machines: the compression library. A Python
built against a different zlib can compress the same files into different
bytes. So the check that always works compares contents, not the zip's own
checksum:

```bash
python scripts/build_extension_zip.py --verify signlytic-extension.zip --ref <the release commit>
```

It hashes every file inside the zip and compares it with the same file at that
commit. It passes only if every file is identical and nothing is extra.

### v0.4.4 and earlier

These zips were made by hand from a Windows working folder, before the script
existed. They cannot be rebuilt to the same checksum, because they carry the
files' real modification times, and five text files in v0.4.4 have Windows line
endings where git stores Unix ones. Their contents can still be checked:

```bash
python scripts/build_extension_zip.py --verify signlytic-extension.zip --ref 0a34f47 --ignore-line-endings
```

Checked on 2026-11-10 against the published v0.4.4 zip (SHA256
`7b6f3e20...a5a295`): all 209 files match commit `0a34f47`, nothing is extra,
and the five files that differ do so only in line endings.

## The desktop app

Electron and electron-builder are pinned by `signlytic-desktop/package-lock.json`,
so install with `npm ci`, not `npm install`.

Windows, on Windows:

```bash
cd signlytic-desktop
npm ci
npm run dist
```

macOS has to be built on a Mac, because the caption helper is compiled there.
The full steps, and the traps on a shared drive, are in
[MACOS_NEXT_SESSION.md](MACOS_NEXT_SESSION.md) and
[signlytic-desktop/README.md](../signlytic-desktop/README.md#building-on-macos).

The installer and the dmg are **not** expected to rebuild to the same checksum.
Both formats record when they were built, and neither build has been set up or
tested for repeatability. What a rebuilder can compare today is the app code
inside them, which is the `signlytic-desktop/main` and `renderer` folders at
the release commit plus the files `npm run sync-vendor` copies in from the
extension.

Making the desktop builds repeatable, and signing them, is tracked on the
[roadmap](../ROADMAP.md).

## Before publishing

`python scripts/check_versions.py --remote` checks that every download link and
version number in both repositories agrees with what is released.
