# Sign data (Session 2/3 - not yet implemented)

Strategy agreed in Session 1: first-run download plus bundled core.

- Full dataset: 5,203 sign JSON files, about 1.45 GB raw (lives in
  extension-data/signs/ in this repo and in the private signs-data GitHub
  repo behind the Vercel proxy).
- Distribution: publish one compressed archive (estimated 150-350 MB) as a
  public GitHub Release asset. The private repo PAT must never ship with
  the app, so the archive needs a public home; licensing check pending.
- First run: download the archive once, extract to
  %LOCALAPPDATA%\Signlytic\signs\, verify a manifest (file count + hashes).
- Bundled in the installer: the 26 fingerspelling letters plus a small core
  set so the app signs immediately while the full set downloads.
- This folder holds only dev-time samples; nothing here ships.
