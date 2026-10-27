# Roadmap

Where Signlytic AI is heading, and how the open issues fit together. Nothing
here has a date: it is a small project, and the order changes as people pick
work up. If something here matters to you, comment on its issue or start a
[discussion](https://github.com/Iyanuoluwa007/Signlytic_AI/discussions).

Issues numbered like `#4` are in this repository. Issues marked `Overlay #4`
are in [Signlytic-Overlay](https://github.com/Iyanuoluwa007/Signlytic-Overlay/issues),
where the released apps live; their code changes still come here.

## Where it is now

- **English to BSL** works in the browser, the Chrome extension and the
  desktop app (Windows and macOS), with a 2D skeleton and a 3D avatar. The 3D
  avatar is the default on the website.
- **BSL to English** works from typed glosses anywhere, and from video and
  webcam on a local GPU install. Video recognition matches isolated signs
  against a dictionary of 5,203 sign videos; it does not yet follow continuous
  signing.
- Signing has **not yet been reviewed by Deaf BSL users** in a structured way.
  Until it has, treat it as a research prototype, not an interpreter.

## Now: make it trustworthy

The most important work is checking that what Signlytic signs is right, and
making that checkable by anyone.

- Build an English to gloss evaluation set with BSL reviewers (#12)
- Build a held-out signer evaluation for sign recognition (#7)
- Fingerspell unknown words with the BSL two-handed manual alphabet (#2)
- Report which everyday words have no sign (#10)
- Improve keyboard and screen-reader accessibility (#8)
- Review the extension's permissions and document its privacy behaviour
  (Overlay #3)
- Publish checksums and reproducible build steps for releases (Overlay #5)

## Next: make it better to watch and easier to use

- Make the 3D avatar signing look realistic (#3)
- Improve transitions between signs, with BSL reviewer input (Overlay #4)
- Make the website signing panel work on phones (#11)
- Improve caption detection across supported websites (Overlay #1)
- Test macOS captions end to end, then finish or retire the macOS Live
  Captions source (Overlay #6, Overlay #7)

## Contributor experience

- Automated tests for gloss to English conversion (#5) and caption sentence
  assembly (Overlay #2). CI already runs on every pull request and picks up new
  test files automatically.
- Fix and validate the quick-start installation instructions (#4)

## Phase 2: mobile apps

Android and iOS apps, starting with English to BSL by text and speech, and
adding camera recognition only once the research above shows it is ready.
The plan, milestones and tasks are in #13; no app code has started.

## Later

These are bigger than one issue and are not started:

- Recognising continuous signing, not just isolated signs
- Facial expressions and other non-manual features, which carry grammar in BSL
- Code-signing and notarising the desktop builds, so neither operating system
  warns on install
- A Chrome Web Store listing for the extension

## What will not change

- Sign pose data stays licensed for research only. It is never committed,
  published or sent anywhere except to play a sign.
- Nothing merges without the maintainer's review and a human check of any
  AI-assisted work (see [CONTRIBUTING.md](CONTRIBUTING.md)).
- Changes to how BSL is signed need a fluent BSL user's review.
