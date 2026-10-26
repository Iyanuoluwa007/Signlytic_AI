# Contributing to Signlytic AI

Thank you for helping. Signlytic AI translates between British Sign Language
(BSL) and English, and it is used by Deaf and hard of hearing people, so
accuracy and care matter more than speed.

## Where things live

- **This repository (Signlytic_AI)** holds all the source code: the Python
  system, the website, the Chrome extension and the desktop app.
- **[Signlytic-Overlay](https://github.com/Iyanuoluwa007/Signlytic-Overlay)**
  holds the released apps and their user documentation. Some issues are filed
  there, but their code changes come here: open your pull request in this
  repository and write `Fixes Iyanuoluwa007/Signlytic-Overlay#<number>` in it.

## Picking something to work on

1. Look at the open issues in both repositories. `good first issue` is a good
   place to start.
2. Comment on the issue to say you are taking it, so two people do not do the
   same work.
3. Every issue has a **Done when** list. Your pull request is ready when every
   item on it is true.

Issues labelled `needs BSL reviewer` change how signing looks. Those need
feedback from a fluent BSL user, ideally Deaf, before they can be merged. If you
are not a BSL user yourself, say so in the issue and ask for a reviewer.

## How a pull request gets merged

Nothing merges into `main` without the maintainer's approval. Every pull request
goes through these steps, in order:

1. **You check your own work, as a person.** The pull request template has a
   checklist. You must confirm that you personally read every changed line, ran
   it, and can explain it. If you used an AI tool, you must say so and confirm
   that you checked everything it produced. An automated check called
   **Human verification** reads the checklist and blocks the pull request until
   it is complete and honest.
2. **The maintainer approves the automated checks to run.** Checks on pull
   requests from outside contributors do not run until the maintainer has
   looked at the change and allowed them.
3. **The maintainer reviews and approves.** Every file is owned by the
   maintainer (see `.github/CODEOWNERS`), and an approving review from them is
   required. If you push new commits after approval, the approval is withdrawn
   and the change must be approved again.
4. **The maintainer merges.** Contributors cannot merge, and pull requests
   cannot merge themselves.

## Using AI tools

AI coding tools are allowed, on these conditions:

- **Say so.** Tick the AI box in the pull request template and describe briefly
  what the tool did.
- **You are responsible for every line.** Read it, run it, test it, and be able
  to explain it in review. Do not submit code you do not understand.
- **Check claims against reality.** AI tools invent APIs, file paths, test
  results and BSL facts. Verify each one.
- **BSL content needs a human BSL reviewer.** Gloss order, handshapes,
  fingerspelling and anything else about how BSL is signed must be checked by a
  fluent BSL user, never taken from an AI tool alone.

## Data you must never commit

- **Sign pose data.** The BSL motion data behind the signing is licensed for
  research only. Never commit or publish frames fetched from `/api/signs`,
  landmarks, features, recordings rendered from them, or anything else derived
  from them. Tests must use small synthetic fixtures you write by hand.
- **Dataset media** (videos, audio, images from BSL datasets).
- **Secrets.** API keys and tokens go in environment variables, never in files.
  If you commit one by mistake, tell the maintainer immediately.
- **Avatar models.** The 3D models are not redistributable; they are served at
  runtime and must not be added to the repository.

## Style

- Match the code around you: naming, comment density and idiom.
- Comments explain *why*, not what.
- House style: no em-dashes in code, comments, documentation or commit
  messages; use commas, parentheses or separate sentences.
- Commit messages explain why the change was made, in plain prose, and state
  what you measured or tested.
- Edit the renderer only in `signlytic-extension/overlay/`. The copies under
  `signlytic-ai-website/public/bsl/` and `signlytic-desktop/renderer/vendor/`
  are generated (`npm run sync-bsl`, `npm run sync-vendor`).

## Licence

Signlytic AI is released under the MIT Licence. By contributing you agree that
your contribution is released under the same licence.
