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

## Set up a development copy

You do not need an NVIDIA GPU, model files or any paid API key to work on the
interface, the APIs or most bugs. The full system needs a GPU only for video and
webcam recognition, speech transcription and the spoken voice.

```bash
git clone https://github.com/Iyanuoluwa007/Signlytic_AI.git
cd Signlytic_AI
```

### The Python server, in mock inference mode

Needs Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

SIGNLYTIC_MOCK_INFERENCE=1 python app_server.py
# Windows PowerShell: $env:SIGNLYTIC_MOCK_INFERENCE="1"; python app_server.py
```

Then open <http://localhost:8000>. A banner across the top of the dashboard
says you are in mock mode.

| Feature | In mock mode |
|---------|--------------|
| English text to BSL glosses | Real (rule-based converter) |
| Signing animation (2D and 3D) | Real signs, fetched from the public website |
| BSL glosses to English | Real (rule-based converter, no API key needed) |
| BSL video and webcam recognition | **Made up**: fixed glosses, marked `"mock": true` |
| Speech to BSL | **Made up**: a fixed sentence, then real glosses and signs |
| Spoken English voice | **Made up**: a short tone |

Mock results are for development only and are never translations. Do not
report them as accuracy, and do not use them in screenshots presented as real
output.

### The website

Needs Node.js 20 or newer.

```bash
cd signlytic-ai-website
npm install
```

Create `signlytic-ai-website/.env.local` so the site uses the live sign and
avatar data (both send CORS headers, so this works from localhost):

```
NEXT_PUBLIC_SIGNS_API_BASE=https://signlytic-ai-website.vercel.app
NEXT_PUBLIC_AVATAR_MALE_URL=https://signlytic-ai-website.vercel.app/api/avatar/male
```

Then `npm run dev` and open <http://localhost:3000>. Without a `GROQ_API_KEY`,
English is turned into glosses by the same rule-based converter the extension
uses, so the signing panel works with no key. Its glosses are plainer than the
hosted site's.

### The desktop app and the extension

- Desktop app: `cd signlytic-desktop`, `npm install`, `npm start`. See
  [signlytic-desktop/README.md](signlytic-desktop/README.md).
- Chrome extension: open `chrome://extensions`, turn on Developer Mode, choose
  Load Unpacked and select the `signlytic-extension` folder.

### Running the checks

These are the same checks CI runs on every pull request:

```bash
pytest tests/ -q                       # server tests, in mock mode, offline
python scripts/check_versions.py       # every version number agrees
cd signlytic-ai-website && npx tsc --noEmit   # website type check
```

New tests go in `tests/` (Python, named `test_*.py`) or next to the code they
test (JavaScript, named `*.test.js`, run with `node --test`). CI picks both up
without any change to the workflow. Tests must not need a GPU, a network
connection or an API key, and must use small fixtures you write yourself,
never sign pose data.

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
   looked at the change and allowed them. Then **CI** runs the tests, the
   version check, the website type check and a check that generated copies are
   up to date. Fix anything it reports.
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
- Edit the renderer only in `signlytic-extension/overlay/`, and the English to
  gloss converter only in `signlytic-extension/gloss/`. The copies under
  `signlytic-ai-website/public/bsl/`, `signlytic-ai-website/gloss/` and
  `signlytic-desktop/renderer/vendor/` are generated (`npm run sync-bsl`,
  `npm run sync-vendor`).

## Community

- Be kind. Everyone taking part agrees to the
  [Code of Conduct](CODE_OF_CONDUCT.md).
- Questions, ideas and show-and-tell go in
  [Discussions](https://github.com/Iyanuoluwa007/Signlytic_AI/discussions);
  bugs and planned work go in issues.
- See [ROADMAP.md](ROADMAP.md) for where the project is heading and
  [ACCESSIBILITY.md](ACCESSIBILITY.md) for the accessibility commitments every
  change has to keep.

## Licence

Signlytic AI is released under the MIT Licence. By contributing you agree that
your contribution is released under the same licence.
