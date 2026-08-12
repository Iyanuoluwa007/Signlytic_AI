# Signlytic AI

**Bidirectional British Sign Language Translation System**

Signlytic AI translates between British Sign Language (BSL) and English using deep learning, large language models, and multi-modal output. The system supports two directions of translation: BSL video to English text and speech, and English speech or text to BSL glosses with animated signing output.

**Website:** [signlytic-ai-website.vercel.app](https://signlytic-ai-website.vercel.app)
&nbsp;|&nbsp;
**Interactive Demo:** [signlytic-ai-website.vercel.app/demo](https://signlytic-ai-website.vercel.app/demo)

---

## System Overview

| Direction | Input | Output |
|-----------|-------|--------|
| BSL to English | BSL video / BSL glosses | English text + speech (Coqui XTTS v2) |
| English to BSL | English speech / text | BSL glosses + signing animation (2D Pose Animator) |

---

## Key Results

| Model | Language | Data | Top-1 | Top-5 |
|-------|----------|------|-------|-------|
| **BSL Dict (Retrieval)** | **British** | **5,203 videos** | **100%** | **100%** |
| BSL-100 | British | 50K/epoch | 72.34% | 95.03% |
| BSL-500 | British | 50K/epoch | 59.26% | 89.04% |
| Pose Recognition | ASL | 1,000 samples | 44.44% | 81.62% |
| Multi-Lingual | ASL+LSF | 1,710 samples | 20.95% | 49.17% |
| SWIN Recognition | ASL | 1,000 samples | 2.64% | 9.53% |

---

## Architecture

```
Direction 1: BSL to English
BSL Video --> Video-SWIN-T --> 768-dim features --> Cosine Retrieval (5,203 signs)
         --> BSL Glosses --> Groq Llama 3.3 70B --> English Text
         --> Coqui XTTS v2 --> Speech Output

Direction 2: English to BSL
English Speech --> OpenAI Whisper (base) --> English Text
             --> Text-to-Gloss Engine --> BSL Glosses
             --> 2D Pose Animator --> Signing Animation (MP4)
```

---

## Technology Stack

| Component | Technology | Details |
|-----------|------------|---------|
| Sign Recognition | Video-SWIN-T | Retrieval on 5,203 pre-extracted 768-dim features |
| Speech Recognition | OpenAI Whisper | Base model, 16 kHz mono |
| Text-to-Speech | Coqui XTTS v2 | Voice cloning with speaker reference |
| Language Model | Groq Llama 3.3 70B | Gloss-to-English (llama-3.3-70b-versatile) |
| Local LLM Fallback | FLAN-T5 Base | Offline text-to-gloss conversion |
| Signing Animation | 2D Pose Animator | Skeleton signing with MP4 export |
| Vocabulary | 11,573+ glosses | BSL-1K + BSLDict datasets |
| Frontend | Gradio | 1,425 lines, 27 functions, 4 tabs |
| Website | Next.js / Vercel | Interactive demo with native React |

---

## Recognition Pipeline

The system uses retrieval-based recognition rather than direct classification. Each of the 5,203 BSL dictionary videos is processed through Video-SWIN-T to produce a 768-dimensional feature vector. At inference time, a query video is compared against all stored features using cosine similarity, returning the top-k most similar signs with confidence scores.

This approach achieves 100% Top-1 accuracy on dictionary signs because cosine similarity finds the exact match in the feature space. For novel user-recorded videos, similarity scores will be lower but the system remains functional due to the semantic consistency of Video-SWIN-T features.

---

## Getting Started

### Requirements

- Python 3.11
- NVIDIA GPU with CUDA support (RTX 4060 or equivalent)
- Conda package manager

### Installation

```bash
git clone https://github.com/Iyanuoluwa007/Signlytic_AI.git
cd Signlytic_AI/code/bsl_translation_project

conda create -n BSL python=3.11
conda activate BSL

pip install -r requirements.txt
```

### Environment Variables

```
GROQ_API_KEY=<your-groq-api-key>
```

### Run the Application

```bash
conda activate BSL
python app.py
# Open http://127.0.0.1:7860
```

---

## Application Interface

The Gradio application provides four tabs:

**BSL to English** — Upload a BSL video, record via webcam, or type BSL glosses. The system produces English text with optional speech output. Video recognition and live camera translation require local GPU.

**English to BSL** — Record audio or type English text. The system produces BSL glosses with word coverage metrics and animated signing preview. Speech input requires Whisper (GPU). Pose animation requires local GPU with 2D Pose Animator.

**Help & Accessibility** — Usage guides for BSL users and hearing users. All outputs include visible text. Nothing relies solely on audio. The interface supports keyboard navigation.

**About & System** — Architecture details, model comparison, and performance metrics.

---

## Data and Models

| Resource | Location |
|----------|----------|
| BSL Dict Videos | `data/videos/bsl_signs/` (5,203 MP4 files) |
| Pre-extracted Features | `data/bsl_dict_features/` (5,203 .npy files) |
| Feature Index | `data/bsl_dict_features/index.json` |
| Retrieval Model | `models/bsl_dict_recognition/retrieval_model.pt` (16 MB) |
| BOBSL SWIN Features | `data/processed/features/bobsl/v1.4/` |
| BSLDict Vocabulary | `data/bsldict/bsldict/bsldict_v1.pkl` |

---

## Project Structure

```
bsl_translation_project/
  app.py                          # Gradio application (1,425 lines)
  src/
    inference/
      bsl_dict_recognizer.py      # SWIN-based video recognizer
      speech_to_bsl.py            # Whisper ASR, TextToGloss, GlossToText, CoquiTTS
      pose_sign_renderer.py       # 2D skeleton signing animation
  scripts/
    extract_bsl_dict_features.py  # SWIN feature extraction
    evaluate_with_subtitles.py    # Evaluation with video output
    train_bsl_fast.py             # BSL training (subsampling + cache)
    train_multilingual.py         # ASL + French LSF training
  data/
    bsl_dict_features/            # Pre-extracted 768-dim features
    videos/bsl_signs/             # 5,203 BSL dictionary videos
  models/
    bsl_dict_recognition/         # Retrieval model
    bsl_recognition/              # Classification models
    pose_recognition/             # Pose-based models
  signlytic-ai-website/           # Next.js website source
    app/
      page.tsx                    # Landing page
      demo/page.tsx               # Interactive demo page
      layout.tsx                  # Root layout with analytics
      globals.css                 # Global styles
```

---

## Accessibility

- All outputs presented as visible text
- Keyboard navigation with ARIA landmarks
- High-contrast colours (navy #1E3A5F, teal #0E7C6B on white #F4F5F7)
- Plain-English labels and BSL word order notes
- Focus states with 2px teal outlines
- Skip navigation link
- Accessibility statement and help documentation in-app

---

## Datasets

- **BSLDict** — 5,203 isolated BSL sign videos with gloss labels
- **BSL-1K** — ~1,000 BSL sign classes with automatic annotations from broadcast subtitles (BOBSL)
- **WLASL** — ASL dataset used for cross-language experiments
- **INCLUDE** — French LSF dataset used for multi-lingual experiments

---

## Limitations

- Performs isolated sign recognition; continuous signing from natural sequences is not yet supported
- 2D Pose Animator produces simplified skeleton animations without facial expressions and non-manual markers
- Requires GPU hardware for video recognition, speech processing, and animation
- Rule-based text-to-gloss covers ~80 words; complex sentences need the Groq API
- 100% accuracy is on dictionary-source videos; novel user videos will have lower similarity scores

---

## Planned Improvements

- Continuous sign language recognition from natural signing sequences
- 3D character animation as an alternative to 2D skeleton rendering
- Live camera translation for real-time BSL recognition
- Chrome extension for screen caption to BSL signing overlay
- User testing with BSL communities
- Expanded rule-based vocabulary

---

## Feedback

If you explore the project and have suggestions or comments, feedback is welcome:

**Google Form:** [https://forms.gle/oTy7Bi414fuThFc1A](https://forms.gle/oTy7Bi414fuThFc1A)

Feedback from researchers, engineers, and BSL community members is especially appreciated.

---

## Author

**Oke Iyanuoluwa Enoch**
Independent Robotics & AI Systems Engineer

**LinkedIn:** [linkedin.com/in/iyanuoluwa-enoch-oke](https://www.linkedin.com/in/iyanuoluwa-enoch-oke/)
**Website:** [signlytic-ai-website.vercel.app](https://signlytic-ai-website.vercel.app)

---

## Third-party assets and licensing

The MIT licence below covers the code in this repository. It does **not**
cover third-party assets used while running or developing the project, which
remain under their own terms.

### 3D avatar models

The 3D avatar models are Mixamo characters and are **not licensed for
redistribution by this project**. They are used here for rendering and
research only, and are not offered for download, resale, or redistribution as
standalone assets. Anyone running this project should obtain their own models
from Mixamo under an Adobe account and accept Adobe's terms directly:
https://www.mixamo.com

If a model file appears anywhere in this repository or its history, that is
unintentional and not an offer to distribute it. Please open an issue and it
will be removed.

### Other third-party material

The same applies to any other third-party asset that may be present without
having been excluded: sign language video or pose data derived from
third-party corpora, pretrained model weights, fonts, icons, and audio. Each
remains the property of its owner and is subject to that owner's licence.
Their presence here is incidental to development and is not a grant of any
right to redistribute them.

Nothing in this repository is intended to distribute third-party assets. If
you believe something has been included that should not have been, please
open an issue and it will be taken down.

---

## License

MIT covers the source code in this repository. Third-party assets are
excluded, as described above.
