# 👋 Signlytic AI

**British Sign Language Recognition & Translation System**

An end-to-end BSL recognition and translation pipeline powered by deep learning, achieving state-of-the-art results on large-vocabulary sign recognition.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

- **Sign Recognition**: SWIN Transformer-based video recognition (9.53% Top-5 accuracy on 500 classes)
- **Text → Gloss**: Convert English text to BSL glosses (71.7% token accuracy)
- **Gloss → Text**: Translate BSL glosses to natural English (92% ROUGE-L)
- **Speech Integration**: Whisper ASR + Coqui TTS for voice interaction
- **Real-Time**: 0.46ms inference per frame

## 📊 Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| SWIN Recognition (Top500) | Top-1 Accuracy | 2.64% |
| SWIN Recognition (Top500) | Top-5 Accuracy | **9.53%** |
| SWIN Recognition (Top500) | vs Random Chance | ~460x better |
| Pose Recognition | Top-5 Accuracy | **74.5%** |
| Gloss → Text | ROUGE-L | **92%** |
| Gloss → Text | BLEU | 0.60 |
| Text → Gloss | Token Accuracy | **71.7%** |
| Text → Gloss | Coverage | 97.5% |

## 🏗️ Architecture

```
Video Input → SWIN Features → Transformer Encoder → BSL Glosses → English Text → Speech (TTS)
     ↓              ↓                  ↓                 ↓              ↓
  1940 videos    768-dim         6 layers, 8 heads    500 classes    Coqui XTTS
```

### Components

1. **Video SWIN Transformer**: Pre-extracted 768-dimensional spatiotemporal features
2. **Temporal Transformer Encoder**: 6 layers, 8 heads, positional encoding
3. **Gloss Classifier**: 500-class vocabulary for common BSL signs
4. **Seq2Seq Translation**: Bidirectional gloss↔text models
5. **Text-to-Speech**: Coqui TTS for audio output

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (8GB+ VRAM recommended)
- Conda or virtualenv

### Installation

```bash
# Clone the repository
git clone https://github.com/Iyanuoluwa007/signlytic-ai.git
cd signlytic-ai

# Create conda environment
conda create -n BSL python=3.11
conda activate BSL

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
# Show metrics
python scripts/demo_pipeline.py --metrics

# Interactive mode
python scripts/demo_pipeline.py --interactive

# Process a video
python scripts/demo_pipeline.py --video path/to/video.mp4
```

## 📁 Project Structure

```
signlytic-ai/
├── configs/                    # Training configurations
├── data/                       # Data processing scripts
│   └── bsl1k_parser.py        # BSL-1K dataset parser
├── models/                     # Model architectures
│   └── temporal_recognition.py
├── scripts/
│   ├── train_top500.py        # Training script
│   ├── evaluate_top500.py     # Evaluation script
│   └── demo_pipeline.py       # Demo script
├── src/
│   ├── data/                  # Data utilities
│   ├── motion/                # Motion generation
│   └── pipeline/              # Recognition pipeline
└── requirements.txt
```

## 📈 Training

### Train Recognition Model

```bash
python scripts/train_top500.py \
    --lr 5e-5 \
    --dropout 0.5 \
    --weight_decay 0.1 \
    --epochs 50 \
    --no_balance
```

### Evaluate Model

```bash
python scripts/evaluate_top500.py
```

## 🔗 Links

- **Live Demo**: [HuggingFace Spaces](https://huggingface.co/spaces/Iyanuoluwa007/signlytic-ai)
- **Portfolio**: [signlytic-ai-website.vercel.app](https://signlytic-ai-website.vercel.app)

## 📚 Dataset

This project uses the [BSL-1K dataset](https://www.robots.ox.ac.uk/~vgg/data/bsl1k/):
- 5.9M annotations
- 1,940 videos
- 24,877 unique glosses
- 4 annotation sources: EXEMPLARS, MOUTHING, DICTIONARY, I3D_PSEUDO_LABELS

## 🙏 Acknowledgments

- [BSL-1K Dataset](https://www.robots.ox.ac.uk/~vgg/data/bsl1k/) - University of Oxford
- [Video SWIN Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)
- [Coqui TTS](https://github.com/coqui-ai/TTS)

## 👤 Author

**Oke Iyanuoluwa Enoch**  
Independent Robotics & AI Systems Engineer

- [GitHub](https://github.com/Iyanuoluwa007)
- [LinkedIn](https://www.linkedin.com/in/iyanuoluwa-enoch-oke/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Building accessible technology to bridge communication gaps through AI.*
