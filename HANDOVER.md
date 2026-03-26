# Signlytic AI - BSL Translation Project Handover

**Date:** March 26, 2026
**Project:** Signlytic_AI - British Sign Language Translation System
**Author:** Oke Iyanuoluwa Enoch
**Attribution:** Independent Robotics & AI Systems Engineer

---

## PENDING TASKS (FOR NEW CHAT)

### 1. Update HuggingFace Space
URL: https://huggingface.co/spaces/Iyanuoluwa007/signlytic-ai
- Update app.py with SWIN recognition
- Add src/inference/bsl_dict_recognizer.py
- Update requirements if needed

### 2. Update Vercel Website  
URL: https://signlytic-ai-website.vercel.app
- Highlight 100% accuracy on BSL recognition
- 5,203 signs supported
- Add demo video/screenshots

### 3. Update Technical Report
- Add BSL Dictionary Recognition results
- Compile on Overleaf -> PDF -> Zenodo DOI

---

## Quick Reference

| Resource | Location |
|----------|----------|
| Main Code | D:\Signlytic_AI\code\bsl_translation_project\ |
| Conda Env | BSL (Python 3.11) |
| GitHub | https://github.com/Iyanuoluwa007/Signlytic_AI.git |
| HuggingFace | https://huggingface.co/spaces/Iyanuoluwa007/signlytic-ai |
| Vercel | https://signlytic-ai-website.vercel.app |

## Key Results

| Model | Accuracy |
|-------|----------|
| BSL Dict (Retrieval) | **100% Top-1** on 5,203 signs |

## Run Commands

conda activate BSL
python app.py          # Local
python app.py --share  # Public link

## Key Files

- src/inference/bsl_dict_recognizer.py (SWIN recognizer)
- models/bsl_dict_recognition/retrieval_model.pt
- data/bsl_dict_features/ (5,203 .npy files)
- evaluation_results/bsl_evaluation_video_h264.mp4

## Latest Commit
fbc8081 - Professional UI redesign + BSL SWIN recognition integration
