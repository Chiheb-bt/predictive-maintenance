---
title: Predictive Maintenance API
emoji: 🔧
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.31.0
app_file: huggingface_spaces/app.py
pinned: false
license: mit
---

# Predictive Maintenance — Hugging Face Spaces

This Space runs the interactive Gradio demo from the
[predictive-maintenance](https://github.com/your-username/predictive-maintenance) project.

Enter real-time sensor readings (temperature, RPM, torque, tool wear) and get back:
- Failure probability and risk level (LOW / MEDIUM / HIGH / CRITICAL)
- Model certainty score
- Top 3 contributing sensor factors
- Sensor-aware maintenance recommendations

## Deploying to Spaces

1. Fork or clone this repo to your Hugging Face account.
2. Download the trained model from the GitHub Release and upload it as `model.pkl`:
   ```
   python scripts/download_model.py --output model.pkl
   ```
   Then drag `model.pkl` into the Space's file browser (or use `git lfs`).

3. The Space reads `MODEL_PATH` from the environment. Set it in the Space settings
   if you place `model.pkl` somewhere other than the repo root.

## Running locally

```bash
pip install -r requirements.txt
python src/models/train.py          # or: python scripts/download_model.py
python huggingface_spaces/app.py
```

The demo opens at http://localhost:7860.
