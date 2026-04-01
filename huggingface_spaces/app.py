"""
Hugging Face Spaces entry point.

Launches the Gradio UI on port 7860 (the Spaces default).
The FastAPI REST server is NOT started here — Spaces only serves the Gradio UI.

To run the full FastAPI server (e.g. locally or on Fly.io / Render):
    python run.py --workers 2

Environment variables:
    MODEL_PATH          — path to model.pkl (default: model.pkl in project root)
    DOWNLOAD_MODEL_URL  — if set and model.pkl is missing, downloads it automatically
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow imports from the project root regardless of where this script is launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Gradio needs a model on disk before it can serve predictions.
# Download one automatically if MODEL_PATH points to a missing file
# and DOWNLOAD_MODEL_URL is set (useful for automated Space deployments).
_model_path = Path(os.getenv("MODEL_PATH", "model.pkl"))
if not _model_path.exists():
    download_url = os.getenv("DOWNLOAD_MODEL_URL", "")
    if download_url:
        print(f"model.pkl not found — downloading from {download_url}")
        from urllib.request import urlretrieve
        urlretrieve(download_url, _model_path)
    else:
        print(
            "WARNING: model.pkl not found and DOWNLOAD_MODEL_URL is not set. "
            "The UI will start in degraded mode — predictions will fail. "
            "Upload model.pkl to the Space or set DOWNLOAD_MODEL_URL."
        )

# Load the model explicitly — the lifespan context manager only runs when the
# full FastAPI app starts. In Spaces, we call load_model() here directly so
# the Gradio UI has an active pipeline when the user clicks Predict.
from src.serving.inference import load_model  # noqa: E402
load_model()

from src.app.main import _build_gradio_app  # noqa: E402

demo = _build_gradio_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )
