"""
Download a pre-trained model.pkl from a GitHub Release or Hugging Face Hub.

Usage:
    # From a GitHub Release (default):
    python scripts/download_model.py

    # Explicit GitHub Release URL:
    python scripts/download_model.py --url https://github.com/your-user/predictive-maintenance/releases/download/v2.2.0/model.pkl

    # From Hugging Face Hub:
    python scripts/download_model.py --hf-repo your-user/predictive-maintenance --hf-filename model.pkl

    # Verify the downloaded file:
    python scripts/download_model.py --verify

This script is useful when you want to run the server without training locally —
for example in CI, in a Docker build, or when deploying to Fly.io / Render / Railway.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_REPO    = "your-username/predictive-maintenance"
DEFAULT_VERSION = "v2.2.0"
DEFAULT_OUTPUT  = "model.pkl"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_url(url: str, dest: Path) -> None:
    log.info("Downloading %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {exc}") from exc
    log.info("Saved %s  (%.1f MB)", dest, dest.stat().st_size / 1_048_576)


def _download_from_github(repo: str, version: str, output: Path) -> None:
    url = f"https://github.com/{repo}/releases/download/{version}/model.pkl"
    _download_url(url, output)


def _download_from_hf(repo: str, filename: str, output: Path) -> None:
    """
    Download from Hugging Face Hub without requiring the huggingface_hub package.
    Falls back to the direct CDN URL if the package isn't installed.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        path = hf_hub_download(repo_id=repo, filename=filename)
        shutil.copy(path, output)
        log.info("Copied from HF cache → %s", output)
    except ImportError:
        log.warning("huggingface_hub not installed — falling back to direct URL download")
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        _download_url(url, output)


def verify(path: Path) -> bool:
    """Smoke-test the downloaded file by loading it with joblib."""
    try:
        import joblib
        pipeline = joblib.load(path)
        _ = pipeline.named_steps["preprocessor"]
        _ = pipeline.named_steps["classifier"]
        log.info("Verification passed — pipeline loaded successfully from %s", path)
        return True
    except Exception as exc:
        log.error("Verification failed: %s", exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a pre-trained model.pkl from GitHub Releases or Hugging Face Hub"
    )
    parser.add_argument("--url",          help="Direct URL to model.pkl")
    parser.add_argument("--hf-repo",      help="Hugging Face repo id (e.g. your-user/predictive-maintenance)")
    parser.add_argument("--hf-filename",  default="model.pkl", help="Filename in the HF repo")
    parser.add_argument("--github-repo",  default=DEFAULT_REPO)
    parser.add_argument("--version",      default=DEFAULT_VERSION)
    parser.add_argument("--output",       default=DEFAULT_OUTPUT)
    parser.add_argument("--verify",       action="store_true", help="Load and smoke-test the downloaded file")
    parser.add_argument("--force",        action="store_true", help="Overwrite an existing file")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.force:
        log.info("%s already exists — skipping download (use --force to overwrite)", output)
    else:
        if args.url:
            _download_url(args.url, output)
        elif args.hf_repo:
            _download_from_hf(args.hf_repo, args.hf_filename, output)
        else:
            _download_from_github(args.github_repo, args.version, output)

    if args.verify:
        ok = verify(output)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
