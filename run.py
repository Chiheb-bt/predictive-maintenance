"""
Entry point for running the Predictive Maintenance API server.

Reads MODEL_THRESHOLD from artifacts/meta.json (written by the training
script) and exports it as an environment variable before starting uvicorn.
This ensures the server always uses the threshold that was optimised at
training time rather than the fallback 0.5.

Logging is configured here — before uvicorn starts — using structlog so
that all output (startup, requests, inference) is structured JSON in
production and readable colour output in a local terminal.

Usage:
    python run.py                     # default: 1 worker, no reload
    python run.py --reload            # development: hot-reload on file changes
    python run.py --workers 2         # production: 2 uvicorn workers
    python run.py --port 9000         # run on a custom port
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import structlog
import uvicorn

# ---------------------------------------------------------------------------
# Structured logging — pretty in a local terminal, JSON in production.
# structlog wraps the stdlib logging so all existing log.info() calls
# automatically emit structured output without any code changes.
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    is_tty = sys.stderr.isatty()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_tty:
        # Human-friendly coloured output for local development
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        # Machine-parseable JSON for production / Docker / CI
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


_configure_logging()
log = structlog.get_logger(__name__)


def _load_threshold_from_meta(meta_path: Path = Path("artifacts/meta.json")) -> float | None:
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        threshold = meta.get("threshold")
        if isinstance(threshold, (int, float)):
            return float(threshold)
        log.warning("threshold_missing_or_invalid", path=str(meta_path))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("meta_json_read_error", path=str(meta_path), error=str(exc))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Predictive Maintenance API")
    parser.add_argument("--host",    default="0.0.0.0", help="Bind address")
    parser.add_argument("--port",    type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")
    parser.add_argument("--reload",  action="store_true", help="Hot-reload (dev only)")
    args = parser.parse_args()

    if args.reload and args.workers > 1:
        log.warning("reload_workers_conflict", message="--reload is incompatible with multiple workers; ignoring --workers.")
        args.workers = 1

    threshold = _load_threshold_from_meta()
    if threshold is not None:
        os.environ.setdefault("MODEL_THRESHOLD", str(threshold))
        log.info("threshold_loaded", value=threshold, source="artifacts/meta.json")
    else:
        log.info("threshold_fallback", message="No meta.json found — using MODEL_THRESHOLD env var or fallback 0.5")

    uvicorn.run(
        "src.app.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
