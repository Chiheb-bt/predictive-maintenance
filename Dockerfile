# Multi-stage build: builder installs deps, final image copies only what's needed.
# Runs as a non-root user for security.
# Deps are installed via pyproject.toml — no separate requirements.txt needed.

FROM python:3.11-slim AS builder

WORKDIR /build

# Copy the full package definition so pip can resolve all deps
COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .


FROM python:3.11-slim AS final

WORKDIR /app

# Non-root user
RUN useradd --create-home appuser
USER appuser

# Deps from builder stage
COPY --from=builder /install /usr/local

# Application source
COPY --chown=appuser:appuser src/      ./src/
COPY --chown=appuser:appuser run.py    ./run.py

# Model artefact (must exist before building — run training first)
COPY --chown=appuser:appuser model.pkl       ./model.pkl
COPY --chown=appuser:appuser artifacts/      ./artifacts/

ENV PORT=8000
ENV MODEL_PATH=/app/model.pkl

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["python", "run.py", "--workers", "2"]
