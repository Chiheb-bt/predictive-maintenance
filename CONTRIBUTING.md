# Contributing

Contributions are welcome — bug fixes, documentation improvements, and new features alike.

## Setup

```bash
git clone https://github.com/Chiheb-bt/predictive-maintenance
cd predictive-maintenance
make install-dev     # installs deps + pre-commit hooks
```

## Running tests

Unit tests run without a trained model:

```bash
make test-unit
```

Full CI pipeline (train on synthetic data, then test):

```bash
make test-ci
```

## Code style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting. After installing dev dependencies, pre-commit runs it automatically on every commit. To run it manually:

```bash
make lint         # check
make lint-fix     # auto-fix where possible
```

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes, keeping commits atomic and messages clear.
3. Add or update tests for any changed behaviour.
4. Open a pull request. Describe what you changed and why.

## Reporting bugs

Open a GitHub Issue with:
- Python version and OS
- Full error traceback
- Minimal reproduction steps
