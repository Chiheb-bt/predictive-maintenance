"""
Generate a synthetic dataset for CI pipelines.

The real AI4I 2020 dataset is not committed to the repository. This script
produces a structurally equivalent CSV so CI can train and smoke-test the
full pipeline without requiring external data.

The real dataset has a ~3.4% failure rate, which is too sparse for small
synthetic samples — every CV fold needs at least a few positive examples.
This generator uses ~15%, which is enough to train and evaluate correctly
on 500–1000 rows while preserving the relative sensor-to-failure relationships.

Usage:
    python scripts/generate_ci_data.py
    python scripts/generate_ci_data.py --rows 1000 --output Data/ci.csv --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    air_temp     = rng.normal(300.0, 2.0, n_rows).clip(295, 305)
    process_temp = (air_temp + rng.normal(10.0, 1.0, n_rows)).clip(305, 315)
    rpm          = rng.integers(1168, 2886, n_rows).astype(float)
    torque       = rng.normal(40.0, 10.0, n_rows).clip(3.8, 76.6)
    tool_wear    = rng.integers(0, 253, n_rows).astype(float)
    machine_type = rng.choice(["L", "M", "H"], n_rows, p=[0.5, 0.3, 0.2])

    # Failure probability is driven by the same sensors as in the real dataset.
    # Scale_pos_weight is ~5.9 (85% healthy, 15% failure) for CI usability.
    failure_score = (
        (tool_wear / 253.0) * 0.40
        + (np.abs(torque - 40.0) / 36.0) * 0.30
        + rng.random(n_rows) * 0.30
    )
    failure = (failure_score > 0.65).astype(int)

    df = pd.DataFrame({
        "UDI":                      range(1, n_rows + 1),
        "Product ID":               [f"CI{i:05d}" for i in range(n_rows)],
        "Type":                     machine_type,
        "Air temperature [K]":      air_temp.round(1),
        "Process temperature [K]":  process_temp.round(1),
        "Rotational speed [rpm]":   rpm,
        "Torque [Nm]":              torque.round(1),
        "Tool wear [min]":          tool_wear,
        "Machine failure":          failure,
        "TWF":                      0,
        "HDF":                      0,
        "PWF":                      0,
        "OSF":                      0,
        "RNF":                      0,
    })

    failure_rate = failure.mean()
    print(f"Generated {n_rows} rows — failure rate: {failure_rate:.1%}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CI training data")
    parser.add_argument("--rows",   type=int, default=500, help="Number of rows to generate")
    parser.add_argument("--output", default="Data/maintenance_ci.csv")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    df  = generate(n_rows=args.rows, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved → {out.resolve()}")


if __name__ == "__main__":
    main()
