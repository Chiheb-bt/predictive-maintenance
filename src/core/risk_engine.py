"""
Risk classification, recommendations, and explainability.

This module contains the "Expert System" rules that translate raw ML probabilities
into actionable business logic. It is intentionally decoupled from sklearn/pandas
to ensure it can be used in low-latency environments or specialized edge deployments
without heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def classify_risk(probability: float) -> RiskLevel:
    """
    Map a failure probability to an operational risk tier.

    Thresholds are chosen to align with common industrial maintenance schedules:
    next-scheduled-window (MEDIUM), expedited within 48h (HIGH), immediate (CRITICAL).

    Args:
        probability: Calibrated failure probability in [0, 1].

    Returns:
        One of "LOW", "MEDIUM", "HIGH", or "CRITICAL".
    """
    if probability < 0.25:
        return "LOW"
    if probability < 0.50:
        return "MEDIUM"
    if probability < 0.75:
        return "HIGH"
    return "CRITICAL"


_BASE_RECOMMENDATIONS: dict[RiskLevel, list[str]] = {
    "LOW": [
        "No action required — continue normal operations.",
        "Log readings and re-evaluate at the next scheduled inspection.",
        "Confirm tool wear tracking is current.",
    ],
    "MEDIUM": [
        "Flag for inspection at the next maintenance window.",
        "Watch the air-to-process temperature differential for upward drift.",
        "Check lubrication and coolant levels against spec.",
        "Verify rotational speed against the reference baseline.",
    ],
    "HIGH": [
        "Schedule an expedited inspection within 24–48 hours.",
        "Reduce machine load by 15–20% as a precautionary measure.",
        "Inspect tooling — replace if within 20% of the wear limit.",
        "Review recent vibration and acoustic logs for anomalies.",
        "Notify the maintenance supervisor.",
    ],
    "CRITICAL": [
        "IMMEDIATE ACTION: Halt this machine now to prevent catastrophic failure.",
        "Dispatch the on-call maintenance engineer for an emergency teardown.",
        "Verify emergency stop (E-stop) functionality and isolate power if necessary.",
        "Document current sensor values for a post-mortem root cause analysis (RCA).",
        "Inspect high-wear components (bearings, spindles) for thermal damage.",
        "Do not restart until a Safety & Quality sign-off is completed.",
    ],
}

# When a specific sensor is the top driver of the prediction, prepend a
# sensor-specific action so operators know exactly where to look first.
_FACTOR_SPECIFIC: dict[str, str] = {
    "Tool Wear":          "Tool wear is the primary driver — inspect and replace tooling before the next run.",
    "Torque":             "Torque is the primary driver — check for mechanical resistance or overload conditions.",
    "Rotational Speed":   "Rotational speed is the primary driver — verify the speed controller and belt tension.",
    "Air Temperature":    "Air temperature is the primary driver — inspect cooling and ambient airflow.",
    "Process Temperature": "Process temperature is the primary driver — check coolant flow and the heat exchanger.",
}


def get_recommendations(risk_level: RiskLevel, top_factor: str | None = None) -> list[str]:
    """
    Return an ordered list of maintenance actions for the given risk level.

    When top_factor matches a known sensor label, a targeted first action is
    prepended so operators immediately know where to focus.
    """
    base = list(_BASE_RECOMMENDATIONS[risk_level])
    if top_factor and top_factor in _FACTOR_SPECIFIC:
        return [_FACTOR_SPECIFIC[top_factor]] + base
    return base


@dataclass
class FeatureFactor:
    feature: str
    label: str
    importance: float
    rank: int


def top_factors(
    feature_importances: list[float],
    feature_names: list[str],
    feature_labels: dict[str, str],
    top_n: int = 3,
) -> list[FeatureFactor]:
    """
    Convert raw feature importances into ranked, human-readable factors.

    Args:
        feature_importances: Raw importance scores aligned with feature_names.
        feature_names: Column names from the preprocessor's get_feature_names_out().
        feature_labels: Mapping from pipeline-prefixed name to display label.
        top_n: How many top factors to return.

    Returns:
        List of FeatureFactor, sorted by descending importance.
    """
    total = sum(feature_importances) or 1.0
    ranked = sorted(
        zip(feature_names, feature_importances, strict=False),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [
        FeatureFactor(
            feature=name,
            label=feature_labels.get(name, name),
            importance=round(imp / total, 4),
            rank=rank,
        )
        for rank, (name, imp) in enumerate(ranked[:top_n], start=1)
    ]
