"""Read-only, GT-blind Long-text Gate V4 shadow observer.

Does not execute OCR, change production routing, or use
ground truth. Candidate thresholds are NOT validated.
"""
from __future__ import annotations

import math
from PIL import Image

from .routing import image_route
from .router_v3_features import FEATURE_NAMES, image_features

VERSION = "longtext_gate_v4_shadow_v0_unvalidated"

MODES = {
    "auto", "baseline", "motion", "artistic", "longtext"
}


def observe(
    image: Image.Image,
    *,
    routing=None,
    mode="auto",
    review_fraction=0.80,
):
    """Compare current STRSystem route with a shadow-only candidate."""

    if mode not in MODES:
        raise ValueError(f"Unknown STRSystem mode: {mode}")

    if not (0.0 < review_fraction < 1.0):
        raise ValueError(
            "review_fraction must be between 0 and 1"
        )

    routing = dict(routing or {})
    w, h = image.size

    if w <= 0 or h <= 0:
        raise ValueError("Invalid image dimensions")

    # Reproduce the existing router without changing its decision.
    if mode == "auto":
        current, router_meta = image_route(
            image, **routing
        )
    else:
        current = mode
        router_meta = {
            "routing_reason": "explicit_mode"
        }

    # Reuse the exact existing V3 feature extractor.
    features = image_features(image)

    if len(features) != len(FEATURE_NAMES):
        raise RuntimeError("V3 feature schema mismatch")

    feature_map = {
        str(k): float(v)
        for k, v in zip(FEATURE_NAMES, features)
    }

    if not all(
        math.isfinite(v)
        for v in feature_map.values()
    ):
        raise RuntimeError("Nonfinite image feature")

    ratio = w / h
    long_min = float(
        routing.get("long_ratio_min", 5.0)
    )

    if not math.isfinite(long_min) or long_min <= 0:
        raise ValueError("Invalid long_ratio_min")

    # Exploratory boundary zone ONLY.
    # This value is not trained or approved.
    lower = review_fraction * long_min

    hypothetical = current
    reason = "no_proposal"

    if mode != "auto":
        reason = "explicit_mode_protected"

    elif current == "longtext":
        reason = "already_longtext_by_existing_router"

    elif lower <= ratio < long_min:
        hypothetical = "longtext"
        reason = (
            "near_horizontal_geometry_boundary_review_only"
        )

    return {
        "policy_version": VERSION,
        "shadow_enabled": True,
        "dispatch_enabled": False,
        "mode": mode,
        "production_route": current,
        "production_route_after_shadow": current,
        "hypothetical_route": hypothetical,
        "route_disagreement": hypothetical != current,
        "protected_route_conflict": (
            hypothetical != current
            and current in {
                "motion", "artistic", "baseline"
            }
        ),
        "reason": reason,
        "aspect": ratio,
        "review_lower_aspect": lower,
        "existing_long_ratio_min": long_min,
        "existing_router_metadata": router_meta,
        "image_features_v3": feature_map,
    }
