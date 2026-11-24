"""Unified V3.2 candidate: FUSE-first, then Motion Gate."""

import hashlib
from pathlib import Path

import joblib
import numpy as np

from .unified_v31 import UnifiedSVTRv31
from .router_motion_features import (
    FEATURE_NAMES,
    motion_features,
)

FUSE_SHA = (
    "77da20ebc07fd7445d62722607b461"
    "f2c7c674fd69a4edf51efc6d822cf42582"
)

MOTION_SHA = (
    "5861578812ecc82812f6f8610b25376"
    "fb1a3a7128dfe769d97afa2777e866b7b"
)


def verified_file(path, expected_sha):
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    if digest != expected_sha:
        raise RuntimeError(
            f"Frozen gate SHA mismatch: {path}"
        )

    return path


class UnifiedSVTRv32(UnifiedSVTRv31):

    def __init__(
        self,
        config,
        policy,
        *,
        hint="none",
        gate_path="outputs/unified_v3/fuse_gate_v3.joblib",
        motion_gate_path="outputs/unified_v3/motion_gate_v1.joblib",
    ):
        fuse_path = verified_file(gate_path, FUSE_SHA)
        motion_path = verified_file(
            motion_gate_path, MOTION_SHA
        )

        super().__init__(
            config,
            policy,
            hint=hint,
            gate_path=str(fuse_path),
        )

        artifact = joblib.load(motion_path)

        if artifact["feature_names"] != FEATURE_NAMES:
            raise RuntimeError(
                "Motion feature schema mismatch"
            )

        self.motion_model = artifact["model"]
        self.motion_threshold = float(
            artifact["threshold"]
        )
        self.motion_active = bool(
            artifact["automatic_gate_active"]
        )

    def choose(self, image, route="auto"):
        selected, details = super().choose(image, route)
        details = dict(details)

        details["router_version"] = (
            "unified_v3_2_fuse_first_candidate"
        )
        details["v32_policy"] = "fuse_first"
        details["v32_motion_gate_active"] = (
            self.motion_active
        )

        # Explicit routes and externally declared hints
        # always have priority.
        if route != "auto" or self.hint != "none":
            details["v32_reason"] = (
                "explicit_route_or_hint"
            )
            return selected, details

        if not self.motion_active:
            details["v32_reason"] = (
                "motion_gate_inactive"
            )
            return selected, details

        # Preserve FUSE selected by V3.1.
        if selected == "artistic":
            details["v32_reason"] = (
                "fuse_priority_protection"
            )
            return selected, details

        # V3.1 already selected S2; do not run
        # another gate unnecessarily.
        if selected == "s2":
            details["v32_reason"] = (
                "existing_s2_retained"
            )
            return selected, details

        if selected != "b0":
            raise RuntimeError(
                f"Unexpected V3.1 auto route: {selected}"
            )

        vector = motion_features(image).reshape(1, -1)

        score = float(
            self.motion_model.predict(vector)[0]
        )

        if not np.isfinite(score):
            raise RuntimeError(
                "Non-finite Motion Gate score"
            )

        details["v32_previous_route"] = "b0"
        details["v32_motion_benefit"] = score
        details["v32_motion_threshold"] = (
            self.motion_threshold
        )
        details["gt_blind"] = True
        details["selection_mode"] = (
            "image_only_fuse_first_candidate"
        )

        if score > self.motion_threshold:
            details["v32_reason"] = (
                "motion_gate_b0_to_s2"
            )
            return "s2", details

        details["v32_reason"] = (
            "motion_gate_kept_b0"
        )
        return "b0", details
