"""Conservative V3 candidate layered on Unified V2."""

import joblib

from .router_v3_features import (
    FEATURE_NAMES, image_features
)
from .unified_v2 import UnifiedSVTRv2


class UnifiedSVTRv3(UnifiedSVTRv2):
    def __init__(
        self,
        config,
        policy,
        *,
        hint="none",
        gate_path="outputs/unified_v3/fuse_gate_v3.joblib",
    ):
        safe_policy = dict(policy)

        # Chưa có LTB dev: không kích hoạt SAGE
        # chỉ vì tỷ lệ chiều rộng/chiều cao.
        safe_policy["enable_longtext_auto"] = False
        safe_policy["enable_artistic_auto"] = False

        super().__init__(
            config, safe_policy, hint=hint
        )

        gate = joblib.load(gate_path)

        if gate["feature_names"] != FEATURE_NAMES:
            raise RuntimeError(
                "V3 feature schema mismatch"
            )

        self.v3_model = gate["model"]
        self.v3_threshold = float(
            gate["threshold"]
        )
        self.v3_active = bool(
            gate["automatic_gate_active"]
        )

    def choose(self, image, route="auto"):
        selected, details = super().choose(
            image, route
        )

        details = dict(details)
        details["router_version"] = (
            "unified_v3_development_candidate"
        )
        details["v3_gate_active"] = self.v3_active

        if route != "auto" or self.hint != "none":
            details["v3_reason"] = (
                "explicit_route_or_external_hint"
            )
            return selected, details

        if not self.v3_active:
            details["v3_reason"] = (
                "unapproved_gate_fallback_v2"
            )
            return selected, details

        # Bảo vệ các route có sẵn:
        # không thay S2 hay long/tall bằng FUSE
        # khi chưa có development hỗ trợ.
        if (
            selected != "b0"
            or details.get("reason")
            == "long_or_tall_geometry"
        ):
            details["v3_reason"] = (
                "protected_existing_route"
            )
            return selected, details

        X = image_features(
            image
        ).reshape(1, -1)

        benefit = float(
            self.v3_model.predict(X)[0]
        )

        selected = (
            "artistic"
            if benefit > self.v3_threshold
            else "b0"
        )

        details.update({
            "v3_reason": "image_only_fuse_gate",
            "v3_expected_net_benefit": benefit,
            "v3_threshold": self.v3_threshold,
            "gt_blind": True,
            "selection_mode": (
                "image_only_dev_candidate"
            ),
        })

        return selected, details
