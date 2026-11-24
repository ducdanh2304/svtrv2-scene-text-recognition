"""V3.1 experimental geometry-guard ablation.

Only reopens the B0 fallback for long/tall images.
Explicit routes, external hints and S2 remain unchanged.
"""

import numpy as np

from .router_v3_features import image_features
from .unified_v3 import UnifiedSVTRv3


class UnifiedSVTRv31(UnifiedSVTRv3):

    def choose(self, image, route="auto"):
        selected, details = super().choose(image, route)
        details = dict(details)

        details["router_version"] = (
            "unified_v3_1_geometry_ablation"
        )

        if route != "auto" or self.hint != "none":
            return selected, details

        if not self.v3_active:
            return selected, details

        # Only reopen the specific V2 geometry guard.
        # Never override S2 or any other selected expert.
        if (
            selected != "b0"
            or details.get("reason")
            != "long_or_tall_geometry"
        ):
            return selected, details

        x = image_features(image).reshape(1, -1)

        benefit = float(self.v3_model.predict(x)[0])

        if not np.isfinite(benefit):
            raise RuntimeError("Non-finite V3.1 gate score")

        selected = (
            "artistic"
            if benefit > self.v3_threshold
            else "b0"
        )

        details.update({
            "v3_reason": "v31_geometry_gate_recheck",
            "v3_expected_net_benefit": benefit,
            "v3_threshold": self.v3_threshold,
            "selection_mode": "image_only_dev_ablation",
            "gt_blind": True,
        })

        return selected, details
