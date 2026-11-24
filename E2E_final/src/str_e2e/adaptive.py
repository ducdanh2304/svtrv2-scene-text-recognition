"""Adaptive SVTRv2 V1: image-only routing over existing OCR backends."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
from PIL import Image

from .routing import laplacian_variance
from .system import STRSystem


class AdaptiveSVTRv2:
    def __init__(self, config: dict, policy: dict):
        self.config = config
        self.policy = policy
        self.system = STRSystem(config)

        self.threshold = float(
            policy.get("blur_laplacian_var_max", 35.0)
        )
        self.min_contrast = float(
            policy.get("min_contrast_std", 12.0)
        )
        self.long_ratio = float(
            policy.get("long_ratio_min", 5.0)
        )
        self.tall_ratio = float(
            policy.get("tall_ratio_max", 0.8)
        )
        self.enable_longtext = bool(
            policy.get("enable_longtext_auto", False)
        )

        if (
            self.threshold <= 0
            or self.min_contrast < 0
            or self.long_ratio <= 1
        ):
            raise ValueError("Invalid adaptive routing policy")

        if policy.get("enable_artistic_auto", False):
            raise ValueError(
                "No validated artistic detector. "
                "Use explicit route='artistic'."
            )

    def _sync(self):
        if str(self.config.get("device", "cpu")).startswith("cuda"):
            import torch
            torch.cuda.synchronize()

    def choose(self, image: Image.Image, route="auto"):
        valid = {"auto", "b0", "s2", "artistic", "longtext"}
        if route not in valid:
            raise ValueError(f"Invalid route: {route}")

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Empty image")

        aspect = width / height

        details = {
            "router_version": "adaptive_svtrv2_v1",
            "aspect": aspect,
            "gt_blind": True,
            "route_request": route,
        }

        if route != "auto":
            details["reason"] = "explicit_route"
            return route, details

        # Protect very long/tall text from unsuitable motion routing.
        if aspect >= self.long_ratio or aspect <= self.tall_ratio:
            chosen = "longtext" if self.enable_longtext else "b0"
            details["reason"] = "long_or_tall_geometry"
            return chosen, details

        gray = np.asarray(
            image.convert("L").resize((128, 48)),
            dtype=np.float32
        )

        contrast = float(np.std(gray))
        blur_score = laplacian_variance(image)

        details.update({
            "contrast_std": contrast,
            "laplacian_variance": blur_score,
            "blur_threshold": self.threshold,
        })

        if contrast < self.min_contrast:
            details["reason"] = "low_contrast_guard"
            return "b0", details

        chosen = "s2" if blur_score < self.threshold else "b0"
        details["reason"] = "laplacian_blur_gate"

        return chosen, details

    def _expert(self, image, route):
        if route == "b0":
            return self.system._backend("motion_b0").predict(image)

        if route == "s2":
            return self.system._backend("s2_aug").predict(image)

        if route == "longtext":
            return self.system._backend("sage").predict_pair(image)[1]

        if route == "artistic":
            from .backends.artistic import choose_fuse

            r3 = self.system._backend("r3_tta").predict(image)
            teacher = self.system._backend("waterec").predict(image)

            return choose_fuse(
                r3,
                teacher,
                float(
                    self.config.get("artistic", {})
                    .get("fuse_delta", 0.15)
                )
            )

        raise ValueError(route)

    @staticmethod
    def _annotate(pred, route, details, elapsed_ms):
        return replace(
            pred,
            route=route,
            latency_ms=round(elapsed_ms, 5),
            details={
                **pred.details,
                **details,
                "latency_scope":
                    "router+selected_expert_synchronized_excludes_disk_io",
            },
        )

    def predict(self, image, route="auto"):
        """Production inference: execute only selected expert."""
        image = image.convert("RGB")

        self._sync()
        start = perf_counter()

        selected, details = self.choose(image, route)
        result = self._expert(image, selected)

        self._sync()
        elapsed = (perf_counter() - start) * 1000

        return self._annotate(
            result, selected, details, elapsed
        )

    def predict_pair(self, image, mode="auto"):
        """Paired evaluation: baseline cost excluded from method latency."""
        image = image.convert("RGB")

        self._sync()
        start = perf_counter()

        selected, details = self.choose(image, mode)
        router_ms = (perf_counter() - start) * 1000

        self._sync()
        start = perf_counter()

        baseline_raw = self._expert(image, "b0")

        self._sync()
        baseline_ms = (perf_counter() - start) * 1000

        baseline = self._annotate(
            baseline_raw, "baseline", details, baseline_ms
        )

        if selected == "b0":
            method = self._annotate(
                baseline_raw, "b0", details,
                baseline_ms + router_ms
            )
            return baseline, method

        self._sync()
        start = perf_counter()

        candidate_raw = self._expert(image, selected)

        self._sync()
        method_ms = (perf_counter() - start) * 1000 + router_ms

        method = self._annotate(
            candidate_raw, selected, details, method_ms
        )

        return baseline, method

    def close(self):
        self.system.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
