"""Unified SVTRv2 V2: B0, Danh S2, Truc FUSE, Thuy SAGE."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

from PIL import Image

from .adaptive import AdaptiveSVTRv2
from .backends.artistic import choose_fuse

HINTS = {"none", "artistic", "longtext", "motion"}
ROUTES = {"auto", "b0", "s2", "artistic", "longtext"}


def audit_experts(config: dict) -> dict[str, Any]:
    """Inspect dependencies and files, without loading any models."""
    assets = config.get("assets", {})
    root = Path(__file__).resolve().parents[2]

    def present(name, suffix=None):
        value = assets.get(name)
        if not value:
            return False
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if suffix:
            path = path / suffix
        return path.exists()

    def package_ready(name):
        try:
            metadata.version(name)
            return True
        except metadata.PackageNotFoundError:
            return False

    def branch(required):
        missing = [
            name for name, ready in required.items()
            if not ready
        ]
        return {
            "assets_ready": not missing,
            "missing": missing,
            "note": "Runtime smoke test still required",
        }

    try:
        sklearn_version = metadata.version("scikit-learn")
    except metadata.PackageNotFoundError:
        sklearn_version = None

    base = {
        "openocr_repo": present(
            "openocr_repo",
            "openrec/modeling/__init__.py"
        ),
        "motion_b0": present(
            "motion_model_dir", "b0_rctc_state.pt"
        ),
        "motion_s2": present(
            "motion_model_dir", "s2_aug_state.pt"
        ),
    }

    sage = {
        "openocr_repo": base["openocr_repo"],
        "official_checkpoint": present("official_checkpoint"),
        "official_config": present("official_config"),
        "sage_selector": present("sage_selector"),
        "scikit-learn==1.6.1": sklearn_version == "1.6.1",
        "joblib": package_ready("joblib"),
        "pandas": package_ready("pandas"),
        "scipy": package_ready("scipy"),
        "rapidfuzz": package_ready("rapidfuzz"),
        "sage_source": (
            root
            / "research_upstream/experiments/long_text"
            / "sage_str_v4/src/sage_gpu_adapter.py"
        ).is_file(),
    }

    fuse = {
        "openocr_repo": base["openocr_repo"],
        "r3_checkpoint": present("r3_checkpoint"),
        "r3_config": present("r3_config"),
        "waterec_repo": present(
            "waterec_repo", "tools/infer_rec.py"
        ),
        "waterec_checkpoint": present("waterec_checkpoint"),
        "waterec_config": present("waterec_config"),
    }

    return {
        "baseline_b0": branch({
            k: v for k, v in base.items()
            if k != "motion_s2"
        }),
        "danh_s2_aug": branch(base),
        "thuy_sage_v4": branch(sage),
        "truc_fuse_r3_waterec": branch(fuse),
        "device": str(config.get("device", "cpu")),
        "notice": (
            "Asset availability is not proof of model "
            "compatibility or successful inference"
        ),
    }


class UnifiedSVTRv2(AdaptiveSVTRv2):
    """One OCR interface for all four expert branches."""

    def __init__(
        self, config: dict, policy: dict, *, hint="none"
    ):
        if hint not in HINTS:
            raise ValueError(f"Unknown hint: {hint}")

        super().__init__(config, policy)
        self.hint = hint

    def choose(self, image: Image.Image, route="auto"):
        if route not in ROUTES:
            raise ValueError(f"Unknown route: {route}")

        # Explicit route always takes priority.
        if route != "auto":
            selected, details = super().choose(image, route)
            return selected, {
                **details,
                "external_hint": self.hint,
                "selection_mode": "explicit_route",
            }

        # A domain hint is supplied externally, not inferred
        # from the ground truth or benchmark labels.
        if self.hint != "none":
            mapping = {
                "artistic": "artistic",
                "longtext": "longtext",
                "motion": "s2",
            }

            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions")

            return mapping[self.hint], {
                "router_version": "unified_v2_domain_conditioned",
                "aspect": width / height,
                "gt_blind": True,
                "external_hint": self.hint,
                "selection_mode": "declared_domain_not_autonomous",
                "reason": "external_domain_hint",
            }

        # Existing GT-blind blur/geometry router.
        selected, details = super().choose(image, "auto")

        return selected, {
            **details,
            "router_version": "unified_v2_image_only",
            "external_hint": "none",
            "selection_mode": "image_only_heuristic",
            "artistic_auto_supported": False,
        }

    def _expert(self, image: Image.Image, route: str):
        # B0, Danh S2_AUG and Thuy SAGE use the existing,
        # independently implemented backend classes.
        if route != "artistic":
            return super()._expert(image, route)

        # Truc EXP-004:
        # R3_lowlr + TTA versus WATERec-RS,
        # original confidence selection with delta=0.15.
        assets = self.config.get("assets", {})

        if "waterec" not in self.system.backends:
            from .backends.waterec import WATERecBackend

            device = str(self.config.get("device", "cpu"))

            # WATERec worker accepts cuda, not cuda:0.
            worker_device = (
                "cuda" if device.startswith("cuda") else "cpu"
            )

            self.system.backends["waterec"] = WATERecBackend(
                waterec_repo=assets.get("waterec_repo"),
                waterec_checkpoint=assets.get(
                    "waterec_checkpoint"
                ),
                waterec_config=assets.get("waterec_config"),
                device=worker_device,
            )

        r3_pred = self.system._backend("r3_tta").predict(image)

        waterec_pred = (
            self.system.backends["waterec"].predict(image)
        )

        delta = float(
            self.config.get("artistic", {}).get(
                "fuse_delta", 0.15
            )
        )

        return choose_fuse(
            r3_pred, waterec_pred, delta
        )
