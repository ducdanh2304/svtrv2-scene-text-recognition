"""EXP-001 S2_AUG and its matched B0 through original audited code."""
from __future__ import annotations
import importlib.util
import sys
from time import perf_counter
from PIL import Image
from ..paths import UPSTREAM, require_dir
from ..types import Prediction

MOTION_SOURCE = UPSTREAM / "experiments/danh_motion_blur/inference/model.py"


def _load_motion_loader():
    # Namespace-specific import avoids colliding with generic module named `model`.
    spec = importlib.util.spec_from_file_location("str_e2e_danh_motion", MOTION_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import archived motion source: {MOTION_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_pipeline


class MotionBackend:
    def __init__(self, *, openocr_repo, motion_model_dir, device="cpu", variant="s2_aug"):
        if variant not in {"b0", "s2_aug", "a123"}:
            raise ValueError(f"Unsupported motion variant: {variant}")
        path = require_dir(motion_model_dir, "motion_model_dir")
        load_pipeline = _load_motion_loader()
        self.pipe = load_pipeline(variant, path, require_dir(openocr_repo, "openocr_repo"), device)
        self.variant = variant

    def predict(self, image: Image.Image) -> Prediction:
        t0 = perf_counter()
        result = self.pipe.predict_crop(image)
        return Prediction(text=result["text"], confidence=float(result["confidence"]),
                          method=f"motion_{self.variant}", route="motion",
                          latency_ms=(perf_counter() - t0) * 1000)
