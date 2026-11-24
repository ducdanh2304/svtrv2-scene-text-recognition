"""Unified end-to-end recognizer interface for *localized* scene-text crops.

Branches preserve their original model logic. This is an integration proposal:
no routed combined-run score is claimed until measured on a held-out cohort.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from PIL import Image
import yaml
from .types import Prediction
from .routing import image_route
from .paths import ROOT

MODES = {"baseline", "motion", "artistic", "longtext", "auto"}


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML object")
    return cfg


class STRSystem:
    def __init__(self, config: dict):
        self.config = config
        self.assets = config.get("assets", {})
        self.device = config.get("device", "cpu")
        self.backends = {}

    def _backend(self, key):
        if key in self.backends:
            return self.backends[key]
        a, d = self.assets, self.device
        if key == "official":
            from .backends.official import OfficialB0
            obj = OfficialB0(openocr_repo=a.get("openocr_repo"),
                             official_checkpoint=a.get("official_checkpoint"),
                             official_config=a.get("official_config"), device=d)
        elif key in ("motion_b0", "s2_aug"):
            from .backends.motion import MotionBackend
            obj = MotionBackend(openocr_repo=a.get("openocr_repo"),
                                motion_model_dir=a.get("motion_model_dir"),
                                device=d, variant="b0" if key == "motion_b0" else "s2_aug")
        elif key == "r3_base" or key == "r3_tta":
            from .backends.artistic import R3TTA
            obj = R3TTA(openocr_repo=a.get("openocr_repo"),
                        r3_checkpoint=a.get("r3_checkpoint"),
                        r3_config=a.get("r3_config"), device=d, tta=key == "r3_tta")
        elif key == "waterec":
            from .backends.waterec import WATERecBackend
            obj = WATERecBackend(waterec_repo=a.get("waterec_repo"),
                                 waterec_checkpoint=a.get("waterec_checkpoint"),
                                 waterec_config=a.get("waterec_config"),
                                 device=("cuda" if d == "cuda:0" else d))
        elif key == "scope":
            from .backends.longtext import ScopeV4
            scope_cfg = self.config.get("longtext", {})
            obj = ScopeV4(openocr_repo=a.get("openocr_repo"),
                          official_checkpoint=a.get("official_checkpoint"),
                          official_config=a.get("official_config"), device=d,
                          batch_size=int(scope_cfg.get("batch_size", 2)),
                          strict_frozen_assets=scope_cfg.get("strict_frozen_assets", True))
        elif key == "sage":
            # Legacy research baseline only; no production route dispatches here.
            from .backends.longtext_sage_legacy import SageV4
            obj = SageV4(openocr_repo=a.get("openocr_repo"),
                         official_checkpoint=a.get("official_checkpoint"),
                         official_config=a.get("official_config"),
                         sage_selector=a.get("sage_selector"), device=d)
        else:
            raise KeyError(key)
        self.backends[key] = obj
        return obj

    def predict_pair(self, image: Image.Image, mode: str | None = None) -> tuple[Prediction, Prediction]:
        if mode is None:
            mode = self.config.get("mode", "auto")
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode}; expected one of {sorted(MODES)}")
        routing_details = {}
        if mode == "auto":
            mode, routing_details = image_route(image, **self.config.get("routing", {}))
        if mode == "baseline":
            base = self._backend("official").predict(image)
            method = base
        elif mode == "motion":
            base = self._backend("motion_b0").predict(image)
            method = self._backend("s2_aug").predict(image)
        elif mode == "artistic":
            from .backends.artistic import choose_fuse
            base = self._backend("r3_base").predict(image)
            r3 = self._backend("r3_tta").predict(image)
            w = self._backend("waterec").predict(image)
            method = choose_fuse(r3, w, float(self.config.get("artistic", {}).get("fuse_delta", .15)))
        elif mode == "longtext":
            base, method = self._backend("scope").predict_pair(image)
        else:
            raise AssertionError(mode)
        # Return explicit route metadata for auditable routed evaluation.
        meta = {"branch": mode, **routing_details}
        return (Prediction(**{**base.as_dict(), "route": mode, "details": {**base.details, **meta}}),
                Prediction(**{**method.as_dict(), "route": mode, "details": {**method.details, **meta}}))

    def predict(self, image: Image.Image, mode: str | None = None) -> Prediction:
        return self.predict_pair(image, mode=mode)[1]

    def close(self):
        for obj in self.backends.values():
            close = getattr(obj, "close", None)
            if callable(close):
                close()
        self.backends.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
