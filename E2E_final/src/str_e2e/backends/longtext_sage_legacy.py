"""SAGE-STR v4, with unchanged upstream candidate generation and learned selector."""
from __future__ import annotations
from time import perf_counter
from pathlib import Path
from PIL import Image
from ..paths import require_file
from ..types import Prediction
from .official import OfficialB0, SAGE_SOURCE


class SageV4:
    def __init__(self, *, openocr_repo, official_checkpoint, official_config,
                 sage_selector, device="cpu"):
        import joblib
        import sklearn
        if sklearn.__version__ != "1.6.1":
            raise RuntimeError("SAGE v4 selector was serialized with scikit-learn 1.6.1; "
                               f"installed={sklearn.__version__}. "
                               "Use a dedicated environment with scikit-learn==1.6.1.")
        self.baseline_model = OfficialB0(openocr_repo=openocr_repo,
                                         official_checkpoint=official_checkpoint,
                                         official_config=official_config, device=device)
        # OfficialB0 initializes the exact archived SAGE module import path.
        from sage_gpu_adapter import SAGEGPUAdapter
        from sage_str_v4.selector import ContextualDualArbiter
        self.adapter = SAGEGPUAdapter(self.baseline_model.net,
                                       self.baseline_model.chars, device=device, batch_size=2)
        self.selector = joblib.load(require_file(sage_selector, "sage_selector"))
        if not isinstance(self.selector, ContextualDualArbiter):
            raise TypeError("SAGE selector is not the expected ContextualDualArbiter")
        if not self.selector.audited:
            raise RuntimeError("SAGE selector has not passed its original development audit")

    def predict_pair(self, image: Image.Image) -> tuple[Prediction, Prediction]:
        start = perf_counter()
        evidence = self.adapter.infer_one(image.convert("RGB"))
        decision = self.selector.select(evidence["observation"])
        ms = (perf_counter() - start) * 1000
        base = Prediction(text=evidence["svtrv2"], method="official_svtrv2_b0",
                          route="longtext", latency_ms=None,
                          details={"cost_timing": "baseline_output_reused_from_sage_candidates; independent_baseline_ms_not_measured"})
        method = Prediction(text=decision["prediction"], method="SAGE_STR_v4",
                            route="longtext", latency_ms=ms,
                            details={"strategy": decision["chosen"],
                                     "reference_strategy": decision["reference"],
                                     "decision_reason": decision["reason"],
                                     "cost_timing": "shared_candidate_generation"})
        return base, method
