"""Official frozen OpenOCR SVTRv2-T baseline, with canonical SCOPE-compatible sources."""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
from time import perf_counter
from PIL import Image
from ..paths import UPSTREAM, require_dir, require_file
from ..types import Prediction

SAGE_SOURCE = UPSTREAM / "experiments/long_text/sage_str_v4/src"
SCOPE_SOURCE = UPSTREAM / "experiments/long-text-recognition/src"
EXPECTED_OPENOCR_COMMIT = "1ccfc6ee6161f7133b192e16af3a9265273997ba"


def init_official_source(openocr: str | Path) -> Path:
    p = require_dir(openocr, "openocr_repo")
    if not (p / "openrec/modeling/__init__.py").is_file():
        raise FileNotFoundError(f"OpenOCR source checkout incomplete: {p}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    # Preserve archived SAGE adapters, but load the frozen SCOPE copy of shared
    # modules first. Their implementation matches except a final newline.
    if not SCOPE_SOURCE.is_dir():
        raise FileNotFoundError(f"Missing frozen SCOPE source directory: {SCOPE_SOURCE}")
    # SAGE is optional for production; its adapters are historical baselines.
    for source in (SAGE_SOURCE, SCOPE_SOURCE):
        if not source.is_dir():
            continue
        if str(source) in sys.path:
            sys.path.remove(str(source))
        sys.path.insert(0, str(source))
    # Avoid mixing live imports from the other official OpenOCR-WATERec tree.
    import subprocess
    if (p / ".git").is_dir():
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=p, text=True).strip()
        if revision != EXPECTED_OPENOCR_COMMIT:
            raise RuntimeError(f"OpenOCR commit mismatch: {revision} != {EXPECTED_OPENOCR_COMMIT}")
    return p


class OfficialB0:
    """GT-blind frozen SVTRv2 original, not post-hoc label-aware."""
    def __init__(self, *, openocr_repo, official_checkpoint, official_config, device="cpu"):
        import torch
        self.torch = torch
        self.device = device
        init_official_source(openocr_repo)
        from longtext_final import load_official_svtr, preprocess_ltb, greedy_ctc, normalize_ltb
        self.preprocess = preprocess_ltb
        self.decode = greedy_ctc
        self.net, self.chars = load_official_svtr(
            openocr_repo, require_file(official_config, "official_config"),
            require_file(official_checkpoint, "official_checkpoint"), device)
        self.net.eval().requires_grad_(False)
        self.method = "official_svtrv2_b0"

    def predict(self, image: Image.Image) -> Prediction:
        t0 = perf_counter()
        x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            logits = self.net(x)[0].detach().cpu().numpy()
        text = self.decode(logits, self.chars)
        return Prediction(text=text, method=self.method, route="baseline",
                          latency_ms=(perf_counter() - t0) * 1000)
