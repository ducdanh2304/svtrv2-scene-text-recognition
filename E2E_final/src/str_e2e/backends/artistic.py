"""EXP-004 FUSE: R3_lowlr + fixed TTA versus WATERec-RS, delta=0.15.

Matching original inference requires its EXACT checkpoint/config/dictionary. We
never silently train an R3 checkpoint, unlike the old Kaggle all-in-one script.
"""
from __future__ import annotations
import copy
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from ..paths import require_file, require_dir
from ..types import Prediction
from .official import init_official_source


def choose_fuse(r3: Prediction, teacher: Prediction, delta: float = 0.15) -> Prediction:
    """Pure replica of EXP-004. Same normalized text -> pick WATERec.

    Different text -> pick WATERec if score_W + delta >= score_R3.
    Comparability of scores is empirical, not mathematically calibrated.
    """
    from ..metrics import normalize
    if r3.confidence is None or teacher.confidence is None:
        raise ValueError("FUSE requires both confidences; no synthetic scores allowed")
    same = normalize(r3.text) == normalize(teacher.text)
    use_teacher = same or float(teacher.confidence) + delta >= float(r3.confidence)
    chosen = teacher if use_teacher else r3
    return Prediction(text=chosen.text, confidence=chosen.confidence,
                      method="EXP004_FUSE_R3TTA_WATERec", route="artistic",
                      latency_ms=(r3.latency_ms or 0.0) + (teacher.latency_ms or 0.0),
                      details={"selected": "waterec" if use_teacher else "r3_tta",
                               "agreement": same, "delta": delta,
                               "r3_text": r3.text, "waterec_text": teacher.text,
                               "r3_score": r3.confidence, "waterec_score": teacher.confidence})


class R3TTA:
    """R3 with original ratio-neighbour and tall-image +/-90 degree TTA."""
    def __init__(self, *, openocr_repo, r3_checkpoint, r3_config, device="cpu", tta=True):
        import torch
        import yaml
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        self.torch = torch
        self.T = T
        self.TF = TF
        self.device = torch.device(device)
        self.tta = tta
        root = init_official_source(openocr_repo)
        with require_file(r3_config, "r3_config").open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        cfg = copy.deepcopy(cfg)
        if not isinstance(cfg, dict) or "Architecture" not in cfg:
            raise ValueError("R3 config must be original OpenOCR YAML with Architecture")
        # Make dictionary path absolute; the default YAML refers to cwd.
        dict_path = root / "tools/utils/EN_symbol_dict.txt"
        if not dict_path.is_file():
            raise FileNotFoundError(dict_path)
        cfg["Global"]["character_dict_path"] = str(dict_path)
        cfg["PostProcess"]["character_dict_path"] = str(dict_path)
        from openrec.modeling import build_model
        from openrec.postprocess import build_post_process
        self.post = build_post_process(cfg["PostProcess"], cfg["Global"])
        cfg["Architecture"]["Decoder"]["out_channels"] = self.post.get_character_num()
        self.model = build_model(cfg["Architecture"])
        ckpt = torch.load(require_file(r3_checkpoint, "r3_checkpoint"), map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            raise ValueError("R3 checkpoint has no state_dict")
        state = {name.removeprefix("module."): tensor for name, tensor in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"R3/config mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
        self.model.to(self.device).eval().requires_grad_(False)
        names = [list(item)[0] for item in cfg["Eval"]["dataset"]["transforms"]]
        if "NaRecTVResize" in names:
            raise ValueError("R3 config unexpectedly uses NaRecTVResize; EXP-004 R3 used RatioDataSetTVResize")
        self.max_ratio = cfg["Eval"]["loader"].get("max_ratio", 12)
        self.base_shape = cfg["Eval"]["dataset"].get("base_shape", [[64, 64], [96, 48], [112, 40], [128, 32]])
        self.base_h = cfg["Eval"]["dataset"].get("base_h", 32)

    def _predict_one(self, img: Image.Image, ratio_override: int | None = None):
        width, height = img.size
        ratio = (int(np.clip(np.around(width / height), 1, self.max_ratio))
                 if ratio_override is None else ratio_override)
        if ratio <= 4:
            iw, ih = self.base_shape[ratio - 1]
        else:
            ih = self.base_h
            iw = self.base_h * ratio
        resized = self.TF.resize(img.convert("RGB"), (ih, iw),
                                 interpolation=self.T.InterpolationMode.BICUBIC)
        tensor = self.TF.to_tensor(resized)
        tensor = self.TF.normalize(tensor, [0.5] * 3, [0.5] * 3)
        with self.torch.inference_mode():
            output = self.model(tensor.unsqueeze(0).to(self.device))
            decoded = self.post(output)
        # Matches original EXP-004 when post returns separate [GTC, CTC] predictions.
        if len(decoded) == 2 and isinstance(decoded[0], list):
            (gtc_text, gtc_score), (ctc_text, ctc_score) = decoded[0][0], decoded[1][0]
            return str(ctc_text), float(ctc_score)
        text, score = decoded[0]
        return str(text), float(score)

    def predict(self, image: Image.Image) -> Prediction:
        start = perf_counter()
        img = image.convert("RGB")
        best_text, best_score = self._predict_one(img)
        if self.tta:
            w, h = img.size
            ratio = int(np.clip(np.around(w / h), 1, self.max_ratio))
            candidates = [(img, r) for r in (ratio - 1, ratio + 1) if 1 <= r <= self.max_ratio]
            if h > 1.3 * w:
                candidates.extend([(img.rotate(90, expand=True), None),
                                   (img.rotate(-90, expand=True), None)])
            for transformed, ratio_override in candidates:
                text, score = self._predict_one(transformed, ratio_override)
                if score > best_score:
                    best_text, best_score = text, score
        return Prediction(text=best_text, confidence=best_score,
                          method="r3_lowlr_tta" if self.tta else "r3_lowlr",
                          route="artistic_component", latency_ms=(perf_counter() - start) * 1000)
