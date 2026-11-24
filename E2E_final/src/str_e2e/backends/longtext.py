"""Frozen, GT-blind SCOPE-CTC V4 long-text adapter for STRSystem.

This file is only an integration adapter. All 17 frozen upstream SCOPE
modules remain untouched under research_upstream/experiments/long-text-recognition/src.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image

from ..paths import UPSTREAM, require_dir, require_file
from ..types import Prediction
from .official import init_official_source, SCOPE_SOURCE, SAGE_SOURCE

# Frozen research release: see scripts/infer.py in the upstream SCOPE artifact.
EXPECTED_CHECKPOINT_SHA256 = "684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6"
EXPECTED_CONFIG_SHA256 = "f1ab35a0f5e44d3c24c940cdf1a07404068ad9191bad9fe232ad4fcf38bbbe29"
SHARED_MODULES = ("longtext_final", "longtext_v3", "density_probe", "orientation_probe", "unified_router")
SCOPE_MODULES = ("alex_ctc", "alex_c1", "alex_e5", "alex_e7", "space_ctc",
                 "scope_e7_fast", "scope_final_runtime_v3", "scope_coverage_runtime",
                 "scope_ctc_jit", "scope_beam_jit_v4")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_scope_sources() -> int:
    """Validate frozen sources; refuse a partial or modified research release."""
    manifest_path = SCOPE_SOURCE / "SOURCE_SHA256.json"
    manifest = json.loads(require_file(manifest_path, "SCOPE SOURCE_SHA256.json").read_text())
    if len(manifest) != 17:
        raise RuntimeError(f"Expected frozen SCOPE manifest with 17 modules, got {len(manifest)}")
    for name, expected in manifest.items():
        if Path(name).name != name or not name.endswith(".py"):
            raise RuntimeError(f"Invalid SCOPE module name in manifest: {name!r}")
        actual = _sha256(require_file(SCOPE_SOURCE / name, f"SCOPE source {name}"))
        if actual != expected:
            raise RuntimeError(f"SCOPE SOURCE HASH MISMATCH: {name}: {actual} != {expected}")
    # The five shared files in the historical SAGE archive differ only by one
    # trailing newline. Verify that this remains true before using shared flat imports.
    for mod in SHARED_MODULES:
        scope_bytes = (SCOPE_SOURCE / (mod + ".py")).read_bytes().rstrip(b"\n")
        sage_file = SAGE_SOURCE / (mod + ".py")
        if sage_file.is_file():
            sage_bytes = sage_file.read_bytes().rstrip(b"\n")
            if scope_bytes != sage_bytes:
                raise RuntimeError(f"Shared SAGE/SCOPE module has diverged: {mod}")
    return len(manifest)


def _check_flat_imports() -> None:
    """Avoid accidental use of arbitrary previously imported flat OCR modules."""
    for mod in SHARED_MODULES:
        existing = sys.modules.get(mod)
        if existing is None:
            continue
        actual = Path(getattr(existing, "__file__", "")).resolve()
        allowed = {(SCOPE_SOURCE / (mod + ".py")).resolve(),
                   (SAGE_SOURCE / (mod + ".py")).resolve()}
        if actual not in allowed:
            raise RuntimeError(f"Conflicting imported {mod}: {actual}; use a clean STR process")
    for mod in SCOPE_MODULES:
        existing = sys.modules.get(mod)
        if existing is None:
            continue
        actual = Path(getattr(existing, "__file__", "")).resolve()
        expected = (SCOPE_SOURCE / (mod + ".py")).resolve()
        if actual != expected:
            raise RuntimeError(f"Conflicting imported {mod}: {actual}; expected {expected}")


class ScopeV4:
    """Single frozen SVTRv2-T with baseline output reused from SCOPE's matched pass."""

    def __init__(self, *, openocr_repo, official_checkpoint, official_config,
                 device="cpu", batch_size=2, strict_frozen_assets=True):
        ckpt = require_file(official_checkpoint, "official_checkpoint")
        cfg = require_file(official_config, "official_config")
        repo = require_dir(openocr_repo, "openocr_repo")
        verify_scope_sources()
        ckpt_sha = _sha256(ckpt)
        cfg_sha = _sha256(cfg)
        frozen_match = (ckpt_sha == EXPECTED_CHECKPOINT_SHA256
                        and cfg_sha == EXPECTED_CONFIG_SHA256)
        if strict_frozen_assets and not frozen_match:
            raise RuntimeError(
                "SCOPE V4 frozen asset SHA mismatch. "
                f"checkpoint={ckpt_sha} expected={EXPECTED_CHECKPOINT_SHA256}; "
                f"config={cfg_sha} expected={EXPECTED_CONFIG_SHA256}. "
                "Do not attribute archived 56.13% LTB to a different checkpoint/config."
            )
        self.device = device
        self.frozen_match = frozen_match
        self.checkpoint_sha256 = ckpt_sha
        self.config_sha256 = cfg_sha
        init_official_source(repo)
        _check_flat_imports()
        import torch
        import longtext_final as LT
        import unified_router as UNI
        import scope_ctc_jit as JIT
        import scope_beam_jit_v4 as BEAM
        import scope_coverage_runtime as SCOPE
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        net, chars = LT.load_official_svtr(repo, cfg, ckpt, device)
        self.recognizer = UNI.UnifiedRecognizer(net, chars, device=device, batch_size=int(batch_size))
        self.torch = torch
        self.scope = SCOPE
        JIT.activate(strict=True)
        JIT.activate_forced(strict=True)
        BEAM.activate(strict=True)

    def predict_pair(self, image: Image.Image) -> tuple[Prediction, Prediction]:
        start = perf_counter()
        with self.torch.inference_mode():
            result = self.scope.predict_image(image.convert("RGB"), self.recognizer)
        elapsed_ms = (perf_counter() - start) * 1000.0
        original = result.get("original_svtrv2_full12")
        safe = result.get("safe_scope")
        if not isinstance(original, str) or not isinstance(safe, str):
            raise RuntimeError("SCOPE backend returned no matched baseline/safe_scope strings")
        if result.get("safe_scope_active") == 0 and original != safe:
            raise AssertionError("SCOPE preservation fallback must equal original baseline")
        base = Prediction(
            text=original,
            method="official_svtrv2_b0",
            route="longtext",
            latency_ms=None,
            details={"source": "SCOPE original_svtrv2_full12 (same model pass)",
                     "baseline_timing": "not independently measured"},
        )
        details = {
            "scope_internal_route": str(result.get("route", "")),
            "scope_policy": result.get("coverage_policy_version"),
            "coverage_decision": result.get("coverage_decision"),
            "safe_scope_active": result.get("safe_scope_active"),
            "safe_scope_original_fallback": result.get("safe_scope_original_fallback"),
            "frozen_asset_hash_match": self.frozen_match,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "cost_timing": "shared forward passes; elapsed_ms is wall-time for paired outputs",
        }
        for key in ("coverage_probe_ms", "scope_decoder_ms", "paired_scale_forward_ms",
                    "safe_scope_total_ms", "active_crops", "extra_crops"):
            value = result.get(key)
            if value is not None:
                details[key] = float(value) if key.endswith("_ms") else int(value)
        method = Prediction(
            text=safe, method="SCOPE_CTC_V4", route="longtext",
            latency_ms=elapsed_ms, details=details,
        )
        return base, method

    def close(self):
        # Keep cleanup symmetrical with other backends; clear Python references.
        self.recognizer = None
        self.scope = None
