"""No-checkpoint integration contracts for the frozen SCOPE long-text backend."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from str_e2e.backends.longtext import ScopeV4, verify_scope_sources
from str_e2e.system import STRSystem
from str_e2e.types import Prediction


class FakeScope:
    def predict_pair(self, image):
        return (Prediction(text="old", method="official_svtrv2_b0", route="longtext"),
                Prediction(text="new", method="SCOPE_CTC_V4", route="longtext"))


class DisallowedSage:
    def predict_pair(self, image):
        raise AssertionError("Production longtext must never call historical SAGE")


def test_scope_manifest_17_sources_verified():
    assert verify_scope_sources() == 17


def test_explicit_longtext_selects_scope_not_sage():
    system = STRSystem({"mode": "longtext", "assets": {}})
    system.backends["scope"] = FakeScope()
    system.backends["sage"] = DisallowedSage()
    base, method = system.predict_pair(Image.new("RGB", (320, 32)), mode="longtext")
    assert base.text == "old"
    assert method.text == "new"
    assert method.method == "SCOPE_CTC_V4"
    assert method.route == "longtext"


def test_auto_longtext_selects_scope_not_sage():
    system = STRSystem({"mode": "auto", "routing": {"long_ratio_min": 5}, "assets": {}})
    system.backends["scope"] = FakeScope()
    system.backends["sage"] = DisallowedSage()
    _, method = system.predict_pair(Image.new("RGB", (320, 32)), mode="auto")
    assert method.method == "SCOPE_CTC_V4"
    assert method.details["branch"] == "longtext"
    assert method.details["routing_reason"] == "geometry"


def _minimal_scope(result):
    backend = ScopeV4.__new__(ScopeV4)
    backend.torch = SimpleNamespace(inference_mode=lambda: _NullContext())
    backend.scope = SimpleNamespace(predict_image=lambda image, recognizer: result)
    backend.recognizer = object()
    backend.frozen_match = True
    backend.checkpoint_sha256 = "test-ckpt"
    backend.config_sha256 = "test-cfg"
    return backend


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_scope_adapter_uses_matched_baseline_and_safe_scope():
    result = {
        "original_svtrv2_full12": "base123", "safe_scope": "base1234",
        "safe_scope_active": 1, "coverage_policy_version": "locked",
        "coverage_decision": "MULTI_CROP_DIRECT", "route": "v3",
        "scope_decoder_ms": 2.0, "safe_scope_total_ms": 12.0,
        "active_crops": 2, "extra_crops": 2,
    }
    b, m = _minimal_scope(result).predict_pair(Image.new("RGB", (320, 32)))
    assert b.text == "base123" and b.latency_ms is None
    assert m.text == "base1234" and m.method == "SCOPE_CTC_V4"
    assert m.details["scope_internal_route"] == "v3"
    assert m.details["scope_decoder_ms"] == 2.0
    assert m.latency_ms >= 0


def test_scope_fallback_must_preserve_original():
    good = {"original_svtrv2_full12": "abc", "safe_scope": "abc", "safe_scope_active": 0}
    b, m = _minimal_scope(good).predict_pair(Image.new("RGB", (120, 32)))
    assert b.text == m.text
    bad = {**good, "safe_scope": "bad"}
    with pytest.raises(AssertionError, match="fallback"):
        _minimal_scope(bad).predict_pair(Image.new("RGB", (120, 32)))


def test_unexpected_response_fails_not_silent_baseline():
    with pytest.raises(RuntimeError, match="no matched"):
        _minimal_scope({"safe_scope": "abc"}).predict_pair(Image.new("RGB", (120, 32)))


def test_frozen_checkpoint_guard_rejects_wrong_hash_before_loading_model(tmp_path):
    ckpt = tmp_path / "random.pth"
    cfg = tmp_path / "random.yml"
    ckpt.write_bytes(b"not the pinned checkpoint")
    cfg.write_bytes(b"not the pinned config")
    with pytest.raises(RuntimeError, match="frozen asset SHA mismatch"):
        ScopeV4(openocr_repo=tmp_path, official_checkpoint=ckpt, official_config=cfg,
                strict_frozen_assets=True)
