"""Typed candidate record contract shared with frozen SVTRv2-T GPU adapter.
The former SAGE v2 binary gate is intentionally NOT used here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from math import log1p, isfinite
from typing import Iterable, Optional, Sequence
import json
import numpy as np
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier

STRATEGIES = {
    "rotate_cw", "rotate_ccw", "full_width2", "geometry_v3",
    "crop2_msr", "crop3_msr", "crop3_width2", "crop3_width3",
}
NUMERIC_FEATURES = [
    "aspect_log", "base_nchars", "base_ctc_pressure", "base_entropy",
    "base_char_margin", "base_edge_occupancy", "base_confidence",
    "cand_nchars", "cand_ctc_pressure", "cand_entropy",
    "cand_char_margin", "cand_edge_occupancy", "cand_confidence",
    "prediction_agreement", "char_growth", "conf_delta", "pressure_delta",
    "candidate_overlap_agreement", "candidate_n_windows",
    "candidate_log_distortion", "candidate_cost_ms",
]
CATEGORICAL_FEATURES = ["strategy"]


def _normalize(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isascii() and ch.isalnum())


def _repeats(s: str) -> int:
    return sum(s[i] == s[i-1] for i in range(1,len(s)))


@dataclass(frozen=True)
class Hypothesis:
    """GT-blind recognizer output and evidence from CTC/raw geometry.

    Expected confidence/entropy/margin are post-softmax CTC aggregates, not
    necessarily comparable across input transforms without calibration.
    distortion = max((W'/H')/(W/H), reciprocal); >=1.
    """
    strategy: str
    text: str
    ctc_T: int
    confidence: float = 0.0
    entropy: float = 0.0
    char_margin: float = 0.0
    edge_occupancy: float = 0.0
    overlap_agreement: float = 0.0
    n_windows: int = 1
    distortion: float = 1.0
    cost_ms: float = 0.0

    def __post_init__(self):
        if not isinstance(self.text,str):raise ValueError("text must be string")
        if self.ctc_T<1 or self.n_windows<1 or self.distortion<1:
            raise ValueError("Invalid T/windows/distortion")
        for k in ("confidence","entropy","char_margin","edge_occupancy",
                  "overlap_agreement","distortion","cost_ms"):
            if not isfinite(float(getattr(self,k))):raise ValueError(k)

    @property
    def normalized(self)->str:return _normalize(self.text)

    @property
    def pressure(self)->float:
        # Necessary CTC slots, NOT a reliable estimate of actual GT length.
        s=self.normalized
        return (len(s)+_repeats(s))/max(1,self.ctc_T)


@dataclass(frozen=True)
class Observation:
    """Inference interface NEVER has ID, filename, ground-truth or dataset."""
    width: int
    height: int
    original: Hypothesis
    alternatives: tuple[Hypothesis,...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.width<=0 or self.height<=0:raise ValueError("Invalid image shape")
        if self.original.strategy!="original":raise ValueError("baseline must be original")
        names=[v.strategy for v in self.alternatives]
        if len(names)!=len(set(names)) or not set(names).issubset(STRATEGIES):
            raise ValueError("Duplicate/unknown candidate strategy")


