"""Additive release API. The original STRSystem and its modes are unchanged."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .system import STRSystem, load_config, MODES
from .paths import ROOT
from .types import Prediction

GLOBAL_MODES = {'global', 'global_spatial'}
CALIBRATION_SHA256 = 'fdcb01ff353c29653b2edaca354b8fe863115db31b053ee833ceff032a573bea'


def spatial_first_table(row):
    """Exact pooling policy used by the full benchmark evaluator."""
    selected=[s if s is not None else f for f,s in zip(row['full_table'],row['spatial_table'])]
    art=[np.asarray(p,dtype=float) for p in selected[2:] if p is not None]
    return [selected[0],selected[1],np.mean(art,axis=0) if art else None]


def locked_calibration(path=None):
    path = Path(path) if path is not None else ROOT/'global_release/global_calibration.locked.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CALIBRATION_SHA256:
        raise RuntimeError('Calibration SHA256 mismatch: '+str(path))
    return json.loads(raw)


class GlobalSTRSystem(STRSystem):
    """Use an explicit global mode; preserve original dispatch for legacy modes.

    A single system owns the persistent WATERec worker. Use sequential calls,
    not concurrent inference. Global pair's baseline is the native SCOPE trace;
    the full benchmark separately invokes OfficialB0 for independent pairing.
    """
    @classmethod
    def from_config(cls, path):
        return cls(load_config(path))

    def predict_pair(self, image, mode=None):
        mode = self.config.get('mode','auto') if mode is None else mode
        if mode not in GLOBAL_MODES:
            return super().predict_pair(image,mode)
        cal = locked_calibration()
        if 'global_joint' not in self.backends:
            from .backends.global_joint import GlobalJoint
            self.backends['global_joint'] = GlobalJoint(self)
        from .backends.global_probability import choose
        row = self.backends['global_joint'].predict_evidence(image)
        table = spatial_first_table(row) if mode == 'global_spatial' else row['table']
        text,mbr,p = choose(row['candidates'],table,cal['weights'],cal['temperatures'])
        return (Prediction(text=row['pred']['b0'],method='official_svtrv2_b0',route='baseline_trace'),
                Prediction(text=text,method='GLOBAL_CTC_AR_SPATIAL_FIRST' if mode=='global_spatial' else 'GLOBAL_CTC_AR',route=mode,latency_ms=row['latency_ms'],details={**row,'posterior':p,'mbr':mbr,'calibration_sha256':CALIBRATION_SHA256}))
