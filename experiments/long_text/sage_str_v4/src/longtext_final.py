"""Long-text STR inference/benchmark: exact LTB Stitch v2 geometry and overlap algorithm.

Model: official SVTRv2-T OpenOCR CTC branch (strict checkpoint/config parity checks).
Methods: full12, full64, stitch_v2, geom_ctc. All share one checkpoint.
Original Stitch v2 crops the ORIGINAL PIL image, then resizes each crop with MSR.
No training, no GT-guided inference, no inference-parameter tuning on LTB.

NOTE: LTB protocol normalizes ASCII alphanumerics to lowercase and strips punctuation.
This is NOT a general-purpose punctuation-preserving OCR output pipeline.
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rapidfuzz.distance import Levenshtein


# ======================= PINNED LEGACY PROCEDURE ==========================
# The two functions below are copied verbatim (algorithm) from user LTB.ipynb,
# cell G2. Do not silently 'improve' coefficients without separate validation.

def make_chunks(img, chunk_ar, overlap):
    w, h = img.size; cw = int(round(h * chunk_ar))
    if w <= cw * 1.2: return [(img, 0, w)]
    stride = max(1, int(cw * (1 - overlap))); xs = list(range(0, w - cw + 1, stride))
    if xs[-1] + cw < w: xs.append(w - cw)
    return [(img.crop((x, 0, x + cw, h)), x, x + cw) for x in xs]


def stitch_v2(parts, spans, tol=0.34):
    out, miss = parts[0], 0
    for j in range(1, len(parts)):
        b = parts[j]; wb = max(spans[j][1] - spans[j][0], 1); exp = max(0, spans[j - 1][1] - spans[j][0]) / wb * len(b)
        if exp < 0.5 or not b or not out: out += b; continue
        Lmax = min(len(out), len(b), int(round(exp * 1.6)) + 2); best = None
        for L in range(1, Lmax + 1):
            d = Levenshtein.distance(out[-L:], b[:L]); sc = L - 2 * d - 0.5 * abs(L - exp)
            if best is None or sc > best[0]: best = (sc, L, d)
        sc, L, d = best
        if sc <= 0 or d > tol * L: miss += 1; L = min(int(round(exp)), len(b)); out += b[L:]
        else: h = L // 2; out = out[:len(out) - L + h] + b[h:]
    return out, miss

# Legacy values selected among eight configurations in original LTB study.
# They are EMPIRICAL LOCKED CONFIG, not theoretic constants or a new grid search.
LEGACY_CHUNK_AR = 6.0
LEGACY_OVERLAP = 0.5


def normalize_ltb(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', str(s).lower())


def target_size_ltb(w: int, h: int, max_ratio: int = 12):
    """Original LTB.ipynb 'msr_round' target_size, copied geometry exactly."""
    if w <= 0 or h <= 0:
        raise ValueError('Invalid image width/height')
    base = [(64, 64), (96, 48), (112, 40), (128, 32)]
    r = min(max(1, round(w / h)), max_ratio)
    return base[r - 1] if r <= 4 else (32 * r, 32)


def preprocess_ltb(img: Image.Image, max_ratio: int = 12) -> torch.Tensor:
    """Original PIL bicubic -> RGB numpy -> [-1,1]."""
    w, h = img.size
    W, H = target_size_ltb(w, h, max_ratio)
    bicubic = Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC
    a = np.asarray(img.convert('RGB').resize((W, H), bicubic), dtype=np.float32) / 255.
    return torch.from_numpy(((a - 0.5) / 0.5).transpose(2, 0, 1).copy())


# ======================= OFFICIAL SVTRv2-T ================================
# Identical tested RCTC extraction and strict load logic from GAWS notebook.

def rctc_logits(dec, x):
    B, C, H, W = x.shape
    x = dec.w_atten_block(x.permute(0, 2, 3, 1).reshape(-1, W, C)).reshape(B, H, W, C).permute(0, 3, 1, 2)
    x_kv = dec.fc_kv(x.flatten(2).transpose(1, 2)).reshape(B, H * W, 2, C).permute(2, 0, 3, 1)
    x_k, x_v = x_kv.unbind(0)
    tok = dec.char_token.tile([B, 1, 1])
    attn = (tok @ x_k).reshape([-1, 1, H, W])
    attn = F.softmax(attn, 2).permute(0, 3, 1, 2)
    x_v = x_v.reshape(B, C, H, W)
    feats = (attn @ x_v.permute(0, 3, 2, 1)).squeeze(2)
    return dec.fc(feats)


class CTCNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transform = getattr(model, 'transform', None)
        self.encoder = model.encoder
        d = model.decoder
        self.rctc = d if hasattr(d, 'w_atten_block') else d.ctc_decoder

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        return rctc_logits(self.rctc, self.encoder(x))


def load_official_svtr(repo, cfg_path, ckpt_path, device='cuda'):
    """Build official architecture, strict checkpoint check, verify extracted logits parity."""
    repo, cfg_path, ckpt_path = map(Path, (repo, cfg_path, ckpt_path))
    if not repo.is_dir() or not cfg_path.is_file() or not ckpt_path.is_file():
        raise FileNotFoundError((str(repo), str(cfg_path), str(ckpt_path)))
    sys.path.insert(0, str(repo))
    from tools.engine.config import Config
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process

    old_dir = Path.cwd()
    try:
        os.chdir(repo)  # official YAML uses repo-relative character dictionary paths
        cfg = Config(str(cfg_path)).cfg
        post = build_post_process(cfg['PostProcess'], cfg['Global'])
        channels = post.get_character_num()
        ccfg = {
            'name': 'CTCLabelDecode',
            'character_dict_path': cfg['PostProcess'].get('character_dict_path') or cfg['Global'].get('character_dict_path'),
            'use_space_char': cfg['PostProcess'].get('use_space_char', cfg['Global'].get('use_space_char', False)),
        }
        chars = list(build_post_process(ccfg, cfg['Global']).character)
        cfg['Architecture']['Decoder']['out_channels'] = channels
        model = build_model(cfg['Architecture'])
    finally:
        os.chdir(old_dir)

    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict):
        raise RuntimeError('Invalid checkpoint format')
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    ref = model.state_dict()
    missing = set(ref) - set(sd)
    extra = set(sd) - set(ref)
    wrong = [k for k in set(ref) & set(sd) if ref[k].shape != sd[k].shape]
    if missing or extra or wrong:
        raise RuntimeError(f'Checkpoint/config mismatch: missing={list(missing)[:4]} extra={list(extra)[:4]} shape={wrong[:4]}')
    model.load_state_dict(sd, strict=True)
    if not chars or str(chars[0]).lower() not in {'blank', '[blank]', '<blank>', 'blank_token'}:
        raise RuntimeError(f'CTC character index 0 must be blank: {chars[:5]}')
    model = model.to(device).eval()
    net = CTCNet(model).to(device).eval()
    if net.rctc.fc.out_features != len(chars):
        raise RuntimeError('CTC output class count differs from dictionary')
    # Check CTCNet exactly matches official RCTC pre-softmax logit.
    probe = torch.randn(1, 3, 32, 192, device=device)
    captured = []
    hook = net.rctc.fc.register_forward_hook(lambda m, arg, output: captured.append(output.detach()))
    try:
        with torch.inference_mode():
            model(probe)
            logits = net(probe)
    finally:
        hook.remove()
    if len(captured) != 2 or not torch.allclose(captured[0], logits, atol=3e-4, rtol=3e-4):
        raise RuntimeError('Official-vs-extracted RCTC logits parity failed')
    del model, probe, captured
    return net, chars


def greedy_ctc(logits: np.ndarray, chars: list[str]) -> str:
    """CTC collapse, blank=0. No blank bias, beam width or confidence threshold."""
    pred = np.asarray(logits).argmax(-1).tolist()
    out, prev = [], -1
    for c in pred:
        if c != prev and c != 0:
            out.append(chars[c])
        prev = c
    return ''.join(out)


def geometry_ctc_same_crops(logits: list[np.ndarray], spans: list[tuple[int, int]], chars: list[str]) -> str:
    """Deterministic center-ownership stitching of CTC timesteps on ORIGINAL crop coordinates.

    SAME crops, backbone, preprocessing, and forward passes as Stitch v2.
    This is a baseline, not a novel or guaranteed accurate CTC alignment algorithm.
    """
    if not logits or len(logits) != len(spans):
        raise ValueError('Need one CTC logit tensor per crop span')
    if len(logits) == 1:
        return greedy_ctc(logits[0], chars)
    centers = [(a + b) / 2 for a, b in spans]
    cuts = [-float('inf')] + [(centers[i] + centers[i+1]) / 2 for i in range(len(centers)-1)] + [float('inf')]
    pieces = []
    for j, (lg, (x0, x1)) in enumerate(zip(logits, spans)):
        t = len(lg)
        if t == 0:
            raise RuntimeError('CTC produced zero timesteps for a crop')
        x = x0 + (np.arange(t) + 0.5) * ((x1 - x0) / t)
        keep = (x >= cuts[j]) & (x < cuts[j+1])
        if not keep.any():
            raise RuntimeError(f'No owned CTC frames for window {j}')
        pieces.append(np.asarray(lg)[keep])
    return greedy_ctc(np.concatenate(pieces, axis=0), chars)


class LongTextRecognizer:
    METHODS = ('full12', 'full64', 'stitch_v2', 'geom_ctc')

    def __init__(self, net: nn.Module, chars: list[str], device: str='cuda', batch_size: int=16):
        self.net = net.to(device).eval()
        self.chars = list(chars)
        self.device = str(device)
        self.batch_size = max(1, int(batch_size))  # memory-only knob, never tuned on accuracy
        if self.chars[0].lower() not in ('blank', '[blank]', '<blank>', 'blank_token'):
            raise ValueError('CTC blank must be class 0')

    @torch.inference_mode()
    def _forward_grouped(self, pils: list[Image.Image], max_ratio: int) -> list[np.ndarray]:
        tens = [preprocess_ltb(im, max_ratio) for im in pils]
        groups = defaultdict(list)
        for i, t in enumerate(tens):
            groups[tuple(t.shape)].append(i)
        ret: list[np.ndarray | None] = [None] * len(tens)
        for shape, ids in groups.items():
            _, h, w = shape
            # Bound temporary tensor volume as image width grows; only affects memory/throughput.
            bs = max(1, min(self.batch_size, int(self.batch_size * 192 / max(w, 1))))
            for start in range(0, len(ids), bs):
                ii = ids[start:start+bs]
                x = torch.stack([tens[i] for i in ii]).to(self.device)
                logits = self.net(x).float().detach().cpu().numpy()
                if logits.ndim != 3 or logits.shape[-1] != len(self.chars):
                    raise RuntimeError(f'Unexpected CTC shape: {logits.shape}')
                for i, lg in zip(ii, logits):
                    ret[i] = lg
        if any(x is None for x in ret):
            raise RuntimeError('Missing CTC results')
        return ret  # type: ignore[return-value]

    def predict_many(self, imgs: list[Image.Image], method: str='stitch_v2', return_info: bool=False):
        if method not in self.METHODS:
            raise ValueError(f'Method must be one of {self.METHODS}')
        if not imgs:
            return []
        imgs = [im.convert('RGB') for im in imgs]
        if method.startswith('full'):
            ratio = 12 if method == 'full12' else 64
            outs = self._forward_grouped(imgs, ratio)
            result = [{'prediction': normalize_ltb(greedy_ctc(lg, self.chars)),
                       'num_windows': 1, 'stitch_misses': 0} for lg in outs]
        else:
            crops, meta = [], []
            for im in imgs:
                left = len(crops)
                parts = make_chunks(im, LEGACY_CHUNK_AR, LEGACY_OVERLAP)
                crops.extend(p for p, _, _ in parts)
                spans = [(a, b) for _, a, b in parts]
                meta.append((left, len(parts), spans))
            lgs = self._forward_grouped(crops, max_ratio=12)
            result = []
            for left, count, spans in meta:
                crop_lgs = lgs[left:left+count]
                raw = [greedy_ctc(x, self.chars) for x in crop_lgs]
                norm_parts = [normalize_ltb(s) for s in raw]
                if method == 'stitch_v2':
                    text, misses = stitch_v2(norm_parts, spans)
                else:
                    text, misses = normalize_ltb(geometry_ctc_same_crops(crop_lgs, spans, self.chars)), 0
                result.append({'prediction': text, 'num_windows': count,
                               'stitch_misses': misses, 'crop_predictions': norm_parts,
                               'crop_spans': spans})
        return result if return_info else [r['prediction'] for r in result]

    def predict(self, img: Image.Image, method: str='stitch_v2', return_info: bool=False):
        return self.predict_many([img], method, return_info)[0]


def score_predictions(df):
    """df columns: id, gt, pred, optional bucket. Output exact-line acc & CER/S/D/I by bucket."""
    import pandas as pd
    rows = []
    for row in df.itertuples():
        gt = normalize_ltb(row.gt)
        pr = normalize_ltb(row.pred)
        ops = Levenshtein.editops(gt, pr)
        s = sum(o.tag == 'replace' for o in ops)
        d = sum(o.tag == 'delete' for o in ops)
        ins = sum(o.tag == 'insert' for o in ops)
        bucket = '26-35' if len(gt) <= 35 else ('36-55' if len(gt) <= 55 else '56+')
        rows.append({'id': row.id, 'gt': gt, 'pred': pr, 'bucket': bucket,
                     'correct': int(gt == pr), 'L': len(gt), 'S': s, 'D': d, 'I': ins})
    out = pd.DataFrame(rows)
    return out


def metric_table(df):
    """A-Avg only if ALL three LTB buckets are present."""
    import pandas as pd
    rows=[]
    for b in ['26-35', '36-55', '56+', 'ALL']:
        sub = df if b == 'ALL' else df[df.bucket == b]
        if len(sub) == 0:
            continue
        L = int(sub.L.sum())
        rows.append({'bucket': b, 'n': len(sub), 'accuracy': 100*sub.correct.mean(),
                     'CER': 100*(sub.S.sum()+sub.D.sum()+sub.I.sum())/max(L,1),
                     'S%': 100*sub.S.sum()/max(L,1),
                     'D%': 100*sub.D.sum()/max(L,1),
                     'I%': 100*sub.I.sum()/max(L,1)})
    present = set(df.bucket)
    if {'26-35','36-55','56+'} <= present:
        rows.append({'bucket': 'A-Avg', 'n':len(df),
                     'accuracy': float(np.mean([x['accuracy'] for x in rows if x['bucket'] in ('26-35','36-55','56+')]))})
    return pd.DataFrame(rows)


def run_regression_tests():
    # Three tests copied from original LTB; plus frame ownership and CTC blank semantics.
    sp = [(0, 100), (50, 150)]
    assert stitch_v2(['abcdefgh', 'efghijkl'], sp)[0] == 'abcdefghijkl'
    assert stitch_v2(['abcdefgh', 'xxxxyyyy'], [(0, 100), (100, 200)])[0] == 'abcdefghxxxxyyyy'
    im = Image.new('RGB',(170,20),'white')
    cs = make_chunks(im, LEGACY_CHUNK_AR, LEGACY_OVERLAP)
    assert [(a,b) for _,a,b in cs] == [(0,120),(50,170)]
    assert len(make_chunks(Image.new('RGB',(100,20)),LEGACY_CHUNK_AR,LEGACY_OVERLAP)) == 1
    # blank=0; two letters same token separated by blank must survive collapse.
    chars = ['blank','a','b']
    logits = np.array([[0,5,0],[0,5,0],[5,0,0],[0,5,0]],dtype=np.float32)
    assert greedy_ctc(logits,chars) == 'aa'
    one = np.array([[6,0,0],[0,8,0],[6,0,0],[0,0,8]],dtype=np.float32)
    two = np.array([[0,8,0],[6,0,0],[0,0,8],[6,0,0]],dtype=np.float32)
    assert geometry_ctc_same_crops([one,two],[(0,100),(50,150)],chars) == 'ab'
    sizes = [target_size_ltb(100,20), target_size_ltb(300,20), target_size_ltb(1500,20,12)]
    assert sizes == [(160,32),(384,32),(384,32)],sizes
    return True


if __name__ == '__main__':
    run_regression_tests()
    print('PASS: deterministic geometry/stitch/CTC/multi-window tests')