"""Research candidate: Stitch v3 evidence-aware post-processing for SVTRv2 CTC.

NOT a validated improvement. Same *original-image* crops, same model and same
inference passes as Stitch v2. We change only the text-merge stage.

Frozen original Stitch v2 is imported from longtext_final.py. Three ablations:
    v3_conf: preserve v2 overlap selection; on *aligned single-character substitutions*
             choose the higher CTC event confidence rather than v2's fixed midpoint.
    v3_geom: preserve v2 successful matches; if v2 rejects a match, select by
             original-image geometry at the midpoint between the crop centres.
    v3_both: both rules together.

No new tuned hyperparameters. Confidence is a diagnostic proxy (not calibrated),
frame-to-pixel locations are approximate, and empirical gains are NOT guaranteed.
This is a coursework research prototype, not a new SOTA or publication claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Sequence
import time

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

from longtext_final import (
    LongTextRecognizer, make_chunks, stitch_v2, normalize_ltb,
    LEGACY_CHUNK_AR, LEGACY_OVERLAP, greedy_ctc, geometry_ctc_same_crops,
)


@dataclass(frozen=True)
class CharEvidence:
    char: str
    confidence: float
    x: float


def ctc_events(logits: np.ndarray, chars: Sequence[str], x0: int, x1: int) -> list[CharEvidence]:
    """Convert greedy CTC character runs to confidence and approximate global x.

    For each nonblank run, its symbol probability is measured at the most
    confident timestep for that symbol; its position is that timestep's centre.
    This is not forced alignment and the position may differ from glyph centre.
    """
    v = np.asarray(logits, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != len(chars) or v.shape[0] < 1:
        raise ValueError(f'Expected nonempty [T,C] logits, got {v.shape}')
    z = v - np.max(v, axis=1, keepdims=True)
    expz = np.exp(z)
    p = expz / np.sum(expz, axis=1, keepdims=True)
    ids = np.argmax(v, axis=1)
    events = []
    t = 0
    while t < len(ids):
        j = t + 1
        while j < len(ids) and ids[j] == ids[t]:
            j += 1
        cls = int(ids[t])
        if cls != 0:
            label = normalize_ltb(str(chars[cls]))
            if len(label) > 1:
                raise ValueError(f'CTC label {chars[cls]!r} normalizes to multiple characters')
            if label:
                peak = t + int(np.argmax(p[t:j, cls]))
                x = float(x0 + (peak + 0.5) * ((x1 - x0) / len(ids)))
                events.append(CharEvidence(label, float(p[peak, cls]), x))
        t = j
    plain = normalize_ltb(greedy_ctc(v, list(chars)))
    if ''.join(e.char for e in events) != plain:
        raise AssertionError('Character-level evidence disagrees with greedy CTC output')
    return events


def _v2_match_info(out: str, b: str, prev_span, next_span, tol=0.34):
    """Exact legacy v2 scoring and acceptance; no change to existing thresholds."""
    wb = max(next_span[1] - next_span[0], 1)
    exp = max(0, prev_span[1] - next_span[0]) / wb * len(b)
    if exp < 0.5 or not b or not out:
        return {'mode': 'concat', 'exp': exp}
    lmax = min(len(out), len(b), int(round(exp * 1.6)) + 2)
    best = None
    for L in range(1, lmax + 1):
        d = Levenshtein.distance(out[-L:], b[:L])
        sc = L - 2*d - 0.5*abs(L-exp)
        if best is None or sc > best[0]:
            best = (sc, L, d)
    if best is None:
        return {'mode':'concat','exp':exp}
    sc,L,d = best
    return {
        'mode': ('reject' if sc<=0 or d>tol*L else 'accept'),
        'L': L, 'd': d, 'sc': sc, 'exp': exp,
    }


def stitch_evidence(
    chunks: Sequence[Sequence[CharEvidence]],
    spans: Sequence[tuple[int, int]],
    *,
    use_confidence: bool=True,
    geometric_fallback: bool=True,
    return_info: bool=False,
):
    """Stitch v3 candidate. Only alters v2 on disagreement or rejected overlap.

    No extra inference, no new learned weights, no threshold fitted to test.
    Comparison against raw v2 is essential; only adopt if robustly better.
    """
    if not chunks or len(chunks) != len(spans):
        raise ValueError('Need nonempty chunks and one span per chunk')
    spans = [tuple(map(int,s)) for s in spans]
    out = list(chunks[0])
    misses = 0
    changes = 0
    geom_misses = 0
    ambiguous = 0
    for j in range(1, len(chunks)):
        nxt = list(chunks[j])
        a = ''.join(e.char for e in out)
        b = ''.join(e.char for e in nxt)
        match = _v2_match_info(a,b,spans[j-1],spans[j])
        mode = match['mode']
        if mode == 'concat':
            out += nxt
            continue
        L = match['L']
        if mode == 'reject':
            misses += 1
            if geometric_fallback:
                # Midpoint between crop centres; lies inside their overlap for
                # equal-size windows. No predicted labels or GT used to route.
                prev_center = (spans[j-1][0] + spans[j-1][1]) / 2
                next_center = (spans[j][0] + spans[j][1]) / 2
                cut_x = (prev_center + next_center) / 2
                out = [e for e in out if e.x < cut_x] + [e for e in nxt if e.x >= cut_x]
                geom_misses += 1
            else:
                strip = min(int(round(match['exp'])),len(nxt))
                out += nxt[strip:]
            continue
        # This was an accepted legacy V2 match. Preserve it unless competing
        # CTC events are positionally aligned 1:1 with a substitution.
        h = L // 2
        left = out[-L:]
        right = nxt[:L]
        if use_confidence and match['d']:
            ops = Levenshtein.editops(''.join(x.char for x in left),
                                      ''.join(x.char for x in right))
            # Only same-index substitutions: no artificial re-indexing of
            # deletion/insertion or repeated-token alignments.
            safe = all(op.tag=='replace' and op.src_pos==op.dest_pos for op in ops)
            if safe:
                consensus = []
                for i in range(L):
                    aa, bb = left[i],right[i]
                    default = aa if i < h else bb
                    if aa.char == bb.char:
                        consensus.append(default)
                    elif aa.confidence > bb.confidence:
                        consensus.append(aa)
                        changes += int(default.char != aa.char)
                    elif bb.confidence > aa.confidence:
                        consensus.append(bb)
                        changes += int(default.char != bb.char)
                    else:
                        consensus.append(default)
                out = out[:-L] + consensus + nxt[L:]
                continue
            ambiguous += 1
        out = out[:-L+h] + nxt[h:]
    text = ''.join(e.char for e in out)
    if return_info:
        return {'prediction':text, 'stitch_misses':misses,
                'confidence_char_flips':changes, 'geometry_fallbacks':geom_misses,
                'unalignable_overlap':ambiguous}
    return text


class EvidenceLongTextRecognizer(LongTextRecognizer):
    """Reuses original crop and GPU logits for all v2/v3/geometry comparisons."""
    CROP_METHODS = ('stitch_v2','v3_conf','v3_geom','v3_both','geom_ctc')
    METHODS = LongTextRecognizer.METHODS + tuple(x for x in CROP_METHODS if x not in LongTextRecognizer.METHODS)

    def predict_all_cropped(self, imgs: list[Image.Image], return_info=False):
        imgs = [im.convert('RGB') for im in imgs]
        all_crops=[]; meta=[]
        for im in imgs:
            i = len(all_crops)
            parts=make_chunks(im,LEGACY_CHUNK_AR,LEGACY_OVERLAP)
            spans=[(a,b) for _,a,b in parts]
            all_crops.extend(p for p,_,_ in parts)
            meta.append((i,len(parts),spans))
        t0=time.perf_counter()
        all_lgs=self._forward_grouped(all_crops, max_ratio=12) if all_crops else []
        forward_seconds=time.perf_counter()-t0
        res={m:[] for m in self.CROP_METHODS}
        cpu_seconds=defaultdict(float)
        evidence_seconds=0.0
        for start,n,spans in meta:
            lgs=all_lgs[start:start+n]
            t1=time.perf_counter()
            evs=[ctc_events(lg,self.chars,a,b) for lg,(a,b) in zip(lgs,spans)]
            parts=[''.join(e.char for e in ev) for ev in evs]
            evidence_seconds+=time.perf_counter()-t1
            t1=time.perf_counter()
            basic,miss=stitch_v2(parts,spans)
            result={'stitch_v2':{'prediction':basic,'stitch_misses':miss}}
            cpu_seconds['stitch_v2']+=time.perf_counter()-t1
            for method,cc,gf in [('v3_conf',True,False),('v3_geom',False,True),('v3_both',True,True)]:
                t1=time.perf_counter()
                result[method]=stitch_evidence(evs,spans,use_confidence=cc,
                                              geometric_fallback=gf,return_info=True)
                cpu_seconds[method]+=time.perf_counter()-t1
            t1=time.perf_counter()
            result['geom_ctc']={'prediction':normalize_ltb(geometry_ctc_same_crops(lgs,spans,self.chars)),
                                'stitch_misses':0}
            cpu_seconds['geom_ctc']+=time.perf_counter()-t1
            for method in self.CROP_METHODS:
                d=dict(result[method]);d['num_windows']=n
                if return_info:
                    d['crop_predictions']=parts
                    d['crop_spans']=spans
                res[method].append(d)
            # Core parity: method with neither intervention is exactly v2.
            if stitch_evidence(evs,spans,use_confidence=False,geometric_fallback=False) != basic:
                raise AssertionError('Stitch v3 core does not reproduce legacy v2')
        return res, {'forward_seconds':forward_seconds,'evidence_seconds':evidence_seconds,
                     'cpu_seconds':dict(cpu_seconds),
                     'crops':len(all_crops), 'images':len(imgs)}

    def predict_many(self,imgs,method='stitch_v2',return_info=False):
        if method in self.CROP_METHODS:
            res,_ = self.predict_all_cropped(imgs,return_info)
            return res[method] if return_info else [x['prediction'] for x in res[method]]
        return super().predict_many(imgs,method,return_info)


def run_regression_tests_v3():
    from longtext_final import run_regression_tests
    assert run_regression_tests()
    chars=['blank','a','b','c','d']
    def logits(seq):
        ans=[]
        for idx in seq:
            row=np.full(len(chars), -8.,dtype=float)
            row[idx]=5.
            ans.append(row)
        return np.asarray(ans)
    lg=logits([1,1,0,1,2,0,2])
    e=ctc_events(lg,chars,10,80)
    assert ''.join(x.char for x in e)=='aabb' and all(10<=x.x<=80 for x in e)
    # Exact agreement is always preserved, no matter which ablation is on.
    p=['abcdef','cdefgh','efghij']
    spans=[(0,100),(50,150),(100,200)]
    evs=[[CharEvidence(c,.9,x0+(i+.5)*100/len(s)) for i,c in enumerate(s)]
          for s,(x0,x1) in zip(p,spans)]
    base=stitch_v2(p,spans)[0]
    for conf in (False,True):
        for geom in (False,True):
            ans=stitch_evidence(evs,spans,use_confidence=conf,geometric_fallback=geom)
            assert ans == base,(conf,geom,ans,base)
    # Exhaustively verify parity with original v2 for all random characters
    # when both interventions are disabled.
    rng=np.random.default_rng(14)
    alphabet='abcde'
    for i in range(200):
        k=int(rng.integers(1,7))
        ss=[''.join(rng.choice(list(alphabet),size=int(rng.integers(0,18)))) for j in range(k)]
        sp=[(j*50,j*50+100) for j in range(k)]
        evs=[[CharEvidence(c,.7,sp[j][0]+(z+.5)*100/max(len(s),1)) for z,c in enumerate(s)]
             for j,s in enumerate(ss)]
        a=stitch_v2(ss,sp)[0]
        b=stitch_evidence(evs,sp,use_confidence=False,geometric_fallback=False)
        assert a==b,(ss,sp,a,b)
    # Repeated letters remain separate events when separated by blank.
    assert ctc_events(logits([1,0,1]),chars,0,30)[0].char == 'a'
    assert ''.join(x.char for x in ctc_events(logits([1,0,1]),chars,0,30))=='aa'
    # Detect a real confidence substitution correction in an accepted overlap.
    a=list('abcxef');b=list('cdefgh');sp=[(0,100),(50,150)]
    ae=[CharEvidence(c, .02 if c=='x' else .8, (j+.5)*100/len(a)) for j,c in enumerate(a)]
    be=[CharEvidence(c, .99 if c=='d' else .8, 50+(j+.5)*100/len(b)) for j,c in enumerate(b)]
    base=stitch_v2([''.join(a),''.join(b)],sp)[0]
    newer=stitch_evidence([ae,be],sp,use_confidence=True,geometric_fallback=False)
    assert base!=newer and 'd' in newer,(base,newer)
    return {'passed':True,'toy_correction':{'v2':base,'v3_conf':newer},'random_parity_cases':200}


if __name__ == '__main__':
    print(run_regression_tests_v3())