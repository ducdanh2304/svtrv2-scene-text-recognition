"""Training-free spatial sequence evidence integration (experimental).

No new checkpoint, learned selector, domain router or artificial AR logits.
The original SCOPE output anchors spatial reconstruction; native CTC experts
are evaluated on the same physical regions, not summed frame by frame.
FUSE is the original proposal generator, not an AR sequence likelihood scorer.
"""
from __future__ import annotations
from collections import defaultdict
from difflib import SequenceMatcher
from threading import RLock
from time import perf_counter
import math
import numpy as np
from ..types import Prediction
from .artistic import choose_fuse

# Trace only inside the single-process, serial inference contract. Do not invoke
# other SCOPE predictions concurrently with this backend (flat module imports).
_TRACE_LOCK = RLock()


def logmeanexp(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a) or np.isnan(a).any() or np.isposinf(a).any():
        raise ValueError('Invalid likelihoods')
    return float(np.logaddexp.reduce(a) - math.log(len(a)))


def local_decode(candidates, logits, chars, *, reference):
    """Uniform *sequence* mixture of CTC recognizers; no frame alignment.

    Per-model forward marginal uses SCOPE's existing 37-class projection.
    Search is restricted to nominated strings; this is not exhaustive MAP.
    """
    import alex_ctc as AX
    import alex_c1 as C1
    import longtext_final as LT
    ref = LT.normalize_ltb(reference)
    strings = list(dict.fromkeys([ref] + [LT.normalize_ltb(x) for x in candidates]))
    if not logits or any(np.asarray(z).ndim != 2 or np.asarray(z).shape[1] != len(chars) or len(z)==0 or not np.isfinite(z).all() for z in logits):
        raise ValueError('Invalid CTC observations')
    lp = [np.log(AX.normalized_probs(np.array(z,copy=True), chars)) for z in logits]
    scores = {s: logmeanexp([C1.ctc_forward_logprob(q, s) for q in lp]) for s in strings}
    # Reference first means exact ties retain it; no tuned margin.
    return max(strings, key=scores.__getitem__), scores


def capture_scope(scope, rec, im):
    import scope_e7_fast as FAST
    import orientation_probe as OP
    cap = {}
    with _TRACE_LOCK:
        original, choose = FAST.decode, OP.choose_orientation
        def traced(ref, base, spans, extra, extra_spans, chars, **kw):
            cap.update(base=base, spans=spans, extra=extra, extra_spans=extra_spans)
            return original(ref, base, spans, extra, extra_spans, chars, **kw)
        def orient(*a, **kw):
            angle, winner = choose(*a, **kw)
            cap['angle'] = angle
            return angle, winner
        FAST.decode, OP.choose_orientation = traced, orient
        try:
            result = scope.predict_image(im, rec)
        finally:
            FAST.decode, OP.choose_orientation = original, choose
    if result['safe_scope_active']:
        if 'base' not in cap:
            raise RuntimeError('Active SCOPE did not expose spatial evidence')
        work = OP.rotated(im, cap['angle']) if result['route'] == 'orientation' else im
        cap['parts'] = [work.crop((int(a), 0, int(b), work.height)) for a, b in cap['spans']]
    else:
        z = rec.base._forward_grouped([im], max_ratio=12)[0]
        cap.update(base=[z], spans=[(0, im.width)], extra=[], extra_spans=[], parts=[im])
    return result, cap


def raw_ctc(model, net, x, chars):
    import torch
    seen = []
    hook = net.rctc.fc.register_forward_hook(lambda m, a, o: seen.append(o.detach()))
    try:
        with torch.inference_mode():
            model(x)
            raw = net(x)
    finally:
        hook.remove()
    if len(seen) != 2 or not torch.allclose(seen[0], raw, rtol=3e-4, atol=3e-4):
        raise RuntimeError('Raw CTC parity failed')
    if raw.ndim != 3 or raw.shape[-1] != len(chars):
        raise RuntimeError('Raw CTC vocabulary mismatch')


def r3_views(r3, im, chars):
    """Preserve original R3 TTA winner, retain only original-orientation logits."""
    import longtext_final as LT
    d = r3.model.decoder
    head = d if hasattr(d, 'w_atten_block') else d.ctc_decoder
    seen, calls = [], []
    original = r3._predict_one
    def tapped(img, ratio_override=None):
        before = len(seen)
        text, score = original(img, ratio_override)
        if len(seen) != before + 1:
            raise RuntimeError('Unexpected R3 CTC head calls')
        z = seen[-1][0]
        if z.ndim != 2 or z.shape[1] != len(chars) or not np.isfinite(z).all():
            raise RuntimeError('Invalid R3 CTC logits')
        if LT.normalize_ltb(LT.greedy_ctc(z, chars)) != LT.normalize_ltb(text):
            raise RuntimeError('R3 CTC extraction parity failed')
        calls.append((img, z, text, float(score)))
        return text, score
    handle = head.fc.register_forward_hook(lambda m,a,o: seen.append(o.detach().float().cpu().numpy()))
    r3._predict_one = tapped
    try:
        pred = r3.predict(im)
    finally:
        r3._predict_one = original
        handle.remove()
    if not calls:
        raise RuntimeError('No R3 observations')
    # R3 original predict() uses the same PIL object for ratio variants.
    unrotated = [c for c in calls if c[0] is calls[0][0]]
    best = max(unrotated, key=lambda c:c[3])
    return pred, best[1]


def spatial_decode(reference, cap, chars, s2, r3, local_texts, fuse_texts):
    """Extend original SCOPE E7 evidence pool, reuse exact original solver.

    No additional spatial groups for models observing the same physical crop.
    The original solver's two-witness/penalty/edit-budget constraints remain.
    Text-only AR proposals are localized by existing reference geometry, then
    verified using real CTC observations; they never become fabricated logits.
    """
    import alex_ctc as AX
    import alex_e7 as E7
    import scope_e7_fast as FAST
    import longtext_final as LT
    reference = LT.normalize_ltb(reference)
    events = E7.mapped_events(reference, AX.reference_events(cap['base'], cap['spans'], chars))
    if not events or not reference:
        return reference, {'reason':'empty_reference_or_geometry'}
    views = []
    group = {tuple(sp):i for i,sp in enumerate(cap['spans'])}
    for family, zs, spans in [('native',cap['base'],cap['spans']),('scale',cap['extra'],cap['extra_spans']),('motion',s2,cap['spans']),('artistic',r3,cap['spans'])]:
        for k,(z,sp) in enumerate(zip(zs,spans)):
            views.append(FAST._make_view(z,sp,group[tuple(sp)],f'{family}:{k}',chars,events))
    proposals,witnesses,nb = E7.generate_proposals(reference,events,views,chars)
    by_key={(p.start,p.end,p.new):p for p in proposals}
    for i,sp in enumerate(cap['spans']):
        a,b=sp
        ids=[j for j,e in enumerate(events) if a<=e.x<=b]
        if not ids or ids!=list(range(ids[0],ids[-1]+1)):
            continue
        local=''.join(reference[j] for j in ids)
        for name,txt in [('joint',local_texts[i]),('fuse',fuse_texts[i])]:
            text=LT.normalize_ltb(txt)
            if not text:
                continue
            # No AR character coordinates are invented. Keep only edits with
            # text anchors on both sides (same constraint as prior composition).
            ops=SequenceMatcher(None,local,text,autojunk=False).get_opcodes()
            for j,op in enumerate(ops):
                if not 0<j<len(ops)-1 or ops[j-1][0]!='equal' or ops[j+1][0]!='equal':
                    continue
                p=E7.new_edit(op,text,events,ids,[],a,b)
                if p is None:
                    continue
                key=(p.start,p.end,p.new)
                by_key.setdefault(key,p)
                witnesses[key].add((i,name))
    return FAST._solve('joint',reference,list(by_key.values()),witnesses,views,nb,{})


class JointRecognizer:
    def __init__(self, system):
        import torch
        self.torch=torch
        self.scope=system._backend('scope')
        self.chars=list(self.scope.recognizer.chars)
        import longtext_final as LT
        from openrec.postprocess.ctc_postprocess import CTCLabelDecode
        from ..paths import require_dir
        self.motion=system._backend('s2_aug')
        self.r3=system._backend('r3_tta')
        self.w=system._backend('waterec')
        self.delta=float(system.config.get('artistic',{}).get('fuse_delta',.15))
        if list(self.motion.pipe.decoder.character)!=self.chars:
            raise RuntimeError('S2 vocabulary order differs from SCOPE')
        r3chars=CTCLabelDecode(character_dict_path=str(require_dir(system.assets['openocr_repo'],'openocr_repo')/'tools/utils/EN_symbol_dict.txt'),use_space_char=False).character
        if list(r3chars)!=self.chars:
            raise RuntimeError('R3 vocabulary order differs from SCOPE')
        self.s2net=LT.CTCNet(self.motion.pipe.model).eval().requires_grad_(False)
        raw_ctc(self.motion.pipe.model,self.s2net,torch.randn(1,3,32,192,device=self.motion.pipe.device),self.chars)

    def predict_pair(self, image):
        import longtext_final as LT
        from alex_e5 import beam_events
        import alex_ctc as AX
        im=image.convert('RGB'); start=perf_counter()
        with self.torch.inference_mode(), _TRACE_LOCK:
            result,cap=capture_scope(self.scope.scope,self.scope.recognizer,im)
            ref=result['safe_scope']; s2=[]; r3=[]; local=[]; fuses=[]; audit=[]
            for crop,z0,sp in zip(cap['parts'],cap['base'],cap['spans']):
                z2=self.s2net(self.motion.pipe.preprocess(crop))[0].detach().float().cpu().numpy()
                rp,z3=r3_views(self.r3,crop,self.chars)
                wp=self.w.predict(crop)
                f=choose_fuse(rp,wp,self.delta)
                zs=[z0,z2,z3]
                if any(z.ndim!=2 or z.shape[1]!=len(self.chars) or not np.isfinite(z).all() for z in zs):
                    raise RuntimeError('Invalid crop CTC observation')
                candidates=[f.text]
                for z in zs:
                    candidates.append(LT.greedy_ctc(z,self.chars))
                    candidates.extend(t for t,_,_ in beam_events(np.log(AX.normalized_probs(z,self.chars)),sp,beam_width=6,topk=5,nbest=3))
                anchor=ref if len(cap['spans'])==1 else LT.greedy_ctc(z0,self.chars)
                text,scores=local_decode(candidates,zs,self.chars,reference=anchor)
                s2.append(z2);r3.append(z3);local.append(text);fuses.append(f.text)
                audit.append({'span':list(sp),'joint':text,'fuse':f.text,'waterec':wp.text,'r3':rp.text,'ctc_candidates':len(scores)})
            if len(cap['spans'])==1:
                output=local[0]; info={'joint_spatial_edits':0}
            else:
                output,info=spatial_decode(ref,cap,self.chars,s2,r3,local,fuses)
            if str(self.scope.device).startswith('cuda'):
                self.torch.cuda.synchronize()
        base=Prediction(text=result['original_svtrv2_full12'],method='official_svtrv2_b0')
        ours=Prediction(text=output,method='SCOPE_JOINT_FROZEN',route='joint',latency_ms=(perf_counter()-start)*1000,
                        details={'scope_reference':ref,'scope_active':result['safe_scope_active'],'views':audit,'reconstruction':info,'training':False,'ar_likelihood':False})
        return base,ours

    def close(self):
        # Backends are owned by STRSystem; do not close shared WATERec twice.
        pass
