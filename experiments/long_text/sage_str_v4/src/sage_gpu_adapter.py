"""GT-blind GPU candidate adapter for SAGE-STR v2; frozen SVTRv2-T only.

Accepts PIL images only. No method here accepts image id, dataset name, labels,
reference predictions, or ground truth.
"""
from __future__ import annotations
from dataclasses import replace
from time import perf_counter
import numpy as np
from PIL import Image
import longtext_final as LT
import longtext_v3 as V3
import orientation_probe as OP
import density_probe as DP
import unified_router as UNI
from sage_str_v2.selector import Hypothesis, Observation
from sage_str_v2.ctc_evidence import from_logits

class SAGEGPUAdapter:
    def __init__(self,net,chars,device='cuda',batch_size=2):
        self.chars=list(chars)
        self.base=LT.LongTextRecognizer(net,self.chars,device=device,batch_size=batch_size)
        self.dense=DP.Probe(net,self.chars,device=device,batch_size=batch_size)
        self.unified=UNI.UnifiedRecognizer(net,self.chars,device=device,batch_size=batch_size)

    def _crop_pass(self,img,parts,method,factor=1):
        """Return stitched text, hypothesis, and agreement; no labels consumed."""
        if factor==1:
            logits=self.base._forward_grouped([p for p,_,_ in parts],max_ratio=12)
        else:
            tensors=[DP.resized_tensor(p,factor) for p,_,_ in parts]
            logits=self.dense._forward_tensors(tensors)
        spans=[(a,b) for _,a,b in parts]
        events=[V3.ctc_events(lg,self.chars,a,b) for lg,(a,b) in zip(logits,spans)]
        if len(logits)==1:
            text=LT.normalize_ltb(LT.greedy_ctc(logits[0],self.chars))
            agreement=0.
        else:
            text=V3.stitch_evidence(events,spans,use_confidence=False,geometric_fallback=True)
            agreement=OP.overlap_agreement(events,spans)
        original_v1_conf=OP.event_confidence(events)
        # Per-crop CTC feature extraction. CTC frames from overlapping crops are
        # NOT concatenated as if they were a single aligned image.
        features=[from_logits(z,self.chars,method) for z in logits]
        weights=np.array([len(z) for z in logits],dtype=float)
        weights=weights/weights.sum()
        def avg(k):return float(sum(w*getattr(f,k) for w,f in zip(weights,features)))
        nw=LT.target_size_ltb(parts[0][0].width,parts[0][0].height,max_ratio=12)[0]
        extra_ratio=float(max(1.,factor))
        hyp=Hypothesis(method,text,ctc_T=sum(len(z) for z in logits),
           confidence=avg('confidence'),entropy=avg('entropy'),
           char_margin=avg('char_margin'),edge_occupancy=avg('edge_occupancy'),
           overlap_agreement=agreement,n_windows=len(logits),
           distortion=extra_ratio,cost_ms=0.)
        return text,hyp,agreement,original_v1_conf

    def infer_one(self,image):
        img=image.convert('RGB')
        t0=perf_counter()
        orig_logits=self.base._forward_grouped([img],max_ratio=12)[0]
        orig=from_logits(orig_logits,self.chars,'original')
        pred=LT.normalize_ltb(LT.greedy_ctc(orig_logits,self.chars))
        assert orig.normalized==pred
        parts=LT.make_chunks(img,LT.LEGACY_CHUNK_AR,LT.LEGACY_OVERLAP)
        n_orig=len(parts)
        candidates=[]
        v3=pred
        rot_results={}
        dense3=None

        def add_candidate(name,parts,fac=1,rot_key=None):
            nonlocal dense3
            started=perf_counter()
            try:
                text,hyp,agree,original_v1_conf=self._crop_pass(img,parts,name,fac)
            except ValueError as exc:
                if 'Expected an expanded width' in str(exc):return None
                raise
            cost=(perf_counter()-started)*1000
            hyp=replace(hyp,cost_ms=cost)
            if name=='geometry_v3':pass
            if name=='crop3_width3':dense3=text
            if rot_key is not None:
                rot_results[rot_key]={'pred':text,'agreement':agree,
                                      'confidence':original_v1_conf}
            if hyp.normalized!=orig.normalized:candidates.append(hyp)
            return text

        if n_orig>1:
            v3=add_candidate('geometry_v3',parts)
        if img.width<=img.height:
            for angle,method in [('cw','rotate_cw'),('ccw','rotate_ccw')]:
                oriented=OP.rotated(img,angle)
                rotated_parts=LT.make_chunks(oriented,LT.LEGACY_CHUNK_AR,LT.LEGACY_OVERLAP)
                add_candidate(method,rotated_parts,fac=1,rot_key=angle)
            assert set(rot_results)=={'cw','ccw'}
            _,choice=OP.choose_orientation(rot_results['cw'],rot_results['ccw'],'overlap')
            v1=choice['pred'];route='orientation'
        elif n_orig==1:
            # Deliberately include v1's aggressive Dense proposal in the
            # experimental candidate pool, even on short words: learned gate
            # must detect and reject its regressions, not hide them via GT.
            c3=DP.forced_crops(img,3)
            add_candidate('crop3_width3',c3,fac=3)
            v1=dense3;route='dense'
            # MSR-conserving crop is a useful ablation for underread lines.
            # This CTC-only probe does not force final adoption.
            if len(pred)>=8:
                add_candidate('crop3_msr',c3,fac=1)
                add_candidate('crop3_width2',c3,fac=2)
        else:
            v1=v3;route='v3'
        # W2 is inexpensive compared with forced crops; run when the baseline
        # predicts at least 8 chars or exhibits high CTC slot pressure.
        if (len(pred)>=8 or orig.pressure>=.4) and img.width>img.height:
            full=[(img,0,img.width)]
            add_candidate('full_width2',full,fac=2)
        obs=Observation(img.width,img.height,orig,tuple(candidates))
        assert all(c.strategy!='original' for c in obs.alternatives)
        return {'svtrv2':pred,'geometry_v3':v3,'unified_v1':v1,
                'route_v1':route,'observation':obs,
                'candidate_time_ms':(perf_counter()-t0)*1000}

    def infer_many(self,images):
        return [self.infer_one(im) for im in images]
