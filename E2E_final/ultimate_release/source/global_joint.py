"""Full-path candidate preservation plus spatial CTC–AR sequence verification."""
from __future__ import annotations
from time import perf_counter
import numpy as np
from ..types import Prediction
from .joint_ar import JointARRecognizer
from .joint import capture_scope, _TRACE_LOCK, spatial_decode
from .artistic import choose_fuse
from .ar_scoring import mixture
from .global_probability import norm,project,coverage_weights,raw_masses,ar_masses,prob,combine_views,average,choose

def r3_observations(r3,im,chars):
    """Capture all real TTA outputs; use native winner for full-sequence
    likelihood, and best unrotated logits only for geometric stitching."""
    import longtext_final as LT
    d=r3.model.decoder; head=d if hasattr(d,'w_atten_block') else d.ctc_decoder
    seen=[];calls=[];original=r3._predict_one
    def tapped(img,ratio_override=None):
        n=len(seen); text,score=original(img,ratio_override)
        if len(seen)!=n+1:raise RuntimeError('R3 CTC trace mismatch')
        z=seen[-1][0]
        if z.ndim!=2 or z.shape[1]!=len(chars) or not np.isfinite(z).all():raise RuntimeError('R3 invalid logits')
        if LT.normalize_ltb(LT.greedy_ctc(z,chars))!=LT.normalize_ltb(text):raise RuntimeError('R3 native decoding mismatch')
        calls.append((img,z,text,float(score)));return text,score
    hook=head.fc.register_forward_hook(lambda m,a,o:seen.append(o.detach().float().cpu().numpy()))
    r3._predict_one=tapped
    try:pred=r3.predict(im)
    finally:r3._predict_one=original;hook.remove()
    best=max(calls,key=lambda c:c[3]); spatial=max([c for c in calls if c[0] is calls[0][0]],key=lambda c:c[3])
    if pred.text!=best[2] or float(pred.confidence)!=best[3]:raise RuntimeError('R3 TTA winner mismatch')
    return pred,best[1],spatial[1],[c[2] for c in calls]

class GlobalJoint(JointARRecognizer):
    def __init__(self,system):
        super().__init__(system)
        import unified_router as UNI
        self.s2rec=UNI.UnifiedRecognizer(self.s2net,self.chars,device=system.device,batch_size=2)

    def predict_evidence(self,image):
        import longtext_final as LT
        import longtext_v3 as V3
        import alex_ctc as AX
        import alex_e7 as E7
        from alex_e5 import beam_events
        im=image.convert('RGB'); start=perf_counter()
        with self.torch.inference_mode(),_TRACE_LOCK:
            result,cap=capture_scope(self.scope.scope,self.scope.recognizer,im)
            ref=result['safe_scope']; b0=result['original_svtrv2_full12']; spans=cap['spans']
            motion=self.motion.predict(im)
            scope_s2=self.scope.scope.predict_image(im,self.s2rec)['safe_scope']
            full_z0=self.scope.recognizer.base._forward_grouped([im],max_ratio=12)[0]
            full_z2=self.s2net(self.motion.pipe.preprocess(im))[0].detach().float().cpu().numpy()
            full_rp,full_z3,full_r3_sp,full_r3_texts=r3_observations(self.r3,im,self.chars)
            full_wp=self.w.predict(im); full_fuse=choose_fuse(full_rp,full_wp,self.delta)
            full_texts=[b0,ref,motion.text,full_rp.text,full_wp.text,full_fuse.text,scope_s2]
            pool_raw=list(full_texts)+full_r3_texts; parts=[]; s2=[]; r3=[]; local=[];fuses=[]
            for crop,z0,sp in zip(cap['parts'],cap['base'],spans):
                same=not cap.get('angle') and tuple(sp)==(0,im.width) and crop.size==im.size and np.array_equal(np.asarray(crop),np.asarray(im))
                if same:
                    z2,rp,z3,z3sp,wp,rt=full_z2,full_rp,full_z3,full_r3_sp,full_wp,full_r3_texts
                    # full image remains the most recent observation only if it is
                    # the first view; predict before every score to enforce identity.
                else:
                    z2=self.s2net(self.motion.pipe.preprocess(crop))[0].detach().float().cpu().numpy()
                    rp,z3,z3sp,rt=r3_observations(self.r3,crop,self.chars);wp=self.w.predict(crop)
                f=choose_fuse(rp,wp,self.delta);zs=[z0,z2,z3]
                raw=[f.text,rp.text,wp.text]+rt
                for z in zs:
                    raw.append(LT.greedy_ctc(z,self.chars))
                    raw.extend(t for t,_,_ in beam_events(np.log(AX.normalized_probs(z,self.chars)),sp,beam_width=6,topk=5,nbest=3))
                raw=list(dict.fromkeys(raw+[ref if len(spans)==1 else LT.greedy_ctc(z0,self.chars)]))
                self.w.predict(crop); ar=self.w.score(raw)['scores']
                pred,_,_=mixture(raw,zs,self.chars,ar,normalizer=LT.normalize_ltb)
                parts.append((crop,zs,raw));s2.append(z2);r3.append(z3sp);local.append(pred);fuses.append(f.text)
                if len(spans)==1:pool_raw.extend(raw+[pred])
            if len(spans)>1:
                def stitch(zs):return V3.stitch_evidence([V3.ctc_events(z,self.chars,*sp) for z,sp in zip(zs,spans)],spans,use_confidence=False,geometric_fallback=True)
                pool_raw.extend([stitch(s2),stitch(r3),LT.stitch_v2([norm(s) for s in fuses],spans)[0],LT.stitch_v2(local,spans)[0]])
                joint,_=spatial_decode(ref,cap,self.chars,s2,r3,local,fuses)
                pool_raw.append(joint)
            candidates=list(dict.fromkeys(norm(s) for s in pool_raw))
            # Native full paths stay in the set regardless of spatial projection.
            full_zs=[full_z0,full_z2,full_z3]
            full=[prob(raw_masses(candidates,z,self.chars,pool_raw)) for z in full_zs]
            a=ar_masses(candidates,self.w,im,pool_raw)
            full.append(prob(a) if a is not None else None)
            spatial=[None]*4; spatial_used=False
            events=E7.mapped_events(norm(ref),AX.reference_events(cap['base'],spans,self.chars)) if norm(ref) else []
            # One identical view must not be counted as independent evidence.
            if events and (len(spans)>1 or cap.get('angle') or tuple(spans[0])!=(0,im.width) or cap['extra'] or not np.array_equal(cap['base'][0],full_z0)):
                coords=[float(e.x) for e in events];weights=coverage_weights(spans,coords)
                tables=[[] for _ in range(4)]
                for (crop,zs,observed),sp in zip(parts,spans):
                    local_c=candidates if len(spans)==1 else [project(norm(ref),s,coords,sp) for s in candidates]
                    for k,z in enumerate(zs):
                        masses=[raw_masses(local_c,z,self.chars,observed)]
                        if k==0:
                            masses.extend(raw_masses(local_c,q,self.chars,observed) for q,ssp in zip(cap['extra'],cap['extra_spans']) if tuple(ssp)==tuple(sp))
                        tables[k].append(np.logaddexp.reduce(np.stack(masses),axis=0)-np.log(len(masses)))
                    tables[3].append(ar_masses(local_c,self.w,crop,observed))
                spatial=[combine_views(v,weights) for v in tables];spatial_used=any(p is not None for p in spatial)
            b,s,r,w=[average(f,p) for f,p in zip(full,spatial)]
            # R3 and WATERec are the two constituents of the original Artistic
            # solution; do not give that solution two independent prior votes.
            art=average(r,w); table=[b,s,art]
            if not any(x is not None for x in table):raise RuntimeError('No full/spatial source supports the hypothesis set')
            uniform,mbr,_=choose(candidates,table)
            if str(self.scope.device).startswith('cuda'):self.torch.cuda.synchronize()
        return dict(candidates=candidates,raw_candidates=pool_raw,table=[p.tolist() if p is not None else None for p in table],
                    full_table=[p.tolist() if p is not None else None for p in full],spatial_table=[p.tolist() if p is not None else None for p in spatial],
                    pred=dict(b0=b0,scope=ref,motion=motion.text,fuse=full_fuse.text,scope_s2=scope_s2,global_uniform=uniform,global_mbr=mbr),
                    spatial_used=spatial_used,crops=len(spans),latency_ms=(perf_counter()-start)*1000)

    def predict_pair(self,image,calibration=None):
        row=self.predict_evidence(image); cal=calibration or {}
        text,mbr,p=choose(row['candidates'],row['table'],cal.get('weights'),cal.get('temperatures'))
        return Prediction(text=row['pred']['b0'],method='official_svtrv2_b0'),Prediction(text=text,method='GLOBAL_CTC_AR',route='global_joint',latency_ms=row['latency_ms'],details={**row,'posterior':p,'mbr':mbr})
