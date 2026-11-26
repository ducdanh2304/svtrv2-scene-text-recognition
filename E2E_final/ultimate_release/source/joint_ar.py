"""Training-free AR extension of Joint Frozen; matched ablation in one pass."""
from __future__ import annotations
import numpy as np
from time import perf_counter
from ..types import Prediction
from .artistic import choose_fuse
from .joint import capture_scope, raw_ctc, r3_views, spatial_decode, local_decode, _TRACE_LOCK
from .ar_scoring import mixture

class JointARRecognizer:
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
        from .waterec_ar import WATERecARBackend
        self.w=WATERecARBackend(waterec_repo=system.assets['waterec_repo'],waterec_checkpoint=system.assets['waterec_checkpoint'],waterec_config=system.assets['waterec_config'],device='cuda' if str(system.device).startswith('cuda') else 'cpu')
        # Lifecycle owned by system, separate from stable WATERec backend.
        system.backends['waterec_ar']=self.w
        self.delta=float(system.config.get('artistic',{}).get('fuse_delta',.15))
        if list(self.motion.pipe.decoder.character)!=self.chars:
            raise RuntimeError('S2 vocabulary order differs from SCOPE')
        r3chars=CTCLabelDecode(character_dict_path=str(require_dir(system.assets['openocr_repo'],'openocr_repo')/'tools/utils/EN_symbol_dict.txt'),use_space_char=False).character
        if list(r3chars)!=self.chars:
            raise RuntimeError('R3 vocabulary order differs from SCOPE')
        self.s2net=LT.CTCNet(self.motion.pipe.model).eval().requires_grad_(False)
        raw_ctc(self.motion.pipe.model,self.s2net,torch.randn(1,3,32,192,device=self.motion.pipe.device),self.chars)

    def predict_pair(self,image):
        import longtext_final as LT
        from alex_e5 import beam_events
        import alex_ctc as AX
        im=image.convert('RGB'); start=perf_counter()
        with self.torch.inference_mode(),_TRACE_LOCK:
            result,cap=capture_scope(self.scope.scope,self.scope.recognizer,im)
            ref=result['safe_scope']; s2=[];r3=[];old_local=[];raw_local=[];ar_local=[];fuses=[];audit=[]
            for crop,z0,sp in zip(cap['parts'],cap['base'],cap['spans']):
                z2=self.s2net(self.motion.pipe.preprocess(crop))[0].detach().float().cpu().numpy()
                rp,z3=r3_views(self.r3,crop,self.chars)
                wp=self.w.predict(crop);f=choose_fuse(rp,wp,self.delta);zs=[z0,z2,z3]
                candidates=[f.text]
                for z in zs:
                    candidates.append(LT.greedy_ctc(z,self.chars))
                    candidates.extend(t for t,_,_ in beam_events(np.log(AX.normalized_probs(z,self.chars)),sp,beam_width=6,topk=5,nbest=3))
                anchor=ref if len(cap['spans'])==1 else LT.greedy_ctc(z0,self.chars)
                old,_=local_decode(candidates,zs,self.chars,reference=anchor)
                raw_strings=list(dict.fromkeys([anchor]+candidates+[rp.text,wp.text]))
                raw,_,_=mixture(raw_strings,zs,self.chars,normalizer=LT.normalize_ltb)
                response=self.w.score(raw_strings); ar_scores=response['scores']
                # If AR cannot represent all lengths, keep the CTC-only control for
                # that crop. Unsupported characters have zero exact raw AR mass.
                ar,_,_=mixture(raw_strings,zs,self.chars,ar_scores,normalizer=LT.normalize_ltb)
                s2.append(z2);r3.append(z3);old_local.append(old);raw_local.append(raw);ar_local.append(ar);fuses.append(f.text)
                audit.append(dict(span=list(sp),old=old,raw_ctc=raw,ar_joint=ar,waterec=wp.text,fuse=f.text,r3=rp.text,
                                  ar_used=ar_scores is not None,ar_reason=response['reason'],prefix_preflight=response['prefix_preflight'],
                                  ar_supported_len=response['supported_len'],ar_projection=response['projection'],candidate_count=len(raw_strings),ar_scores=ar_scores))
            if len(cap['spans'])==1:
                old,raw,ar=old_local[0],raw_local[0],ar_local[0];info={'joint_spatial_edits':0}
            else:
                old,_=spatial_decode(ref,cap,self.chars,s2,r3,old_local,fuses)
                raw,_=spatial_decode(ref,cap,self.chars,s2,r3,raw_local,fuses)
                ar,info=spatial_decode(ref,cap,self.chars,s2,r3,ar_local,fuses)
            if str(self.scope.device).startswith('cuda'):self.torch.cuda.synchronize()
        base=Prediction(text=result['original_svtrv2_full12'],method='official_svtrv2_b0')
        ours=Prediction(text=ar,method='SCOPE_JOINT_AR',route='joint_ar',latency_ms=(perf_counter()-start)*1000,
                        details=dict(scope_reference=ref,scope_active=result['safe_scope_active'],views=audit,reconstruction=info,
                                     joint_frozen_replay=old,raw_ctc=raw,training=False,ar_likelihood=True))
        return base,ours

    def close(self):pass
