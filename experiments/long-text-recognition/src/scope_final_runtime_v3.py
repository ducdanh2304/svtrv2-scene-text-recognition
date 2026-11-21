"""One-pass matched inference and CUDA-synchronised stage timing for SCOPE final.

Matched same SVTRv2-T checkpoint:
  SVTRv2-T raw full12 single-view CTC greedy (actual model pass),
  Geometry v3, Unified, ALEX C0/C1, SCOPE native and SCOPE cross-scale.
The SCOPE fast decoder reuses crop logits and has unchanged evidence rules.
"""
from __future__ import annotations
import time
from PIL import Image
import torch
import longtext_final as LT
import longtext_v3 as V3
import orientation_probe as OP
import density_probe as DP
import unified_router as UNI
import alex_ctc as AX
import alex_c1 as C1
import alex_e7 as E7
import scope_e7_fast as FAST


def _sync(device):
    if 'cuda' in str(device) and torch.cuda.is_available():torch.cuda.synchronize()


def predict_image(image,recognizer,*,run_original=True,check_legacy=False,skip_structurally_impossible=True,native_logits_override=None,native_forward_ms_override=0.,dense_logits_override=None,dense_forward_ms_override=0.,orientation_logits_override=None,orientation_forward_ms_override=0.):
    """No GT, image ID, or archived prediction enters inference."""
    im=image.convert('RGB');chars=recognizer.chars;device=recognizer.base.device
    _sync(device);tic=time.perf_counter();st={}
    def timed(name,fn):
        _sync(device);a=time.perf_counter();value=fn();_sync(device)
        st[name]=st.get(name,0.)+1000*(time.perf_counter()-a)
        return value
    parts=LT.make_chunks(im,LT.LEGACY_CHUNK_AR,LT.LEGACY_OVERLAP)
    native_spans=[(int(a),int(b)) for _,a,b in parts]
    native_logits=(timed('main_forward_ms',lambda:recognizer.base._forward_grouped([p for p,_,_ in parts],max_ratio=12)) if native_logits_override is None else native_logits_override)
    if native_logits_override is not None:
        assert len(native_logits)==len(parts)==1, 'Only native single-crop cache supported'
        st['main_forward_ms']=float(native_forward_ms_override)
    native_e=[V3.ctc_events(z,chars,*sp) for z,sp in zip(native_logits,native_spans)]
    v3=LT.normalize_ltb(V3.stitch_evidence(native_e,native_spans,use_confidence=False,geometric_fallback=True))
    route=UNI.UnifiedRecognizer.classify(im.width,im.height,len(parts))
    base_logits=native_logits;base_spans=native_spans;base_parts=parts;unified=v3
    if route=='orientation':
        scores={}
        for angle in ('cw','ccw'):
            rotated=OP.rotated(im,angle)
            p=LT.make_chunks(rotated,LT.LEGACY_CHUNK_AR,LT.LEGACY_OVERLAP)
            spans=[(int(a),int(b)) for _,a,b in p]
            if orientation_logits_override is not None:
                lg=orientation_logits_override[angle]
                assert len(lg)==len(p),(angle,len(lg),len(p))
                st['orientation_forward_ms']=float(orientation_forward_ms_override)
            else:
                lg=timed('orientation_forward_ms',lambda:recognizer.base._forward_grouped([c for c,_,_ in p],max_ratio=12))
            ev=[V3.ctc_events(q,chars,*sp) for q,sp in zip(lg,spans)]
            pred=LT.normalize_ltb(V3.stitch_evidence(ev,spans,use_confidence=False,geometric_fallback=True))
            scores[angle]={'pred':pred,'agreement':OP.overlap_agreement(ev,spans),
                           'confidence':OP.event_confidence(ev),'logits':lg,'spans':spans,'parts':p}
        _,winner=OP.choose_orientation(scores['cw'],scores['ccw'],kind='overlap')
        base_logits=winner['logits'];base_spans=winner['spans'];base_parts=winner['parts'];unified=winner['pred']
    elif route=='dense':
        base_parts=DP.forced_crops(im,3)
        base_spans=[(int(a),int(b)) for _,a,b in base_parts]
        if dense_logits_override is not None:
            base_logits=dense_logits_override
            assert len(base_logits)==len(base_parts),(len(base_logits),len(base_parts))
            st['dense_forward_ms']=float(dense_forward_ms_override)
        else:
            base_logits=timed('dense_forward_ms',lambda:recognizer.dense._forward_tensors([DP.resized_tensor(p,3) for p,_,_ in base_parts]))
        ev=[V3.ctc_events(q,chars,*sp) for q,sp in zip(base_logits,base_spans)]
        unified=LT.normalize_ltb(V3.stitch_evidence(ev,base_spans,use_confidence=False,geometric_fallback=True))
    c0=timed('c0_ms',lambda:AX.decode(base_logits,base_spans,chars))
    c1_sub,c1,c1_info=timed('c1_ms',lambda:C1.decode(base_logits,base_spans,chars,c0=c0,return_info=True))
    _sync(device);c1_end=time.perf_counter()
    st['matched_c1_ms']=1000*(c1_end-tic)
    if skip_structurally_impossible and len(base_spans)<2:
        alt_logits=[];alt_spans=[]
    else:
        alt_logits=timed('paired_scale_forward_ms',lambda:recognizer.dense._forward_tensors([DP.resized_tensor(p,2) for p,_,_ in base_parts]))
        alt_spans=[(int(a),int(b)) for _,a,b in base_parts]
        assert base_spans==alt_spans
    native,cross,info=timed('scope_decoder_ms',lambda:FAST.decode(c1,base_logits,base_spans,alt_logits,alt_spans,chars,return_info=True))
    if check_legacy:
        legacy_alt=alt_logits
        legacy_spans=alt_spans
        if len(base_spans)<2 and skip_structurally_impossible:
            # Old SCOPE was forced to run an unnecessary second scale here.
            # Its output is provably unaltered: no two spatial crop groups.
            assert native==c1 and cross==c1
        else:
            a,b,old=timed('parity_legacy_overhead_ms',lambda:E7.decode(c1,base_logits,base_spans,legacy_alt,legacy_spans,chars,return_info=True))
            assert (a,b)==(native,cross),f'Fast decoder changed text: {(a,b)!r} vs {(native,cross)!r}'
            for k in old:
                if k in info and k not in ('native_gain','cross_gain'):
                    assert old[k]==info[k],(k,old[k],info[k])
    _sync(device);scope_end=time.perf_counter()
    st['scope_total_ms']=1000*(scope_end-tic)-st.get('parity_legacy_overhead_ms',0.)
    # The direct original-model baseline uses the ORIGINAL image, no crop,
    # one MSR target_size_ltb(..., max_ratio=12), and CTC greedy.
    if run_original:
        if len(native_logits)==1 and native_spans==[(0,im.width)]:
            raw_logits=native_logits[0]
            st['original_extra_forward_ms']=0.
            st['original_reused_from_native']=1
        else:
            raw_logits=timed('original_extra_forward_ms',lambda:recognizer.base._forward_grouped([im],max_ratio=12))[0]
            st['original_reused_from_native']=0
        original=LT.normalize_ltb(LT.greedy_ctc(raw_logits,chars))
    else:
        original='';st['original_extra_forward_ms']=float('nan')
        st['original_reused_from_native']=0
    st.setdefault('paired_scale_forward_ms',0.)
    st.setdefault('orientation_forward_ms',0.)
    st.setdefault('dense_forward_ms',0.)
    st['num_primary_forward_crops']=len(native_logits)+ (6 if route=='orientation' else 3 if route=='dense' else 0)
    return {
        'original_svtrv2_full12':original,
        'v3_geom':v3,'unified_v1':unified,'c0_route':c0,'c1_route':c1,
        'e7_native':native,'e7_cross':cross,'route':route,
        'active_crops':len(base_spans),'extra_crops':len(alt_logits),
        'active_frames':sum(len(q) for q in base_logits),
        'extra_frames':sum(len(q) for q in alt_logits),
        'e7_native_candidates':info['native_candidates'],
        'e7_native_accepted':info['native_accepted'],
        'e7_cross_candidates':info['cross_candidates'],
        'e7_cross_accepted':info['cross_accepted'],
        'e7_cross_edits_path':info['cross_edits'],
        'e7_cross_insertion_chars':info['cross_insertion_chars'],
        'e7_cross_deleted_chars':info['cross_deleted_chars'],
        'e7_cross_gain':info['cross_gain'],
        'diagnostic_ms':round(st['scope_total_ms'],3),
        **st
    }


def cpu_tests():
    class Dummy(torch.nn.Module):
        def forward(self,xx):
            b,_,h,w=xx.shape;t=max(4,w//4)
            y=torch.full((b,t,3),-6.,device=xx.device)
            y[:,:,0]=6.;y[:,max(1,t//3),1]=10.;y[:,max(2,t*2//3),2]=10.
            return y
    core=UNI.UnifiedRecognizer(Dummy(),['blank','a','b'],device='cpu',batch_size=2)
    old=__import__('alex_e7_experiment')
    for im in [Image.new('RGB',(18,210),'white'),Image.new('RGB',(120,30),'white'),Image.new('RGB',(330,20),'white')]:
        new=predict_image(im,core,check_legacy=True)
        base=old.predict_image(im,core)
        assert new['original_svtrv2_full12']==LT.normalize_ltb(core.base.predict(im,'full12'))
        for k in ('v3_geom','unified_v1','c0_route','c1_route','e7_native','e7_cross','route'):
            assert new[k]==base[k],(k,new[k],base[k])
    return {'passed':True,'three_routes':True,'original_full12':True,'fast_legacy_prediction_parity':True}
