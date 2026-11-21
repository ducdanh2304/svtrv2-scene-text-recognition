"""SCOPE-CTC final candidate: native baseline protection plus counterfactual crop-coverage probe.

Frozen SVTRv2-T only; decision uses pixels, CTC logits and predicted strings.
No GT, dataset name, image ID, historical correctness or test-set membership.

Design: long-text recall requires detecting grossly under-decoded native views.
For a one-crop image, use SAME primary route observations as full SCOPE to
compare the amount of supported crop text to native CTC text; if evidence is
insufficient, return the actual original-model prediction. If activated, reuse
probe logits in full SCOPE, preventing duplicate heavy primary forward passes.

The evidence rule below was designed after reviewing LTB/external benchmark
failures: it is DEVELOPMENT-TUNED, not independent validation or a published
universal guarantee. Inference never receives GT and never uses dataset IDs.
"""
from __future__ import annotations
import time
import math
import numpy as np
import torch
from PIL import Image
import longtext_final as LT
import longtext_v3 as V3
import orientation_probe as OP
import density_probe as DP
import scope_final_runtime_v3 as FULL

POLICY_VERSION='SCOPE_FINAL_COVERAGE_PROBE_V3_LOCKED_20261121'
RAW_DIRECT_DENSE_TRIGGER=16
COVERAGE_EXTRA_MIN=3
COVERAGE_RELATIVE_GAIN=0.5


def _sync(device):
    if 'cuda' in str(device) and torch.cuda.is_available():torch.cuda.synchronize()


def _native_crop_count(image):
    return len(LT.make_chunks(image,LT.LEGACY_CHUNK_AR,LT.LEGACY_OVERLAP))


def _evidence_threshold(raw):
    return max(COVERAGE_EXTRA_MIN,int(math.ceil(COVERAGE_RELATIVE_GAIN*len(raw))))


def _coverage_witness(raw,probe):
    """Raw vs multi-view text evidence; no truth oracle or prediction confidence label."""
    if not isinstance(raw,str) or not isinstance(probe,str):raise ValueError('Expected transcript strings')
    gain=len(probe)-len(raw)
    return bool(gain>=_evidence_threshold(raw)),int(gain)


def _pred_from_crops(logits,spans,chars):
    if not logits:return ''
    events=[V3.ctc_events(x,chars,a,b) for x,(a,b) in zip(logits,spans)]
    return LT.normalize_ltb(V3.stitch_evidence(events,spans,use_confidence=False,geometric_fallback=True))


def _orientation_probe(im,recognizer):
    """Reuse exactly the two orientation probe forwards/full candidate selector."""
    scores={};cache={};device=recognizer.base.device;elapsed=0.0
    for angle in ('cw','ccw'):
        rotated=OP.rotated(im,angle)
        parts=LT.make_chunks(rotated,LT.LEGACY_CHUNK_AR,LT.LEGACY_OVERLAP)
        spans=[(int(a),int(b)) for _,a,b in parts]
        _sync(device);start=time.perf_counter()
        logits=recognizer.base._forward_grouped([c for c,_,_ in parts],max_ratio=12)
        _sync(device);elapsed+=1000*(time.perf_counter()-start)
        ev=[V3.ctc_events(z,recognizer.chars,*span) for z,span in zip(logits,spans)]
        text=LT.normalize_ltb(V3.stitch_evidence(ev,spans,use_confidence=False,geometric_fallback=True))
        scores[angle]={'pred':text,'agreement':OP.overlap_agreement(ev,spans),
                       'confidence':OP.event_confidence(ev),'logits':logits,'spans':spans,'parts':parts}
        cache[angle]=logits
    angle,winner=OP.choose_orientation(scores['cw'],scores['ccw'],kind='overlap')
    return winner['pred'],cache,elapsed


def _fill_fallback(raw,forward_ms,im,meta):
    return dict(original_svtrv2_full12=raw,safe_scope=raw,
          route=('orientation' if im.width<=im.height else 'dense'),
          v3_geom=raw,active_crops=1,extra_crops=0,
          active_frames=int(meta.get('raw_frames',0)),extra_frames=0,
          main_forward_ms=float(forward_ms),dense_forward_ms=0.,orientation_forward_ms=0.,
          paired_scale_forward_ms=0.,scope_decoder_ms=0.,matched_c1_ms=float('nan'),
          original_extra_forward_ms=0.,original_reused_from_native=1,
          num_primary_forward_crops=1,
          scope_total_ms=float(forward_ms)+float(meta.get('probe_ms',0.)),
          safe_scope_total_ms=float(forward_ms)+float(meta.get('probe_ms',0.)),
          safe_scope_active=0,safe_scope_original_fallback=1,safe_scope_base_compared=1)


def predict_image(image,recognizer,*,run_original=True,check_legacy=False,skip_structurally_impossible=True):
    if not run_original:raise ValueError('Matched SVTRv2-T original baseline mandatory')
    im=image.convert('RGB')
    device=recognizer.base.device
    start=time.perf_counter()
    parts=LT.make_chunks(im,LT.LEGACY_CHUNK_AR,LT.LEGACY_OVERLAP)
    if len(parts)>1:
        out=FULL.predict_image(im,recognizer,run_original=True,check_legacy=check_legacy,
           skip_structurally_impossible=skip_structurally_impossible)
        out['safe_scope']=out['e7_cross']
        out['safe_scope_active']=1
        out['safe_scope_original_fallback']=0
        out['safe_scope_base_compared']=1
        out['safe_scope_total_ms']=float(out['scope_total_ms'])
        out['coverage_decision']='MULTI_CROP_DIRECT'
        out['coverage_probe_ms']=0.
        out['coverage_probe_extra_chars']=0
        out['coverage_probe_candidate_chars']=-1
        out['coverage_raw_chars']=len(out['original_svtrv2_full12'])
        out['coverage_probe_crops']=0
        out['coverage_native_crops']=len(parts)
        out['coverage_policy_version']=POLICY_VERSION
        _sync(device)
        out['scope_core_total_ms']=float(out['scope_total_ms'])
        out['safe_scope_total_ms']=1000*(time.perf_counter()-start)
        out['scope_total_ms']=out['safe_scope_total_ms']
        return out

    # A single native crop: first obtain a matched original SVTRv2-T transcript.
    _sync(device);t0=time.perf_counter()
    raw_logits=recognizer.base._forward_grouped([im],max_ratio=12)
    _sync(device)
    raw_forward_ms=1000*(time.perf_counter()-t0)
    if len(raw_logits)!=1:raise AssertionError('Raw full12 must be a single view')
    raw=LT.normalize_ltb(LT.greedy_ctc(raw_logits[0],recognizer.chars))
    meta={'raw_frames':len(raw_logits[0]),'probe_ms':0.,'probe_crops':0,
          'candidate_chars':-1,'extra_chars':0,'raw_chars':len(raw)}
    dense_logits=None;orient_logits=None
    active=False
    reason=''

    if im.width>im.height and len(raw)>=RAW_DIRECT_DENSE_TRIGGER:
        active=True;reason='DENSE_RAW_TEXT_LOAD'
    elif im.width>im.height:
        # The single-crop case that v2 incorrectly declared short. Generate
        # the primary Dense observations and compare transcript COVERAGE to
        # original raw. If activated these logits are REUSED by SCOPE.
        crop_parts=DP.forced_crops(im,3)
        _sync(device);start_probe=time.perf_counter()
        dense_logits=recognizer.dense._forward_tensors(
              [DP.resized_tensor(p,3) for p,_,_ in crop_parts])
        _sync(device);meta['probe_ms']=1000*(time.perf_counter()-start_probe)
        spans=[(int(a),int(b)) for _,a,b in crop_parts]
        candidate=_pred_from_crops(dense_logits,spans,recognizer.chars)
        active,meta['extra_chars']=_coverage_witness(raw,candidate)
        meta['candidate_chars']=len(candidate)
        meta['probe_crops']=len(dense_logits)
        reason='DENSE_COVERAGE_SUPPORTED' if active else 'DENSE_PRESERVE_RAW'
    else:
        # This also covers vertically oriented one-crop images, which v2
        # previously early-exited without ever checking the rotated content.
        candidate,orient_logits,probe_ms=_orientation_probe(im,recognizer)
        meta['probe_ms']=float(probe_ms)
        meta['candidate_chars']=len(candidate)
        meta['probe_crops']=sum(map(len,orient_logits.values()))
        active,meta['extra_chars']=_coverage_witness(raw,candidate)
        reason='ORIENTATION_COVERAGE_SUPPORTED' if active else 'ORIENTATION_PRESERVE_RAW'

    if active:
        out=FULL.predict_image(im,recognizer,run_original=True,check_legacy=check_legacy,
            skip_structurally_impossible=skip_structurally_impossible,
            native_logits_override=raw_logits,native_forward_ms_override=raw_forward_ms,
            dense_logits_override=dense_logits,dense_forward_ms_override=meta['probe_ms'] if dense_logits is not None else 0.,
            orientation_logits_override=orient_logits,orientation_forward_ms_override=meta['probe_ms'] if orient_logits is not None else 0.)
        if out['original_svtrv2_full12']!=raw:
            raise AssertionError('Raw baseline changed when reusing primary logits')
        out['safe_scope']=out['e7_cross']
        out['safe_scope_total_ms']=float(out['scope_total_ms'])+raw_forward_ms+meta['probe_ms']
        out['safe_scope_active']=1
        out['safe_scope_original_fallback']=0
        out['safe_scope_base_compared']=1
    else:
        out=_fill_fallback(raw,raw_forward_ms,im,meta)

    out['coverage_decision']=reason
    out['coverage_probe_ms']=float(meta['probe_ms'])
    out['coverage_probe_extra_chars']=int(meta['extra_chars'])
    out['coverage_probe_candidate_chars']=int(meta['candidate_chars'])
    out['coverage_raw_chars']=len(raw)
    out['coverage_probe_crops']=int(meta['probe_crops'])
    out['coverage_native_crops']=len(parts)
    out['coverage_policy_version']=POLICY_VERSION
    _sync(device)
    out['scope_core_total_ms']=float(out['scope_total_ms'])
    out['safe_scope_total_ms']=1000*(time.perf_counter()-start)
    out['scope_total_ms']=out['safe_scope_total_ms']
    return out


def cpu_tests():
    import unified_router as UNI
    class Dummy(torch.nn.Module):
        def forward(self,x):
            batch,_,_,w=x.shape
            T=max(4,w//4)
            z=torch.full((batch,T,3),-6.,device=x.device)
            z[:,:,0]=5.
            z[:,max(1,T//3),1]=10.
            z[:,max(2,2*T//3),2]=10.
            return z
    model=UNI.UnifiedRecognizer(Dummy(),['blank','a','b'],device='cpu',batch_size=2)
    for im in [Image.new('RGB',(120,30),'white'),Image.new('RGB',(20,120),'white'),
               Image.new('RGB',(330,20),'white'),Image.new('RGB',(18,210),'white')]:
        x=predict_image(im,model)
        assert isinstance(x['safe_scope'],str) and isinstance(x['original_svtrv2_full12'],str)
        assert x['safe_scope_total_ms']>=0
        if x['safe_scope_active']==0:
            assert x['safe_scope']==x['original_svtrv2_full12']
        if x['safe_scope_active']==1 and x['route']=='dense':
            old=FULL.predict_image(im,model)
            assert x['safe_scope']==old['e7_cross']
    # The logic and normalization of the coverage comparator are deliberate.
    assert _coverage_witness('abc','abcdefgh')[0]
    assert not _coverage_witness('abc','abcde')[0]
    assert not _coverage_witness('abc','ab')[0]
    assert _coverage_witness('', 'abcd')[0]
    return {'passed':True,'test_routes':4,'fallback_matches_original':True}
