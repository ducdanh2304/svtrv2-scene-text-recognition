"""E7 matched baseline + cross-scale posterior segment reconstruction adapter.

Inputs are pixels and frozen model only. Labels/IDs/evaluation archives are NOT read.
"""
import time
from PIL import Image
import longtext_final as LT
import longtext_v3 as V3
import orientation_probe as OP
import density_probe as DP
import unified_router as UNI
import alex_ctc as AX
import alex_c1 as C1
import alex_e7 as E7


def predict_image(image, recognizer):
    im=image.convert('RGB');chars=recognizer.chars
    tic=time.perf_counter()
    parts=LT.make_chunks(im,LT.LEGACY_CHUNK_AR,LT.LEGACY_OVERLAP)
    native_spans=[(int(a),int(b)) for _,a,b in parts]
    native_logits=recognizer.base._forward_grouped([p for p,_,_ in parts],max_ratio=12)
    native_e=[V3.ctc_events(z,chars,*sp) for z,sp in zip(native_logits,native_spans)]
    v3=LT.normalize_ltb(V3.stitch_evidence(native_e,native_spans,use_confidence=False,geometric_fallback=True))
    route=UNI.UnifiedRecognizer.classify(im.width,im.height,len(parts))
    base_logits=native_logits;base_spans=native_spans
    base_parts=parts
    unified=v3
    if route=='orientation':
        scores={}
        for angle in ('cw','ccw'):
            rotated=OP.rotated(im,angle)
            p=LT.make_chunks(rotated,LT.LEGACY_CHUNK_AR,LT.LEGACY_OVERLAP)
            spans=[(int(a),int(b)) for _,a,b in p]
            lg=recognizer.base._forward_grouped([c for c,_,_ in p],max_ratio=12)
            ev=[V3.ctc_events(q,chars,*sp) for q,sp in zip(lg,spans)]
            pred=LT.normalize_ltb(V3.stitch_evidence(ev,spans,use_confidence=False,geometric_fallback=True))
            scores[angle]={'pred':pred,'agreement':OP.overlap_agreement(ev,spans),
                           'confidence':OP.event_confidence(ev),'logits':lg,
                           'spans':spans,'parts':p}
        chosen,winner=OP.choose_orientation(scores['cw'],scores['ccw'],kind='overlap')
        base_logits=winner['logits'];base_spans=winner['spans'];base_parts=winner['parts']
        unified=winner['pred']
    elif route=='dense':
        base_parts=DP.forced_crops(im,3)
        base_spans=[(int(a),int(b)) for _,a,b in base_parts]
        base_logits=recognizer.dense._forward_tensors([DP.resized_tensor(p,3) for p,_,_ in base_parts])
        ev=[V3.ctc_events(q,chars,*sp) for q,sp in zip(base_logits,base_spans)]
        unified=LT.normalize_ltb(V3.stitch_evidence(ev,base_spans,use_confidence=False,geometric_fallback=True))
    c0=AX.decode(base_logits,base_spans,chars)
    c1_sub,c1,c1_meta=C1.decode(base_logits,base_spans,chars,c0=c0,return_info=True)
    # Matched second observation: same raw pixel crop spans, different horizontal MSR.
    # For Dense, the secondary view is x2 and active primary is x3.
    alt_logits=recognizer.dense._forward_tensors([DP.resized_tensor(p,2) for p,_,_ in base_parts])
    secondary_spans=[(int(a),int(b)) for _,a,b in base_parts]
    assert base_spans==secondary_spans
    local,cross,meta=E7.decode(c1,base_logits,base_spans,alt_logits,secondary_spans,chars,return_info=True)
    return {'v3_geom':v3,'unified_v1':unified,'c0_route':c0,'c1_route':c1,
            'e7_native':local,'e7_cross':cross,'route':route,
            'active_crops':len(base_spans),'extra_crops':len(alt_logits),
            'active_frames':sum(len(z) for z in base_logits),
            'extra_frames':sum(len(z) for z in alt_logits),
            'e7_native_candidates':meta['native_candidates'],
            'e7_native_accepted':meta['native_accepted'],
            'e7_cross_candidates':meta['cross_candidates'],
            'e7_cross_accepted':meta['cross_accepted'],
            'e7_cross_insertion_chars':meta['cross_insertion_chars'],
            'e7_cross_deleted_chars':meta['cross_deleted_chars'],
            'e7_cross_edits_path':meta['cross_edits'],
            'diagnostic_ms':round((time.perf_counter()-tic)*1000,3)}


def cpu_tests():
    import torch
    class Dummy(torch.nn.Module):
        def forward(self,xx):
            b,_,h,w=xx.shape
            t=max(4,w//4)
            y=torch.full((b,t,3),-6.,device=xx.device)
            y[:,:,0]=6.;y[:,max(1,t//3),1]=10.;y[:,max(2,t*2//3),2]=10.
            return y
    core=UNI.UnifiedRecognizer(Dummy(),['blank','a','b'],device='cpu',batch_size=2)
    for im in [Image.new('RGB',(18,210),'white'),Image.new('RGB',(120,30),'white'),Image.new('RGB',(330,20),'white')]:
        row=predict_image(im,core)
        assert isinstance(row['e7_cross'],str) and row['e7_cross']
        base,_=core.infer_batch([im])
        assert row['unified_v1']==LT.normalize_ltb(base[0]['pred']),row
        assert row['extra_crops']==row['active_crops']
    assert E7.cpu_tests()['passed']
    return {'passed':True,'routes':['orientation','dense','v3'],
            'baseline_parity':True,'extra_crop_alignment':True}
