"""GT-blind orientation probe. Frozen original crop-before-resize + v3 geometry.

Never call this module with label text. All selector criteria use predictions,
CTC event geometry and confidence only. Scores are exploratory, not calibrated.
"""
from __future__ import annotations
from collections import defaultdict
from difflib import SequenceMatcher
import math,time
import numpy as np
import torch
from PIL import Image
import longtext_final as LT
import longtext_v3 as V3

METHODS=('original','cw_v3','ccw_v3','cw_v2','ccw_v2','cw_geom_ctc','ccw_geom_ctc','select_overlap','select_charconf')

def rotated(im,which):
    if which == 'original':return im.convert('RGB')
    if which == 'cw':return im.convert('RGB').transpose(Image.Transpose.ROTATE_270)
    if which == 'ccw':return im.convert('RGB').transpose(Image.Transpose.ROTATE_90)
    raise ValueError(which)

def overlap_agreement(event_lists,spans):
    match=0.;denom=0
    for i in range(len(spans)-1):
        left,right=event_lists[i],event_lists[i+1]
        a=max(spans[i][0],spans[i+1][0]);b=min(spans[i][1],spans[i+1][1])
        if b<=a:continue
        s1=''.join(e.char for e in left if a<=e.x<=b)
        s2=''.join(e.char for e in right if a<=e.x<=b)
        # Weight by shared recognized event count; empty/empty gives no false credit
        c=len(s1)+len(s2)
        if c:
            match+=c*SequenceMatcher(None,s1,s2,autojunk=False).ratio()
            denom+=c
    return float(match/denom) if denom else 0.0

def event_confidence(events):
    vals=[e.confidence for sub in events for e in sub]
    return float(np.mean(vals)) if vals else 0.0

def choose_orientation(a,b,kind='overlap'):
    # Positive direction names are literal rotations; no ground-truth, no label length
    if kind=='overlap':
        key_a=(a['agreement'],a['confidence'],len(a['pred']))
        key_b=(b['agreement'],b['confidence'],len(b['pred']))
    elif kind=='charconf':
        key_a=(a['confidence'],a['agreement'],len(a['pred']))
        key_b=(b['confidence'],b['agreement'],len(b['pred']))
    else:raise ValueError(kind)
    return ('cw',a) if key_a>=key_b else ('ccw',b)

class OrientationProbe:
    def __init__(self,net,chars,device='cuda',batch_size=2):
        self.core=LT.LongTextRecognizer(net,chars,device=device,batch_size=batch_size)
        self.chars=list(chars)

    def infer_batch(self,images):
        images=[im.convert('RGB') for im in images]
        # Original is 1-crop for this cohort. Orientation candidates use exact
        # make_chunks(raw pixels) THEN individual resized crop.
        allparts=[];meta=[]
        for n,im in enumerate(images):
            for angle in ('original','cw','ccw'):
                candidate=rotated(im,angle)
                parts=LT.make_chunks(candidate,LT.LEGACY_CHUNK_AR,LT.LEGACY_OVERLAP)
                start=len(allparts)
                allparts.extend(p for p,_,_ in parts)
                meta.append((n,angle,start,len(parts),[(x0,x1) for _,x0,x1 in parts]))
        t0=time.perf_counter()
        logits=self.core._forward_grouped(allparts,max_ratio=12)
        gpu_s=time.perf_counter()-t0
        record=[{} for _ in images]
        for n,angle,start,count,spans in meta:
            cur=logits[start:start+count]
            events=[V3.ctc_events(lg,self.chars,a,b) for lg,(a,b) in zip(cur,spans)]
            v3=V3.stitch_evidence(events,spans,use_confidence=False,geometric_fallback=True)
            parts=[''.join(e.char for e in ev) for ev in events]
            v2,_=LT.stitch_v2(parts,spans)
            geom=LT.normalize_ltb(LT.geometry_ctc_same_crops(cur,spans,self.chars))
            result={'pred':v3,'v2':v2,'geom_ctc':geom,'n_crops':count,
                    'agreement':overlap_agreement(events,spans),
                    'confidence':event_confidence(events),'T_sum':sum(len(z) for z in cur)}
            record[n][angle]=result
        outputs=[]
        for r in record:
            cw=r['cw'];ccw=r['ccw']
            a=choose_orientation(cw,ccw,'overlap')
            b=choose_orientation(cw,ccw,'charconf')
            d={'original':r['original']['pred'],'cw_v3':cw['pred'],'ccw_v3':ccw['pred'],
               'cw_v2':cw['v2'],'ccw_v2':ccw['v2'],
               'cw_geom_ctc':cw['geom_ctc'],'ccw_geom_ctc':ccw['geom_ctc'],
               'select_overlap':a[1]['pred'],'select_charconf':b[1]['pred'],
               'overlap_choice':a[0],'charconf_choice':b[0],
               'cw_agreement':cw['agreement'],'ccw_agreement':ccw['agreement'],
               'cw_confidence':cw['confidence'],'ccw_confidence':ccw['confidence'],
               'cw_n_crops':cw['n_crops'],'ccw_n_crops':ccw['n_crops'],
               'original_T':r['original']['T_sum']}
            outputs.append(d)
        return outputs,{'num_tensors':len(allparts),'gpu_forward_seconds':gpu_s}

def run_cpu_tests():
    from PIL import Image
    class Dummy(torch.nn.Module):
        def forward(self,x):
            b,_,h,w=x.shape
            v=torch.zeros(b,w//4,3,device=x.device);v[:,:,0]=2.;v[:,(w//8):((w//8)+2),1]=3.
            return v
    im=Image.new('RGB',(18,210),'white')
    assert rotated(im,'cw').size==(210,18) and rotated(im,'ccw').size==(210,18)
    model=OrientationProbe(Dummy(),['blank','a','b'],device='cpu',batch_size=2)
    x,perf=model.infer_batch([im,Image.new('RGB',(31,220),'white')])
    assert len(x)==2 and all(set(METHODS).issubset(z.keys()) for z in x)
    assert all(z['overlap_choice'] in ('cw','ccw') for z in x)
    assert all(isinstance(z[k],str) for z in x for k in METHODS)
    assert any(z['cw_n_crops']>=2 for z in x)
    assert perf['num_tensors']>=10
    # Selector cannot access GT because it accepts only orientation dictionaries
    assert choose_orientation({'pred':'a','agreement':0.8,'confidence':0.3},
                              {'pred':'b','agreement':0.5,'confidence':0.9},'overlap')[0]=='cw'
    return {'passed':True,'methods':list(METHODS),'dummy_tensors':perf['num_tensors']}
