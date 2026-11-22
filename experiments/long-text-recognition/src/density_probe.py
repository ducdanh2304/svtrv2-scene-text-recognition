"""Input-resolution/forced-crop probe for high-density LTB lines.

All transformations GT-blind and predetermined. No model training or test tuning.
"""
from __future__ import annotations
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
import time
from typing import Sequence
import longtext_final as LT
import longtext_v3 as V3

METHODS = ('baseline','width_x2','width_x3','force_2','force_3','force_3_width_x3')
MAX_INPUT_W = 768


def resized_tensor(img: Image.Image, factor: int) -> torch.Tensor:
    """Original MSR when factor=1; direct resize raw pixels (no double interpolation).

    factor>1 enlarges *horizontal* input dimension while keeping MSR height,
    a diagnostic OOD intervention on the frozen checkpoint.
    """
    if factor == 1:
        return LT.preprocess_ltb(img,max_ratio=12)
    w, h = LT.target_size_ltb(*img.size,max_ratio=12)
    w2 = int(min(MAX_INPUT_W, 4*round(w*factor/4)))
    if w2 <= w: raise ValueError('Expected an expanded width')
    mode = Image.Resampling.BICUBIC if hasattr(Image,'Resampling') else Image.BICUBIC
    a = np.asarray(img.convert('RGB').resize((w2,h),mode),dtype=np.float32)/255.
    return torch.from_numpy(((a-0.5)/0.5).transpose(2,0,1).copy())


def forced_crops(img: Image.Image, n: int):
    """Overlapping 2- or 3-view raw-image tiling, no GT or recognition output."""
    if n not in (2,3): raise ValueError('Only 2 and 3 fixed raw-image views')
    w,h=img.size
    if w < 8: return [(img,0,w)]  # no-op for truly narrow images; log number of crops
    cw=min(w-1,max(4,int(round(w*(0.65 if n==2 else 0.52)))))
    starts = [0,w-cw] if n==2 else [0,int(round((w-cw)/2)),w-cw]
    if len(set(starts)) != n: raise ValueError('Degenerate forced-crop geometry')
    assert starts[0]==0 and starts[-1]+cw == w
    return [(img.crop((x,0,x+cw,h)),x,x+cw) for x in starts]


class Probe:
    def __init__(self,net,chars,device='cuda',batch_size=8):
        self.net=net.to(device).eval();self.chars=list(chars)
        self.device=device;self.batch_size=batch_size
        if not self.chars or str(self.chars[0]).lower() not in ('blank','[blank]','<blank>','blank_token'):
            raise ValueError('Expected blank class zero')

    @torch.inference_mode()
    def _forward_tensors(self,tensors: Sequence[torch.Tensor]) -> list[np.ndarray]:
        groups=defaultdict(list)
        for i,t in enumerate(tensors):groups[tuple(t.shape)].append(i)
        ret=[None]*len(tensors)
        for shape,ids in groups.items():
            _,h,w=shape
            volume=max(1,w*h)
            bs=max(1,min(self.batch_size,int(self.batch_size*192*32/volume)))
            for start in range(0,len(ids),bs):
                ii=ids[start:start+bs]
                xx=torch.stack([tensors[i] for i in ii],dim=0).to(self.device)
                raw=self.net(xx).detach().float().cpu().numpy()
                assert raw.ndim==3 and raw.shape[2]==len(self.chars),(raw.shape,len(self.chars))
                for i,x in zip(ii,raw):ret[i]=x
        assert all(x is not None for x in ret)
        return ret

    def batch(self, imgs: Sequence[Image.Image]):
        imgs=[im.convert('RGB') for im in imgs]
        inputs=[];mapping=[]
        for i,im in enumerate(imgs):
            for m in METHODS:
                if m=='baseline': parts=[(im,0,im.width)]; factor=1
                elif m=='width_x2':parts=[(im,0,im.width)]; factor=2
                elif m=='width_x3':parts=[(im,0,im.width)]; factor=3
                elif m=='force_2':parts=forced_crops(im,2);factor=1
                elif m=='force_3':parts=forced_crops(im,3);factor=1
                elif m=='force_3_width_x3':parts=forced_crops(im,3);factor=3
                else:raise AssertionError(m)
                start=len(inputs)
                inputs.extend(resized_tensor(p,factor) for p,_,_ in parts)
                spans=[(a,b) for _,a,b in parts]
                mapping.append((i,m,start,len(parts),spans))
        t0=time.perf_counter();logits=self._forward_tensors(inputs)
        if str(self.device).startswith('cuda'):torch.cuda.synchronize()
        inference_s=time.perf_counter()-t0
        outputs={m:[None]*len(imgs) for m in METHODS}
        for i,m,start,n,spans in mapping:
            cur=logits[start:start+n]
            if n==1:
                pred=LT.normalize_ltb(LT.greedy_ctc(cur[0],self.chars))
            else:
                events=[V3.ctc_events(x,self.chars,a,b) for x,(a,b) in zip(cur,spans)]
                pred=V3.stitch_evidence(events,spans,use_confidence=False,geometric_fallback=True)
            outputs[m][i]={'prediction':pred,'num_crops':n,'ctc_Ts':[len(x) for x in cur],
                           'spans':spans}
        return outputs, {'forward_s':inference_s,'tensors':len(inputs)}


def run_cpu_tests():
    from PIL import Image
    import torch
    class Dummy(torch.nn.Module):
        def forward(self,x):
            b,_,h,w=x.shape
            t=w//4
            out=torch.zeros(b,t,3,device=x.device)
            out[...,0]=1.0
            out[...,1]=-0.1
            out[:,t//3,1]=2
            return out
    img=Image.new('RGB',(120,30),'white')
    assert [int(x.shape[-1]) for x in [resized_tensor(img,1),resized_tensor(img,2),resized_tensor(img,3)]]==[128,256,384]
    assert len(forced_crops(img,2))==2 and len(forced_crops(img,3))==3
    assert len(forced_crops(img,3)[0][0].getdata())>0
    p=Probe(Dummy(),['blank','a','b'],'cpu',batch_size=4)
    result,perf=p.batch([img,Image.new('RGB',(210,30),'white')])
    assert set(result)==set(METHODS) and perf['tensors']==2*11
    assert all(len(v)==2 and isinstance(v[0]['prediction'],str) for v in result.values())
    assert result['baseline'][0]['ctc_Ts']==[32]
    assert result['width_x3'][0]['ctc_Ts']==[96]
    assert result['force_3'][0]['num_crops']==3
    return {'passed':True,'methods':list(METHODS),'total_tensors_for_2_imgs':perf['tensors']}
