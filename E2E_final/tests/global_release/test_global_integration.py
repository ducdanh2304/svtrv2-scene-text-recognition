from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from PIL import Image
import str_e2e.backends as pkg

from str_e2e.types import Prediction
import str_e2e.backends.global_joint as G

CHARS=['blank','a','b','c','x','A','B','C','X']
def logits(t):
    ids=[0]
    for c in t:ids += [CHARS.index(c),0]
    z=np.full((len(ids),len(CHARS)),-4.,np.float32);z[np.arange(len(ids)),ids]=5.;return z

def test_r3_rotated_winner_is_scored_not_unrotated(monkeypatch):
    class R:
        def __init__(self):
            self.model=SimpleNamespace(decoder=SimpleNamespace(w_atten_block=True,fc=torch.nn.Identity()))
        def _predict_one(self,img,ratio_override=None):
            text,score=('abc',.4) if img.width<img.height else ('axc',.9)
            self.model.decoder.fc(torch.from_numpy(logits(text))[None]);return text,score
        def predict(self,img):
            a=self._predict_one(img);b=self._predict_one(img.rotate(90,expand=True));t,s=max([a,b],key=lambda p:p[1]);return Prediction(t,s)
    r=R(); original=r._predict_one
    pred,best,unrotated,views=G.r3_observations(r,Image.new('RGB',(10,30)),CHARS)
    assert pred.text=='axc' and np.array_equal(best,logits('axc')) and np.array_equal(unrotated,logits('abc')) and views==['abc','axc']
    assert r._predict_one==original and not r.model.decoder.fc._forward_hooks

@pytest.mark.parametrize('multi',[False,True])
def test_global_evidence_full_and_spatial_paths(monkeypatch,multi):
    im=Image.new('RGB',(60,20));spans=[(0,35),(25,60)] if multi else [(0,60)];parts=[im.crop((a,0,b,20)) for a,b in spans];base=[logits('ab'),logits('bc')] if multi else [logits('abc')]
    cap=dict(parts=parts,spans=spans,base=base,extra=[],extra_spans=[])
    monkeypatch.setattr(G,'capture_scope',lambda *a:(dict(safe_scope='abc',original_svtrv2_full12='abc'),cap))
    monkeypatch.setattr(G,'spatial_decode',lambda *a:('abc',{}))
    monkeypatch.setattr(G,'r3_observations',lambda r,c,chars:(Prediction('abc',.8),logits('abc'),logits('abc'),['abc']))
    m=G.GlobalJoint.__new__(G.GlobalJoint);m.torch=torch;m.chars=CHARS;m.delta=.15;m.r3=object();m.s2rec=object()
    m.scope=SimpleNamespace(scope=SimpleNamespace(predict_image=lambda *a:{'safe_scope':'axc'}),recognizer=SimpleNamespace(base=SimpleNamespace(_forward_grouped=lambda *a,**k:[logits('abc')])),device='cpu')
    m.motion=SimpleNamespace(predict=lambda im:Prediction('axc',.9),pipe=SimpleNamespace(preprocess=lambda im:im.width))
    m.s2net=lambda width:torch.from_numpy(logits('axc'))[None]
    class W:
        def predict(self,im):return Prediction('abc',.9)
        def score(self,strings):return {'scores':{s:(-.01 if s=='abc' else -5.) for s in strings}}
    m.w=W();out=m.predict_evidence(im)
    assert {'abc','axc'}<=set(out['candidates']);assert out['pred']['motion']=='axc' and out['pred']['scope_s2']=='axc'
    assert out['crops']==len(spans) and out['spatial_used']==multi
    assert all(p is None or np.isclose(sum(p),1.) for p in out['table'])
