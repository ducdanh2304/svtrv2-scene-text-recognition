import math
import itertools
import numpy as np
import pytest
import torch
from str_e2e.backends.ar_scoring import raw_ctc_logprob, mixture, ReplayAR


def test_ctc_exact_against_exhaustive_paths():
    z=np.array([[.2,.3,.5],[.4,.8,.1],[.2,.4,.7]])
    p=torch.softmax(torch.tensor(z),-1).numpy(); chars=['blank','A','a']
    for text in ('','A','a','AA','Aa','aA','aaa','X'):
        mass=0.
        for path in itertools.product(range(3),repeat=3):
            seq=''.join(chars[c] for i,c in enumerate(path) if c and (i==0 or c!=path[i-1]))
            if seq==text:mass+=np.prod([p[t,c] for t,c in enumerate(path)])
        got=raw_ctc_logprob(z,text,chars)
        assert np.isclose(np.exp(got),mass)


def test_ar_probability_changes_result_without_fake_ctc_frames():
    z=np.array([[-10.,2.,1.]])
    args=(['A','B'],[z]*3,['blank','A','B'])
    # Use a close CTC disagreement so AR support can change ranking.
    z[0,1]=.1;z[0,2]=0.
    assert mixture(*args)[0]=='A'
    assert mixture(*args,ar={'A':-10.,'B':-.01})[0]=='B'
    assert mixture(*args,ar=None)[0]=='A'


def test_case_events_are_deduped():
    z=np.zeros((1,3)); y,s,_=mixture(['A','A','a'],[z],['blank','A','a'],normalizer=str.lower)
    assert y=='a' and np.isclose(np.exp(s['a']),2/3)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.fc=torch.nn.Linear(3,3,bias=False)
        with torch.no_grad(): self.fc.weight.copy_(torch.tensor([[1.,1.,4.],[4.,1.,0.],[1.,4.,0.]]))
    def forward(self,x):
        prev=torch.tensor([0]); out=[]
        for t in range(4):
            if t==0: feat=torch.tensor([[1.,0.,0.]])
            elif t==2:feat=torch.tensor([[0.,0.,1.]])
            else:feat=torch.nn.functional.one_hot(prev,num_classes=3).float()
            z=self.fc(feat); out.append(z.softmax(-1));prev=z.argmax(-1)
            if int(prev)==0:break
        return torch.stack(out,dim=1)


class Post:
    character=['<eos>','a','b','<bos>','<pad>']
    def __call__(self,out):
        ids=out.argmax(-1)[0].tolist();text=''
        for i in ids:
            if i==0:break
            text+=self.character[i]
        return [(text,1.)]


class Rec:
    def __init__(self): self.model=torch.nn.Module();self.model.decoder=Decoder().eval();self.post=Post()
    def __call__(self,**kwargs):
        text,score=self.post(self.model.decoder(torch.tensor(0.))[0:1])[0];return [{'text':text,'score':score}]


def test_replay_scores_eos_and_prefixes():
    r=Rec();s=ReplayAR(r,r.post,3);native=s.observe('dummy');assert native['text']=='ab'
    scores=s.score(['aa','ab','ba','', 'x']);assert scores['ab']>scores['aa'] and scores['x'] is None
    assert scores['']==pytest.approx(float(torch.log_softmax(torch.tensor([1.,4.,1.]),0)[0]))
    assert s.supported_len==3 and s.checked
    assert not r.model.decoder.fc._forward_hooks


def test_replay_failure_restores_hooks():
    r=Rec();s=ReplayAR(r,r.post,3);s.observe('dummy')
    with pytest.raises(RuntimeError):s.replay([100])
    assert not r.model.decoder.fc._forward_hooks
