import math
import sys
import numpy as np
import pytest
from pathlib import Path
import importlib.util
import types

# Run the same module in a temporary package; its only sibling dependency
# is the original, already tested AR scoring implementation.
name='_final_testpkg';pkg=types.ModuleType(name);pkg.__path__=[str(Path(__file__).parent)];sys.modules[name]=pkg
from str_e2e.backends.final_evidence import conditional_decode,pooled_logodds,variants,crop_pair,ar_pair_bf,spatial_final

def test_conditional_pool_normalized_and_offset_invariant():
    z=np.array([[0.,2.,1.]])
    c=['blank','a','b'];strings=['a','b','a']
    y,p,n=conditional_decode(strings,[z,z],c,{'a':-30.,'b':-29.},str.lower)
    yy,pp,nn=conditional_decode(strings,[z+5,z-7],c,{'a':-3.,'b':-2.},str.lower)
    assert y==yy and n==nn==3
    assert sum(np.exp(list(p.values())))==pytest.approx(1.)
    assert p==pytest.approx(pp)

def test_unsupported_expert_and_dedup():
    z=np.array([[0.,2.,1.]])
    a=conditional_decode(['a','b'],[z],['blank','a','b'],None,str.lower)
    b=conditional_decode(['a','a','b'],[z],['blank','a','b'],{'a':None,'b':None},str.lower)
    assert a==b

def test_odds_mixture_extremes():
    assert pooled_logodds([1000.,1000.])==pytest.approx(1000.)
    assert pooled_logodds([-1000.,1000.])==pytest.approx(0.)
    assert pooled_logodds([3.])==pytest.approx(3.)
    assert pooled_logodds([3.9,3.9,-.2,15.])>1.6>pooled_logodds([3.9,3.9,-.2])

def test_pair_variant_mass_and_unsupported():
    obs=['AbC','aBc'];v=variants('abc',obs,str.lower)
    assert len(v)==len(set(v)) and set(obs)<=set(v)
    a=variants('abc',[],str.lower);b=variants('axc',[],str.lower)
    table={x:-12. for x in a};table.update({x:-2. for x in b})
    assert ar_pair_bf(('abc','axc'),table,[],str.lower)==pytest.approx(10.)
    assert ar_pair_bf(('abc','axc'),None,[],str.lower) is None
    table.pop(a[0]);assert ar_pair_bf(('abc','axc'),table,[],str.lower) is None

def test_spatial_real_solver_ar_changes_edit(monkeypatch):
    import alex_e7 as E7
    import alex_ctc as AX
    import scope_e7_fast as FAST
    import longtext_v3 as V3
    ref='abc'; p=E7.SpanEdit(1,2,'x',50.)
    events=[V3.CharEvidence(c,1.,x) for c,x in zip(ref,[40.,50.,60.])]
    monkeypatch.setattr(AX,'reference_events',lambda *a:events)
    monkeypatch.setattr(FAST,'_make_view',lambda z,sp,g,scale,*a:dict(group=g,span=sp,scale=scale))
    monkeypatch.setattr(E7,'generate_proposals',lambda *a:([p],{(1,2,'x'):{(0,'native'),(1,'native')}},2))
    monkeypatch.setattr(E7,'local_bf',lambda v,*a: -.2 if v['family']=='artistic' else 3.9)
    class Worker:
        def __init__(self,active):self.active=active;self.calls=0
        def predict(self,im):self.calls+=1
        def score(self,strings):return {'scores':{s:(-1. if s.lower()=='axc' else -16.) for s in strings} if self.active else None}
    z=np.zeros((3,5));cap=dict(base=[z,z],spans=[(0,80),(20,100)],extra=[],extra_spans=[],parts=[0,1])
    def run(w):return spatial_final(ref,cap,['blank','a','b','c','x'],[z,z],[z,z],['axc']*2,['axc']*2,['axc']*2,[['abc','axc']]*2,w,[24,24],[True,True])
    a,b,info=run(Worker(True));assert a==ref and b=='axc' and info['ar_pairs']==2 and info['ar']['accepted']==1
    a,b,info=run(Worker(False));assert a==b==ref and info['ar_pairs']==0

def test_geometry_and_scope_native_tests():
    import alex_e7 as E7
    import scope_e7_fast as FAST
    p=E7.SpanEdit(1,2,'x',50.)
    assert crop_pair(p,'abc',[0,1,2],(0,100))==('abc','axc')
    assert crop_pair(p,'abc',[0,2],(0,100)) is None
    assert crop_pair(p,'abc',[0,1,2],(49,100)) is None
    assert E7.cpu_tests()['passed'] and FAST.cpu_tests()['passed']
