import math,sys,types
from pathlib import Path
import numpy as np
import pytest
pkg=types.ModuleType('_global_testpkg');pkg.__path__=[str(Path(__file__).parent)];sys.modules[pkg.__name__]=pkg
from str_e2e.backends.global_probability import project,coverage_weights,prob,combine_views,raw_masses,ar_masses,choose,fit_calibration,pool


def test_projection_insertion_deletion_and_substitution():
    coords=[5.,15.,25.,35.]
    for alt in ('abcd','xabcd','abxcd','ab','abzzd','abxyz','', 'abcde'):
        assert project('abcd',alt,coords,(0,40))==alt
    assert project('abcd','abxcd',coords,(0,21))=='abx'
    assert project('abcd','abxcd',coords,(21,40))=='cd'
    assert project('abcd','abcd',coords,(10,30))=='bc'


def test_overlap_weights():
    w=coverage_weights([(0,20),(10,30)],[5,15,25])
    assert np.allclose(w,[.5,.5]) and sum(w)==pytest.approx(1.)
    assert np.allclose(coverage_weights([(0,10),(0,10),(10,30)],[5,15,25]),[1/6,1/6,2/3])


def test_spatial_composite_and_empty_support():
    a=np.log([.9,.1]);b=np.log([.6,.4]);q=combine_views([a,b],[.5,.5]);assert q[0]>q[1] and sum(q)==pytest.approx(1.)
    assert combine_views([np.array([0.,-np.inf]),np.array([-np.inf,0.])],[.5,.5]) is None
    assert np.allclose(combine_views([a,None],[.5,.5]),np.exp(a))


def test_case_mass_dedup_and_raw_event():
    z=np.array([[0.,2.,1.,-3.]])
    v=raw_masses(['a','b','a'],z,['blank','a','A','b'],['a','A','a'])
    q=np.exp(z[0]-np.logaddexp.reduce(z[0]));assert np.exp(v[0])==pytest.approx(q[1]+q[2]);assert v[0]==v[2]


def test_ar_missing_and_case_mass():
    class W:
        def predict(self,im): self.im=im
        def score(self,strings):return {'scores':{s:(math.log(.3) if s=='a' else math.log(.2) if s=='A' else None) for s in strings}}
    w=W();q=ar_masses(['a','b'],w,'img',['A']);assert np.exp(q[0])==pytest.approx(.5) and np.isneginf(q[1]) and w.im=='img'


def test_full_path_not_constrained_by_reference():
    candidates=['abc','xyzz','ab']; table=[[.2,.75,.05],[.1,.85,.05],[.2,.7,.1]]
    assert choose(candidates,table)[0]=='xyzz'
    assert np.allclose(pool([None,[.1,.9],None]),[.1,.9])


def test_calibration_only_constant_parameters():
    rows=[]
    for i in range(60):
        rows.append(dict(gt='a',candidates=['a','b'],table=[[.1,.9],[.2,.8],[.8,.2]],source_group_id=str(i)))
    cal=fit_calibration(rows);assert cal['nll_after']<cal['nll_before'];assert len(cal['weights'])==len(cal['temperatures'])==3
    assert choose(['a','b'],rows[0]['table'],cal['weights'],cal['temperatures'])[0]=='a'


def test_no_out_of_pool_fit():
    rows=[dict(gt='absent',candidates=['a','b'],table=[[.1,.9],[.2,.8],[.8,.2]],source_group_id=str(i)) for i in range(50)]
    with pytest.raises(RuntimeError,match='Insufficient'):fit_calibration(rows)
