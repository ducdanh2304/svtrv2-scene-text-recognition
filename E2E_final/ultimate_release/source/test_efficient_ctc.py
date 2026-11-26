import math,time
import numpy as np
from efficient_ctc import Scorer
from ar_scoring import raw_ctc_logprob

def main():
    rng=np.random.default_rng(20261125);vocab=['blank','a','b','c','A','B','1'];cases=[]
    for t in (1,2,3,7,15,32,64,128):
        for scale in (0.,1.,15.,100.):
            z=rng.normal(size=(t,len(vocab)))*scale
            for s in ('','a','aa','aaa','ab','aba','A1B','invalid','a'*t, ''.join(rng.choice(vocab[1:],size=max(1,t//2)))):cases.append((z,s,vocab))
    s=Scorer(raw_ctc_logprob,True,max_views=4,max_bytes=8192);delta=0.
    for z,text,chars in cases+cases:
        a=raw_ctc_logprob(z,text,chars);b=s(z,text,chars)
        assert math.isfinite(a)==math.isfinite(b),(a,b)
        if math.isfinite(a):
            delta=max(delta,abs(a-b));assert abs(a-b)<=2e-8,(a,b)
        else:assert a==b
    z=np.ones((12,len(vocab)));a=s(z,'ab',vocab);z[0,1]+=10;b=s(z,'ab',vocab);assert abs(b-raw_ctc_logprob(z,'ab',vocab))<2e-8 and a!=b
    for z in (np.zeros((0,len(vocab))),np.full((4,len(vocab)),np.nan)):
        try:s(z,'a',vocab)
        except ValueError:pass
        else:raise AssertionError('invalid input accepted')
    assert s.stats['peak_bytes']<=8192
    print('CTC CPU PARITY PASS',len(cases)*2,'max_abs_delta',delta,'stats',s.stats)
    z=rng.normal(size=(96,len(vocab)));texts=[''.join(rng.choice(vocab[1:],size=20)) for _ in range(40)]
    start=time.perf_counter();[raw_ctc_logprob(z,t,vocab) for t in texts];base=time.perf_counter()-start
    s=Scorer(raw_ctc_logprob,True);start=time.perf_counter();[s(z,t,vocab) for t in texts];fast=time.perf_counter()-start
    print('Synthetic CTC-only microbenchmark (not GPU pipeline):',base,fast,'speedup',base/fast)
if __name__=='__main__':main()
