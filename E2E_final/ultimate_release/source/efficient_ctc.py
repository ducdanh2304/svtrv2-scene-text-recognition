"""E3: exact finite-event CTC scoring with bounded image-local reuse.
No pruning, fastmath, altered logits, or new thresholds in recognition.
"""
from collections import OrderedDict
from contextlib import contextmanager
import math,time
import numpy as np
from numba import njit

@njit(cache=True,fastmath=False)
def forward(logp,ids):
    n=2*len(ids)+1
    states=np.zeros(n,np.int64)
    for i in range(len(ids)): states[2*i+1]=ids[i]
    a=np.full(n,-np.inf); b=np.empty(n,np.float64)
    a[0]=logp[0,0]; a[1]=logp[0,ids[0]]
    for t in range(1,logp.shape[0]):
        for j in range(n):
            step=a[j-1] if j>=1 else -np.inf
            jump=a[j-2] if j>=2 and states[j]!=0 and states[j]!=states[j-2] else -np.inf
            b[j]=np.logaddexp(np.logaddexp(a[j],step),jump)+logp[t,states[j]]
        a,b=b,a
    return np.logaddexp(a[-1],a[-2])

class Scorer:
    def __init__(self,original,fast,max_bytes=64*1024*1024,max_views=32,max_scores=4096):
        self.original=original;self.fast=fast;self.cache=OrderedDict();self.nbytes=0
        self.max_bytes=max_bytes;self.max_views=max_views;self.max_scores=max_scores
        self.stats=dict(calls=0,ms=0.,view_hits=0,view_misses=0,score_hits=0,dp_calls=0,peak_bytes=0)
    def __call__(self,z,text,chars):
        t=time.perf_counter();self.stats['calls']+=1
        try:
            return self.score(z,text,chars) if self.fast else self.original(z,text,chars)
        finally:self.stats['ms']+=(time.perf_counter()-t)*1000
    def score(self,z,text,chars):
        z=np.asarray(z,dtype=np.float64)
        if z.ndim!=2 or z.shape[1]!=len(chars) or not len(z) or not np.isfinite(z).all():raise ValueError('Invalid CTC logits')
        chars=tuple(chars); lookup={c:i for i,c in enumerate(chars) if i and len(c)==1}
        if any(c not in lookup for c in text):return -math.inf
        ids=[lookup[c] for c in text]
        if len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))>len(z):return -math.inf
        key=(z.shape,chars,z.tobytes(order='C'))
        entry=self.cache.get(key)
        if entry is None:
            self.stats['view_misses']+=1
            lp=np.ascontiguousarray(z-np.logaddexp.reduce(z,axis=1,keepdims=True)); table={}
            size=len(key[2])+lp.nbytes
            if size<=self.max_bytes:
                while self.cache and (len(self.cache)>=self.max_views or self.nbytes+size>self.max_bytes):
                    _,(_,_,n)=self.cache.popitem(last=False);self.nbytes-=n
                self.cache[key]=(lp,table,size);self.nbytes+=size
                self.stats['peak_bytes']=max(self.stats['peak_bytes'],self.nbytes)
        else:
            self.stats['view_hits']+=1; self.cache.move_to_end(key);lp,table,_=entry
        if text in table:
            self.stats['score_hits']+=1;return table[text]
        if not ids: value=float(lp[:,0].sum())
        else:
            self.stats['dp_calls']+=1;value=float(forward(lp,np.asarray(ids,dtype=np.int64)))
        if len(table)<self.max_scores:table[text]=value
        return value

@contextmanager
def scoring_context(fast):
    from str_e2e.backends import ar_scoring as a,global_probability as g
    original=a.raw_ctc_logprob;alias=g.raw_ctc_logprob
    assert original is alias,'unexpected CTC alias'
    scorer=Scorer(original,fast)
    a.raw_ctc_logprob=g.raw_ctc_logprob=scorer
    try:yield scorer
    finally:
        a.raw_ctc_logprob=original;g.raw_ctc_logprob=alias
        scorer.cache.clear();scorer.nbytes=0
