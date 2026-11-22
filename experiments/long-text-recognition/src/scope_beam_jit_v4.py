"""Parity-guarded Numba replacement for Python CTC prefix beam-search hot loop.

Algorithm unchanged: blank+repeat CTC prefix recurrence, top-k pruning,
beam width and order, n-best. No labels, ground truth or dataset information.
Requires numerical and prediction parity before activation.
"""
from __future__ import annotations
import math
import numpy as np
import space_ctc as SPACE
from longtext_final import normalize_ltb

_ORIGINAL=SPACE.prefix_beam_search
_READY=False
_ORIGINAL_FALLBACKS=0
_PARITY_CHECKED=0
_AUDIT_REMAINING=0


def _kernel_factory():
    from numba import njit
    @njit(cache=True)
    def la(a,b):
        if a==-np.inf:return b
        if b==-np.inf:return a
        m=a if a>b else b
        return m+math.log(math.exp(a-m)+math.exp(b-m))
    @njit(cache=True)
    def _run(lp,topmask,beam_width,nbest):
        T,C=lp.shape
        B=beam_width
        # All surviving prefixes have length <= T.
        current=np.zeros((B,T+1),dtype=np.int16)
        clens=np.zeros(B,dtype=np.int32)
        cpb=np.full(B,-np.inf,dtype=np.float64)
        cpnb=np.full(B,-np.inf,dtype=np.float64)
        cpb[0]=0.
        nb=1
        K=B*(C+1)
        ambiguity=False
        cand=np.zeros((K,T+1),dtype=np.int16)
        lens=np.zeros(K,dtype=np.int32)
        pb=np.full(K,-np.inf,dtype=np.float64)
        pnb=np.full(K,-np.inf,dtype=np.float64)
        selected=np.zeros(C,dtype=np.bool_)
        for t in range(T):
            for c in range(C):selected[c]=topmask[t,c]
            for i in range(nb):
                if clens[i]>0:selected[current[i,clens[i]-1]]=True
            count=0
            for i in range(nb):
                plen=clens[i]
                pblank=cpb[i]
                pnonblank=cpnb[i]
                whole=la(pblank,pnonblank)
                # Blank: same prefix, blank accumulator.
                index=-1
                for j in range(count):
                    if lens[j]==plen:
                        equal=True
                        for k in range(plen):
                            if cand[j,k]!=current[i,k]:equal=False;break
                        if equal:index=j;break
                if index==-1:
                    index=count;count+=1
                    lens[index]=plen
                    for k in range(plen):cand[index,k]=current[i,k]
                    pb[index]=-np.inf;pnb[index]=-np.inf
                pb[index]=la(pb[index],whole+lp[t,0])
                last=current[i,plen-1] if plen>0 else -1
                for c in range(1,C):
                    if not selected[c]:continue
                    pc=lp[t,c]
                    if c==last:
                        # Repeating label without blank: retain same prefix.
                        index=-1
                        for j in range(count):
                            if lens[j]==plen:
                                equal=True
                                for k in range(plen):
                                    if cand[j,k]!=current[i,k]:equal=False;break
                                if equal:index=j;break
                        if index==-1:
                            index=count;count+=1
                            lens[index]=plen
                            for k in range(plen):cand[index,k]=current[i,k]
                            pb[index]=-np.inf;pnb[index]=-np.inf
                        pnb[index]=la(pnb[index],pnonblank+pc)
                        branch_score=pblank+pc
                    else:
                        branch_score=whole+pc
                    extended=plen+1
                    index=-1
                    for j in range(count):
                        if lens[j]==extended and cand[j,plen]==c:
                            equal=True
                            for k in range(plen):
                                if cand[j,k]!=current[i,k]:equal=False;break
                            if equal:index=j;break
                    if index==-1:
                        index=count;count+=1
                        lens[index]=extended
                        for k in range(plen):cand[index,k]=current[i,k]
                        cand[index,plen]=c
                        pb[index]=-np.inf;pnb[index]=-np.inf
                    pnb[index]=la(pnb[index],branch_score)
            if count==0:raise ValueError('no beam')
            # Conservative fallback on exact/near ties: Python's set/dict
            # iteration can break equal-score ties in a different order.
            totals=np.empty(count,dtype=np.float64)
            for j in range(count):totals[j]=la(pb[j],pnb[j])
            for j in range(count):
                if totals[j]==-np.inf:continue
                for k in range(j+1,count):
                    if abs(totals[j]-totals[k])<=1e-10:
                        ambiguity=True
            # Stable top B by total likelihood; no string semantics in sorting.
            keep=np.ones(count,dtype=np.bool_)
            new_n=min(count,B)
            for i in range(new_n):
                best=-1;maxscore=-np.inf
                for j in range(count):
                    if not keep[j]:continue
                    score=la(pb[j],pnb[j])
                    if best==-1 or score>maxscore:
                        best=j;maxscore=score
                keep[best]=False
                clens[i]=lens[best]
                for k in range(lens[best]):current[i,k]=cand[best,k]
                cpb[i]=pb[best];cpnb[i]=pnb[best]
            nb=new_n
        out=np.zeros((min(nb,nbest),T+1),dtype=np.int16)
        outlen=np.zeros(len(out),dtype=np.int32)
        outscore=np.zeros(len(out),dtype=np.float64)
        # current already sorted by prefix likelihood
        for i in range(len(out)):
            outlen[i]=clens[i]
            outscore[i]=la(cpb[i],cpnb[i])
            for k in range(clens[i]):out[i,k]=current[i,k]
        return out,outlen,outscore,ambiguity
    return _run

_KERNEL=None


def prefix_beam_search(log_probs,chars,beam_width=8,topk=6,nbest=8):
    global _ORIGINAL_FALLBACKS,_PARITY_CHECKED,_AUDIT_REMAINING
    lp=np.asarray(log_probs,dtype=np.float64)
    if lp.ndim!=2 or lp.shape[1]!=len(chars) or lp.shape[0]<1:
        raise ValueError('Invalid prefix beam input')
    if beam_width<1 or topk<1:
        raise ValueError('Beam width and topk must be >=1')
    topmask=np.zeros(lp.shape,dtype=np.bool_)
    k=min(topk,len(chars))
    topinds=np.argpartition(lp,-k,axis=1)[:,-k:]
    np.put_along_axis(topmask,topinds,True,axis=1)
    seq,lens,scores,ambiguous=_KERNEL(lp,topmask,beam_width,nbest)
    if ambiguous:
        _ORIGINAL_FALLBACKS += 1
        return _ORIGINAL(log_probs,chars,beam_width=beam_width,topk=topk,nbest=nbest)
    out=[]
    for i in range(len(lens)):
        pred=normalize_ltb(''.join(chars[int(j)] for j in seq[i,:lens[i]]))
        out.append({'pred':pred,'logp':float(scores[i])})
    if _AUDIT_REMAINING>0:
        old=_ORIGINAL(log_probs,chars,beam_width=beam_width,topk=topk,nbest=nbest)
        if len(old)!=len(out) or any(a['pred']!=b['pred'] or abs(a['logp']-b['logp'])>1e-8 for a,b in zip(old,out)):
            raise RuntimeError('PREFIX_BEAM_PARITY_FAILURE: stop inference, do not use scores')
        _PARITY_CHECKED+=1
        _AUDIT_REMAINING-=1
    return out


def activate(strict=True,audit_first=128):
    global _KERNEL,_READY,_AUDIT_REMAINING,_ORIGINAL_FALLBACKS,_PARITY_CHECKED
    if _READY:return {'status':'already_active','available':True}
    try:
        _KERNEL=_kernel_factory()
        # First compile and compare random, low blank, high blank, repeated letters and ties.
        rng=np.random.default_rng(20261121)
        for T in (1,2,3,4,9,16,48,80):
            for case in range(6):
                q=rng.uniform(1e-6,1,size=(T,37))
                if case==1:q[:,0]*=70
                if case==2:q[:,0]*=.05
                if case==3:q.fill(.5)
                if case==4:q[:,1]=q[:,2]=10
                if case==5:q[:,0]=q[:,1]=10
                q/=q.sum(axis=1,keepdims=True)
                for b,top,n in ((6,5,3),(3,3,3)):
                    old=_ORIGINAL(np.log(q),['']+list('0123456789abcdefghijklmnopqrstuvwxyz'),beam_width=b,topk=top,nbest=n)
                    new=prefix_beam_search(np.log(q),['']+list('0123456789abcdefghijklmnopqrstuvwxyz'),beam_width=b,topk=top,nbest=n)
                    if [x['pred'] for x in old]!=[x['pred'] for x in new] or not all(abs(x['logp']-y['logp'])<1e-9 for x,y in zip(old,new)):
                        raise AssertionError((T,case,b,top,old[:2],new[:2]))
        SPACE.prefix_beam_search=prefix_beam_search
        _AUDIT_REMAINING=int(audit_first)
        _PARITY_CHECKED=0
        _ORIGINAL_FALLBACKS=0
        _READY=True
        return {'status':'numba_prefix_beam_parity_pass','available':True}
    except Exception as exc:
        SPACE.prefix_beam_search=_ORIGINAL
        _READY=False
        if strict:raise
        return {'status':str(exc),'available':False}


def benchmark(n=50,T=48):
    global _AUDIT_REMAINING
    preserved_audit=_AUDIT_REMAINING
    _AUDIT_REMAINING=0
    rng=np.random.default_rng(201)
    q=rng.uniform(.0001,1,(T,37));q[:,0]*=5;q/=q.sum(axis=1,keepdims=True)
    args=(np.log(q),['']+list('0123456789abcdefghijklmnopqrstuvwxyz'))
    import time
    a=time.perf_counter()
    for i in range(n):_ORIGINAL(*args,beam_width=6,topk=5,nbest=3)
    b=time.perf_counter()
    for i in range(n):prefix_beam_search(*args,beam_width=6,topk=5,nbest=3)
    c=time.perf_counter()
    _AUDIT_REMAINING=preserved_audit
    return {'original_ms':(b-a)*1e3/n,'fast_ms':(c-b)*1e3/n,'ratio':(b-a)/(c-b),'auditing_paused_during_microbenchmark':True}


def audit_status():
    return {'audit_checked':int(_PARITY_CHECKED),'audit_remaining':int(_AUDIT_REMAINING),
            'tie_fallbacks':int(_ORIGINAL_FALLBACKS),'active':bool(_READY)}
