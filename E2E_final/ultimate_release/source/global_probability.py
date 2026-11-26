"""Global hypothesis decoding. No recognizer weights are updated.
Candidate scores are restricted raw-string masses, not full equivalence classes.
Spatial composite likelihood assumes aligned crop projections; dependence is
mitigated by overlap coverage weights, not assumed absent.
"""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np
from rapidfuzz.distance import Levenshtein
from .ar_scoring import raw_ctc_logprob


def norm(s):
    return ''.join(c for c in str(s).lower() if c in 'abcdefghijklmnopqrstuvwxyz0123456789')


def project(ref, alt, coords, span):
    """Minimum-edit alignment, complete path retained; assign insertion to
    nearest boundary midpoint. Projection is evidence only, never a filter.
    """
    if not ref or len(coords)!=len(ref):
        raise ValueError('Missing reference geometry')
    pos=[]
    for op in Levenshtein.opcodes(ref,alt):
        a,b,c,d=op.src_start,op.src_end,op.dest_start,op.dest_end
        if op.tag=='equal': pos.extend(zip(alt[c:d],coords[a:b]))
        elif d>c:
            if b>a:
                x=np.interp(np.linspace(a,b-1,d-c),np.arange(len(ref)),coords)
            else:
                x=np.full(d-c, (coords[a-1]+coords[a])/2 if 0<a<len(ref) else coords[0] if a==0 else coords[-1])
            pos.extend(zip(alt[c:d],x))
    assert ''.join(c for c,x in pos)==alt
    return ''.join(c for c,x in pos if span[0]<=x<=span[1])


def coverage_weights(spans,coords):
    a=np.array([[l<=x<=r for x in coords] for l,r in spans],float)
    count=a.sum(axis=0); valid=count>0
    if not valid.any():return np.ones(len(spans))/len(spans)
    return (a[:,valid]/count[valid]).sum(axis=1)/valid.sum()


def variants(text,observed):
    return list(dict.fromkeys([text,text.upper(),text.title()]+[s for s in observed if norm(s)==text]))


def lse(a):
    a=np.asarray(a,dtype=float)
    return float(np.logaddexp.reduce(a)) if a.size else -math.inf


def prob(logmass):
    a=np.asarray(logmass,dtype=float); d=lse(a)
    return np.exp(a-d) if math.isfinite(d) else None


def raw_masses(strings,logits,chars,observed):
    """Exact CTC raw probabilities, sum over a finite variant set."""
    cache={}
    for t in set(strings):
        cache[t]=lse([raw_ctc_logprob(logits,s,chars) for s in variants(t,observed)])
    return np.asarray([cache[t] for t in strings])


def ar_masses(strings,worker,image,observed):
    sets={t:variants(t,observed) for t in set(strings)}
    raw=sorted({s for vs in sets.values() for s in vs})
    worker.predict(image)
    table=worker.score(raw)['scores']
    if table is None:return None
    return np.array([lse([table[s] for s in sets[t] if table[s] is not None]) for t in strings])


def combine_views(rows,weights):
    """Coverage-weighted composite likelihood over the *same* paths.
    If support intersection is empty, this source has no coherent spatial
    likelihood; retain its independent full-image distribution, if any.
    """
    valid=[(np.asarray(x),float(w)) for x,w in zip(rows,weights) if x is not None and w>0 and np.isfinite(x).any()]
    if not valid:return None
    total=sum(w for x,w in valid)
    val=sum(x*(w/total) for x,w in valid)
    return prob(val)


def average(*ps):
    valid=[p for p in ps if p is not None]
    return np.mean(valid,axis=0) if valid else None


def distribution(p, temp):
    if p is None:return None
    with np.errstate(divide='ignore'):
        return prob(np.log(np.asarray(p))/float(temp))


def pool(table,weights=None,temperatures=None):
    """Static mixture; missing source is renormalized for the whole image."""
    n=len(table);weights=np.ones(n)/n if weights is None else np.asarray(weights,float)
    temperatures=np.ones(n) if temperatures is None else np.asarray(temperatures,float)
    available=[(distribution(p,t),w) for p,w,t in zip(table,weights,temperatures) if p is not None and w>0]
    if not available:raise ValueError('No supported source distribution')
    p=sum(q*w for q,w in available)/sum(w for q,w in available)
    return p/p.sum()


def choose(candidates,table,weights=None,temperatures=None):
    p=pool(table,weights,temperatures)
    # MAP for exact-line; MBR is a distinct CER-oriented ablation.
    map_text=candidates[int(p.argmax())]
    mbr=min(candidates,key=lambda t:sum(float(w)*Levenshtein.distance(t,y) for w,y in zip(p,candidates)))
    return map_text,mbr,p.tolist()


def fit_calibration(rows):
    """Only five scalar degrees of freedom: three temperatures and two
    independent mixture logits. Existing dev only; no classifier or backbone
    training. Conditional NLL is defined only when the target is nominated.
    """
    from scipy.optimize import minimize
    from scipy.special import softmax
    usable=[]
    for r in rows:
        gt=norm(r['gt']); cand=r['candidates']; table=r['table']
        if gt in cand and any(p is not None and p[cand.index(gt)]>0 for p in table):
            usable.append((table,cand.index(gt),r['source_group_id']))
    if len(usable)<30:raise RuntimeError('Insufficient supported calibration examples')
    counts=defaultdict(int)
    for _,_,g in usable:counts[g]+=1
    def pars(x):return softmax([x[0],x[1],0.]),np.exp(x[2:])
    def loss(x):
        w,t=pars(x)
        return sum(-math.log(max(float(pool(p,w,t)[j]),1e-300))/counts[g] for p,j,g in usable)/len(counts)
    initial=np.zeros(5);base=loss(initial)
    # Bounds are numerical safeguards; not image-dependent decision thresholds.
    fit=minimize(loss,initial,method='L-BFGS-B',bounds=[(-5,5)]*2+[(-2.3,2.3)]*3,options={'maxiter':200})
    if not fit.success or not np.isfinite(fit.fun) or fit.fun>base+1e-9:raise RuntimeError(f'Calibration failed: {fit.message}')
    w,t=pars(fit.x)
    return dict(weights=w.tolist(),temperatures=t.tolist(),n=len(rows),supported=len(usable),source_groups=len(counts),nll_before=base,nll_after=float(fit.fun),fit_status=str(fit.message))
