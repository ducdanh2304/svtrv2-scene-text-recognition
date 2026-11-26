"""Frozen candidate-conditional pooling and AR-aware SCOPE edit evidence.
All evidence is GT-blind. No fitted calibration or domain-dependent weights.
"""
from __future__ import annotations
import math
from collections import defaultdict
from difflib import SequenceMatcher
import numpy as np
from .ar_scoring import raw_ctc_logprob


def lse(values):
    a=np.asarray(list(values),dtype=np.float64)
    return float(np.logaddexp.reduce(a)) if len(a) else -math.inf


def conditional_decode(strings,zs,chars,ar,normalize):
    """Equal mixture of each expert's posterior conditional on the nominated raw set."""
    strings=list(dict.fromkeys(strings)); tables=[]
    for z in zs:
        tables.append([raw_ctc_logprob(z,s,chars) for s in strings])
    if ar is not None:
        tables.append([ar[s] if ar[s] is not None else -math.inf for s in strings])
    rows=[]
    for t in tables:
        den=lse(t)
        if math.isfinite(den): rows.append(np.asarray(t)-den)
    if not rows: raise RuntimeError('No expert supports any candidate')
    scores=np.logaddexp.reduce(np.stack(rows),axis=0)-math.log(len(rows))
    grouped={}
    for s,v in zip(strings,scores):
        key=normalize(s);grouped[key]=float(np.logaddexp(grouped.get(key,-math.inf),v))
    return max(grouped,key=grouped.get),grouped,len(rows)


def pooled_logodds(bfs):
    """logit(mean(sigmoid(BF))); stable in extreme odds, no clipping."""
    a=np.asarray(list(bfs),dtype=np.float64)
    if not len(a) or np.isnan(a).any():raise ValueError('Invalid evidence odds')
    pos=lse(-np.logaddexp(0.,-a)); neg=lse(-np.logaddexp(0.,a))
    return pos-neg


def variants(text,observed,normalize):
    """Restricted raw variants, NOT the full normalized-string event."""
    return list(dict.fromkeys([text,text.upper(),text.title()]+[s for s in observed if normalize(s)==text]))


def crop_pair(p,reference,ids,span):
    if not ids or ids!=list(range(ids[0],ids[-1]+1)):return None
    a,b=span
    if not a+.10*(b-a)<=p.x<=b-.10*(b-a):return None
    l,r=ids[0],ids[-1]+1
    if not l<=p.start<=p.end<=r:return None
    plain=reference[l:r];alt=reference[l:p.start]+p.new+reference[p.end:r]
    return (plain,alt) if plain!=alt else None


def ar_pair_bf(pair,table,observed,normalize):
    if table is None:return None
    a,b=pair
    aa=variants(a,observed,normalize);bb=variants(b,observed,normalize)
    if not all(s in table for s in aa+bb):return None
    def mass(ss):return lse(table[s] for s in ss if table[s] is not None)
    x,y=mass(aa),mass(bb)
    # Unsupported / absent event is not evidence against a proposal.
    if not math.isfinite(x) or not math.isfinite(y):return None
    return y-x


def spatial_final(reference,cap,chars,s2,r3,local_texts,fuse_texts,waterec_texts,raw_candidates,worker,ar_capacity,ar_available):
    """Reuse E7 geometry, proposals, two-crop witnesses, penalty, and lattice.
    Match CTC-only and CTC+AR with the same proposal pool and pooling objective.
    AR contributes full-crop counterfactual odds for the SAME local edit, not
    fabricated CTC frames. CTC contributes original E7 segment-level odds.
    """
    import alex_ctc as AX
    import alex_e7 as E7
    import scope_e7_fast as FAST
    import longtext_final as LT
    reference=LT.normalize_ltb(reference)
    events=E7.mapped_events(reference,AX.reference_events(cap['base'],cap['spans'],chars))
    if not events or not reference:return reference,reference,dict(reason='empty_geometry',ar_pairs=0)
    group={tuple(sp):i for i,sp in enumerate(cap['spans'])};views=[]
    for family,zs,spans in [('native',cap['base'],cap['spans']),('native',cap['extra'],cap['extra_spans']),('motion',s2,cap['spans']),('artistic',r3,cap['spans'])]:
        for k,(z,sp) in enumerate(zip(zs,spans)):
            v=FAST._make_view(z,sp,group[tuple(sp)],f'{family}:{len(views)}',chars,events);v['family']=family;views.append(v)
    proposals,witnesses,nb=E7.generate_proposals(reference,events,views,chars)
    by_key={(p.start,p.end,p.new):p for p in proposals}; ids_by_group={}
    for i,sp in enumerate(cap['spans']):
        a,b=sp;ids=[j for j,e in enumerate(events) if a<=e.x<=b];ids_by_group[i]=ids
        if not ids or ids!=list(range(ids[0],ids[-1]+1)):continue
        local=''.join(reference[j] for j in ids)
        for name,txt in [('joint',local_texts[i]),('fuse',fuse_texts[i]),('ar',waterec_texts[i])]:
            text=LT.normalize_ltb(txt)
            if not text:continue
            ops=SequenceMatcher(None,local,text,autojunk=False).get_opcodes()
            for j,op in enumerate(ops):
                if not 0<j<len(ops)-1 or ops[j-1][0]!='equal' or ops[j+1][0]!='equal':continue
                p=E7.new_edit(op,text,events,ids,[],a,b)
                if p is None:continue
                key=(p.start,p.end,p.new);by_key.setdefault(key,p);witnesses[key].add((i,name))
    eligible=[p for k,p in by_key.items() if len({g for g,_ in witnesses[k]})>=2]
    ctc={};pairs=defaultdict(dict); requests=defaultdict(set)
    for p in eligible:
        g=defaultdict(lambda:defaultdict(list))
        for v in views:
            bf=E7.local_bf(v,p,reference)
            if bf is not None and np.isfinite(bf):g[v['group']][v['family']].append(float(bf))
        ctc[p]={i:[float(np.median(x)) for x in families.values()] for i,families in g.items()}
        if len(g)<2:continue
        for i in g:
            if not ar_available[i]:continue
            pair=crop_pair(p,reference,ids_by_group[i],cap['spans'][i])
            if pair is None:continue
            vv=variants(pair[0],raw_candidates[i],LT.normalize_ltb)+variants(pair[1],raw_candidates[i],LT.normalize_ltb)
            if any(len(s)>ar_capacity[i] for s in vv):continue
            pairs[i][p]=pair;requests[i].update(vv)
    ar_evidence=defaultdict(dict);ar_requests=0
    for i,strings in sorted(requests.items()):
        worker.predict(cap['parts'][i]);response=worker.score(sorted(strings));ar_requests+=len(strings)
        table=response['scores']
        for p,pair in pairs[i].items():
            bf=ar_pair_bf(pair,table,raw_candidates[i],LT.normalize_ltb)
            if bf is not None:ar_evidence[p][i]=bf
    def solve(with_ar):
        scores={};details={}
        for p,g in ctc.items():
            if len(g)<2:continue
            vals=[]
            for i,bfs in g.items():
                b=list(bfs)
                if with_ar and i in ar_evidence[p]:b.append(ar_evidence[p][i])
                vals.append(pooled_logodds(b))
            if sum(v>0 for v in vals)<2:continue
            score=float(np.median(vals))-(1.0+.6*max(p.end-p.start,len(p.new)))
            if score>0:scores[p]=score;details[p]=vals
        text,ops,gain=E7.select_edits(reference,scores)
        return text,dict(scored=len(scores),accepted=len(ops),gain=float(gain),edits=[dict(start=p.start,end=p.end,new=p.new,group_odds=details[p],ar_groups=len(ar_evidence[p])) for p in ops])
    c,ci=solve(False);a,ai=solve(True)
    return c,a,dict(candidates=len(by_key),eligible=len(eligible),beam_hypotheses=nb,ar_requests=ar_requests,ar_pairs=sum(len(v) for v in ar_evidence.values()),ctc=ci,ar=ai)
