"""SCOPE-CTC E7 exact-logic acceleration; no training and no GT.

Guarantee sought: bitwise identical final transcripts to E7.decode for the same
CTC crop logits. Deduplicate beam proposal generation and local CTC scoring.
No scale skipping or threshold modification in the parity-preserving mode.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import alex_e7 as LEG
import alex_ctc as AX
import alex_e5 as E5


def _key(p):
    return (p.start,p.end,p.new)


def _make_view(raw,span,group,scale,chars,ref_events):
    q=AX.normalized_probs(raw,chars)
    a,b=span
    ids=[j for j,e in enumerate(ref_events) if a<=e.x<=b]
    q=np.asarray(q,dtype=np.float64)
    if not ids or ids!=list(range(ids[0],ids[-1]+1)):
        frames=None
    else:
        tx=''.join(ref_events[i].char for i in ids)
        frames,_=E5.fast_forced_frames(q,tx)
    return {'logits':raw,'span':tuple(span),'group':int(group),'scale':str(scale),
            'q':q,'logq':np.log(np.maximum(q,1e-35)),'indices':ids,'frames':frames}


def _score(p,reference,views,bf_cache):
    groups=defaultdict(list)
    for v in views:
        k=(v['group'],v['scale'],p)
        if k not in bf_cache:
            bf_cache[k]=LEG.local_bf(v,p,reference)
        d=bf_cache[k]
        if d is not None and np.isfinite(d):
            groups[v['group']].append(float(d))
    if len(groups)<2:return None
    values=[float(np.median(g)) for g in groups.values()]
    support=sum(s>0 for s in values)
    if support<2:return None
    raw=float(np.median(values))
    return raw-(1.0+0.6*max(p.end-p.start,len(p.new)))


def _solve(scope,reference,ps,witnesses,views,nb,bf_cache):
    scores={}
    for p in ps:
        key=_key(p)
        if len({g for g,s in witnesses.get(key,set())})<2:continue
        score=_score(p,reference,views,bf_cache)
        if score is not None and score>0:scores[p]=score
    text,ops,gain=LEG.select_edits(reference,scores)
    info={'candidates':len(ps),'scored':len(scores),'accepted':len(ops),
          'insertion_chars':sum(len(p.new) for p in ops if p.start==p.end),
          'deleted_chars':sum(p.end-p.start for p in ops if not p.new),
          'beam_hypotheses':nb,'gain':float(gain),
          'edits':'|'.join(f'{p.start}:{p.end}:{p.new}' for p in ops)}
    return text,{scope+'_'+k:v for k,v in info.items()}


def decode(reference,base_logits,base_spans,extra_logits,extra_spans,chars,*,return_info=False):
    if len(base_logits)!=len(base_spans) or len(extra_logits)!=len(extra_spans):
        raise ValueError('Logits/spans mismatch')
    raw_ref=AX.reference_events(base_logits,base_spans,chars)
    events=LEG.mapped_events(reference,raw_ref)
    if not events or not reference:
        fields={'candidates':0,'scored':0,'accepted':0,'insertion_chars':0,
                'deleted_chars':0,'beam_hypotheses':0,'gain':0.,'edits':''}
        info={f'{s}_{k}':v for s in ('native','cross') for k,v in fields.items()}
        return (reference,reference,info) if return_info else reference
    base=[_make_view(q,sp,i,'base',chars,events) for i,(q,sp) in enumerate(zip(base_logits,base_spans))]
    group_map={tuple(sp):i for i,sp in enumerate(base_spans)}
    extra=[]
    for q,sp in zip(extra_logits,extra_spans):
        if tuple(sp) not in group_map:raise ValueError('Extra view span not matched to base crop')
        extra.append(_make_view(q,sp,group_map[tuple(sp)],'alt',chars,events))
    ps_base,wit_base,nb_base=LEG.generate_proposals(reference,events,base,chars)
    bf_cache={}
    native,info_native=_solve('native',reference,ps_base,wit_base,base,nb_base,bf_cache)
    if not extra:
        cross,info_cross=_solve('cross',reference,ps_base,wit_base,base,nb_base,bf_cache)
    else:
        ps_extra,wit_extra,nb_extra=LEG.generate_proposals(reference,events,extra,chars)
        all_candidates={_key(p):p for p in ps_base}
        for p in ps_extra:all_candidates[_key(p)]=p
        all_witnesses=defaultdict(set)
        for k,v in wit_base.items():all_witnesses[k].update(v)
        for k,v in wit_extra.items():all_witnesses[k].update(v)
        cross,info_cross=_solve('cross',reference,list(all_candidates.values()),
                                 all_witnesses,base+extra,nb_base+nb_extra,bf_cache)
    info={**info_native,**info_cross}
    return (native,cross,info) if return_info else cross


def cpu_tests():
    assert LEG.cpu_tests()['passed']
    chars=['blank','a','b','c']
    def logit(seq):
        arr=np.full((len(seq),4),-8.,np.float32)
        for i,j in enumerate(seq):arr[i,j]=8.
        return arr
    X=logit([1,0,2,0,3,0,1,0,2,0,3])
    for ref,base,spans,extra in [
        ('abcabc',[X,X],[(0,60),(0,60)],[]),
        ('', [np.tile(logit([0]),(9,1))],[(0,60)],[]),
    ]:
        old=LEG.decode(ref,base,spans,extra,spans if extra else [],chars,return_info=True)
        new=decode(ref,base,spans,extra,spans if extra else [],chars,return_info=True)
        assert old==new,(old,new)
    return {'passed':True,'no_math_or_threshold_changes':True}
