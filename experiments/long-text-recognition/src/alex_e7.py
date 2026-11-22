"""ALEX E7 / SCOPE-CTC: scale-paired, segment-level counterfactual CTC decoding.

Research prototype. Uses frozen SVTRv2-T crop logits, no GT/IDs in inference.
The full CTC forward score is exact only for the nominated local substring and
crop-frame slice; the spatial segmentation and view dependence are approximations.
This is NOT a proof of literature novelty or generalization.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from collections import defaultdict
from itertools import combinations
import numpy as np
import longtext_v3 as V3
import alex_ctc as AX
import alex_c1 as C1
import alex_e5 as E5

@dataclass(frozen=True)
class SpanEdit:
    start: int
    end: int
    new: str
    x: float


def mapped_events(reference, source_events):
    """Interpolate character geometry for a C1-edited reference (GT-blind)."""
    source=''.join(e.char for e in source_events)
    if not source_events:
        return []
    coords=np.full(len(reference),np.nan)
    matcher=SequenceMatcher(None,source,reference,autojunk=False)
    for tag,a,b,c,d in matcher.get_opcodes():
        if d==c:continue
        if tag=='equal':
            coords[c:d]=[e.x for e in source_events[a:b]]
        else:
            lo=float(source_events[a-1].x) if a>0 else float(source_events[0].x)-3.
            hi=float(source_events[b].x) if b<len(source_events) else float(source_events[-1].x)+3.
            coords[c:d]=np.linspace(lo,hi,d-c+2)[1:-1]
    if np.isnan(coords).any():raise ValueError('Reference coordinate mapping incomplete')
    coords=np.maximum.accumulate(coords + np.arange(len(coords))*1e-5)
    return [V3.CharEvidence(ch,1.,float(x)) for ch,x in zip(reference,coords)]


def new_edit(proposal, text, ref_events, ids, observed_events, src_a,src_b):
    tag,u0,u1,v0,v1=proposal
    if tag=='equal':return None
    if tag not in ('insert','delete','replace'):return None
    if (u1-u0)>6 or (v1-v0)>6:return None
    if tag=='insert' and v0==v1:return None
    start=ids[u0] if u0<len(ids) else ids[-1]+1
    end=ids[u1-1]+1 if u1>u0 else start
    if end-start>6 or len(text[v0:v1])>6 or (end==start and v0==v1):return None
    new=text[v0:v1]
    if any(ch not in AX.ALPH for ch in new):return None
    if start==end:
        if not ref_events:return None
        if start==0:x=ref_events[0].x
        elif start==len(ref_events):x=ref_events[-1].x
        else:x=.5*(ref_events[start-1].x+ref_events[start].x)
    else:x=np.mean([e.x for e in ref_events[start:end]])
    if not (src_a+.12*(src_b-src_a) <= x <= src_b-.12*(src_b-src_a)):
        return None
    if v1>v0 and observed_events:
        wx=np.mean([e.x for e in observed_events[v0:v1]])
        if abs(wx-x)>max(5.,.19*(src_b-src_a)):
            return None
    return SpanEdit(int(start),int(end),new,float(x))


def generate_proposals(reference, ref_events, views, chars, beam_width=6, nbest=3):
    """GT-blind proposals from CTC greedy AND N-best on each view."""
    props={}; witnesses=defaultdict(set); beam_count=0
    for v in views:
        a,b=v['span']
        ids=[j for j,e in enumerate(ref_events) if a<=e.x<=b]
        if not ids or ids!=list(range(ids[0],ids[-1]+1)):continue
        # One virtual reference anchor on each side disambiguates an internal
        # insertion near overlap boundaries; only centrally witnessed edits
        # can survive the following spatial check.
        ids=list(range(max(0,ids[0]-1),min(len(reference),ids[-1]+2)))
        local=''.join(reference[j] for j in ids)
        raw=v['logits']
        greedy=V3.ctc_events(raw,chars,a,b)
        source=[(''.join(e.char for e in greedy),greedy)]
        q=AX.normalized_probs(raw,chars)
        lp=np.log(np.maximum(q,1e-35))
        beams=E5.beam_events(lp,(a,b),beam_width=beam_width,topk=5,nbest=nbest)
        beam_count+=len(beams)
        source.extend((text,ev) for text,ev,_ in beams)
        seen=set()
        for txt,events in source:
            if txt in seen:continue
            seen.add(txt)
            for opc in SequenceMatcher(None,local,txt,autojunk=False).get_opcodes():
                p=new_edit(opc,txt,ref_events,ids,events,a,b)
                if p is None:continue
                key=(p.start,p.end,p.new)
                props[key]=p;witnesses[key].add((v['group'],v['scale']))
    return list(props.values()),witnesses,beam_count


def make_view(raw,span,group,scale,chars,ref_events):
    q=AX.normalized_probs(raw,chars)
    a,b=span
    ids=[j for j,e in enumerate(ref_events) if a<=e.x<=b]
    q=np.asarray(q,dtype=np.float64)
    if not ids or ids!=list(range(ids[0],ids[-1]+1)):
        frames=None
    else:
        tx=''.join(ref_events[i].char for i in ids)
        frames,_=AX.ctc_viterbi(q,tx)
    return {'logits':raw,'span':tuple(span),'group':int(group),'scale':str(scale),
            'q':q,'logq':np.log(np.maximum(q,1e-35)),'indices':ids,'frames':frames}


def local_bf(view,p,reference,anchors=2):
    ids=view['indices']; frames=view['frames']
    if frames is None or not ids:return None
    a,b=view['span']
    if not (a+.10*(b-a) <= p.x <= b-.10*(b-a)):return None
    if p.start<ids[0] or p.end>ids[-1]+1:return None
    local_l=max(ids[0],p.start-anchors)
    local_r=min(ids[-1]+1,p.end+anchors)
    if local_r<=local_l:return None
    first=local_l-ids[0];last=local_r-1-ids[0]
    q=view['logq'];T=len(q)
    slack=max(3,int(round(T*.05)))
    projected=int(np.clip(round((p.x-a)/max(b-a,1)*T),0,T-1))
    # Do not crop away the very gap we are attempting to recover: a missing
    # span may have no reference-aligned token near its actual visual frames.
    tl=max(0,min(frames[first],projected)-slack)
    tr=min(T,max(frames[last],projected)+slack+1)
    if tr-tl<4:return None
    plain=reference[local_l:local_r]
    alt=reference[local_l:p.start]+p.new+reference[p.end:local_r]
    if alt==plain:return None
    qpiece=q[tl:tr]
    lp_plain=C1.ctc_forward_logprob(qpiece,plain)
    lp_alt=C1.ctc_forward_logprob(qpiece,alt)
    if not np.isfinite(lp_plain) or not np.isfinite(lp_alt):return None
    return float(lp_alt-lp_plain)


def robust_score(p,reference,views):
    """Median across scales in a crop, then across distinct crop groups."""
    groups=defaultdict(list)
    for v in views:
        d=local_bf(v,p,reference)
        if d is not None and np.isfinite(d):groups[v['group']].append(float(d))
    if len(groups)<2:return None
    scores=[float(np.median(g)) for g in groups.values()]
    support=sum(s>0 for s in scores)
    if support<2:return None
    # Group averaging is robust to repeated scale views within one image region.
    score=float(np.median(scores))
    cost=1.0+.6*max(p.end-p.start,len(p.new))
    return score-cost,{'group_count':len(groups),'support_groups':support,
                       'group_median':score,'cost':cost}


def compatible(a,b):
    # insertion slot occupies a virtual half-character interval.
    sa=2*a.start if a.start==a.end else 2*a.start+1
    ea=sa+1 if a.start==a.end else 2*a.end+1
    sb=2*b.start if b.start==b.end else 2*b.start+1
    eb=sb+1 if b.start==b.end else 2*b.end+1
    return ea<=sb or eb<=sa


def select_edits(reference,weighted,max_candidates=20,max_accepted=3,max_char_budget=8):
    # Fixed-size sparse segmental lattice search, no GT or image ID.
    ranked=sorted(((p,float(v)) for p,v in weighted.items() if v>0),
                  key=lambda z:(-z[1],z[0].start,z[0].end,z[0].new))[:max_candidates]
    best=[];best_gain=0.
    for k in range(1,min(max_accepted,len(ranked))+1):
        for subset in combinations(ranked,k):
            chars=sum(max(p.end-p.start,len(p.new)) for p,_ in subset)
            if chars>max_char_budget:continue
            ps=sorted((p for p,_ in subset),key=lambda a:(a.start,a.end,a.new))
            if any(not compatible(ps[i],ps[j]) for i in range(len(ps)) for j in range(i+1,len(ps))):
                continue
            gain=sum(score for _,score in subset)
            if gain>best_gain+1e-9:
                best_gain=gain;best=ps
    result=reference
    for p in sorted(best,key=lambda p:(p.start,p.end),reverse=True):
        result=result[:p.start]+p.new+result[p.end:]
    return result,best,best_gain


def decode(reference,base_logits,base_spans,extra_logits,extra_spans,chars,*,return_info=False):
    if len(base_logits)!=len(base_spans) or len(extra_logits)!=len(extra_spans):
        raise ValueError('Logits/spans mismatch')
    raw_ref=AX.reference_events(base_logits,base_spans,chars)
    events=mapped_events(reference,raw_ref)
    if not events or not reference:
        empty_fields={'candidates':0,'scored':0,'accepted':0,
                      'insertion_chars':0,'deleted_chars':0,
                      'beam_hypotheses':0,'gain':0.0,'edits':''}
        info={f'{scope}_{key}':value for scope in ('native','cross')
              for key,value in empty_fields.items()}
        return (reference,reference,info) if return_info else reference
    base=[make_view(q,sp,i,'base',chars,events) for i,(q,sp) in enumerate(zip(base_logits,base_spans))]
    extra=[]
    # Pair cross-scale evidence by exact original-image crop span, not timestep.
    group_map={tuple(sp):i for i,sp in enumerate(base_spans)}
    for q,sp in zip(extra_logits,extra_spans):
        if tuple(sp) not in group_map:
            raise ValueError('Extra view span not matched to base crop')
        extra.append(make_view(q,sp,group_map[tuple(sp)],'alt',chars,events))
    pred=[];metadata=[]
    for scope,views in [('native',base),('cross',base+extra)]:
        ps,wit,nb=generate_proposals(reference,events,views,chars)
        scores={};dets={}
        for p in ps:
            # A positive CTC Bayes factor alone may favor repeated letters even
            # when no observation actually proposed the character. Require
            # independent spatial *candidate witnesses*, not just likelihood.
            key=(p.start,p.end,p.new)
            if len({group for group,_scale in wit.get(key,set())})<2:
                continue
            s,detail=None,None
            out=robust_score(p,reference,views)
            if out is not None:s,detail=out
            if s is not None and s>0:
                scores[p]=s;dets[p]=detail
        text,ops,gain=select_edits(reference,scores)
        pred.append(text)
        metadata.append({'scope':scope,'candidates':len(ps),'scored':len(scores),
                         'accepted':len(ops),'insertion_chars':sum(len(p.new) for p in ops if p.start==p.end),
                         'deleted_chars':sum(p.end-p.start for p in ops if not p.new),
                         'beam_hypotheses':nb,'gain':float(gain),
                         'edits':'|'.join(f'{p.start}:{p.end}:{p.new}' for p in ops)})
    info={'native_'+k:v for k,v in metadata[0].items() if k!='scope'}
    info.update({'cross_'+k:v for k,v in metadata[1].items() if k!='scope'})
    return (pred[0],pred[1],info) if return_info else pred[1]


def cpu_tests():
    # GT-blind text event coordinate mapping when C1 changed string length.
    source=[V3.CharEvidence(c,1.,float(i*10+5)) for i,c in enumerate('abcdef')]
    mapped=mapped_events('abcxdef',source)
    assert ''.join(v.char for v in mapped)=='abcxdef'
    assert all(mapped[i].x<=mapped[i+1].x for i in range(len(mapped)-1))
    p=SpanEdit(1,1,'x',10.)
    assert select_edits('ab',{p:3.})[0]=='axb'
    p=SpanEdit(1,2,'',10.)
    assert select_edits('abc',{p:3.})[0]=='ac'
    p=SpanEdit(1,3,'xy',10.)
    assert select_edits('abcd',{p:3.})[0]=='axyd'
    # Exact local likelihood verification with two independent spatial crops.
    alphabet=['blank','a','b','c']
    def logits(v):
        z=np.full((len(v),4),-8.,np.float32)
        for i,k in enumerate(v):z[i,k]=8.
        return z
    q=logits([1,0,2,0,3,0,1,0,2,0,3])
    native,cross,info=decode('abcabc',[q,q],[(0,60),(0,60)],[],[],alphabet,return_info=True)
    assert native=='abcabc' and cross=='abcabc'
    assert info['cross_accepted']==0
    blank=np.full((12,4),-8.,np.float32)
    blank[:,0]=8.
    for reference in ('','ab'):
        native,cross,meta=decode(reference,[blank],[(0,80)],[blank],[(0,80)],alphabet,return_info=True)
        assert native==cross==reference
        assert meta['native_candidates']==meta['cross_candidates']==0
        assert meta['native_accepted']==meta['cross_accepted']==0
        assert meta['cross_edits']==''
    # Posterior-only character recovery: neither base view's greedy contains c,
    # paired higher-resolution views expose it in TWO overlapping crop groups.
    def paired_logits(group,extra=False):
        p=np.full((18,4),1e-5);p[:,0]=.99997
        for frame,lab in ((3,1),(13,2)):
            p[frame]=[.001,.001,.001,.001];p[frame,lab]=.997
        t=9 if group==0 else 6
        p[t]=[.80,.001,.001,.198] if not extra else [.01,.001,.001,.988]
        p/=p.sum(axis=1,keepdims=True)
        return np.log(p)
    psp=[(0,80),(20,100)]
    lows=[paired_logits(0),paired_logits(1)]
    highs=[paired_logits(0,True),paired_logits(1,True)]
    ref=AX.decode(lows,psp,alphabet)
    assert ref=='ab'
    a,b,debug=decode(ref,lows,psp,highs,psp,alphabet,return_info=True)
    assert a=='ab' and b=='acb' and debug['cross_accepted']==1,(a,b,debug)
    return {'passed':True,'span_insertion_deletion_replacement':3,'ref_coordinates':True,
            'no_unwarranted_edit':True,'posterior_only_cross_scale_missing_char':'ab -> acb'}
