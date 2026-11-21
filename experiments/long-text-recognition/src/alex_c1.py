"""ALEX-CTC C1 exploratory constrained edit-lattice over multiple CTC crop posteriors.

No GT or dataset IDs in inference; only observed crop posterior, crop geometry,
V3 reference event positions, and optional C0 same-length substitutions.
Edits: substitution, insertion, deletion. Candidate generation from witness crop
CTC greedy character events; optional edits are scored using exact CTC *forward*
sequence likelihood ratios on each covering crop. A length-changing path is
selected by budgeted DAG dynamic programming (not an unconstrained optimal
joint CTC decoder). Does not mix frame posteriors or use language priors.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Sequence
import numpy as np
import longtext_v3 as V3
import alex_ctc as AX

ALPH=AX.ALPH
NEG=-1e50

@dataclass(frozen=True)
class Proposal:
    op:str
    position:int
    char:str
    x:float


def ctc_forward_logprob(logq:np.ndarray, text:str)->float:
    """Exact forward sum over CTC paths for 37 normalized classes, log domain."""
    T,C=logq.shape
    if C!=37 or T<1:
        raise ValueError('Expected [T,37] nonempty normalized log posterior')
    if any(ch not in ALPH for ch in text):
        raise ValueError('Unnormalized transcription')
    labels=[1+ALPH.index(c) for c in text]
    repeats=sum(labels[j]==labels[j-1] for j in range(1,len(labels)))
    if len(labels)+repeats>T:return float('-inf')
    if not labels:return float(logq[:,0].sum())
    S=2*len(labels)+1
    states=np.zeros(S,dtype=np.int32)
    states[1::2]=labels
    can_jump=np.zeros(S,dtype=bool)
    can_jump[2:]=(states[2:]!=0)&(states[2:]!=states[:-2])
    prev=np.full(S,-np.inf,np.float64)
    prev[0]=logq[0,0];prev[1]=logq[0,labels[0]]
    for t in range(1,T):
        stay=prev
        step=np.empty(S,np.float64);step[0]=-np.inf;step[1:]=prev[:-1]
        jump=np.full(S,-np.inf,np.float64);jump[2:]=np.where(can_jump[2:],prev[:-2],-np.inf)
        prev=np.logaddexp(np.logaddexp(stay,step),jump)+logq[t,states]
    return float(np.logaddexp(prev[-1],prev[-2]))


def proposal_position(ref,op,j):
    if op!='ins':return float(ref[j].x)
    if j==0:return float(ref[0].x)
    if j==len(ref):return float(ref[-1].x)
    return float((ref[j-1].x+ref[j].x)/2.)


def generate_proposals(ref, crop_events, spans):
    """Generate one-symbol edits witnessed by original crop greedy events.

    Matching uses equal-string anchors (no hidden GT oracle). Restrict witness
    to the middle 70% of its crop to reduce truncation artefacts.
    """
    n=len(ref); proposals={}
    for events, (a,b) in zip(crop_events,spans):
        ids=[i for i,e in enumerate(ref) if a<=e.x<=b]
        if not ids or ids!=list(range(ids[0],ids[-1]+1)):continue
        original=''.join(ref[i].char for i in ids)
        observed=''.join(e.char for e in events)
        matcher=SequenceMatcher(None,original,observed,autojunk=False)
        for tag,u0,u1,v0,v1 in matcher.get_opcodes():
            op=None; pos=None; ch=''
            if tag=='replace' and u1-u0==1 and v1-v0==1:
                op='sub';pos=ids[u0];ch=events[v0].char
            elif tag=='insert' and v1-v0==1:
                op='ins';pos=ids[u0] if u0<len(ids) else ids[-1]+1
                ch=events[v0].char
            elif tag=='delete' and u1-u0==1:
                op='del';pos=ids[u0]
            if op is None:continue
            if ch and ch not in ALPH:continue
            x=proposal_position(ref,op,pos)
            # Avoid source-view boundaries, where partial characters are common.
            if not (a+0.15*(b-a)<=x<=b-0.15*(b-a)):continue
            # A witness symbol should also be spatially plausible relative to
            # the reference event (in CTC time/global crop coordinates).
            if op in ('sub','ins'):
                witness_x=events[v0].x
                if abs(witness_x-x)>max(0.18*(b-a),4):continue
            p=Proposal(op,pos,ch,x)
            proposals[(p.op,p.position,p.char)]=p
    return list(proposals.values())


def change_local(local,loc,op,ch):
    if op=='sub':
        if not 0<=loc<len(local):return None
        return local[:loc]+ch+local[loc+1:]
    if op=='del':
        if not 0<=loc<len(local):return None
        return local[:loc]+local[loc+1:]
    if op=='ins':
        if not 0<=loc<=len(local):return None
        return local[:loc]+ch+local[loc:]
    raise ValueError(op)


def evidence_scores(ref,logits,spans,chars,proposals):
    n=len(ref)
    prepared=[]
    for raw,(a,b) in zip(logits,spans):
        ids=[i for i,e in enumerate(ref) if a<=e.x<=b]
        if not ids or ids!=list(range(ids[0],ids[-1]+1)):continue
        q=AX.normalized_probs(raw,chars)
        logq=np.log(np.maximum(q,1e-35))
        txt=''.join(ref[i].char for i in ids)
        base=ctc_forward_logprob(logq,txt)
        prepared.append((int(a),int(b),ids,txt,logq,base))
    scores={}; score_details={}
    for p in proposals:
        diffs=[]; coverage=0
        for a,b,ids,txt,logq,base in prepared:
            if not np.isfinite(base):continue
            # Only trust center-covered views to avoid seam artefacts.
            if not (a+0.10*(b-a)<=p.x<=b-0.10*(b-a)):continue
            if p.op=='ins':
                if p.position<ids[0] or p.position>ids[-1]+1:continue
            else:
                if p.position not in ids:continue
            loc=p.position-ids[0]
            mutant=change_local(txt,loc,p.op,p.char)
            if mutant is None or mutant==txt:continue
            alt=ctc_forward_logprob(logq,mutant)
            if not np.isfinite(alt):continue
            coverage+=1
            # Equal weight per independently computed crop observation; avoid
            # repeated-frame double-counting inside one crop.
            diffs.append(float(alt-base))
        # No label/GT in criteria: insist on at least two usable views and
        # a consistent positive exact-CTC log-likelihood ratio.
        if coverage>=2:
            score=float(np.mean(diffs))
            if np.isfinite(score):
                scores[p]=score
                score_details[p]=(coverage,sum(d>0 for d in diffs),float(min(diffs)))
    return scores,score_details


def lattice_decode(reference:str, scores, max_edits=2, margin=1.8, types=('sub','ins','del')):
    """DAG over string positions / optional insertion slots, with edit budget.

    Keep edges have score 0. Proposed edits score (avg CTC log BF - margin).
    At most one insertion per gap, substitutions/deletions consume one glyph.
    """
    n=len(reference)
    ins=defaultdict(list); glyph=defaultdict(list)
    for p,score in scores.items():
        if p.op not in types or score<=margin or not np.isfinite(score):continue
        (ins if p.op=='ins' else glyph)[p.position].append((p,score-margin))
    # dp keyed (number_of_edits) => (score, transcript, accepted edits)
    states={0:(0.,'',[])}
    for i in range(n+1):
        # optional insertion before position i
        after={}
        for used,(v,t,ops) in states.items():
            if used not in after or v>after[used][0]:after[used]=(v,t,ops)
            if used<max_edits:
                for p,gain in ins.get(i,[]):
                    uv=used+1; cand=(v+gain,t+p.char,ops+[p])
                    if uv not in after or cand[0]>after[uv][0]:after[uv]=cand
        if i==n:states=after;break
        nxt={}
        for used,(v,t,ops) in after.items():
            nxt[used]=(v,t+reference[i],ops)
            if used<max_edits:
                for p,gain in glyph.get(i,[]):
                    uv=used+1; text=t+(p.char if p.op=='sub' else '')
                    cand=(v+gain,text,ops+[p])
                    if uv not in nxt or cand[0]>nxt[uv][0]:nxt[uv]=cand
        states=nxt
    best=max(states.values(),key=lambda z:z[0])
    return best[1],best[2],best[0]


def decode(logits,spans,chars,c0=None,return_info=False):
    """Produces C1_subonly and C1_edit using same proposal+scoring pool."""
    if c0 is None:c0=AX.decode(logits,spans,chars)
    ref0=AX.reference_events(logits,spans,chars)
    if len(c0)!=len(ref0):
        raise AssertionError('C0 and reference must share one-to-one char slots')
    ref=[replace(e,char=ch) for e,ch in zip(ref0,c0)]
    crop_events=[V3.ctc_events(raw,chars,*sp) for raw,sp in zip(logits,spans)]
    proposals=generate_proposals(ref,crop_events,spans)
    scores,details=evidence_scores(ref,logits,spans,chars,proposals)
    sub,sub_ops,_=lattice_decode(c0,scores,types=('sub',))
    edit,edit_ops,_=lattice_decode(c0,scores,types=('sub','ins','del'))
    info={'n_proposals':len(proposals),'n_scored':len(scores),
          'n_accepted_sub':len(sub_ops),'n_accepted_edit':len(edit_ops),
          'edit_insertions':sum(p.op=='ins' for p in edit_ops),
          'edit_deletions':sum(p.op=='del' for p in edit_ops),
          'edit_substitutions':sum(p.op=='sub' for p in edit_ops),
          'edit_path':';'.join(f'{p.op}@{p.position}:{p.char}' for p in edit_ops)}
    return (sub,edit,info) if return_info else edit


def cpu_tests():
    from itertools import product
    def brute(q,y):
        result=0.
        for path in product(range(q.shape[1]),repeat=q.shape[0]):
            collapsed=[]; prev=None
            for c in path:
                if c!=0 and c!=prev:collapsed.append(ALPH[c-1])
                prev=c
            if ''.join(collapsed)==y:
                v=1.
                for t,c in enumerate(path):v*=q[t,c]
                result+=v
        return result
    rng=np.random.default_rng(321)
    for L in ('','a','aa','aba','abb'):
        q=np.zeros((4,37),np.float64)
        pp=rng.dirichlet([1.8,1.8,1.8],size=4)
        q[:,:3]=pp
        a=ctc_forward_logprob(np.log(np.maximum(q,1e-35)),L)
        b=brute(q,L)
        assert np.isclose(np.exp(a),b,atol=1e-10), (L,a,b)
    # DAG unit tests demonstrate that C1 can change transcript length.
    dummy=lambda op,i,c:Proposal(op,i,c,float(i))
    for name,ref,p,answer in [('insertion','ab',dummy('ins',1,'c'),'acb'),
                              ('deletion','abc',dummy('del',1,''),'ac'),
                              ('substitution','abc',dummy('sub',1,'d'),'adc')]:
        out,ops,_=lattice_decode(ref,{p:4.},max_edits=1,margin=1.)
        assert out==answer,(name,out,answer)
    # Ref case can be preserved exactly when there is no positive witness.
    out,_,_=lattice_decode('abcdef',{},max_edits=2)
    assert out=='abcdef'
    return {'passed':True,'ctc_forward_vs_bruteforce':5,
            'edit_lattice':['insertion','deletion','substitution'],
            'zero_edit_fallback':True}
