"""ALEX E5: posterior-generated CTC candidate edits + matched C1 scoring.

EXPLORATORY. For each crop, CTC prefix beam search provides non-greedy
sequence hypotheses; Viterbi places their characters in crop coordinates.
Differences to C0 reference become (sub/ins/del) hypotheses and are scored by
exact CTC forward per crop using *the SAME scoring and DAG* as C1. This is an
ablation of candidate discovery, not an exact joint multi-crop CTC solver.
No labels, GT, IDs or historical predictions are accepted by decode().
"""
from __future__ import annotations
from collections import Counter
import numpy as np
import alex_ctc as AX
import alex_c1 as C1
import longtext_v3 as V3
import space_ctc as SPACE

ALPH=AX.ALPH
BEAM_CHARS=['']+list(ALPH)


def fast_forced_frames(q:np.ndarray, text:str):
    """Forced Viterbi CTC alignment: exact max-path with blank/repeats."""
    tmax=len(q)
    if not text:return [],0.
    labels=np.fromiter((1+ALPH.index(c) for c in text),dtype=np.int16)
    if len(labels)+sum(labels[j]==labels[j-1] for j in range(1,len(labels)))>tmax:
        return None, -np.inf
    states=np.zeros(2*len(labels)+1,dtype=np.int16)
    states[1::2]=labels
    size=len(states)
    allowed=np.zeros(size,dtype=bool)
    allowed[2:]=(states[2:]!=0)&(states[2:]!=states[:-2])
    logq=np.log(np.maximum(q,1e-30))
    prev=np.full(size,-np.inf,dtype=np.float64)
    prev[0]=logq[0,0]
    prev[1]=logq[0,labels[0]]
    trace=np.full((tmax,size),-1,np.int16)
    minus=np.full(size,-np.inf,dtype=np.float64)
    for t in range(1,tmax):
        step=minus.copy();step[1:]=prev[:-1]
        jump=minus.copy();jump[2:]=np.where(allowed[2:],prev[:-2],-np.inf)
        choices=np.stack((prev,step,jump),axis=0)
        which=np.argmax(choices,axis=0)
        trace[t]=np.arange(size)-which
        prev=choices[which,np.arange(size)]+logq[t,states]
    terminal=size-1 if prev[-1]>=prev[-2] else size-2
    best=float(prev[terminal])
    if not np.isfinite(best):return None, -np.inf
    state_path=np.zeros(tmax,np.int16)
    current=terminal
    for t in range(tmax-1,-1,-1):
        state_path[t]=current
        if t:current=int(trace[t,current])
    frames=[]
    for j,lab in enumerate(labels):
        ts=np.flatnonzero(state_path==2*j+1)
        if len(ts)==0:return None,-np.inf
        frames.append(int(ts[np.argmax(q[ts,lab])]))
    return frames,best


def beam_events(logq,span,beam_width=8,topk=5,nbest=5):
    """Return alternative character-event streams from actual 37-class posterior."""
    q=np.exp(logq)
    nbest_rows=SPACE.prefix_beam_search(logq,BEAM_CHARS,
            beam_width=beam_width,topk=topk,nbest=nbest)
    out=[];seen=set()
    a,b=span;T=len(q)
    for row in nbest_rows:
        text=row['pred']
        if not text or text in seen:continue
        seen.add(text)
        frames,score=fast_forced_frames(q,text)
        if frames is None:continue
        evs=[V3.CharEvidence(c,float(q[t,1+ALPH.index(c)]),
               float(a+(t+0.5)*(b-a)/T)) for c,t in zip(text,frames)]
        out.append((text,evs,float(row['logp'])))
    return out


def get_proposals(ref,logits,spans,chars,beam_width=8,topk=5,nbest=5):
    """Get greedy-only and unique non-greedy proposals on same raw evidence."""
    greedy_events=[V3.ctc_events(raw,chars,*sp) for raw,sp in zip(logits,spans)]
    greedy=C1.generate_proposals(ref,greedy_events,spans)
    greedy_keys={(p.op,p.position,p.char) for p in greedy}
    posterior={};counts=Counter(); n_uniq_alt=0;n_failed_align=0
    for raw,span,grev in zip(logits,spans,greedy_events):
        q=AX.normalized_probs(raw,chars)
        lp=np.log(np.maximum(q,1e-30))
        crop_greedy=''.join(ev.char for ev in grev)
        alternatives=beam_events(lp,span,beam_width,topk,nbest)
        for text,evs,logprob in alternatives:
            if text==crop_greedy:continue
            n_uniq_alt+=1
            for p in C1.generate_proposals(ref,[evs],[span]):
                k=(p.op,p.position,p.char)
                if k not in greedy_keys:
                    posterior[k]=p
                    counts[k]+=1
    info={'greedy_proposals':len(greedy),'posterior_only_proposals':len(posterior),
          'non_greedy_crop_hypotheses':n_uniq_alt,
          'posterior_repeated_witnesses':sum(v>=2 for v in counts.values())}
    return greedy, list(posterior.values()),info


def decode(logits,spans,chars,*,beam_width=8,frame_topk=5,nbest=5,
           max_edits=2,margin=1.8,return_info=False):
    c0=AX.decode(logits,spans,chars)
    ref0=AX.reference_events(logits,spans,chars)
    if len(ref0)!=len(c0):raise RuntimeError('C0 length mismatch')
    from dataclasses import replace
    ref=[replace(e,char=c) for e,c in zip(ref0,c0)]
    gp,pp,stats=get_proposals(ref,logits,spans,chars,beam_width,frame_topk,nbest)
    g_scores,_=C1.evidence_scores(ref,logits,spans,chars,gp)
    p_scores,_=C1.evidence_scores(ref,logits,spans,chars,pp)
    base,b_ops,_=C1.lattice_decode(c0,g_scores,max_edits=max_edits,margin=margin)
    joint={**g_scores,**p_scores}
    union,u_ops,_=C1.lattice_decode(c0,joint,max_edits=max_edits,margin=margin)
    posterior_only,p_ops,_=C1.lattice_decode(c0,p_scores,max_edits=max_edits,margin=margin)
    inf={**stats,'posterior_scored':len(p_scores),
        'posterior_positive_scores':sum(s>margin for s in p_scores.values()),
        'changed_by_posterior':int(union!=base),
        'union_edits':len(u_ops),
        'union_insertions':sum(p.op=='ins' for p in u_ops),
        'union_deletions':sum(p.op=='del' for p in u_ops),
        'union_substitutions':sum(p.op=='sub' for p in u_ops),
        'union_posterior_only_edits':sum((p.op,p.position,p.char) not in {(g.op,g.position,g.char) for g in gp} for p in u_ops)}
    return (base,posterior_only,union,inf) if return_info else union


def cpu_tests():
    import itertools
    from numpy.testing import assert_allclose
    rng=np.random.default_rng(9)
    for text in ('a','aa','ab','aba','abb'):
        q=np.full((6,37),1e-30,float)
        q[:,:3]=rng.dirichlet([2,3,4],size=6)
        q/=q.sum(axis=1,keepdims=True)
        fast,v=fast_forced_frames(q,text)
        slow,s=AX.ctc_viterbi(q,text)
        assert_allclose(v,s,rtol=0,atol=1e-9)
        assert fast==slow,(text,fast,slow)
    q=np.zeros((4,37),float)
    q[:,0]=.4;q[:,1]=.45;q[:,2]=.15
    out=beam_events(np.log(np.maximum(q,1e-30)),(0,100),beam_width=12,topk=3,nbest=8)
    assert len(out)>=2 and all(len(y)==len(ev) for y,ev,_ in out)
    return {'passed':True,'forced_viterbi_vs_reference':5,'non_greedy_beam_candidates':len(out)}
