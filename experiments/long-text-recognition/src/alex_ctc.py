"""ALEX-CTC E1: Alignment-Latent cross-view character evidence (C0 prototype).

GT-blind, frozen SVTRv2-T. Reference-anchored CTC Viterbi alignment to every
visible crop, preserving baseline spatial order, then multi-crop posterior
inference over possible character substitutions. No thresholds on GT scores.
Not an exact joint decoder for insertions/deletions (future C1).
"""
from __future__ import annotations
import numpy as np
import longtext_v3 as V3
from longtext_final import normalize_ltb

ALPH='0123456789abcdefghijklmnopqrstuvwxyz'

def normalized_probs(logits,chars):
    z=np.asarray(logits,np.float64);z-=z.max(axis=1,keepdims=True)
    p=np.exp(z);p/=p.sum(axis=1,keepdims=True)
    q=np.zeros((len(p),37),np.float64)
    for i,c in enumerate(chars):
        norm=normalize_ltb(c) if i else ''
        if len(norm)==1 and norm in ALPH:q[:,1+ALPH.index(norm)]+=p[:,i]
        else:q[:,0]+=p[:,i]
    q=np.maximum(q,1e-30);q/=q.sum(axis=1,keepdims=True)
    return q


def reference_events(logits,spans,chars):
    evs=[V3.ctc_events(q,chars,*sp) for q,sp in zip(logits,spans)]
    out=list(evs[0])
    for j in range(1,len(evs)):
        nxt=list(evs[j]); a=''.join(x.char for x in out);b=''.join(e.char for e in nxt)
        match=V3._v2_match_info(a,b,spans[j-1],spans[j])
        if match['mode']=='concat':out+=nxt;continue
        if match['mode']=='reject':
            cut=((spans[j-1][0]+spans[j-1][1])/2+(spans[j][0]+spans[j][1])/2)/2
            out=[e for e in out if e.x<cut]+[e for e in nxt if e.x>=cut]
            continue
        L=match['L'];h=L//2
        out=out[:-L+h]+nxt[h:]
    s=normalize_ltb(''.join(e.char for e in out))
    actual=normalize_ltb(V3.stitch_evidence(evs,spans,use_confidence=False,geometric_fallback=True))
    assert s==actual,(s,actual)
    return out


def ctc_viterbi(q,txt):
    """CTC forced Viterbi path for normalized text, returns per-letter peak frame.

    q is 37-class posterior with punctuation mass mapped to blank. Unlike greedy,
    this marginal remains conditional on a hypothesized string, not ground truth.
    """
    T,C=q.shape
    if not txt:return [],0.
    lab=[1+ALPH.index(c) for c in txt]
    S=2*len(lab)+1
    minimum_frames=len(lab)+sum(lab[j]==lab[j-1] for j in range(1,len(lab)))
    if minimum_frames>T:return None,-np.inf
    st=np.zeros(S,np.int16);st[1::2]=lab
    L=np.log(np.maximum(q,1e-30))
    dp=np.full((T,S),-np.inf,np.float64)
    back=np.full((T,S),-1,np.int16)
    dp[0,0]=L[0,0]
    if S>1:dp[0,1]=L[0,st[1]]
    for t in range(1,T):
        for s in range(S):
            paths=[(dp[t-1,s],s)]
            if s>0:paths.append((dp[t-1,s-1],s-1))
            if s>1 and st[s]!=0 and st[s]!=st[s-2]:paths.append((dp[t-1,s-2],s-2))
            best=max(paths,key=lambda a:a[0]);dp[t,s]=best[0]+L[t,st[s]];back[t,s]=best[1]
    if S==1:end=0
    else:end=int(np.argmax([dp[T-1,S-2],dp[T-1,S-1]]))+S-2
    score=float(dp[T-1,end])
    if not np.isfinite(score):return None,-np.inf
    states=np.empty(T,dtype=int)
    s=end
    for t in range(T-1,-1,-1):
        states[t]=s
        if t>0:s=int(back[t,s]);assert s>=0
    frames=[]
    for k,c in enumerate(lab):
        ts=np.where(states==(2*k+1))[0]
        if not len(ts):return None,-np.inf
        best=int(ts[np.argmax(q[ts,c])]);frames.append(best)
    return frames,score


def decode(logits,spans,chars,return_info=False):
    ev=reference_events(logits,spans,chars)
    base=''.join(e.char for e in ev)
    if not ev:return ('',{'n_aligned_crops':0,'changed_chars':0}) if return_info else ''
    votes=[[] for _ in ev]
    feasible=0
    for k,(raw,(a,b)) in enumerate(zip(logits,spans)):
        ids=[j for j,e in enumerate(ev) if a<=e.x<=b]
        if not ids:continue
        txt=''.join(ev[j].char for j in ids)
        q=normalized_probs(raw,chars)
        frames,score=ctc_viterbi(q,txt)
        if frames is None:continue
        feasible+=1
        for j,t in zip(ids,frames):
            x=ev[j].x
            w=.5+.5*max(0,1-2*abs(x-(a+b)/2)/max(b-a,1))
            votes[j].append((w,q[t]))
    out=[];changed=0
    for j,e in enumerate(ev):
        if not votes[j]:out.append(e.char);continue
        v=votes[j];ww=np.array([s for s,_ in v],np.float64);ww/=ww.sum()
        # Bayesian geometric mean from multiple observation windows: all
        # crops vote once, so overlapping frame count cannot swamp output.
        score=sum(float(w)*np.log(np.maximum(d[1:],1e-30)) for w,(_,d) in zip(ww,v))
        c=ALPH[int(score.argmax())];out.append(c);changed+=int(c!=e.char)
    result=''.join(out)
    return (result,{'n_aligned_crops':feasible,'changed_chars':changed,'baseline_chars':len(ev)}) if return_info else result


def cpu_tests():
    chars=['blank','a','b','c']
    def logits(ids):
        x=np.full((len(ids),4),-8.,np.float32)
        for i,j in enumerate(ids):x[i,j]=8.
        return x
    a=logits([1,0,1,0,2]);assert decode([a],[(0,100)],chars)=='aab'
    q=normalized_probs(a,chars)
    frm,score=ctc_viterbi(q,'aab');assert len(frm)==3 and np.isfinite(score)
    assert decode([a,a],[(0,100),(0,100)],chars)=='aab'
    return {'passed':True,'repeat_ctc':True,'reference_parity':True}
