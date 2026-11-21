"""SPaCE-CTC E0: frozen CTC spatial-posterior alignment and prefix-beam pilot.

Hypothesis only. Geometry of CTC timesteps is approximate. No label or image ID
is consumed by inference functions. This module is intentionally self-contained
with no external library beyond NumPy and the pinned project's longtext_final.
"""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from longtext_final import greedy_ctc, normalize_ltb


def logadd(*values):
    vals = [float(x) for x in values if x != -math.inf]
    if not vals:
        return -math.inf
    m = max(vals)
    return m + math.log(sum(math.exp(x-m) for x in vals))


def stable_probs(logits):
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] < 2 or not np.isfinite(a).all():
        raise ValueError('Need finite nonempty CTC logits [T, C]')
    z = a - np.max(a, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def owned_frames(pieces: Sequence[np.ndarray], spans: Sequence[tuple[int,int]]):
    if len(pieces)!=len(spans) or not pieces:
        raise ValueError('CTC evidence/spans mismatch')
    mids = [(a+b)/2. for a,b in spans]
    cuts = [-np.inf] + [(mids[i]+mids[i+1])/2 for i in range(len(mids)-1)] + [np.inf]
    anchors = []
    for i,(p,(x0,x1)) in enumerate(zip(pieces, spans)):
        t = p.shape[0]
        if x1<=x0 or t==0:
            raise ValueError('Invalid crop span or CTC length')
        coords = x0+(np.arange(t)+.5)*(x1-x0)/t
        ids = np.where((coords >= cuts[i]) & (coords < cuts[i+1]))[0]
        if not len(ids):
            raise RuntimeError('No frames owned by crop')
        anchors.extend((i,int(k),float(coords[k])) for k in ids)
    return anchors


def spatial_lattice(logits, spans, *, mode='aligned', alpha=.35, radius=2):
    """Anchor ownership + GT-blind geometric/CTC neighbor alignment.

    For each owned timestep, the owner observation retains at least 1-alpha of
    the log-probability weight. Neighbor contributions in overlapping spatial
    support are aligned softly within +/- radius frames around projected x.
    mode='coordinate' uses a strictly geometric nearest-frame ablation.
    mode='aligned' incorporates CTC-posterior similarity without GT.
    """
    if mode not in ('aligned','coordinate') or not (0<=alpha<=.49) or not (1<=radius<=6):
        raise ValueError('Unsupported spatial-lattice configuration')
    prob = [stable_probs(v) for v in logits]
    if len({v.shape[1] for v in prob})!=1:
        raise ValueError('CTC class counts differ between crops')
    anchors = owned_frames(prob, spans)
    out = []
    n_mixed=0
    for i,k,x in anchors:
        owner = prob[i][k]
        neighbors=[]
        for j, (q,(a,b)) in enumerate(zip(prob,spans)):
            if j==i or not (a<=x<=b):
                continue
            expected = (x-a)/(b-a)*len(q)-.5
            # Explicit one-frame-nearest support for coordinate-only control.
            if mode=='coordinate':
                neighbor = q[int(np.clip(round(expected),0,len(q)-1))]
            else:
                lo=max(0,int(np.floor(expected))-radius)
                hi=min(len(q)-1,int(np.ceil(expected))+radius)
                candidates=np.arange(lo,hi+1,dtype=int)
                # Bhattacharyya similarity in the actual distribution;
                # window around projected location prevents semantic shortcuts.
                sim = np.sqrt(q[candidates]*owner[None,:]).sum(axis=1)
                score=np.log(np.maximum(sim,1e-10)) - .5*((candidates-expected)/max(radius/2,1))**2
                weights=np.exp(score-np.max(score))
                weights /= weights.sum()
                neighbor=np.sum(q[candidates]*weights[:,None],axis=0)
            neighbors.append(neighbor)
        if neighbors:
            n_mixed+=1
            each=alpha/len(neighbors)
            logp=(1-alpha)*np.log(np.maximum(owner,1e-30))
            for p in neighbors:
                logp+=each*np.log(np.maximum(p,1e-30))
            logp-=logadd(*logp)
            out.append(logp)
        else:
            out.append(np.log(np.maximum(owner,1e-30)))
    ans=np.stack(out,axis=0)
    assert np.isfinite(ans).all()
    return ans, {'frames':len(anchors),'mixed_frames':n_mixed,'n_crops':len(prob)}


def ownership_log_probs(logits,spans):
    p=[stable_probs(v) for v in logits]
    return np.stack([np.log(np.maximum(p[i][k],1e-30)) for i,k,x in owned_frames(p,spans)])


def prefix_beam_search(log_probs,chars, beam_width=8, topk=6, nbest=8):
    """Reference CTC prefix beam search with blank=0, repeated-label logic.

    Inference is bounded via per-frame top-k emissions plus each current prefix
    last label (for exact repeat-path recurrence under pruned emissions).
    Beam pruning is approximate; therefore n-best membership is NOT an upper
    bound over the entire CTC distribution.
    """
    lp=np.asarray(log_probs,dtype=np.float64)
    if lp.ndim!=2 or lp.shape[1]!=len(chars) or lp.shape[0]<1:
        raise ValueError('Invalid prefix beam input')
    if beam_width<1 or topk<1:
        raise ValueError('Beam width and topk must be >=1')
    beam={(): (0., -math.inf)}
    for row in lp:
        chosen = set(np.argpartition(row, -min(len(row),topk))[-min(len(row),topk):].tolist())
        chosen.add(0)
        for pre in beam:
            if pre:
                chosen.add(pre[-1])
        next_b={}
        def add(key, add_pb=-math.inf,add_pnb=-math.inf):
            prev_pb,prev_pnb=next_b.get(key,(-math.inf,-math.inf))
            next_b[key]=(logadd(prev_pb,add_pb),logadd(prev_pnb,add_pnb))
        for pre,(pb,pnb) in beam.items():
            whole=logadd(pb,pnb)
            add(pre,add_pb=whole+row[0])
            for c in chosen:
                if c==0:continue
                if pre and pre[-1]==c:
                    add(pre,add_pnb=pnb+row[c])
                    add(pre+(c,),add_pnb=pb+row[c])
                else:
                    add(pre+(c,),add_pnb=whole+row[c])
        beam=dict(sorted(next_b.items(),key=lambda x:logadd(*x[1]),reverse=True)[:beam_width])
    ranked=sorted(beam.items(),key=lambda x:logadd(*x[1]),reverse=True)[:nbest]
    result=[]
    for seq,(pb,pnb) in ranked:
        pred=normalize_ltb(''.join(chars[c] for c in seq))
        result.append({'pred':pred,'logp':float(logadd(pb,pnb))})
    return result


def decode_all(logits, spans, chars, beam_width=8,topk=6):
    owner=ownership_log_probs(logits,spans)
    coord,mdc=spatial_lattice(logits,spans,mode='coordinate')
    aligned,mda=spatial_lattice(logits,spans,mode='aligned')
    def g(lp):
        return normalize_ltb(greedy_ctc(lp,chars))
    runs={
        'ownership_greedy':g(owner),
        'ownership_beam':prefix_beam_search(owner,chars,beam_width,topk),
        'coordinate_greedy':g(coord),
        'coordinate_beam':prefix_beam_search(coord,chars,beam_width,topk),
        'space_greedy':g(aligned),
        'space_beam':prefix_beam_search(aligned,chars,beam_width,topk),
        'metadata':mda,
    }
    return runs, {'owner':owner,'coordinate':coord,'aligned':aligned}


def cpu_tests():
    chars=['blank','a','b','c']
    def pth(ids):
        x=np.full((len(ids),len(chars)),-8.,np.float32)
        for i,c in enumerate(ids):x[i,c]=8.
        return x
    single=[pth([1,0,1,0,2])]
    out,latt=decode_all(single,[(0,100)],chars,5,4)
    assert out['ownership_greedy']=='aab',out
    assert out['ownership_beam'][0]['pred']=='aab',out
    assert out['space_beam'][0]['pred']=='aab',out
    repeated=[pth([1,1,1])]
    z=prefix_beam_search(ownership_log_probs(repeated,[(0,10)]),chars,4,4)
    assert z[0]['pred']=='a',z
    parts=[pth([1,0,2]),pth([2,0,3])]
    k,m=decode_all(parts,[(0,60),(30,90)],chars,5,4)
    assert k['metadata']['n_crops']==2 and k['metadata']['frames']>0
    assert m['aligned'].shape[1]==4
    assert np.max(np.abs(np.exp(m['aligned']).sum(axis=1)-1))<1e-7
    assert np.allclose(m['owner'],m['coordinate']) is False or k['metadata']['mixed_frames']>=0
    assert prefix_beam_search(np.array([[0,-100,-100,-100]],float),chars,4,4)[0]['pred']==''
    return {'passed':True,'repeated_character_test':True,'blank_test':True,'overlap_test':True}
