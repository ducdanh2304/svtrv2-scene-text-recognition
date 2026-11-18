"""Frozen SVTRv2-T CTC-logit evidence for SAGE-STR v2.

The class 0 is blank; logits are shaped [T,C]. No GT/label input is accepted.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .selector import Hypothesis


def _softmax(x):
    x=x-np.max(x,axis=-1,keepdims=True)
    e=np.exp(x)
    return e/np.sum(e,axis=-1,keepdims=True)


def from_logits(logits:np.ndarray,chars:Sequence[str],strategy:str,
                n_windows:int=1,overlap_agreement:float=0.,
                distortion:float=1.,cost_ms:float=0.) -> Hypothesis:
    """Compute a GT-free evidence summary from a *single* CTC sequence.

    For a tiled proposal, compute one summary per crop and independently
    aggregate events in the GPU adapter; stitched-text evidence cannot be
    obtained by simply concatenating logits from overlapping windows.
    """
    z=np.asarray(logits,dtype=np.float64)
    if z.ndim!=2 or len(z)==0 or len(z[0])!=len(chars):
        raise ValueError('Expect non-empty CTC logits [T, len(chars)]')
    p=_softmax(z)
    ids=np.argmax(p,axis=1)
    if len(chars)<2:raise ValueError('CTC charset needs blank and characters')
    tokens=[];event_steps=[];event_confs=[];event_margins=[]
    prev=0
    for t,k in enumerate(ids):
        k=int(k)
        if k!=0 and k!=prev:
            tokens.append(chars[k]);event_steps.append(t)
            event_confs.append(float(p[t,k]))
            sorted2=np.partition(p[t],-2)[-2:]
            event_margins.append(float(sorted2.max()-sorted2.min()))
        prev=k
    entropy=-(p*np.log(np.clip(p,1e-12,1))).sum(axis=1)
    mean_entropy=float(np.mean(entropy/np.log(len(chars))))
    if event_steps:
        edge=float(np.mean([(t<.1*len(z) or t>=.9*len(z)) for t in event_steps]))
    else:edge=0.
    return Hypothesis(strategy=strategy,text=''.join(tokens),ctc_T=len(z),
                      confidence=float(np.mean(event_confs)) if event_confs else 0.,
                      entropy=mean_entropy,
                      char_margin=float(np.mean(event_margins)) if event_margins else 0.,
                      edge_occupancy=edge,overlap_agreement=overlap_agreement,
                      n_windows=n_windows,distortion=distortion,cost_ms=cost_ms)
