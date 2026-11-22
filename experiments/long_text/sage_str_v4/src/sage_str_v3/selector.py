"""SAGE-STR v3 — deterministic dual-arbiter experiment, NOT validated superiority.

Frozen SVTRv2-T proposals are represented by the immutable v2 Hypothesis/Observation.
Training and tuning can see labels, but select() only receives an Observation.
No source ID, filenames, GT, benchmark labels, wall clock, or GPU latency feature at inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import log1p
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sage_str_v2.selector import Hypothesis, Observation
from rapidfuzz.distance.Levenshtein import distance as edit_distance

SEED = 20261118
NEUTRAL, GAIN, HARM = 0, 1, 2
METHOD_COST_UNITS = {"rotate_cw":1., "rotate_ccw":1.,"full_width2":.7,
                     "crop2_msr":1.4,"crop3_msr":1.8,"crop3_width2":2.0,
                     "crop3_width3":2.3,"geometry_v3":1.5,"original":0.}
NUMERIC = ["aspect_log","original_len","original_slot_pressure","original_entropy",
 "original_edge_occupancy","reference_len","reference_is_geometry",
 "candidate_len","relative_len_change","candidate_n_windows", "candidate_log_distortion",
 "candidate_overlap_agreement", "candidate_entropy", "candidate_confidence",
 "candidate_char_margin","sequence_agreement","candidate_longer_than_ref",
 "length_ratio", "reference_confidence", "reference_char_margin"]
CATEGORICAL = ["strategy"]
COLUMNS = CATEGORICAL + NUMERIC


def _norm(s):
    return ''.join(ch for ch in s.lower() if ch.isascii() and ch.isalnum())


def reference(obs:Observation)->Hypothesis:
    # Geometry is a safer long-line reference than single full-image MSR when
    # its prediction differs. If it equals original, same text is retained.
    return next((c for c in obs.alternatives if c.strategy=="geometry_v3"),obs.original)


def candidates(obs:Observation):
    ref=reference(obs)
    cs=[c for c in obs.alternatives if c.strategy!="geometry_v3" and c.normalized!=ref.normalized]
    if ref.strategy=="geometry_v3" and obs.original.normalized!=ref.normalized:
        cs.append(obs.original)  # Original remains a *possible*, not mandatory fallback.
    cs=sorted(cs,key=lambda c:(c.strategy,c.normalized))
    assert len({c.strategy for c in cs})==len(cs)
    return ref,cs


def group_for(c):
    return "orientation" if c.strategy in ("rotate_cw","rotate_ccw") else "recovery"


def features(obs:Observation,ref:Hypothesis,c:Hypothesis):
    a,b=ref.normalized,c.normalized
    return {
        "strategy":c.strategy,
        "aspect_log":float(np.log(obs.width/obs.height)),
        "original_len":float(len(obs.original.normalized)),
        "original_slot_pressure":float(min(2.,obs.original.pressure)),
        "original_entropy":float(obs.original.entropy),
        "original_edge_occupancy":float(obs.original.edge_occupancy),
        "reference_len":float(len(a)),
        "reference_is_geometry":float(ref.strategy=="geometry_v3"),
        "candidate_len":float(len(b)),
        "relative_len_change":float(len(b)-len(a)),
        "candidate_n_windows":float(c.n_windows),
        "candidate_log_distortion":float(log1p(c.distortion-1)),
        "candidate_overlap_agreement":float(c.overlap_agreement),
        "candidate_entropy":float(c.entropy),
        "candidate_confidence":float(c.confidence),
        "candidate_char_margin":float(c.char_margin),
        "sequence_agreement":float(SequenceMatcher(None,a,b,autojunk=False).ratio()),
        "candidate_longer_than_ref":float(len(b)>len(a)),
        "length_ratio":float((len(b)+1)/(len(a)+1)),
        "reference_confidence":float(ref.confidence),
        "reference_char_margin":float(ref.char_margin),
    }


def matrix(xs):
    z=pd.DataFrame(xs,columns=COLUMNS)
    if len(z)==0:return z
    assert np.isfinite(z[NUMERIC].to_numpy(dtype=float)).all()
    return z


def _pre():
    return ColumnTransformer([
        ("num",StandardScaler(),NUMERIC),
        ("cat",OneHotEncoder(handle_unknown="ignore"),CATEGORICAL),
    ])


def _model_clf(X,y,sample_weight):
    if len(set(y))==1:
        return DummyClassifier(strategy="constant",constant=y[0]).fit(X,y,sample_weight=sample_weight)
    m=Pipeline([("pre",_pre()),("clf",LogisticRegression(C=.4,solver="lbfgs",max_iter=1400,random_state=SEED))])
    m.fit(X,y,clf__sample_weight=sample_weight)
    return m


def _model_reg(X,y,sample_weight):
    if len(set(float(round(z,8)) for z in y))==1:
        return DummyRegressor(strategy="constant",constant=float(y[0])).fit(X,y,sample_weight=sample_weight)
    m=Pipeline([("pre",_pre()),("reg",Ridge(alpha=8.0))])
    m.fit(X,y,reg__sample_weight=sample_weight)
    return m


def _triples(records):
    """Input (Observation,GT,source) triples; source is for reporting ONLY."""
    pools={"orientation":[],"recovery":[]}
    for obs,gt,source in records:
        truth=_norm(gt)
        if not truth:continue
        ref,cs=candidates(obs)
        ref_ok=(ref.normalized==truth)
        ref_edit=edit_distance(ref.normalized,truth)
        k=max(1,len(cs))
        for c in cs:
            if c.normalized==ref.normalized:continue
            ok=(c.normalized==truth)
            label=GAIN if ok and not ref_ok else HARM if ref_ok and not ok else NEUTRAL
            cer_delta=(ref_edit-edit_distance(c.normalized,truth))/max(1,len(truth))
            pools[group_for(c)].append((features(obs,ref,c),label,float(np.clip(cer_delta,-1.,1.)),1./k))
    return pools


def _temp_prob(model,X,temp:float):
    p=model.predict_proba(X)
    out=np.zeros((len(X),3),dtype=float)
    for j,cls in enumerate(model.classes_):out[:,int(cls)]=p[:,j]
    if temp!=1. and len(model.classes_)>1:
        rows=out>0
        logp=np.where(rows,np.log(np.clip(out,1.e-12,1.)), -1.e6)
        logp=logp/temp
        logp=logp-np.max(logp,axis=1,keepdims=True)
        ex=np.exp(logp);out=ex/np.sum(ex,axis=1,keepdims=True)
    assert np.allclose(out.sum(axis=1),1.,atol=1.e-9)
    return out


@dataclass
class _Head:
    classifier:object
    regressor:object
    temperature:float=1.
    count:int=0


class DualArbiter:
    """Three-class separate arbiter heads, CER-aware reward, locked tuning.

    NO guarantee of risk control on OOD data. Audit is a diagnostic gate only.
    """
    def __init__(self,harm_weight=2.5,cer_weight=1.0,cost_weight=.002):
        self.harm_weight=float(harm_weight)
        self.cer_weight=float(cer_weight)
        self.cost_weight=float(cost_weight)
        self.heads={};self.thresholds={"orientation":float("inf"),"recovery":float("inf")}
        self.audited=False;self.audit_report={};self.tuning_report={}

    def fit(self,train):
        pools=_triples(train)
        self.heads={}
        for group,rows in pools.items():
            if not rows:continue
            X=matrix([v[0] for v in rows]); y=np.array([v[1] for v in rows],dtype=int)
            ce=np.array([v[2] for v in rows],dtype=float)
            weights=np.array([v[3] for v in rows],dtype=float)
            clf=_model_clf(X,y,weights)
            reg=_model_reg(X,ce,weights)
            self.heads[group]=_Head(clf,reg,1.,len(rows))
        if not self.heads:raise RuntimeError("No training candidates")
        self.audited=False
        return self

    def calibrate(self,records):
        pools=_triples(records);report={}
        for group,head in self.heads.items():
            rows=pools[group]
            if len(rows)<20 or len(head.classifier.classes_)<2:
                report[group]={"temperature":1.,"n":len(rows),"reason":"insufficient_data"};continue
            X=matrix([v[0] for v in rows]);targets=np.array([v[1] for v in rows],dtype=int)
            def nll(logt):
                prob=_temp_prob(head.classifier,X,float(np.exp(logt)))
                return -float(np.mean(np.log(np.clip(prob[np.arange(len(targets)),targets],1.e-12,1.))))
            result=minimize_scalar(nll,bounds=(-1.4,1.4),method="bounded",options={"xatol":1.e-5})
            head.temperature=float(np.exp(result.x))
            report[group]={"temperature":head.temperature,"n":len(rows),"nll":float(result.fun)}
        return report

    def _rank(self,obs):
        ref,cs=candidates(obs)
        ranked=[]
        for group in ("orientation","recovery"):
            use=[c for c in cs if group_for(c)==group]
            if not use or group not in self.heads:continue
            head=self.heads[group];X=matrix([features(obs,ref,c) for c in use])
            p=_temp_prob(head.classifier,X,head.temperature)
            cer=np.clip(head.regressor.predict(X),-.75,.75)
            for c,pr,ce in zip(use,p,cer):
                fixed_cost=self.cost_weight*METHOD_COST_UNITS.get(c.strategy,2.)
                utility=float(pr[GAIN]-self.harm_weight*pr[HARM]+self.cer_weight*ce-fixed_cost)
                ranked.append({"hyp":c,"group":group,"strategy":c.strategy,
                               "reference":ref.strategy,"score":utility,
                               "p_gain":float(pr[GAIN]),"p_harm":float(pr[HARM]),
                               "p_neither":float(pr[NEUTRAL]),
                               "predicted_cer_gain":float(ce),
                               "fixed_cost":float(fixed_cost)})
        ranked.sort(key=lambda r:(-r["score"],r["strategy"]))
        return ref,ranked

    @staticmethod
    def _decision_from_ranked(ref,ranked,thresholds):
        eligible=[x for x in ranked if x["score"]>thresholds[x["group"]]]
        return eligible[0]["hyp"] if eligible else ref

    def propose(self,obs,thresholds=None):
        ref,ranked=self._rank(obs)
        return self._decision_from_ranked(ref,ranked,self.thresholds if thresholds is None else thresholds)

    def select(self,obs):
        ref,ranked=self._rank(obs)
        if not self.audited:
            chosen=ref;reason="audit_failed_reference_fallback"
        else:
            chosen=self._decision_from_ranked(ref,ranked,self.thresholds)
            reason="accepted_evidence" if chosen is not ref else "reference_fallback"
        return {"prediction":chosen.normalized,"chosen":chosen.strategy,
                "reference":ref.strategy,"reason":reason}

    def rank_trace(self,obs):
        ref,r=self._rank(obs)
        selected=self.select(obs)
        return [{k:v for k,v in d.items() if k!='hyp'}|{
            "prediction":d['hyp'].normalized,"accepted":d['strategy']==selected['chosen'],
            "threshold":self.thresholds[d['group']]}
            for d in r]

    def _evaluate_thresholds(self,precomputed,thresholds):
        aggregate={"wins":0,"harms":0,"edit_gain":0,"promoted":0,"n":0}
        by_source={}
        for r in precomputed:
            ref,ranked,gt,src=r
            d=self._decision_from_ranked(ref,ranked,thresholds)
            winner=int(d.normalized==gt and ref.normalized!=gt)
            harm=int(d.normalized!=gt and ref.normalized==gt)
            edit_gain=edit_distance(ref.normalized,gt)-edit_distance(d.normalized,gt)
            for a in (aggregate,by_source.setdefault(src,{"wins":0,"harms":0,"edit_gain":0,"promoted":0,"n":0})):
                a['wins']+=winner;a['harms']+=harm;a['edit_gain']+=edit_gain
                a['promoted']+=int(d.normalized!=ref.normalized);a['n']+=1
        return aggregate,by_source

    def _precompute(self,records):
        return [(self._rank(o)[0],self._rank(o)[1],_norm(gt),src)
                for o,gt,src in records if _norm(gt)]

    def choose_thresholds(self,records):
        """Tuning split ONLY; score predeclared and risk is descriptive."""
        if not records:raise ValueError("Empty policy tune")
        rows=self._precompute(records)
        GRID=(-.20,-.10,0.,.10,.20,.40,.65,float('inf'))
        best=None
        for t_ori in GRID:
            for t_rec in GRID:
                ts={"orientation":t_ori,"recovery":t_rec}
                agg,src=self._evaluate_thresholds(rows,ts)
                if agg['harms']/max(1,agg['n'])>.025:continue
                if any(v['n']>=35 and v['harms']-v['wins']>2 for v in src.values()):continue
                objective=agg['wins']-2.5*agg['harms']+0.03*agg['edit_gain']
                key=(objective,-agg['harms'],-agg['promoted'])
                if best is None or key>best[0]:best=(key,ts,agg,src)
        self.thresholds=best[1] if best else {"orientation":float('inf'),"recovery":float('inf')}
        self.audited=False
        self.tuning_report={"thresholds":self._printable_thresholds(),
                            "tune":best[2] if best else {},"by_source":best[3] if best else {}}
        return self.tuning_report

    def audit(self,records,min_n=100):
        if not records:raise ValueError('Empty audit')
        rows=self._precompute(records)
        agg,by_source=self._evaluate_thresholds(rows,self.thresholds)
        # This is an empirical audit guard, NOT a guaranteed bound under shift.
        passed=bool(agg['n']>=min_n and agg['wins']>=agg['harms'] and
            agg['harms']/agg['n']<=.025 and
            all(v['wins']+1>=v['harms'] for v in by_source.values() if v['n']>=30))
        self.audited=passed
        self.audit_report={"passed":passed,"n":agg['n'],"summary":agg,"by_source":by_source,
                           "note":"Development audit; no OOD guarantee or conformal claim"}
        return self.audit_report

    def _printable_thresholds(self):
        return {k:(v if np.isfinite(v) else None) for k,v in self.thresholds.items()}

    def manifest(self):
        return {"version":"SAGE-STR-v3-experimental","seed":SEED,
           "thresholds":self._printable_thresholds(),"audit":self.audit_report,
           "calibration_temps":{k:v.temperature for k,v in self.heads.items()},
           "features":COLUMNS,"cost":"fixed token units, not GPU timings",
           "training_classes":"mutually exclusive neutral/gain/harm",
           "reference":"geometry_v3 when available, otherwise original",
           "risk_note":"No guarantee on unseen distributions"}
