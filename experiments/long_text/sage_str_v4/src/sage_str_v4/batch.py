"""Deterministic vectorized inference for frozen-candidate SAGE v3/v4 selectors.

No GT, source labels, image IDs or wall time enters scoring. A reference implementation
of per-image select() is checked for parity in the notebook.
"""
from collections import defaultdict
import numpy as np


def score_observations(selector, observations, module):
    """Return (decision list, per-observation GT-blind candidate-score lists)."""
    refs=[]
    ranked=[[] for _ in observations]
    grouped=defaultdict(list)
    for i,obs in enumerate(observations):
        ref,cs=module.candidates(obs)
        refs.append(ref)
        for c in cs:
            group=module.group_for(c)
            if group in selector.heads:
                grouped[group].append((i,c,module.features(obs,ref,c)))
    for group,records in grouped.items():
        head=selector.heads[group]
        X=module.matrix([r[2] for r in records])
        proba=module._temp_prob(head.classifier,X,head.temperature)
        delta=np.clip(head.regressor.predict(X),-.75,.75)
        for (idx,c,_),pr,ce in zip(records,proba,delta):
            fixed=selector.cost_weight*module.METHOD_COST_UNITS.get(c.strategy,2.)
            utility=float(pr[module.GAIN]-selector.harm_weight*pr[module.HARM]+selector.cer_weight*ce-fixed)
            ranked[idx].append({'hyp':c,'group':group,'strategy':c.strategy,
                    'reference':refs[idx].strategy,'score':utility,
                    'p_gain':float(pr[module.GAIN]),'p_harm':float(pr[module.HARM]),
                    'p_neither':float(pr[module.NEUTRAL]),
                    'predicted_cer_gain':float(ce),'fixed_cost':float(fixed)})
    decisions=[]; trace=[]
    for ref,rr in zip(refs,ranked):
        rr.sort(key=lambda d:(-d['score'],d['strategy']))
        if selector.audited:
            chosen=selector._decision_from_ranked(ref,rr,selector.thresholds)
            reason='accepted_evidence' if chosen is not ref else 'reference_fallback'
        else:
            chosen=ref;reason='audit_failed_reference_fallback'
        decisions.append({'prediction':chosen.normalized,'chosen':chosen.strategy,
                          'reference':ref.strategy,'reason':reason})
        trace.append([{k:v for k,v in d.items() if k!='hyp'} | {
            'prediction':d['hyp'].normalized,
            'accepted':d['strategy']==chosen.strategy,
            'threshold':selector.thresholds[d['group']]}
            for d in rr])
    return decisions,trace
