import json,sys
from pathlib import Path
import numpy as np
from rapidfuzz.distance import Levenshtein
from full_global_eval import norm
D=('iiit5k','u14m_artistic','blur_f008','blur_f018','blur_f030','ltb_clean_eval')
MODES=('e3','e4_cache','e4_batch')
def pair_ok(p):
    return bool(p and all(p[k] for k in ('candidate_equal','support_equal','trace_equal','counts_equal','text_equal')) and p['delta']<=2e-4)
def summarize(rows,mode):
    rs=[r for r in rows if r['results'].get(mode) is not None];n=len(rs)
    if n!=len(rows) or not n:return dict(n=n,expected=len(rows),complete=False)
    v=[r['results'][mode] for r in rs];times=np.asarray([x['ms'] for x in v]);assert np.isfinite(times).all() and min(times)>0
    chars=sum(len(norm(r['gt'])) for r in rs);valid=sum(bool(norm(r['gt'])) for r in rs)
    return dict(n=n,complete=True,mean_ms=float(times.mean()),p95_ms=float(np.percentile(times,95)),accuracy_pct=100*sum(norm(r['gt'])==norm(x['text']) and bool(norm(r['gt'])) for r,x in zip(rs,v))/valid,cer_pct=100*sum(Levenshtein.distance(norm(r['gt']),norm(x['text'])) for r,x in zip(rs,v) if norm(r['gt']))/chars,historical_mismatches=sum(x['text']!=r['historical'] for r,x in zip(rs,v)),vision={k:sum(x['vision'][k] for x in v) for k in v[0]['vision']})
def main(o):
    rows=[json.loads(l) for d in D for l in (o/f'e4_{d}.jsonl').read_text().splitlines() if l.strip()]
    expected={(r['dataset'],r['index']):(r['rgb_sha256'],r['gt'],r['pred']['spatial_first']) for r in json.loads((o/'efficient_manifest.json').read_text())}
    actual={(r['dataset'],r['index']):(r['rgb_sha256'],r['gt'],r['historical']) for r in rows}
    assert len(rows)==len(actual)==len(expected)==130 and actual==expected
    sr=[r for r in rows if 'reference' in r['results']];sent={(r['dataset'],r['index']) for r in json.loads((o/'final_sentinel_manifest.json').read_text())};assert len(sr)==len(sent)==24 and {(r['dataset'],r['index']) for r in sr}==sent
    groups={'overall':rows,**{d:[r for r in rows if r['dataset']==d] for d in D}}
    for tag,lo,hi in [('26-35',26,35),('36-55',36,55),('56+',56,10**9)]:groups['ltb/'+tag]=[r for r in groups['ltb_clean_eval'] if lo<=len(norm(r['gt']))<=hi]
    metrics={g:{m:summarize(rr,m) for m in MODES} for g,rr in groups.items()};ref=summarize(sr,'reference');sm={m:summarize(sr,m) for m in MODES}
    reference_ok=ref['historical_mismatches']==0 and all(pair_ok(r['pairs'].get('reference')) and r['baseline']['parity'] for r in sr)
    bt=[r['baseline']['ms'] for r in sr];assert np.isfinite(bt).all() and min(bt)>0
    baseline=dict(n=len(bt),mean_ms=float(np.mean(bt)),p95_ms=float(np.percentile(bt,95)))
    base=metrics['overall']['e3'];base_ok=base['historical_mismatches']==0 and reference_ok
    def speed_ref(m):return sm[m]['complete'] and ref['mean_ms']/sm[m]['mean_ms']>=1.2 and sm[m]['p95_ms']<=ref['p95_ms']
    gates={};accepted=[]
    if base_ok and speed_ref('e3'):accepted.append('e3')
    for m in MODES[1:]:
        v=metrics['overall'][m];parity=v['complete'] and v['historical_mismatches']==0 and all(pair_ok(r['pairs'].get(m)) for r in rows)
        speed=bool(v['complete'] and base['mean_ms']/v['mean_ms']>=1.1 and v['p95_ms']<=base['p95_ms'] and all(metrics[g][m]['complete'] and metrics[g][m]['mean_ms']<=1.05*metrics[g]['e3']['mean_ms'] for g in groups if g!='overall'))
        used=bool(v['complete'] and v['vision']['ctc_net_hits']>0 and (m!='e4_batch' or v['vision']['r3_items']>v['vision']['r3_batches'] or v['vision']['s2_items']>v['vision']['s2_batches']))
        gates[m]=dict(parity=parity,speed=speed,optimization_used=used,reference_speed_gate=speed_ref(m))
        if base_ok and parity and speed and used and speed_ref(m):accepted.append(m)
    mode=min(accepted,key=lambda m:metrics['overall'][m]['mean_ms']) if accepted else None
    summary=dict(baseline_same_session=baseline,selected_mode=mode,decision='SELECT_'+mode.upper() if mode else 'HOLD_REVIEW',reference_parity=reference_ok,e3_parity=base_ok,gates=gates,metrics=metrics,reference_mean_ms=ref['mean_ms'],reference_p95_ms=ref['p95_ms'],selected_speedup_vs_reference=ref['mean_ms']/sm[mode]['mean_ms'] if mode else 0,selected_p95_ms=sm[mode]['p95_ms'] if mode else None,sentinel=sm,selected_over_baseline_latency_ratio=sm[mode]['mean_ms']/baseline['mean_ms'] if mode else None,note='Historical 130-row pilot, same-session original on fixed 24-row subset. Selection locked before full inference; no per-image labels used.')
    for name in ('e4_summary.json','final_sentinel_summary.json'):(o/name).write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('E4 DECISION',summary['decision'],json.dumps(metrics['overall']),flush=True)
    if not mode:raise RuntimeError('No validated mode; no full run')
if __name__=='__main__':main(Path(sys.argv[1]))
