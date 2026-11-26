"""Final E1+E2+E3 evaluation against frozen full historical predictions. No fit."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from full_global_eval import norm,digest,spatial_first_table
from rapidfuzz.distance import Levenshtein
from efficient_ctc import scoring_context
COUNTS={'iiit5k':3000,'u14m_artistic':900,'blur_f008':200,'blur_f018':200,'blur_f030':200,'ltb_clean_eval':4789}

def dump(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def load_rows(p):
    with p.open(encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]
def parity(r):
    return r['text']==r['historical_text'] and r['native_equal'] and r['trace_equal'] and r['global_equal']
def metric(rows):
    valid=[r for r in rows if norm(r['gt'])];n=len(valid);chars=sum(len(norm(r['gt'])) for r in valid)
    out={'total':len(rows),'scored':n,'gt_chars':chars,'raw_prediction_mismatches':sum(r['text']!=r['historical_text'] for r in rows),'normalized_prediction_mismatches':sum(norm(r['text'])!=norm(r['historical_text']) for r in rows),'native_mismatches':sum(not r['native_equal'] for r in rows),'trace_mismatches':sum(not r['trace_equal'] for r in rows),'global_mismatches':sum(not r['global_equal'] for r in rows),'cache_hits':sum(r['cache_hits'] for r in rows),'cache_misses':sum(r['cache_misses'] for r in rows),'fast_scored':sum(r['fast_scored'] for r in rows)}
    out['ctc_calls']=sum(r['ctc']['calls'] for r in rows);out['ctc_dp_calls']=sum(r['ctc']['dp_calls'] for r in rows);out['ctc_view_hits']=sum(r['ctc']['view_hits'] for r in rows);out['ctc_score_hits']=sum(r['ctc']['score_hits'] for r in rows);out['ctc_mean_ms']=float(np.mean([r['ctc']['ms'] for r in rows])) if rows else 0.
    out['vision']={k:sum(r['vision'][k] for r in rows) for k in rows[0]['vision']} if rows else {}
    if not n:return out
    for key in ('baseline','historical_text','text'):
        correct=sum(norm(r[key])==norm(r['gt']) for r in valid);edits=sum(Levenshtein.distance(norm(r[key]),norm(r['gt'])) for r in valid)
        out[key]={'correct':correct,'edits':edits,'accuracy_pct':100*correct/n,'cer_pct':100*edits/chars,'gains_vs_b0':sum(norm(r[key])==norm(r['gt']) and norm(r['baseline'])!=norm(r['gt']) for r in valid),'harms_vs_b0':sum(norm(r[key])!=norm(r['gt']) and norm(r['baseline'])==norm(r['gt']) for r in valid)}
    out['paired_wins_vs_historical']=sum(norm(r['text'])==norm(r['gt']) and norm(r['historical_text'])!=norm(r['gt']) for r in valid)
    out['paired_losses_vs_historical']=sum(norm(r['text'])!=norm(r['gt']) and norm(r['historical_text'])==norm(r['gt']) for r in valid)
    for k in ('historical_baseline_ms','historical_evidence_ms','evidence_ms','total_ms'):
        values=[r[k] for r in rows];out[k]={'mean':float(np.mean(values)),'p95':float(np.percentile(values,95)),'sum_hours':float(sum(values)/3600000)}
    out['optimized_over_baseline_latency_ratio_cross_session']=out['total_ms']['mean']/out['historical_baseline_ms']['mean']
    out['historical_evidence_speedup_cross_session']=out['historical_evidence_ms']['mean']/out['evidence_ms']['mean']
    return out

def aggregate(out,journal_dir,mode):
    rows=[];groups={}
    for name,n in COUNTS.items():
        part=load_rows(journal_dir/f'final_{name}.jsonl');assert len(part)==n and [r['index'] for r in part]==list(range(n)) and all(r['dataset']==name and r['mode']==mode for r in part),name
        groups[name]=metric(part);rows.extend(part)
    groups['overall']=metric(rows)
    for subset,rr in [('all',rows),('excluding_known_calibration',[r for r in rows if not r['known_calibration_overlap']])]:
        groups[subset+'/overall']=metric(rr)
        for name in COUNTS:groups[subset+'/'+name]=metric([r for r in rr if r['dataset']==name])
        bins=[]
        for tag,lo,hi in [('26-35',26,35),('36-55',36,55),('56+',56,10**9)]:
            b=metric([r for r in rr if r['dataset']=='ltb_clean_eval' and lo<=len(norm(r['gt']))<=hi]);groups[subset+'/ltb/'+tag]=b;bins.append(b)
        groups[subset+'/ltb/A-Avg']={k:sum(b[k]['accuracy_pct'] for b in bins)/3 for k in ('baseline','historical_text','text')}
    g=groups['overall'];assert g['total']==9289 and g['scored']==9287
    ok=all(parity(r) for r in rows)
    pre=json.loads((out/'final_sentinel_summary.json').read_text()); assert pre['selected_mode']==mode
    speed_ok=pre['selected_speedup_vs_reference']>=1.2 and pre['selected_p95_ms']<=pre['reference_p95_ms']
    used_ok=g['fast_scored']>0 and g['cache_hits']>0 and g['ctc_dp_calls']>0 and g['ctc_view_hits']>0 and (mode=='e3' or g['vision']['ctc_net_hits']>0) and (mode!='e4_batch' or g['vision']['r3_items']>g['vision']['r3_batches'] or g['vision']['s2_items']>g['vision']['s2_batches'])
    decision='PASS_FINAL_'+mode.upper() if ok and speed_ok and used_ok else 'HOLD_REVIEW'
    result={'decision':decision,'mode':mode,'full_prediction_and_trace_parity':ok,'matched_sentinel_speed_gate':speed_ok,'optimization_exercised':used_ok,'groups':groups,'baseline_same_session_24':pre['baseline_same_session'],'selected_over_baseline_latency_ratio_24':pre['selected_over_baseline_latency_ratio'],'note':'Historical matched engineering benchmark, not independent holdout. Full historical latency comparison is cross-session; same-session original-to-final speed evidence is the 24-row subset of the matched 130-row pilot. No reference predictions or GT used for inference or fallback decisions per image.'}
    dump(out/'final_summary.json',result)
    table=[]
    for group,v in groups.items():
        if 'scored' not in v:continue
        for method in ('baseline','historical_text','text'):
            if method not in v:continue
            table.append(dict(group=group,method=method,total=v['total'],scored=v['scored'],**v[method]))
    with (out/'final_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with (out/'final_latency.csv').open('w',newline='',encoding='utf-8') as f:
        keys=['group','historical_evidence_mean_ms','historical_evidence_p95_ms','optimized_evidence_mean_ms','optimized_evidence_p95_ms','optimized_total_mean_ms','optimized_total_p95_ms','cross_session_evidence_speedup'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for k,v in groups.items():
            if 'evidence_ms' in v:w.writerow(dict(zip(keys,[k,v['historical_evidence_ms']['mean'],v['historical_evidence_ms']['p95'],v['evidence_ms']['mean'],v['evidence_ms']['p95'],v['total_ms']['mean'],v['total_ms']['p95'],v['historical_evidence_speedup_cross_session']])))
    print('FINAL DECISION:',decision,json.dumps(g,ensure_ascii=False),flush=True)

def evaluate(a):
    import lmdb,torch,cv2,yaml
    from str_e2e.system import STRSystem
    from str_e2e.backends.efficient_global import GlobalEfficient
    from str_e2e.backends.global_probability import choose
    sys.path.insert(0,str(Path.cwd()/'scripts'))
    from evaluate_unified_e2e_v1 import DATA,read_item
    name=a.dataset;db_path,n=DATA[name];assert n==COUNTS[name]
    cal=json.loads((a.out/'global_calibration.locked.json').read_text());refs=load_rows(a.out/f'historical_{name}.jsonl');assert len(refs)==n and [r['index'] for r in refs]==list(range(n))
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    cfg=yaml.safe_load(Path('configs/system.local.yaml').read_text());journal=a.journal_dir/f'final_{name}.jsonl';identity=a.journal_dir/f'final_{name}.identity.json'
    ident={'dataset':name,'count':n,'mode':a.mode,'historical_sha256':hashlib.sha256((a.out/f'historical_{name}.jsonl').read_bytes()).hexdigest(),'source_hashes':json.loads((a.out/'e4_lock.json').read_text())}
    if identity.exists():assert json.loads(identity.read_text())==ident
    else:assert not journal.exists();dump(identity,ident)
    rows=load_rows(journal) if journal.exists() else [];assert [r['index'] for r in rows]==list(range(len(rows))) and len(rows)<=n
    for r in rows:
        h=refs[r['index']];assert r['rgb_sha256']==h['rgb_sha256'] and r['gt']==h['gt'] and r['mode']==a.mode and r['historical_text']==h['pred']['spatial_first']
    def infer(model,im):
        model.w.efficient_mode='e2';model.w.efficient_scored=0
        torch.cuda.synchronize();t=time.perf_counter()
        model.w.begin_e2()
        try:
            with scoring_context(True) as scorer:
                ev=model.predict_evidence(im);c=ev['candidates'];text=choose(c,spatial_first_table(ev),cal['weights'],cal['temperatures'])[0];orig=choose(c,ev['table'],cal['weights'],cal['temperatures'])[0]
        finally:
            model.w.end_e2()
        torch.cuda.synchronize();return ev,text,orig,(time.perf_counter()-t)*1000,dict(scorer.stats)
    if len(rows)<n:
        with lmdb.open(db_path,readonly=True,lock=False,readahead=False) as db,db.begin() as txn,STRSystem(cfg) as system,journal.open('a',encoding='utf-8') as f:
            assert int(txn.get(b'num-samples'))==n;model=GlobalEfficient(system);model.vision_mode=a.mode
            im,_=read_item(txn,0);infer(model,im);infer(model,im)
            for i in range(len(rows),n):
                im,gt=read_item(txn,i);h=refs[i];sha=digest(im);assert h['dataset']==name and h['rgb_sha256']==sha and h['gt']==gt,(name,i,'historical identity mismatch')
                ev,text,orig,elapsed,ctc=infer(model,im)
                p=ev['pred'];hp=h['pred'];native=all(p[k]==hp[k] for k in ('scope','motion','fuse')) and p['b0']==hp['global_b0'] and norm(p['b0'])==norm(hp['b0'])
                trace=ev['crops']==h['crops'] and ev['spatial_used']==h['spatial_used'] and len(ev['candidates'])==h['candidate_count']
                r=dict(dataset=name,index=i,gt=gt,rgb_sha256=sha,mode=a.mode,known_calibration_overlap=h['known_calibration_overlap'],baseline=hp['b0'],historical_text=hp['spatial_first'],text=text,native_equal=native,trace_equal=trace,global_equal=orig==hp['global'],historical_baseline_ms=h['baseline_latency_ms'],historical_evidence_ms=h['global_evidence_latency_ms'],evidence_ms=ev['latency_ms'],total_ms=elapsed,cache_hits=getattr(model.w,'cache_hits',0) if a.mode in ('e3','e4_cache','e4_batch') else 0,cache_misses=getattr(model.w,'cache_misses',0) if a.mode in ('e3','e4_cache','e4_batch') else 0,fast_scored=model.w.efficient_scored,candidate_count=len(ev['candidates']),crops=ev['crops'],spatial_used=ev['spatial_used'],native_predictions=p,global_prediction=orig,ctc=ctc,vision=dict(model.vision_stats))
                rows.append(r);f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                if (i+1)%50==0 or i+1==n:print(name,i+1,n,'mismatches',sum(not parity(q) for q in rows),flush=True)
    dump(a.out/f'final_{name}_summary.json',metric(rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--journal-dir',type=Path,required=True);p.add_argument('--mode',choices=('e3','e4_cache','e4_batch'),required=True);p.add_argument('--dataset',choices=COUNTS);p.add_argument('--aggregate',action='store_true');a=p.parse_args()
    if a.aggregate:aggregate(a.out,a.journal_dir,a.mode)
    else:assert a.dataset;evaluate(a)
if __name__=='__main__':main()
