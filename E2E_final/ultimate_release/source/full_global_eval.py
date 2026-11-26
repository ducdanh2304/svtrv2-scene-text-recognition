"""Frozen full-corpus evaluation; no refitting, GT used only for evaluation."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from rapidfuzz.distance import Levenshtein

def norm(s): return ''.join(c for c in str(s).casefold() if c.isalnum())
def digest(im): return hashlib.sha256(f'{im.width}x{im.height}:'.encode()+im.tobytes()).hexdigest()
def spatial_first_table(row):
    selected=[s if s is not None else f for f,s in zip(row['full_table'],row['spatial_table'])]
    art=[np.asarray(p,dtype=float) for p in selected[2:] if p is not None]
    return [selected[0],selected[1],np.mean(art,axis=0) if art else None]
def report(rows, cohort, subset):
    rr=[r for r in rows if norm(r['gt'])];n=len(rr);chars=sum(len(norm(r['gt'])) for r in rr)
    if not n:return []
    out=[]
    for m in ('b0','global','spatial_first','scope','motion','fuse'):
        correct=edits=gain=harm=empty=0
        for r in rr:
            g=norm(r['gt']);p=norm(r['pred'][m]);b=norm(r['pred']['b0']);correct+=p==g;edits+=Levenshtein.distance(p,g);gain+=p==g and b!=g;harm+=p!=g and b==g;empty+=not p
        out.append(dict(dataset=cohort,subset=subset,method=m,total=len(rows),scored=n,correct=correct,accuracy_pct=100*correct/n,edits=edits,gt_chars=chars,cer_pct=100*edits/chars,gains=gain,harms=harm,empty=empty))
    return out

def summaries(rows,name):
    groups={'all':rows,'excluding_known_calibration':[r for r in rows if not r['known_calibration_overlap']]}
    if name=='ltb_clean_eval':
        groups['historical_eval240']=[r for r in rows if r['historical_eval240']]
    result=[]
    for label,rs in groups.items():
        result+=report(rs,name,label)
        if name=='ltb_clean_eval':
            bins=[]
            for tag,lo,hi in [('26-35',26,35),('36-55',36,55),('56+',56,10**9),('0-25',0,25)]:
                sub=report([r for r in rs if lo<=len(norm(r['gt']))<=hi],name,label+'/'+tag);result+=sub
                if tag!='0-25':bins.append(sub)
            if all(bins):
                for m in ('b0','global','spatial_first','scope','motion','fuse'):
                    v=[next(r for r in b if r['method']==m) for b in bins]
                    result.append(dict(dataset=name,subset=label+'/A-Avg',method=m,total=sum(x['total'] for x in v),scored=sum(x['scored'] for x in v),accuracy_pct=sum(x['accuracy_pct'] for x in v)/3))
    return result

def main():
    import lmdb,torch,cv2,yaml
    from str_e2e.system import STRSystem
    sys.path.insert(0,str(Path.cwd()/'scripts'))
    from evaluate_unified_e2e_v1 import DATA,read_item,indices_for,MAP
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True);p.add_argument('--dataset',choices=DATA,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=None);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    import str_e2e.backends as pkg
    pkg.__path__.insert(0,str(a.overlay))
    from str_e2e.backends.global_joint import GlobalJoint
    from str_e2e.backends.global_probability import choose
    cal=json.loads((a.overlay/'global_calibration.locked.json').read_text());dev_path=a.overlay/'calibration_manifest.json';assert hashlib.sha256(dev_path.read_bytes()).hexdigest()==cal['manifest_sha256']
    dev=json.loads(dev_path.read_text());dev_groups={r['source_group_id'] for r in dev};dev_hashes={r['rgb_sha256'] for r in dev};dev_ids={(r['cohort'],r['index']) for r in dev}
    refs=json.loads((a.overlay/'global_reference940.json').read_text());ref_by={(r['cohort'],r['index']):r for r in refs}
    cfg=yaml.safe_load(Path('configs/system.local.yaml').read_text());torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    name=a.dataset;db_path,n=DATA[name];stop=n if a.stop is None else a.stop;assert 0<=a.start<stop<=n
    eligible=set(indices_for(name)) if name=='ltb_clean_eval' else set()
    split={int(r['index_0based']):r['split'] for r in csv.DictReader(MAP.open(encoding='utf-8-sig'))} if name=='ltb_clean_eval' else {}
    stem=f'{name}_{a.start}_{stop}';journal=a.out/(stem+'.jsonl');identity=a.out/(stem+'.identity.json');identity_data=dict(dataset=name,start=a.start,stop=stop,dataset_total=n,cal_sha256=hashlib.sha256((a.overlay/'global_calibration.locked.json').read_bytes()).hexdigest(),source_sha256={k:hashlib.sha256((a.overlay/k).read_bytes()).hexdigest() for k in ('global_joint.py','global_probability.py','full_global_eval.py')})
    if identity.exists():assert json.loads(identity.read_text())==identity_data
    else:
        assert not journal.exists();identity.write_text(json.dumps(identity_data,indent=2))
    rows=[]
    if journal.exists():
        with journal.open() as f:
            rows=[json.loads(l) for l in f if l.strip()]
        assert [r['index'] for r in rows]==list(range(a.start,a.start+len(rows)))
    with lmdb.open(db_path,readonly=True,lock=False,readahead=False) as db,db.begin() as txn:
        assert int(txn.get(b'num-samples'))==n
        if len(rows)<stop-a.start:
            with STRSystem(cfg) as system,journal.open('a',encoding='utf-8') as f:
                model=GlobalJoint(system); official=system._backend('official')
                for i in range(a.start+len(rows),stop):
                    im,gt=read_item(txn,i);sha=digest(im);group=f'iiit_val:{i:06d}' if name.startswith('blur_') else f'ltb:{sha}' if name=='ltb_clean_eval' else f'{name}:{i:06d}'
                    t=time.perf_counter();base=official.predict(im).text;torch.cuda.synchronize();base_ms=(time.perf_counter()-t)*1000
                    evidence=model.predict_evidence(im);pred=evidence['pred'];c=evidence['candidates']
                    original=choose(c,evidence['table'],cal['weights'],cal['temperatures'])[0];spatial=choose(c,spatial_first_table(evidence),cal['weights'],cal['temperatures'])[0]
                    ref=ref_by.get((name,i));known_reference=bool(ref is not None)
                    if ref is not None:
                        assert ref['rgb_sha256']==sha and ref['gt']==gt,(name,i,'reference identity')
                        for k in ('b0','scope','motion','fuse'):
                            if k in ref['pred']:assert norm(ref['pred'][k])==norm(pred[k]),(name,i,k,'recognizer parity')
                    row=dict(dataset=name,index=i,gt=gt,rgb_sha256=sha,source_group_id=group,known_calibration_overlap=group in dev_groups or sha in dev_hashes or (name,i) in dev_ids,historical_ltb_split=split.get(i),historical_eval240=i in eligible,known_reference=known_reference,pred=dict(b0=base,global_b0=pred['b0'],scope=pred['scope'],motion=pred['motion'],fuse=pred['fuse'],spatial_first=spatial,**{'global':original}),b0_parity=norm(base)==norm(pred['b0']),baseline_latency_ms=base_ms,global_evidence_latency_ms=evidence['latency_ms'],crops=evidence['crops'],spatial_used=evidence['spatial_used'],candidate_count=len(c),candidate_oracle=norm(gt) in {norm(x) for x in c})
                    rows.append(row);f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                    if (i-a.start+1)%50==0:print(name,i+1,stop,flush=True)
    assert len(rows)==stop-a.start
    results=summaries(rows,name)
    with (a.out/(stem+'_metrics.csv')).open('w',newline='',encoding='utf-8') as f:
        keys=['dataset','subset','method','total','scored','correct','accuracy_pct','edits','gt_chars','cer_pct','gains','harms','empty'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(results)
    status=dict(dataset=name,start=a.start,stop=stop,total=n,complete_dataset=a.start==0 and stop==n,rows=len(rows),scored=sum(bool(norm(r['gt'])) for r in rows),known_calibration_overlap=sum(r['known_calibration_overlap'] for r in rows),independent_b0_mismatches=sum(not r['b0_parity'] for r in rows),note='Full historical corpus, not independent holdout. Excluding known calibration is not project-wide train/test decontamination. Spatial-first is post-hoc, fixed before this run. No new fit.')
    (a.out/(stem+'_status.json')).write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
if __name__=='__main__':main()
