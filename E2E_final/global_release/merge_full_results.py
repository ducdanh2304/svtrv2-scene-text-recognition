"""Merge CS406 full-benchmark ZIPs or directories. Standard library only."""
import argparse
import csv
import hashlib
import json
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

EXPECTED = dict(iiit5k=3000,u14m_artistic=900,blur_f008=200,blur_f018=200,blur_f030=200,ltb_clean_eval=4789)
METHODS = ('b0','global','spatial_first','scope','motion','fuse')
PAT = re.compile(r'^(iiit5k|u14m_artistic|blur_f008|blur_f018|blur_f030|ltb_clean_eval)_(\d+)_(\d+)\.jsonl$')

def norm(s):
    return ''.join(c for c in str(s).casefold() if c.isalnum())

try:
    from rapidfuzz.distance.Levenshtein import distance
except ImportError:
    def distance(a,b):
        if len(a)<len(b): a,b=b,a
        prev=list(range(len(b)+1))
        for i,x in enumerate(a,1):
            cur=[i]
            for j,y in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
            prev=cur
        return prev[-1]

def items(path):
    if path.is_dir():
        for f in sorted(path.rglob('*.jsonl')):
            if PAT.fullmatch(f.name):
                yield str(f),f.read_text(encoding='utf-8'),json.loads(f.with_suffix('.identity.json').read_text(encoding='utf-8'))
    else:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if PAT.fullmatch(PurePosixPath(n).name):
                    identity=n[:-6]+'.identity.json'
                    yield f'{path}:{n}',z.read(n).decode('utf-8'),json.loads(z.read(identity))

def same_prediction(a,b):
    keys=('dataset','index','gt','rgb_sha256','source_group_id','known_calibration_overlap','historical_eval240','pred','b0_parity')
    return all(a[k]==b[k] for k in keys)

def merge(paths):
    rows={}; lock=None; duplicates=0; journals=[]
    for p in paths:
        for label,raw,identity in items(p):
            name,start,stop=PAT.fullmatch(PurePosixPath(label.split(':')[-1]).name).groups();start=int(start);stop=int(stop)
            assert (identity['dataset'],identity['start'],identity['stop'],identity['dataset_total'])==(name,start,stop,EXPECTED[name]),label
            assert 0<=start<stop<=EXPECTED[name],label
            this=(identity['cal_sha256'],identity['source_sha256'])
            assert set(this[1])=={'global_joint.py','global_probability.py','full_global_eval.py'},label
            assert all(re.fullmatch('[a-f0-9]{64}',x) for x in [this[0],*this[1].values()]),label
            if lock is None: lock=this
            assert lock==this, f'Mixed calibration/source versions: {label}'
            parsed=[]
            for line in raw.splitlines():
                if line.strip(): parsed.append(json.loads(line))
            assert len(parsed)<=stop-start,label
            assert [r['index'] for r in parsed]==list(range(start,start+len(parsed))),f'Non-contiguous journal: {label}'
            for r in parsed:
                assert r['dataset']==name and set(METHODS)<=set(r['pred']),label
                assert isinstance(r['gt'],str) and all(isinstance(r['pred'][m],str) for m in METHODS),label
                assert re.fullmatch('[a-f0-9]{64}',r['rgb_sha256']),label
                k=(name,r['index'])
                if k in rows:
                    assert same_prediction(rows[k],r),f'Conflicting duplicate {k}'
                    duplicates+=1
                else: rows[k]=r
            journals.append({'journal':label,'start':start,'stop':stop,'rows':len(parsed)})
    if not journals: raise ValueError('No matching full benchmark JSONL journals found.')
    return rows,lock,duplicates,journals

def metric(rows,name,subset,method):
    valid=[r for r in rows if norm(r['gt'])]; n=len(valid)
    if not n: return None
    correct=edits=gains=harms=empty=chars=0
    for r in valid:
        g=norm(r['gt']); p=norm(r['pred'][method]); b=norm(r['pred']['b0'])
        correct+=p==g; edits+=distance(p,g); chars+=len(g); gains+=p==g and b!=g; harms+=p!=g and b==g; empty+=not p
    assert gains-harms==correct-sum(norm(r['pred']['b0'])==norm(r['gt']) for r in valid)
    latkey='baseline_latency_ms' if method=='b0' else 'global_evidence_latency_ms' if method in ('global','spatial_first') else None
    lats=[r[latkey] for r in rows if latkey and r.get(latkey) is not None]
    return dict(dataset=name,subset=subset,method=method,total=len(rows),scored=n,correct=correct,accuracy_pct=100*correct/n,edits=edits,gt_chars=chars,cer_pct=100*edits/chars,gains=gains,harms=harms,empty=empty,median_recorded_latency_ms=statistics.median(lats) if lats else None)

def metrics(rows,name):
    result=[]
    subsets={'all':rows,'excluding_known_calibration':[r for r in rows if not r['known_calibration_overlap']]}
    if name=='ltb_clean_eval': subsets['historical_eval240']=[r for r in rows if r['historical_eval240']]
    for label,rs in subsets.items():
        for m in METHODS:
            v=metric(rs,name,label,m)
            if v: result.append(v)
            if name!='ltb_clean_eval': continue
            bins=[]
            for tag,lo,hi in [('26-35',26,35),('36-55',36,55),('56+',56,10**9),('0-25',0,25)]:
                b=metric([r for r in rs if lo<=len(norm(r['gt']))<=hi],name,label+'/'+tag,m)
                if b: result.append(b)
                if tag!='0-25': bins.append(b)
            if all(bins): result.append(dict(dataset=name,subset=label+'/A-Avg',method=m,total=sum(b['total'] for b in bins),scored=sum(b['scored'] for b in bins),accuracy_pct=sum(b['accuracy_pct'] for b in bins)/3))
    return result

def ranges(nums):
    out=[]
    for i in sorted(nums):
        if out and i==out[-1][1]:out[-1][1]+=1
        else:out.append([i,i+1])
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=Path);p.add_argument('--out',type=Path,default=Path('CS406_FULL_MERGED'));p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    rows,lock,dups,journals=merge(a.inputs);a.out.mkdir(parents=True,exist_ok=True)
    coverage={};table=[];allrows=[]
    for name,n in EXPECTED.items():
        rr=[rows[(name,i)] for i in range(n) if (name,i) in rows];allrows.extend(rr)
        missing=set(range(n))-{r['index'] for r in rr}
        coverage[name]=dict(expected=n,observed=len(rr),missing=len(missing),missing_ranges=ranges(missing),complete=not missing,b0_mismatches=sum(not r['b0_parity'] for r in rr))
        table.extend(metrics(rr,name))
    complete=all(v['complete'] for v in coverage.values());parity=all(v['b0_mismatches']==0 for v in coverage.values())
    table.extend(metrics(allrows,'overall_full' if complete else 'overall_partial'))
    fields=['dataset','subset','method','total','scored','correct','accuracy_pct','edits','gt_chars','cer_pct','gains','harms','empty','median_recorded_latency_ms']
    with (a.out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
    with (a.out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for r in allrows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    status=dict(complete=complete,b0_parity=parity,expected=sum(EXPECTED.values()),observed=len(rows),duplicates_removed=dups,cal_sha256=lock[0],source_sha256=lock[1],datasets=coverage,journals=journals,note='Historical full corpus; not independent holdout. Source/calibration hashes checked for consistency across shards, not independently authenticated. Spatial-first was developed post-hoc. Latency is recorded evidence latency, not end-to-end deployment latency. Missing ranges use [start, stop).')
    (a.out/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(f'Coverage: {len(rows)}/{sum(EXPECTED.values())}; duplicates removed: {dups}; B0 parity: {parity}')
    for k,v in coverage.items(): print(k,v['observed'],'/',v['expected'],'missing',v['missing_ranges'])
    print('Outputs:',a.out)
    if a.require_complete and (not complete or not parity): raise SystemExit(2)

if __name__=='__main__': main()
