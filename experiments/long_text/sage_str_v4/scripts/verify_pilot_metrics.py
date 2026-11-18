from pathlib import Path
import csv, re

here=Path(__file__).resolve().parents[1]
results=here.parents[2]/'reports/long_text/sage_str_v4/results'

def norm(text): return re.sub('[^0-9a-z]','',str(text).lower())

def edit(a,b):
    if len(a)<len(b): a,b=b,a
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def rows(filename):
    with (results/filename).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

preds=rows('evaluation_predictions.csv'); metrics=rows('evaluation_accuracy_cer_by_dataset.csv')
methods={'SVTRv2-T_single':'svtrv2','SVTRv2-T_v3':'geometry_v3',
         'Unified_v1':'unified_v1','Safe_reference':'reference_prediction',
         'SAGE_v3_locked':'sage_v3','SAGE_v4':'sage_v4'}
assert len(preds)==461, len(preds)
for m in metrics:
    ds=m['dataset']; method=m['method']; data=[r for r in preds if r['dataset']==ds]
    valid=[(norm(r['gt']),norm(r[methods[method]])) for r in data if norm(r['gt'])]
    n=len(valid); correct=sum(x==y for x,y in valid)
    edits=sum(edit(x,y) for x,y in valid); chars=sum(len(x) for x,_ in valid)
    assert n==int(m['n']) and correct==int(m['exact_correct']), (ds,method,'accuracy')
    assert edits==int(m['total_edit_errors']), (ds,method,'edit')
    assert abs(100*edits/chars-float(m['CER_pct']))<1e-7, (ds,method,'CER')
print(f'PILOT METRICS PASS: {len(preds)} eval images, {len(metrics)} metric rows; exact-line, CER, edit counts match exported CSV')
