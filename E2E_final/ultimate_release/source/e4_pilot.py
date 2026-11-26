import argparse,json,sys,time,traceback,itertools
from pathlib import Path
import numpy as np
from full_global_eval import digest,spatial_first_table,norm
from efficient_pilot import compare
from efficient_ctc import scoring_context

MODES=('e3','e4_cache','e4_batch')
def evaluate(model,im,mode,cal,torch):
    model.vision_mode='e3' if mode=='reference' else mode
    model.w.efficient_mode='reference' if mode=='reference' else 'e2';model.w.efficient_scored=0
    torch.cuda.synchronize();t=time.perf_counter()
    if mode!='reference':model.w.begin_e2()
    try:
        with scoring_context(mode!='reference') as scorer:
            ev=model.predict_evidence(im);text,_,posterior=choose_evidence(ev,cal)
    finally:
        if mode!='reference':model.w.end_e2()
    torch.cuda.synchronize()
    return ev,dict(text=text,ms=(time.perf_counter()-t)*1000,posterior=posterior,fast_scored=model.w.efficient_scored,cache_hits=getattr(model.w,'cache_hits',0) if mode!='reference' else 0,cache_misses=getattr(model.w,'cache_misses',0) if mode!='reference' else 0,ctc=dict(scorer.stats),vision=dict(model.vision_stats))
def choose_evidence(ev,cal):
    from str_e2e.backends.global_probability import choose
    return choose(ev['candidates'],spatial_first_table(ev),cal['weights'],cal['temperatures'])
def compare_pair(base,candidate,br,cr,reference=False):
    equal,support,delta=compare(base,candidate)
    if equal:delta=max(delta,float(np.max(np.abs(np.asarray(br['posterior'])-np.asarray(cr['posterior'])))))
    trace=all(base[k]==candidate[k] for k in ('pred','crops','spatial_used'))
    counts=reference or (all(br[k]==cr[k] for k in ('fast_scored','cache_hits','cache_misses')) and br['ctc']['calls']==cr['ctc']['calls'])
    return dict(candidate_equal=equal,support_equal=support,trace_equal=trace,counts_equal=counts,delta=delta,text_equal=br['text']==cr['text'])
def main():
    import lmdb,torch,cv2,yaml
    from str_e2e.system import STRSystem
    from str_e2e.backends.efficient_global import GlobalEfficient
    sys.path.insert(0,str(Path.cwd()/'scripts'))
    from evaluate_unified_e2e_v1 import DATA,read_item
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--dataset',required=True);a=p.parse_args();o=a.out
    cal=json.loads((o/'global_calibration.locked.json').read_text());subset=[r for r in json.loads((o/'efficient_manifest.json').read_text()) if r['dataset']==a.dataset];assert subset
    sent={(r['dataset'],r['index']) for r in json.loads((o/'final_sentinel_manifest.json').read_text())}
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    cfg=yaml.safe_load(Path('configs/system.local.yaml').read_text());db_path,n=DATA[a.dataset];disabled={}
    with lmdb.open(db_path,readonly=True,lock=False,readahead=False) as db,db.begin() as txn,STRSystem(cfg) as system,(o/f'e4_{a.dataset}.jsonl').open('w',encoding='utf-8') as f:
        assert int(txn.get(b'num-samples'))==n;model=GlobalEfficient(system)
        im,gt=read_item(txn,subset[0]['index']);assert digest(im)==subset[0]['rgb_sha256'] and gt==subset[0]['gt']
        def safe(im,mode):
            if mode in disabled:return None,None
            try:return evaluate(model,im,mode,cal,torch)
            except Exception as exc:
                if mode in ('e3','reference'):raise
                disabled[mode]=repr(exc);print('CANDIDATE_DISABLED',mode,traceback.format_exc(),flush=True)
                if isinstance(exc,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
                return None,None
        official=system._backend('official');official.predict(im);torch.cuda.synchronize()
        for mode in ('reference',)+MODES:safe(im,mode)
        for j,r in enumerate(subset):
            im,gt=read_item(txn,r['index']);assert digest(im)==r['rgb_sha256'] and gt==r['gt']
            modes=('reference',)+MODES if (a.dataset,r['index']) in sent else MODES
            perms=list(itertools.permutations(modes));order=perms[j%len(perms)];ev={};res={}
            for mode in order:ev[mode],res[mode]=safe(im,mode)
            pairs={mode:compare_pair(ev['e3'],ev[mode],res['e3'],res[mode],mode=='reference') for mode in modes if mode!='e3' and ev[mode] is not None}
            baseline=None
            if 'reference' in modes:
                torch.cuda.synchronize();t=time.perf_counter();bt=official.predict(im).text;torch.cuda.synchronize();baseline=dict(text=bt,ms=(time.perf_counter()-t)*1000,parity=norm(bt)==norm(ev['e3']['pred']['b0']))
            row=dict(baseline=baseline,dataset=a.dataset,index=r['index'],gt=gt,rgb_sha256=r['rgb_sha256'],historical=r['pred']['spatial_first'],results=res,pairs=pairs,disabled=disabled.copy(),order=list(order),crops=ev['e3']['crops'])
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            print(a.dataset,j+1,len(subset),{k:round(v['ms']) if v else None for k,v in res.items()},flush=True)
if __name__=='__main__':main()
