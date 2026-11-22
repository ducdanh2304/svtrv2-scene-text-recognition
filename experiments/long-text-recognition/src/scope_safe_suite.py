"""One-pass cross-dataset evaluation for the pinned SCOPE-CTC E7 FINAL.

This module ONLY handles data acquisition, locked evaluation, crash recovery and
reporting. It does not alter model weights, crop policy, beam, or edit thresholds.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from rapidfuzz.distance import Levenshtein
from scipy.stats import binomtest

DATASET_REPO = 'topdu/OpenOCR-Data'
DATASET_REV = 'd59c13643f1d7f92831625fe161e2c19ec3d7f52'

# Official OpenOCR test variants, NOT the larger IC13_1015 / IC15_2077 variants.
SPECS = [
    ('Common','IIIT5K','test/IIIT5k',3000),
    ('Common','SVT','test/SVT',647),
    ('Common','IC13-857','test/IC13_857',857),
    ('Common','IC15-1811','test/IC15_1811',1811),
    ('Common','SVTP','test/SVTP',645),
    ('Common','CUTE80','test/CUTE80',288),
    ('Union14M','Curve','u14m/curve',2426),
    ('Union14M','Multi-Oriented','u14m/multi_oriented',1369),
    ('Union14M','Artistic','u14m/artistic',900),
    ('Union14M','Contextless','u14m/contextless',779),
    ('Union14M','Salient','u14m/salient',1585),
    ('Union14M','Multi-Words','u14m/multi_words',829),
    ('OST','Weak','OST/weak',None),
    ('OST','Heavy','OST/heavy',None),
]
GENERAL = ('Union14M','General','u14m/general',400000)
METHODS = ['original_svtrv2_full12','safe_scope']
CANDIDATE_METHOD = 'safe_scope'


def checksum(obj):
    if isinstance(obj,(bytes,bytearray)): payload=obj
    else: payload=json.dumps(obj,sort_keys=True,ensure_ascii=False,default=str).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def fetch_lmdb(spec, data_home, *, revision=DATASET_REV):
    from huggingface_hub import hf_hub_download
    family,name,folder,expected=spec
    data_home=Path(data_home)
    local=data_home/folder/'data.mdb'
    if local.is_file() and local.stat().st_size>4096:
        return local.parent
    data_home.mkdir(parents=True, exist_ok=True)
    err=None
    for attempt in range(1,4):
        try:
            result=hf_hub_download(repo_id=DATASET_REPO,filename=folder+'/data.mdb',
                                   repo_type='dataset',revision=revision,
                                   local_dir=str(data_home),etag_timeout=60)
            result=Path(result)
            if result.stat().st_size<=4096:raise RuntimeError(f'Download suspiciously small: {result}')
            return result.parent
        except Exception as ex:
            err=ex
            print(f'DOWNLOAD RETRY {name} {attempt}/3: {type(ex).__name__}: {ex}',flush=True)
            if attempt<3:time.sleep(2*attempt)
    raise RuntimeError(f'Failed to download {name} LMDB: {err}') from err


class LmdbSamples:
    def __init__(self,folder,*,expected=None):
        import lmdb
        self.folder=Path(folder)
        self.env=lmdb.open(str(self.folder),readonly=True,lock=False,readahead=False,
                           meminit=False,max_readers=32)
        with self.env.begin(write=False) as tr:
            count=tr.get(b'num-samples')
            if count is None:raise RuntimeError(f'Missing num-samples: {folder}')
            self.n=int(count)
            if self.n<=0:raise RuntimeError(f'Empty LMDB: {folder}')
            for i in sorted(set((1, max(1,self.n//2),self.n))):
                if not tr.get(f'image-{i:09d}'.encode()):
                    raise RuntimeError(f'Missing image-{i:09d} in {folder}')
                if tr.get(f'label-{i:09d}'.encode()) is None:
                    raise RuntimeError(f'Missing label-{i:09d} in {folder}')
        if expected is not None and self.n!=expected:
            raise RuntimeError(f'Unexpected sample count {folder}: expected {expected}, got {self.n}')
    def get_image(self,idx):
        idx=int(idx)
        if not 1<=idx<=self.n:raise IndexError(idx)
        with self.env.begin(write=False) as tr:
            b=tr.get(f'image-{idx:09d}'.encode())
        if not b:raise RuntimeError(f'Missing/corrupt image idx={idx} in {self.folder}')
        with Image.open(io.BytesIO(b)) as pic:
            return pic.convert('RGB')
    def get_label(self,idx):
        idx=int(idx)
        with self.env.begin(write=False) as tr:
            b=tr.get(f'label-{idx:09d}'.encode())
        if b is None:raise RuntimeError(f'Missing label idx={idx} in {self.folder}')
        return b.decode('utf-8','replace')
    def close(self):
        self.env.close()


def sample_indices(n, limit, seed):
    if limit is None or int(limit)==0 or int(limit)>=n:return list(range(1,n+1))
    if int(limit)<0:raise ValueError('limit must be nonnegative')
    return sorted(random.Random(int(seed)).sample(range(1,n+1),int(limit)))


def bucket(text):
    length=len(text)
    if length<=25:return '01-25'
    if length<=35:return '26-35'
    if length<=55:return '36-55'
    return '56+'


def resume_journal(path, valid_indices):
    path=Path(path);result={};bad_tail=False
    if path.exists():
        with path.open('r',encoding='utf-8') as f:
            for no,line in enumerate(f,1):
                if not line.strip():continue
                try:r=json.loads(line)
                except json.JSONDecodeError:
                    bad_tail=True
                    print(f'WARNING: ignored unfinished checkpoint line {no} in {path.name}')
                    break
                idx=int(r['sample_index'])
                if idx not in valid_indices:raise RuntimeError(f'Resume ID outside selected cohort: {idx}')
                result[idx]=r
    if bad_tail:
        temp=path.with_suffix('.repaired')
        with temp.open('w',encoding='utf-8') as f:
            for idx in sorted(result):f.write(json.dumps(result[idx],ensure_ascii=False,default=str)+'\n')
        temp.replace(path)
    return result


def run_dataset(spec, *, model, out_root, data_home, model_fingerprint,
                general_mode='off', general_n=2000, seed=20261120, max_per_dataset=0,
                flush_every=25, strict_image_errors=False, infer_fn=None,
                dataset_opener=None):
    from tqdm.auto import tqdm
    if infer_fn is None:
        import scope_guarded_runtime as runtime
        infer_fn=runtime.predict_image
    family,name,folder,expected=spec
    if name=='General' and general_mode=='off':
        raise ValueError('General is disabled')
    ds_key=family+'__'+name.replace(' ','_')
    out=Path(out_root)/ds_key
    out.mkdir(parents=True,exist_ok=True)
    if dataset_opener is not None:
        ds=dataset_opener(spec)
    else:
        folder_path=fetch_lmdb(spec,data_home)
        ds=LmdbSamples(folder_path,expected=expected)
    try:
        if name=='General' and general_mode=='sample':
            selected=sample_indices(ds.n,general_n,seed)
            kind='DETERMINISTIC_SUBSET_NOT_FULL_BENCHMARK'
        else:
            selected=sample_indices(ds.n,max_per_dataset,seed)
            kind='FULL' if len(selected)==ds.n else 'DETERMINISTIC_SUBSET_NOT_FULL_BENCHMARK'
        spec_data={
            'family':family,'dataset':name,'folder':folder,'source_repo':DATASET_REPO,
            'source_revision':DATASET_REV,'available_samples':ds.n,'evaluated_samples':len(selected),
            'sample_kind':kind,'selection_seed':seed,'selection_sha256':checksum(selected),
            'model_fingerprint':model_fingerprint,
            'inference_module':'scope_coverage_runtime.predict_image',
            'control_methods':METHODS,'run_original':True,
            'ctc_processing':'frozen SVTRv2-T, coverage probe and reused primary CTC, JIT exact scoring',
        }
        spec_file=out/'run_spec.json'
        if spec_file.exists():
            old=json.loads(spec_file.read_text())
            if old!=spec_data:raise RuntimeError(f'Unsafe resume: changed dataset config for {ds_key}')
        else:spec_file.write_text(json.dumps(spec_data,indent=2,ensure_ascii=False))
        journal=out/'inference_GT_blind_journal.jsonl'
        done=resume_journal(journal,set(selected))
        pending=[idx for idx in selected if idx not in done]
        print(f'\n{ds_key}: {len(done)} completed, {len(pending)} remaining / {len(selected)} '+
              f'({kind}), available {ds.n}',flush=True)
        buf=[];tick=time.perf_counter()
        try:
            for idx in tqdm(pending,desc=ds_key):
                try:
                    io_start=time.perf_counter()
                    image=ds.get_image(idx)
                    io_ms=(time.perf_counter()-io_start)*1000
                    pred=infer_fn(image,model,run_original=True,check_legacy=False,
                                  skip_structurally_impossible=True)
                    del image
                    r={k:(v.item() if isinstance(v,np.generic) else v) for k,v in pred.items()}
                    r.update(sample_index=int(idx),dataset=name,family=family,
                             status='OK',image_io_ms=round(io_ms,3))
                except (KeyboardInterrupt,SystemExit):
                    raise
                except Exception as exc:
                    r={'sample_index':int(idx),'dataset':name,'family':family,
                       'status':'FAILED','error_type':type(exc).__name__,
                       'error_summary':str(exc)[:500]}
                    if strict_image_errors:raise
                done[idx]=r;buf.append(r)
                if len(buf)>=int(flush_every):
                    with journal.open('a',encoding='utf-8') as f:
                        for z in buf:f.write(json.dumps(z,ensure_ascii=False,default=str)+'\n')
                        f.flush();os.fsync(f.fileno())
                    buf=[]
                    if len(done)%250<flush_every:
                        print(f'{ds_key}: {len(done)}/{len(selected)} completed, elapsed {round(time.perf_counter()-tick,1)}s',flush=True)
        finally:
            if buf:
                with journal.open('a',encoding='utf-8') as f:
                    for z in buf:f.write(json.dumps(z,ensure_ascii=False,default=str)+'\n')
                    f.flush();os.fsync(f.fileno())
        if len(done)!=len(selected):
            raise RuntimeError(f'Incomplete run {ds_key}: {len(done)} != {len(selected)}')
        rows=pd.DataFrame([done[idx] for idx in selected])
        outpath=out/'predictions_GT_blind.csv'
        rows.to_csv(outpath,index=False)
        return ds_key,ds,rows,out
    except Exception:
        ds.close()
        raise


def score_dataset(ds,rows,out,normalize_fn,*,fail_if_incomplete=False):
    data=rows.copy()
    assert data['sample_index'].is_unique, 'Duplicate LMDB IDs'
    labels=[ds.get_label(idx) for idx in data['sample_index'].astype(int)]
    data['gt_original']=labels
    data['gt']=[normalize_fn(s) for s in labels]
    data['gt_length']=data['gt'].str.len()
    data['length_bucket']=data['gt'].map(bucket)
    data['gt_symbols_removed']=data['gt_original'].map(lambda s:sum(not('a'<=ch.lower()<='z' or '0'<=ch<='9') for ch in s))
    data['eligible']=data['status'].eq('OK') & data['gt_length'].gt(0)
    for m in METHODS:
        if m not in data:raise RuntimeError(f'Missing method column {m}')
        data[m]=data[m].fillna('')
    N=len(data);valid=data[data['eligible']].copy()
    if valid.empty:raise RuntimeError('No valid samples for scoring')
    missing=N-len(valid)
    print(f'SCORED {len(valid)}/{N}; excluded {missing} (inference failure/zero normalized GT)',flush=True)
    if fail_if_incomplete and missing:raise RuntimeError('Incomplete benchmark, see scored CSV')
    for m in METHODS:
        valid[m+'_ok']=valid[m].eq(valid['gt'])
        ops=[Levenshtein.editops(a,b) for a,b in zip(valid[m],valid['gt'])]
        # Order is prediction -> GT; insert is missing characters, delete is extra output.
        valid[m+'_S']=[sum(x.tag=='replace' for x in op) for op in ops]
        valid[m+'_D']=[sum(x.tag=='insert' for x in op) for op in ops]
        valid[m+'_I']=[sum(x.tag=='delete' for x in op) for op in ops]
        valid[m+'_edits']=(valid[m+'_S']+valid[m+'_D']+valid[m+'_I'])
    valid['SCOPE_vs_original']=valid['safe_scope_ok'].astype(int)-valid['original_svtrv2_full12_ok'].astype(int)
    for col in valid.columns:
        if col.endswith(('_ok','_S','_D','_I','_edits')) or col=='SCOPE_vs_original':
            data[col]=data['sample_index'].map(valid.set_index('sample_index')[col])
    data.to_csv(Path(out)/'predictions_scored.csv',index=False)
    return data,valid


def summarise_scored(scored_datasets, group_name='External'):
    """Collect per-dataset and length rows; per-dataset exact and CER use one cohort."""
    quality=[];paired=[];latency=[];by_length=[]
    for spec,df,all_count in scored_datasets:
        family,name,*_=spec
        if df.empty:continue
        d=df.copy()
        subsets=[(name,d)] + [(f'LEN:{b}',sub) for b,sub in d.groupby('length_bucket',sort=True)]
        for subsection,part in subsets:
            if subsection.startswith('LEN:'):
                suffix={'length_bucket':subsection[4:]}
            else:suffix={'length_bucket':'ALL'}
            for m in METHODS:
                total_gt=int(part['gt_length'].sum())
                row={'family':family,'dataset':name,'length_bucket':suffix['length_bucket'],
                     'method':m,'n_evaluated':len(part),'n_expected':all_count,
                     'correct':int(part[m+'_ok'].sum()),
                     'accuracy':float(part[m+'_ok'].mean()),
                     'CER':float(part[m+'_edits'].sum()/total_gt) if total_gt else float('nan'),
                     'S':int(part[m+'_S'].sum()),'D':int(part[m+'_D'].sum()),'I':int(part[m+'_I'].sum()),
                     'gt_chars':total_gt}
                (by_length if subsection.startswith('LEN:') else quality).append(row)
        for bname,subset in [('ALL',d)]+[(b,t) for b,t in d.groupby('length_bucket',sort=True)]:
            for baseline in ('original_svtrv2_full12',):
                w=int((subset['safe_scope_ok']&~subset[baseline+'_ok']).sum())
                l=int((~subset['safe_scope_ok']&subset[baseline+'_ok']).sum())
                paired.append({'family':family,'dataset':name,'length_bucket':bname,
                               'baseline':baseline,'method':'safe_scope',
                               'n_evaluated':len(subset),'rescued':w,'regressed':l,'net':w-l,
                               'mcnemar_exact_p':float(binomtest(w,w+l,.5).pvalue) if w+l else 1.})
        for stage in ('safe_scope_total_ms','scope_total_ms','coverage_probe_ms','matched_c1_ms','main_forward_ms','paired_scale_forward_ms',
                      'scope_decoder_ms','image_io_ms'):
            if stage not in d:continue
            values=pd.to_numeric(d[stage],errors='coerce').dropna()
            if not len(values):continue
            latency.append({'family':family,'dataset':name,'stage':stage,'n':len(values),
                            'mean_ms':float(values.mean()),'median_ms':float(values.median()),
                            'p95_ms':float(values.quantile(.95)),'total_s':float(values.sum()/1000)})
        # Reused raw original when one native crop, matching original FINAL latency accounting.
        if 'original_extra_forward_ms' in d and 'original_reused_from_native' in d:
            raw_ms=np.where(pd.to_numeric(d['original_reused_from_native'],errors='coerce').fillna(0).eq(1),
                            pd.to_numeric(d['main_forward_ms'],errors='coerce'),
                            pd.to_numeric(d['original_extra_forward_ms'],errors='coerce'))
            r=pd.Series(raw_ms).dropna()
            if len(r):
                latency.append({'family':family,'dataset':name,'stage':'raw_svtr_ms','n':len(r),
                                'mean_ms':float(r.mean()),'median_ms':float(r.median()),
                                'p95_ms':float(r.quantile(.95)),'total_s':float(r.sum()/1000)})
    quality=pd.DataFrame(quality);paired=pd.DataFrame(paired)
    latency=pd.DataFrame(latency);by_length=pd.DataFrame(by_length)
    return quality,paired,latency,by_length


def aggregate_families(scored_datasets):
    summary=[]
    families=sorted(set(spec[0] for spec,_,_ in scored_datasets))
    if any(spec[0]=='Union14M' for spec,_,_ in scored_datasets):
        families.append('Union14M_6_NON_GENERAL')
    for family in families+['ALL_EXTERNAL']:
        selected=[(spec,d) for spec,d,_ in scored_datasets if
                  spec[0]==family or
                  (family=='Union14M_6_NON_GENERAL' and spec[0]=='Union14M' and spec[1]!='General') or
                  family=='ALL_EXTERNAL']
        if not selected:continue
        ds_rows=[]
        for m in METHODS:
            n=sum(len(d) for _,d in selected)
            hits=sum(int(d[m+'_ok'].sum()) for _,d in selected)
            edits=sum(int(d[m+'_edits'].sum()) for _,d in selected)
            chars=sum(int(d['gt_length'].sum()) for _,d in selected)
            means=[float(d[m+'_ok'].mean()) for _,d in selected]
            summary.append({'family':family,'method':m,'datasets_included':len(selected),
                            'n_evaluated':n,'correct':hits,'micro_accuracy':hits/n if n else float('nan'),
                            'dataset_macro_accuracy':float(np.mean(means)),
                            'micro_CER':edits/chars if chars else float('nan'),
                            'aggregation_note':('FULL six non-General Union14M subsets' if family=='Union14M_6_NON_GENERAL' else
                                'Exploratory: if General is sampled this family includes a PARTIAL dataset; '
                                'macro weights named datasets equally, not a claimed official ranking')})
    return pd.DataFrame(summary)


def pack_results(zip_path,root,ltb_out,meta):
    zip_path=Path(zip_path);root=Path(root);ltb_out=Path(ltb_out)
    zip_path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        z.writestr('MULTIDATASET_PROTOCOL.json',json.dumps(meta,indent=2,ensure_ascii=False))
        for p in sorted(root.rglob('*')):
            if p.is_file() and ('/.' not in str(p)):
                z.write(p,arcname='cross_dataset/'+str(p.relative_to(root)))
        for module_name in ['longtext_final.py','longtext_v3.py','density_probe.py','orientation_probe.py',
                            'unified_router.py','alex_ctc.py','alex_c1.py','alex_e5.py','space_ctc.py',
                            'alex_e7.py','scope_e7_fast.py','scope_final_runtime_v3.py',
                            'scope_coverage_runtime.py','scope_ctc_jit.py','scope_beam_jit_v4.py','scope_safe_suite.py']:
            source_file=Path(__file__).resolve().parent/module_name
            if source_file.exists():z.write(source_file,arcname='source_modules/'+module_name)
        for name in ['quality_by_bucket.csv','paired_vs_original.csv','latency_by_bucket_stage.csv',
                     'predictions_GT_blind.csv','predictions_scored_full.csv','ltb_coverage_decisions.csv',
                     'e7_decoder_parity.csv','protocol.json','provenance.json']:
            f=ltb_out/name
            if f.is_file():z.write(f,arcname='LTB_extended/'+name)
    return zip_path


def local_logic_smoke_test():
    """No network/GPU needed; validate scoring, groups, paired, and sparse output."""
    class Fake:
        def get_label(self,idx):
            return {1:'Ab 12',2:'ac',3:'zb'}[int(idx)]
    rows=[]
    for idx in (1,2,3):
        p='ab12' if idx==1 else 'ac' if idx==2 else 'zz'
        q='ab12' if idx==1 else 'ac' if idx==2 else 'zb'
        r={m:p for m in METHODS}
        r['safe_scope']=q
        r.update(sample_index=idx,status='OK',scope_total_ms=20.,matched_c1_ms=5.,
                 main_forward_ms=3.,original_extra_forward_ms=0.,original_reused_from_native=1,
                 image_io_ms=1.)
        rows.append(r)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        norm=lambda s:''.join(ch for ch in s.lower() if ('a'<=ch<='z' or '0'<=ch<='9'))
        scored,valid=score_dataset(Fake(),pd.DataFrame(rows),d,norm)
        assert len(valid)==3
        q,p,l,b=summarise_scored([(('Common','TEST','/fake',3),valid,3)])
        assert q.query("method == 'safe_scope'")['correct'].iloc[0]==3
        assert p.query("baseline == 'original_svtrv2_full12' and length_bucket == 'ALL'")['rescued'].iloc[0]==1
        assert l.query("stage == 'raw_svtr_ms'")['mean_ms'].iloc[0]==3.0
        assert b.query("length_bucket == '01-25'")['n_evaluated'].max()==3
        assert (scored['gt'].tolist()==['ab12','ac','zb'])
        assert sample_indices(50,8,17)==sample_indices(50,8,17)
    return True
