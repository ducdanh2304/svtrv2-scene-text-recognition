"""Per-image exact CTC cache and shape-preserving vision microbatches.
No changes to model parameters, native transform list, or candidate selection.
"""
from collections import OrderedDict,defaultdict
from contextlib import contextmanager,ExitStack
from time import perf_counter
import hashlib
import numpy as np
import torch
from ..types import Prediction

class ExactTensorCache:
    def __init__(self,max_entries=32,max_bytes=32*1024*1024):
        self.items=OrderedDict();self.bytes=0;self.max_entries=max_entries;self.max_bytes=max_bytes
        self.hits=0;self.misses=0
    def call(self,fn,x):
        # Exact input including batch composition, dtype, shape, and device.
        if not isinstance(x,torch.Tensor) or x.dtype!=torch.float32 or not x.is_contiguous():
            return fn(x)
        raw=x.detach().cpu().numpy().tobytes()
        key=(tuple(x.shape),str(x.dtype),str(x.device),hashlib.sha256(raw).digest())
        if key in self.items:
            saved,y,size=self.items[key]
            if saved!=raw:raise RuntimeError('CTC cache digest collision')
            self.items.move_to_end(key);self.hits+=1;return y.clone()
        self.misses+=1;y=fn(x)
        if not torch.is_tensor(y):raise RuntimeError('Unexpected raw CTC output')
        size=len(raw)+y.numel()*y.element_size()
        if size<=self.max_bytes:
            while self.items and (len(self.items)>=self.max_entries or self.bytes+size>self.max_bytes):
                self.bytes-=self.items.popitem(last=False)[1][2]
            self.items[key]=(raw,y.detach().clone(),size);self.bytes+=size
        return y

@contextmanager
def cache_context(model,stats):
    # Only wrapper CTCNet objects, never shared lower-level encoder/decoder hooks.
    nets={id(n):n for n in (model.scope.recognizer.base.net,model.s2net)}
    saved=[]
    try:
        for net in nets.values():
            if net.training:raise RuntimeError('Cache requires eval mode')
            cache=ExactTensorCache();old=net.forward;had='forward' in net.__dict__
            saved.append((net,old,had,cache))
            def wrapped(x,_old=old,_cache=cache):return _cache.call(_old,x)
            net.forward=wrapped
        yield
    finally:
        for net,old,had,cache in reversed(saved):
            if had:net.forward=old
            else:del net.forward
            stats['ctc_net_hits']+=cache.hits;stats['ctc_net_misses']+=cache.misses

def groups(tensors,max_batch=4):
    by=defaultdict(list)
    for i,t in enumerate(tensors):
        if t.ndim!=4 or t.shape[0]!=1:raise RuntimeError('Expected single NCHW tensor')
        by[(tuple(t.shape),str(t.device),str(t.dtype))].append(i)
    for ids in by.values():
        # Limit spatial activation footprint. No padding or shape changes.
        t=tensors[ids[0]];size=int(t.shape[-1]*t.shape[-2]);bs=max(1,min(max_batch,49152//size))
        for i in range(0,len(ids),bs):yield ids[i:i+bs]

def batch_s2(model,images,stats):
    if not images:return []
    ts=[model.motion.pipe.preprocess(im) for im in images];result=[None]*len(ts)
    for ids in groups(ts):
        z=model.s2net(torch.cat([ts[i] for i in ids],dim=0)).detach().float().cpu().numpy()
        if z.ndim!=3 or len(z)!=len(ids) or z.shape[-1]!=len(model.chars):raise RuntimeError('S2 batch output shape')
        stats['s2_batches']+=1;stats['s2_items']+=len(ids)
        for i,row in zip(ids,z):result[i]=row
    return result

def slice_output(output,i,b):
    if torch.is_tensor(output):
        if output.ndim==0 or output.shape[0]!=b:raise RuntimeError('Unsupported R3 output batch dimension')
        return output[i:i+1]
    if isinstance(output,list):return [slice_output(v,i,b) for v in output]
    if isinstance(output,tuple):return tuple(slice_output(v,i,b) for v in output)
    if isinstance(output,dict):return {k:slice_output(v,i,b) for k,v in output.items()}
    raise RuntimeError('Unsupported R3 output type')

def native_views(r3,image):
    # Reuse exact native transform enumeration; no inference inside this call.
    old=r3._predict_one;had='_predict_one' in r3.__dict__;views=[]
    def collect(im,ratio_override=None):
        views.append((im,ratio_override));return '',0.0
    r3._predict_one=collect
    try:r3.predict(image)
    finally:
        if had:r3._predict_one=old
        else:del r3._predict_one
    if not views:raise RuntimeError('Missing R3 views')
    return views

def preprocess_r3(r3,img,ratio_override):
    width,height=img.size
    ratio=int(np.clip(np.around(width/height),1,r3.max_ratio)) if ratio_override is None else ratio_override
    if ratio<=4:iw,ih=r3.base_shape[ratio-1]
    else:ih=r3.base_h;iw=ih*ratio
    resized=r3.TF.resize(img.convert('RGB'),(ih,iw),interpolation=r3.T.InterpolationMode.BICUBIC)
    return r3.TF.normalize(r3.TF.to_tensor(resized),[0.5]*3,[0.5]*3).unsqueeze(0).to(r3.device)

def batch_r3(r3,images,chars,stats):
    if not images:return []
    import longtext_final as LT
    t0=perf_counter();meta=[];ts=[];views=[]
    for im in images:
        cur=native_views(r3,im);meta.append((len(views),len(cur)));views.extend(cur)
        ts.extend(preprocess_r3(r3,img,ratio) for img,ratio in cur)
    result=[None]*len(ts);seen=[];d=r3.model.decoder;head=d if hasattr(d,'w_atten_block') else d.ctc_decoder
    hook=head.fc.register_forward_hook(lambda m,a,o:seen.append(o.detach().float().cpu().numpy()))
    try:
        for ids in groups(ts):
            seen.clear();out=r3.model(torch.cat([ts[i] for i in ids],dim=0))
            if len(seen)!=1 or seen[0].ndim!=3 or seen[0].shape[0]!=len(ids) or seen[0].shape[-1]!=len(chars):raise RuntimeError('R3 batch trace shape')
            stats['r3_batches']+=1;stats['r3_items']+=len(ids)
            for j,i in enumerate(ids):
                decoded=r3.post(slice_output(out,j,len(ids)))
                if len(decoded)==2 and isinstance(decoded[0],list):text,score=decoded[1][0]
                else:text,score=decoded[0]
                z=seen[0][j];text=str(text);score=float(score)
                if not np.isfinite(z).all() or LT.normalize_ltb(LT.greedy_ctc(z,chars))!=LT.normalize_ltb(text):raise RuntimeError('R3 batch native decoding mismatch')
                result[i]=(views[i][0],z,text,score)
    finally:hook.remove()
    ret=[]
    for start,n in meta:
        calls=result[start:start+n];best=max(calls,key=lambda c:c[3]);spatial=max((c for c in calls if c[0] is calls[0][0]),key=lambda c:c[3])
        pred=Prediction(text=best[2],confidence=best[3],method='r3_lowlr_tta' if r3.tta else 'r3_lowlr',route='artistic_component',latency_ms=(perf_counter()-t0)*1000)
        ret.append((pred,best[1],spatial[1],[c[2] for c in calls]))
    return ret

def same_view(cap,im,crop,sp):
    return not cap.get('angle') and tuple(sp)==(0,im.width) and crop.size==im.size and np.array_equal(np.asarray(crop),np.asarray(im))

def stats_dict():return dict(ctc_net_hits=0,ctc_net_misses=0,s2_batches=0,s2_items=0,r3_batches=0,r3_items=0)
