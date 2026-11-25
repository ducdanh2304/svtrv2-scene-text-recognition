"""Frozen AR conditional probability via replay of native greedy decoding.
Fail closed if the archived NaARDecoder violates the explicit trace contract.
No training-mode switch and no replacement of the recognizer's encoder.
"""
from __future__ import annotations
import inspect, math
import torch


def raw_ctc_logprob(z, text, chars):
    import numpy as np
    z=np.asarray(z,dtype=np.float64)
    if z.ndim!=2 or z.shape[1]!=len(chars) or not len(z) or not np.isfinite(z).all():
        raise ValueError('Invalid CTC logits')
    lookup={c:i for i,c in enumerate(chars) if i and len(c)==1}
    if any(c not in lookup for c in text): return -math.inf
    ids=[lookup[c] for c in text]
    if len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))>len(z): return -math.inf
    logp=z-np.logaddexp.reduce(z,axis=1,keepdims=True)
    if not ids: return float(logp[:,0].sum())
    states=np.zeros(2*len(ids)+1,dtype=int); states[1::2]=ids
    jumpable=(states[2:]!=0)&(states[2:]!=states[:-2])
    v=np.full(len(states),-np.inf);v[:2]=logp[0,states[:2]]
    for row in logp[1:]:
        step=np.r_[-np.inf,v[:-1]];jump=np.r_[-np.inf,-np.inf,np.where(jumpable,v[:-2],-np.inf)]
        v=np.logaddexp(np.logaddexp(v,step),jump)+row[states]
    return float(np.logaddexp(v[-1],v[-2]))


def mixture(candidates, zs, chars, ar=None, normalizer=None):
    """Exact raw-string events. Sum distinct nominated variants after scoring.
    The sum is a restricted candidate-set mass, not the entire normalized event.
    Missing AR for a crop omits it for *all* candidates, not candidate by candidate.
    """
    import numpy as np
    normalizer=normalizer or (lambda x:x)
    strings=list(dict.fromkeys(candidates)); groups={}; ctc={}
    for text in strings:
        probs=[raw_ctc_logprob(z,text,chars) for z in zs]
        ctc[text]=float(np.logaddexp.reduce(probs)-math.log(len(probs)))
        if ar is not None:
            probs.append(float(ar[text]) if ar[text] is not None else -math.inf)
        score=float(np.logaddexp.reduce(probs)-math.log(len(probs)))
        key=normalizer(text)
        groups[key]=float(np.logaddexp(groups.get(key,-math.inf),score))
    if not groups or not any(math.isfinite(v) for v in groups.values()):
        raise RuntimeError('No supported candidate')
    return max(groups,key=groups.get),groups,ctc


class ReplayAR:
    """Capture native decoder inputs, identify final per-step projection by
    observed greedy parity, then force prefixes at the output projection.
    Only untouched log-softmax probabilities enter the sequence score.
    """
    def __init__(self, recognizer, post, max_len):
        self.rec=recognizer; self.decoder=recognizer.model.decoder;self.post=post
        self.chars=list(post.character);self.max_len=int(max_len)
        eos=[i for i,c in enumerate(self.chars) if str(c).lower() in ('<eos>','eos','</s>','[eos]')]
        if len(eos)!=1: raise RuntimeError(f'Cannot identify EOS: {self.chars[:4]} ... {self.chars[-4:]}')
        self.eos=eos[0];self.lookup={c:i for i,c in enumerate(self.chars) if len(c)==1}
        if len(self.lookup)!=sum(len(c)==1 for c in self.chars):raise RuntimeError('Duplicate AR tokens')
        self.head=None;self.checked=False
        self.source=inspect.getsource(type(self.decoder))
        if self.decoder.training:raise RuntimeError('AR decoder not in eval mode')

    @staticmethod
    def last(z):
        if not torch.is_tensor(z) or z.ndim not in (2,3) or z.shape[0]!=1:
            raise RuntimeError('Unsupported AR projection shape')
        return z[0] if z.ndim==2 else z[0,-1]

    def text(self, ids):
        out=[]
        for i in ids:
            if i==self.eos:break
            if i>=len(self.chars) or len(self.chars[i])!=1: raise RuntimeError('Unexpected predicted special token')
            out.append(self.chars[i])
        return ''.join(out)

    def post_text(self, output):
        result=self.post(output)
        if not isinstance(result,(list,tuple)) or len(result)!=1 or not isinstance(result[0],(list,tuple)):
            raise RuntimeError('Unsupported AR postprocessor result')
        return str(result[0][0])

    def observe(self, path):
        handles=[]; calls={}; captured=[]
        def pre(m,args,kwargs):captured.append((args,kwargs))
        handles.append(self.decoder.register_forward_pre_hook(pre,with_kwargs=True))
        def record(name):
            def hook(m,args,z): calls.setdefault(name,[]).append(self.last(z).detach().float().cpu())
            return hook
        modules=dict(self.decoder.named_modules())
        for name,m in modules.items():
            if isinstance(m,torch.nn.Linear) and len(self.chars)-3<=m.out_features<=len(self.chars):
                handles.append(m.register_forward_hook(record(name)))
        try:
            with torch.inference_mode(): native=self.rec(img_path=str(path),batch_num=1)[0]
        finally:
            for h in handles:h.remove()
        if len(captured)!=1:raise RuntimeError('Expected exactly one decoder call per image')
        valid=[]
        for name,values in calls.items():
            ids=[int(z.argmax()) for z in values]
            try:
                if self.text(ids)==str(native['text']) and len(ids)>=len(str(native['text'])) and (self.eos in ids or len(ids)>=self.max_len):valid.append(name)
            except RuntimeError: pass
        if self.head is None:
            if len(valid)!=1:raise RuntimeError(f'Cannot identify unique autoregressive projection: {valid}; calls={[(k,len(v)) for k,v in calls.items()]}')
            self.head=valid[0]
        if self.head not in valid:raise RuntimeError('Native AR greedy trace mismatch')
        self.args,self.kwargs=captured[0];self.native_logits=calls[self.head]
        self.native_ids=[int(z.argmax()) for z in self.native_logits]
        self.output_classes=self.native_logits[0].numel()
        if self.eos>=self.output_classes:raise RuntimeError('EOS outside AR output support')
        return native

    def replay(self, ids=None, *, allow_truncated=False):
        observed=[]; chosen=[]
        def hook(m,args,z):
            row=self.last(z); lp=torch.log_softmax(row.float(),dim=-1).detach().cpu();i=len(observed)
            token=int(row.argmax()) if ids is None else ids[i] if i<len(ids) else self.eos
            if not 0<=token<len(lp):raise RuntimeError('Target AR token outside classifier')
            observed.append(lp);chosen.append(token)
            if ids is None:return None
            modified=z.clone(); dest=self.last(modified);dest.fill_(-1e4);dest[token]=1e4
            return modified
        handle=dict(self.decoder.named_modules())[self.head].register_forward_hook(hook)
        try:
            with torch.inference_mode():out=self.decoder(*self.args,**self.kwargs)
        finally:handle.remove()
        if not observed:raise RuntimeError('No AR step observations')
        # Native postprocessor must see the same sequence as the traced projection.
        if self.post_text(out)!=self.text(chosen):raise RuntimeError('Decoder output does not match autoregressive trace')
        if ids is not None and len(observed)<len(ids) and not allow_truncated:raise RuntimeError('Decoder stopped before target EOS')
        return observed,chosen

    def validate(self):
        obs,ids=self.replay()
        if ids!=self.native_ids or len(obs)!=len(self.native_logits):raise RuntimeError('Native decoder replay parity failed')
        for lp,z in zip(obs,self.native_logits):
            if not torch.allclose(lp,torch.log_softmax(z,dim=-1),atol=2e-4,rtol=2e-4):raise RuntimeError('Native decoder replay logits differ')
        # A forced native prefix must reproduce the unforced probabilities through EOS.
        if self.eos in ids:
            target=ids[:ids.index(self.eos)+1];forced,_=self.replay(target)
            for a,b in zip(obs[:len(target)],forced[:len(target)]):
                if not torch.allclose(a,b,atol=2e-4,rtol=2e-4):raise RuntimeError('Forced native prefix probability mismatch')
        # Verify conditional prefix semantics with two deliberately different continuations.
        alphabet=[i for i in self.lookup.values() if i<self.output_classes]
        if len(alphabet)<2:raise RuntimeError('AR vocabulary too small')
        a,b=alphabet[:2]
        p,_=self.replay([a,a,self.eos]);q,_=self.replay([a,b,self.eos])
        if not all(torch.allclose(p[i],q[i],atol=2e-4,rtol=2e-4) for i in range(2)):
            raise RuntimeError('AR prefix causality check failed')
        r,_=self.replay([b,a,self.eos])
        if torch.allclose(p[1],r[1],atol=1e-5,rtol=1e-5):raise RuntimeError('AR step does not depend on forced previous token')
        cap,_=self.replay([a]*(self.max_len+2),allow_truncated=True)
        self.supported_len=min(self.max_len,len(cap)-1)
        if self.supported_len<2:raise RuntimeError('AR EOS capacity too small')
        self.checked=True

    def score(self, strings):
        if not self.checked:self.validate()
        results={}
        for s in dict.fromkeys(strings):
            if len(s)>self.supported_len:raise ValueError('Overlength candidate must be excluded at crop level')
            if any(c not in self.lookup or self.lookup[c]>=self.output_classes for c in s):results[s]=None;continue
            ids=[self.lookup[c] for c in s]+[self.eos]
            obs,chosen=self.replay(ids)
            if chosen[:len(ids)]!=ids:raise RuntimeError('Forced AR token mismatch')
            value=sum(float(lp[i]) for lp,i in zip(obs,ids))
            if not math.isfinite(value) or value>1e-5:raise RuntimeError('Invalid AR sequence likelihood')
            results[s]=value
        return results
