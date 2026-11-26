"""Experimental batched causal scoring; frozen native decoder weights and EOS."""
import hashlib, inspect, math
from collections import defaultdict
import torch
EXPECTED_SOURCE = '06cbc5f16429b6575f92d32d29be4bc699a52b8bee17888a9c309c3db347d099'

def fast_score(scorer, strings, audit=False, batch_size=8):
    if not scorer.checked: scorer.validate()
    d=scorer.decoder
    if hashlib.sha256(inspect.getsource(type(d)).rstrip().encode()).hexdigest()!=EXPECTED_SOURCE:
        raise RuntimeError('Archived decoder source mismatch; refuse fast path')
    if d.training or scorer.head!='tgt_word_prj' or scorer.eos!=d.eos:
        raise RuntimeError('Decoder trace contract mismatch')
    bound=inspect.signature(d.forward).bind(*scorer.args,**scorer.kwargs)
    mem,mask=bound.arguments['src']
    if mem.ndim!=3 or mem.shape[0]!=1 or (mask is not None and (mask.ndim!=2 or mask.shape[0]!=1)):
        raise RuntimeError('Unexpected encoder memory/mask shape')
    groups=defaultdict(list); result={}
    for s in dict.fromkeys(strings):
        if len(s)>scorer.supported_len: raise ValueError('Overlength candidate')
        if any(c not in scorer.lookup or scorer.lookup[c]>=scorer.output_classes for c in s): result[s]=None; continue
        groups[len(s)].append((s,[scorer.lookup[c] for c in s]))
    with torch.inference_mode():
        for length,items in groups.items():
            for offset in range(0,len(items),batch_size):
                batch=items[offset:offset+batch_size]; b=len(batch)
                inp=torch.tensor([[d.bos]+ids for _,ids in batch],dtype=torch.long,device=mem.device)
                targets=torch.tensor([ids+[d.eos] for _,ids in batch],dtype=torch.long,device=mem.device)
                tgt=d.positional_encoding(d.embedding(inp))
                causal=d.generate_square_subsequent_mask(length+1,mem.get_device() if mem.is_cuda else mem.device)
                out=d.decoder(tgt,mem.expand(b,-1,-1).contiguous(),tgt_mask=causal,memory_key_padding_mask=None if mask is None else mask.expand(b,-1).contiguous())
                lp=torch.log_softmax(d.tgt_word_prj(out).float(),dim=-1)
                vals=lp.gather(-1,targets.unsqueeze(-1)).squeeze(-1).cpu().tolist()
                for (s,_),steps in zip(batch,vals):
                    value=sum(steps)
                    if not math.isfinite(value) or value>1e-5: raise RuntimeError('Invalid fast AR likelihood')
                    result[s]=value
    if audit:
        ref=scorer.score(strings)
        for s in ref:
            if ref[s] is None or result[s] is None:
                if ref[s] != result[s]: raise RuntimeError('AR support mismatch')
            elif abs(ref[s]-result[s])>0.002:
                raise RuntimeError(f'AR warmup score mismatch: {s!r} {ref[s]} vs {result[s]}')
    return result
