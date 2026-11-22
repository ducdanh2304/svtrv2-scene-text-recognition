"""Optional exact-scoring CTC forward JIT, preserving C1's recurrence.

No GT, learned weights, threshold changes, or modification of the allowed CTC paths.
Numba disabled/fallback if absent or its results fail a parity self-test.
"""
import numpy as np
import alex_c1 as C1
import alex_ctc as AX

_ORIGINAL_FORWARD = C1.ctc_forward_logprob
_NUMBA_FORWARD = None
_JIT_READY = False
_JIT_STATUS = 'not_installed'


def _make_kernel():
    from numba import njit

    @njit(cache=True)
    def forward_kernel(logq, labels):
        T, C = logq.shape
        n = len(labels)
        if n == 0:
            result = 0.0
            for t in range(T):
                result += logq[t, 0]
            return result
        repeats=0
        for i in range(1,n):
            if labels[i]==labels[i-1]:
                repeats += 1
        if n+repeats>T:
            return -np.inf
        S=2*n+1
        states=np.zeros(S,np.int32)
        for i in range(n):
            states[2*i+1]=labels[i]
        previous=np.empty(S,np.float64)
        for s in range(S):
            previous[s]=-np.inf
        previous[0]=logq[0,0]
        previous[1]=logq[0,labels[0]]
        current=np.empty(S,np.float64)
        for t in range(1,T):
            for s in range(S):
                stay=previous[s]
                step=previous[s-1] if s>0 else -np.inf
                jump=previous[s-2] if (s>1 and states[s]!=0 and states[s]!=states[s-2]) else -np.inf
                current[s]=np.logaddexp(np.logaddexp(stay,step),jump)+logq[t,states[s]]
            sw=previous;previous=current;current=sw
        return np.logaddexp(previous[S-1],previous[S-2])
    return forward_kernel


def _as_labels(text):
    return np.fromiter((1+AX.ALPH.index(x) for x in text),dtype=np.int32,count=len(text))


def _fast(logq,text):
    arr=np.asarray(logq,dtype=np.float64)
    if arr.ndim!=2 or arr.shape[1]!=37 or arr.shape[0]<1:
        raise ValueError('Expected [T,37] nonempty normalized log posterior')
    return float(_NUMBA_FORWARD(arr,_as_labels(text)))


def activate(strict=True):
    """Perform deterministic edge/random parity tests before patching C1 globally."""
    global _NUMBA_FORWARD,_JIT_READY,_JIT_STATUS
    if _JIT_READY:
        return {'available':True,'status':_JIT_STATUS}
    try:
        _NUMBA_FORWARD = _make_kernel()
        g=np.random.default_rng(20261121)
        seqs=['','a','ab','aaa','abba','abcdef','000000','abc1a','z9z9z9z9']
        for T in [1,2,3,4,7,15,32,64,120]:
            for trial in range(2):
                data=g.uniform(.001,1,size=(T,37))
                data=data/data.sum(axis=1,keepdims=True)
                logq=np.log(data)
                for txt in seqs:
                    old=_ORIGINAL_FORWARD(logq,txt)
                    new=_fast(logq,txt)
                    if not (np.isneginf(old) and np.isneginf(new)):
                        if not np.isclose(old,new,rtol=1e-11,atol=1e-10):
                            raise AssertionError((T,txt,old,new))
        # tie case and stable high blank probabilities
        z=np.full((25,37),-12.,np.float64)
        z[:,0]=0.
        for txt in ('','a','aa','aba'):
            old=_ORIGINAL_FORWARD(z,txt);new=_fast(z,txt)
            if not np.isclose(old,new,rtol=1e-11,atol=1e-10):
                raise AssertionError((txt,old,new))
        C1.ctc_forward_logprob=_fast
        _JIT_READY=True
        _JIT_STATUS='jit_exact_recurrence_parity_pass'
        return {'available':True,'status':_JIT_STATUS}
    except Exception as exc:
        C1.ctc_forward_logprob=_ORIGINAL_FORWARD
        _JIT_READY=False
        _JIT_STATUS=f'fallback_original_{type(exc).__name__}:{exc}'
        if strict:
            raise
        return {'available':False,'status':_JIT_STATUS}


def cpu_benchmark(n=80,T=64):
    import time
    g=np.random.default_rng(734)
    q=g.uniform(.001,1,size=(T,37));q=np.log(q/q.sum(axis=1,keepdims=True))
    s='correctreconstruction'
    t0=time.perf_counter()
    for _ in range(n):_ORIGINAL_FORWARD(q,s)
    t1=time.perf_counter()
    for _ in range(n):C1.ctc_forward_logprob(q,s)
    t2=time.perf_counter()
    return {'original_ms':1000*(t1-t0)/n,'patched_ms':1000*(t2-t1)/n,
            'ratio':max(1e-9,(t1-t0))/max(1e-9,(t2-t1))}

_ORIGINAL_FORCED = None
_FORCED_KERNEL = None


def _make_forced_kernel():
    from numba import njit

    @njit(cache=True)
    def kernel(q, labels):
        T, C=q.shape
        n=len(labels)
        if n==0:
            return np.empty(0,np.int32), 0.0, True
        repeats=0
        for k in range(1,n):
            if labels[k]==labels[k-1]:repeats+=1
        if n+repeats>T:
            return np.empty(0,np.int32),-np.inf,False
        S=2*n+1
        states=np.zeros(S,np.int32)
        for k in range(n):states[2*k+1]=labels[k]
        lq=np.log(np.maximum(q,1e-30))
        prev=np.empty(S,np.float64)
        curr=np.empty(S,np.float64)
        for s in range(S):prev[s]=-np.inf
        prev[0]=lq[0,0];prev[1]=lq[0,labels[0]]
        trace=np.empty((T,S),np.int16)
        trace[:,:]=-1
        for t in range(1,T):
            for s in range(S):
                stay=prev[s]
                step=prev[s-1] if s>=1 else -np.inf
                jump=prev[s-2] if (s>=2 and states[s]!=0 and states[s]!=states[s-2]) else -np.inf
                if stay>=step and stay>=jump:
                    curr[s]=stay+lq[t,states[s]];trace[t,s]=s
                elif step>=jump:
                    curr[s]=step+lq[t,states[s]];trace[t,s]=s-1
                else:
                    curr[s]=jump+lq[t,states[s]];trace[t,s]=s-2
            tmp=prev;prev=curr;curr=tmp
        terminal=S-1 if prev[S-1]>=prev[S-2] else S-2
        score=prev[terminal]
        if not np.isfinite(score):return np.empty(0,np.int32),-np.inf,False
        state_path=np.empty(T,np.int32)
        state=terminal
        for t in range(T-1,-1,-1):
            state_path[t]=state
            if t:state=int(trace[t,state])
        frames=np.empty(n,np.int32)
        for k in range(n):
            best=-1
            bestval=-1.0
            target=2*k+1
            for t in range(T):
                if state_path[t]==target:
                    v=q[t,labels[k]]
                    if best<0 or v>bestval:
                        best=t;bestval=v
            if best<0:return np.empty(0,np.int32),-np.inf,False
            frames[k]=best
        return frames,float(score),True
    return kernel


def activate_forced(strict=True):
    """Enable JIT forced alignment only after matching E5 on deterministic cases."""
    global _ORIGINAL_FORCED,_FORCED_KERNEL
    import alex_e5 as E5
    if _ORIGINAL_FORCED is not None and E5.fast_forced_frames is not _ORIGINAL_FORCED:
        return {'available':True,'status':'already_active'}
    _ORIGINAL_FORCED=E5.fast_forced_frames
    try:
        _FORCED_KERNEL=_make_forced_kernel()
        def fast_forced_frames(q, text):
            if not text:return [],0.0
            labels=_as_labels(text)
            frames,score,ok=_FORCED_KERNEL(np.asarray(q,dtype=np.float64),labels)
            if not ok:return None,-np.inf
            return [int(x) for x in frames],float(score)
        rg=np.random.default_rng(20261121)
        seqs=['','a','ab','aaa','abba','abcdef','0000','z9z9z9z9']
        for T in [2,3,4,8,16,32,70]:
            for seed in range(2):
                q=rg.uniform(.001,1,size=(T,37));q/=q.sum(axis=1,keepdims=True)
                for txt in seqs:
                    old=_ORIGINAL_FORCED(q,txt)
                    new=fast_forced_frames(q,txt)
                    if old[0] != new[0] or not (
                      (np.isneginf(old[1]) and np.isneginf(new[1])) or
                      np.isclose(old[1],new[1],rtol=1e-11,atol=1e-10)):
                        raise AssertionError((T,txt,old,new))
        E5.fast_forced_frames=fast_forced_frames
        return {'available':True,'status':'forced_alignment_parity_pass'}
    except Exception as e:
        E5.fast_forced_frames=_ORIGINAL_FORCED
        if strict:raise
        return {'available':False,'status':f'forced_alignment_fallback:{e}'}


def forced_benchmark(n=80,T=70):
    import time
    import alex_e5 as E5
    g=np.random.default_rng(734)
    q=g.uniform(.001,1,size=(T,37));q/=q.sum(axis=1,keepdims=True)
    txt='correctreconstruction'
    a=time.perf_counter()
    for _ in range(n):_ORIGINAL_FORCED(q,txt)
    b=time.perf_counter()
    for _ in range(n):E5.fast_forced_frames(q,txt)
    c=time.perf_counter()
    return {'original_ms':1000*(b-a)/n,'patched_ms':1000*(c-b)/n,'ratio':(b-a)/max(1e-9,c-b)}
