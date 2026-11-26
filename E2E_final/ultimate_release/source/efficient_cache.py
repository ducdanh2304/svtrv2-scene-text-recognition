"""Per-observation LRU, exact RGB identity; never shared across input images."""
from collections import OrderedDict
import hashlib, struct

FIELDS = ('args','kwargs','native_logits','native_ids','output_classes')

def rgb_key(image):
    im=image.convert('RGB')
    return hashlib.sha256(struct.pack('>II',*im.size)+im.tobytes()).hexdigest()

class ObservationCache:
    def __init__(self, max_entries=8, max_bytes=64*1024*1024):
        self.max_entries=max_entries; self.max_bytes=max_bytes; self.clear()
    def clear(self):
        self.items=OrderedDict(); self.bytes=0
    def restore(self,key,scorer):
        if key not in self.items:return None
        state,response,size=self.items.pop(key); self.items[key]=(state,response,size)
        for name,value in state.items():setattr(scorer,name,value)
        return dict(response)
    def store(self,key,scorer,response):
        state={name:getattr(scorer,name) for name in FIELDS}
        def size(v):
            if hasattr(v,'numel') and hasattr(v,'element_size'):return v.numel()*v.element_size()
            if isinstance(v,(list,tuple)):return sum(size(x) for x in v)
            if isinstance(v,dict):return sum(size(x) for x in v.values())
            return 0
        n=size(state)
        if n>self.max_bytes:return
        if key in self.items:self.bytes-=self.items.pop(key)[2]
        while self.items and (len(self.items)>=self.max_entries or self.bytes+n>self.max_bytes):
            self.bytes-=self.items.popitem(last=False)[1][2]
        self.items[key]=(state,dict(response),n); self.bytes+=n
