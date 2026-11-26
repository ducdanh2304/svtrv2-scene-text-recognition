"""CPU tests of cache, grouping, native R3 enumeration and slicing with mock network."""
import sys,types
import numpy as np
import torch
from PIL import Image
from str_e2e.backends.efficient_vision import ExactTensorCache,groups,slice_output,cache_context,batch_s2,batch_r3,stats_dict
from str_e2e.backends.artistic import R3TTA
from str_e2e.types import Prediction

class Net(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x):self.calls+=1;return x.mean((2,3)).unsqueeze(1)

class DummyHead(torch.nn.Module):
    def __init__(self):super().__init__();self.fc=torch.nn.Identity();self.w_atten_block=True
    def forward(self,x):return self.fc(x)
class RNet(torch.nn.Module):
    def __init__(self):super().__init__();self.decoder=DummyHead();self.calls=0
    def forward(self,x):
        self.calls+=1
        m=x.mean((1,2,3));out=torch.stack([torch.zeros_like(m),m+3,torch.zeros_like(m)-2],dim=-1)[:,None,:].repeat(1,3,1)
        return self.decoder(out)
class TF:
    @staticmethod
    def resize(im,shape,interpolation):return im.resize((shape[1],shape[0]),Image.Resampling.BICUBIC)
    @staticmethod
    def to_tensor(im):return torch.tensor(np.array(im).copy()).permute(2,0,1).float()/255
    @staticmethod
    def normalize(t,mean,std):return (t-0.5)/0.5

def main():
    torch.set_num_threads(1)
    c=ExactTensorCache(max_entries=2,max_bytes=4096);m=Net().eval();x=torch.ones(1,3,4,4)
    with torch.inference_mode():
        a=c.call(m,x);b=c.call(m,x.clone());assert torch.equal(a,b) and m.calls==1
        b.add_(100);assert torch.equal(c.call(m,x),a)
        c.call(m,x*2);c.call(m,x*3);assert len(c.items)==2 and c.bytes<=4096
        c.call(m,x);assert m.calls==4
        model=types.SimpleNamespace(scope=types.SimpleNamespace(recognizer=types.SimpleNamespace(base=types.SimpleNamespace(net=m))),s2net=m)
        stats=stats_dict()
        with cache_context(model,stats):m(x);m(x)
        assert stats['ctc_net_hits']==1 and stats['ctc_net_misses']==1 and 'forward' not in m.__dict__
        try:
            with cache_context(model,stats):raise ValueError('test')
        except ValueError:pass
        assert 'forward' not in m.__dict__
        ts=[torch.ones(1,3,4,w) for w in [8,4,8,8,4]];assert list(groups(ts))==[[0,2,3],[1,4]]
        out={'a':[torch.ones(3,2),torch.ones(3,4)]};assert slice_output(out,2,3)['a'][0].shape==(1,2)
        try:slice_output(torch.ones(2,3),0,3);raise AssertionError('missing shape guard')
        except RuntimeError:pass
        model.motion=types.SimpleNamespace(pipe=types.SimpleNamespace(preprocess=lambda im:torch.tensor(np.array(im).copy()).permute(2,0,1).float().unsqueeze(0)))
        model.chars=['blank','a','b'];images=[Image.new('RGB',(8,4),(i,i,i)) for i in range(4)]
        ref=[model.s2net(model.motion.pipe.preprocess(im))[0].numpy() for im in images];stats=stats_dict();got=batch_s2(model,images,stats)
        assert all(np.array_equal(a,b) for a,b in zip(ref,got)) and stats['s2_batches']==1 and stats['s2_items']==4
        r3=R3TTA.__new__(R3TTA);r3.torch=torch;r3.TF=TF;r3.T=types.SimpleNamespace(InterpolationMode=types.SimpleNamespace(BICUBIC='bicubic'));r3.device='cpu';r3.tta=True;r3.max_ratio=12;r3.base_shape=[[64,64],[96,48],[112,40],[128,32]];r3.base_h=32;r3.model=RNet().eval()
        r3.post=lambda z:[('a',float(row.softmax(-1).max())) for row in z]
        images=[Image.new('RGB',(80,32),(60+i,60+i,60+i)) for i in range(3)]+[Image.new('RGB',(18,60),'white')]
        from str_e2e.backends.global_joint import r3_observations
        refs=[r3_observations(r3,i,model.chars) for i in images];calls=r3.model.calls;stats=stats_dict();got=batch_r3(r3,images,model.chars,stats)
        for a,b in zip(refs,got):
            assert a[0].text==b[0].text and abs(a[0].confidence-b[0].confidence)<1e-6 and a[3]==b[3]
            assert np.allclose(a[1],b[1],atol=1e-6) and np.allclose(a[2],b[2],atol=1e-6)
        assert stats['r3_items']==calls and stats['r3_batches']<calls
    print('VISION_CPU_TEST_PASS',stats)
if __name__=='__main__':main()
