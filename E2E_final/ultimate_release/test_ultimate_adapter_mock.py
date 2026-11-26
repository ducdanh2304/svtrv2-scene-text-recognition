from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from str_e2e.ultimate_pipeline import UltimateSTRSystem
import str_e2e.backends.efficient_global as eg
from str_e2e.backends import ar_scoring,global_probability
import numpy as np
class Worker:
    def __init__(self):self.calls=[]
    def begin_e2(self):self.calls.append('begin')
    def end_e2(self):self.calls.append('end')
class Model:
    def __init__(self,system):self.w=Worker();self.vision_stats={};self.torch=None;self.system=system
    def predict_evidence(self,image):
        assert self.vision_mode=='e4_batch' and self.w.efficient_mode=='e2'
        assert ar_scoring.raw_ctc_logprob is global_probability.raw_ctc_logprob
        assert image.mode=='RGB'
        return dict(pred={'b0':'a'},candidates=['a','b'],full_table=[[.8,.2],[.7,.3],[.1,.9],[.1,.9]],spatial_table=[[.1,.9],[.1,.9],[.1,.9],[.1,.9]],latency_ms=1.)
orig=ar_scoring.raw_ctc_logprob
with patch.object(eg,'GlobalEfficient',Model):
    with UltimateSTRSystem({'device':'cpu'}) as s:
        base,p=s.predict_pair(Image.new('L',(10,10)))
        assert base.text=='a' and p.text=='b' and p.route=='ultimate'
        assert s.backends['ultimate'].w.calls==['begin','end']
        assert ar_scoring.raw_ctc_logprob is orig
        model=s.backends['ultimate']
        with patch.object(model,'predict_evidence',side_effect=RuntimeError('mock')):
            try:s.predict(Image.new('RGB',(10,10)))
            except RuntimeError as e:assert str(e)=='mock'
            else:raise AssertionError('missing error')
        assert model.w.calls==['begin','end','begin','end'] and ar_scoring.raw_ctc_logprob is orig
print('ULTIMATE_ADAPTER_MOCK_PASS')
