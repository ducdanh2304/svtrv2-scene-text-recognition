import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from str_e2e.global_pipeline import GlobalSTRSystem, locked_calibration, spatial_first_table
from str_e2e.types import Prediction

def test_spatial_first_exact_policy():
    row={'full_table':[[.9,.1],[.8,.2],[.4,.6],[.7,.3]],'spatial_table':[[.1,.9],None,[.2,.8],None]}
    result=spatial_first_table(row)
    assert np.allclose(result,[[.1,.9],[.8,.2],[.45,.55]])

def test_locked_calibration_and_bad_hash(tmp_path):
    assert len(locked_calibration()['weights'])==3
    path=tmp_path/'bad.json';path.write_text('{}')
    with pytest.raises(RuntimeError,match='SHA256'):locked_calibration(path)

def test_dispatch_no_router_and_legacy():
    image=Image.new('RGB',(50,10));system=GlobalSTRSystem({})
    row={'candidates':['a','b'],'table':[[.9,.1]]*3,'full_table':[[.9,.1]]*4,'spatial_table':[[.1,.9]]*4,'pred':{'b0':'a'},'latency_ms':0.0}
    system.backends['global_joint']=SimpleNamespace(predict_evidence=lambda im:row)
    system.backends['official']=SimpleNamespace(predict=lambda im:Prediction('native'))
    assert system.predict(image,'global').text=='a'
    assert system.predict(image,'global_spatial').text=='b'
    assert system.predict(image,'baseline').text=='native'
