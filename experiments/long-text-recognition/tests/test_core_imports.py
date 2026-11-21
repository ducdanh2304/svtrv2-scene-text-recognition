import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def test_decoder_frozen_api():
    import alex_e7 as E7
    import scope_e7_fast as FAST
    assert E7.cpu_tests()['passed']
    assert FAST.cpu_tests()['passed']

def test_guard_fallback():
    import scope_coverage_runtime as COVER
    assert COVER.cpu_tests()['passed']
