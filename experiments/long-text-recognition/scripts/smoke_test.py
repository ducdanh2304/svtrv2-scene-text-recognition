from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

def main():
    import longtext_final as LT
    import longtext_v3 as V3
    import density_probe as DP
    import orientation_probe as OP
    import unified_router as UNI
    import alex_ctc as AX
    import alex_c1 as C1
    import space_ctc as SPACE
    import alex_e5 as E5
    import alex_e7 as E7
    import scope_e7_fast as FAST
    import scope_final_runtime_v3 as FULL
    import scope_coverage_runtime as COVER
    checks={
        'geometry_v2':LT.run_regression_tests,
        'geometry_v3':V3.run_regression_tests_v3,
        'dense':DP.run_cpu_tests,
        'orientation':OP.run_cpu_tests,
        'routing':UNI.run_cpu_tests,
        'CTC_viterbi':AX.cpu_tests,
        'CTC_edit':C1.cpu_tests,
        'CTC_prefix_beam':SPACE.cpu_tests,
        'CTC_posterior':E5.cpu_tests,
        'SCOPE_E7':E7.cpu_tests,
        'SCOPE_fast':FAST.cpu_tests,
        'three_route_parity':FULL.cpu_tests,
        'guard_and_fallback':COVER.cpu_tests,
    }
    for name, fn in checks.items():
        result=fn()
        if isinstance(result,dict) and result.get('passed') is False:
            raise AssertionError((name,result))
        print('PASS',name, str(result)[:140], flush=True)
    print('ALL CPU SMOKE TESTS PASS')

if __name__=='__main__':
    main()
