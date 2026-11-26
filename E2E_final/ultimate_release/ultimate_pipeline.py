"""Deployment adapter for the frozen E4-Batch computation; no GT or reference needed."""
from time import perf_counter
from threading import RLock
from .global_pipeline import GlobalSTRSystem, locked_calibration, spatial_first_table, CALIBRATION_SHA256
from .types import Prediction
_LOCK = RLock()
class UltimateSTRSystem(GlobalSTRSystem):
    def predict_pair(self, image, mode=None):
        mode = 'ultimate' if mode is None else mode
        if mode != 'ultimate':
            raise ValueError('Use GlobalSTRSystem in the original runtime for other modes.')
        from .backends.efficient_global import GlobalEfficient
        from .backends.efficient_ctc import scoring_context
        from .backends.global_probability import choose
        with _LOCK:
            if 'ultimate' not in self.backends:
                model = GlobalEfficient(self)
                model.vision_mode = 'e4_batch'
                self.backends['ultimate'] = model
            model = self.backends['ultimate']
            cal = locked_calibration()
            model.w.efficient_mode = 'e2'
            model.w.efficient_scored = 0
            if str(self.device).startswith('cuda'):
                model.torch.cuda.synchronize()
            start = perf_counter()
            model.w.begin_e2()
            try:
                with scoring_context(True) as scorer:
                    ev = model.predict_evidence(image.convert('RGB'))
                    text, mbr, posterior = choose(ev['candidates'], spatial_first_table(ev), cal['weights'], cal['temperatures'])
            finally:
                model.w.end_e2()
            if str(self.device).startswith('cuda'):
                model.torch.cuda.synchronize()
            total_ms = (perf_counter()-start)*1000
            return (Prediction(text=ev['pred']['b0'], method='official_svtrv2_b0', route='baseline_trace'),
                    Prediction(text=text, method='EFFICIENT_GLOBAL_SPATIAL_FIRST', route='ultimate', latency_ms=total_ms,
                               details={'evidence_ms':ev['latency_ms'], 'candidates':ev['candidates'], 'posterior':posterior, 'mbr':mbr,
                                        'calibration_sha256':CALIBRATION_SHA256, 'ctc':dict(scorer.stats), 'vision':dict(model.vision_stats)}))
