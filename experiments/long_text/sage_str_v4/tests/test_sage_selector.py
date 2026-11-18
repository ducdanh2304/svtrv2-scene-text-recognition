import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sage_str_v2.selector import Hypothesis, Observation
from sage_str_v4.selector import features, candidates, reference, group_for, COLUMNS


def hyp(name, text, confidence=0.75, windows=1):
    return Hypothesis(name,text,ctc_T=48,confidence=confidence,entropy=.15,
       char_margin=.22,edge_occupancy=.04,overlap_agreement=.5,n_windows=windows,
       distortion=1.,cost_ms=0.)

class TestSageContract(unittest.TestCase):
    def test_reference_is_geometry_when_present(self):
        obs=Observation(450,25,hyp('original','short'),(hyp('geometry_v3','muchlonger',windows=3),))
        r=reference(obs)
        self.assertEqual(r.strategy,'geometry_v3')
        r,cs=candidates(obs)
        self.assertEqual([c.strategy for c in cs],['original'])

    def test_original_reference_on_short_text(self):
        obs=Observation(180,36,hyp('original','clear'),(hyp('crop3_width3','cllear'),))
        self.assertEqual(reference(obs).strategy,'original')
        r,cs=candidates(obs)
        self.assertEqual(len(cs),1)
        self.assertEqual(group_for(cs[0]),'recovery')
        f=features(obs,r,cs[0])
        self.assertEqual(set(f),set(COLUMNS))

    def test_rotation_context_and_determinism(self):
        obs=Observation(40,180,hyp('original','hfl'),
            (hyp('rotate_cw','half',.92),hyp('rotate_ccw','hf',.4)))
        r,cs=candidates(obs)
        self.assertEqual([c.strategy for c in cs],['rotate_ccw','rotate_cw'])
        self.assertTrue(all(group_for(c)=='orientation' for c in cs))
        self.assertEqual(features(obs,r,cs[1]),features(obs,r,cs[1]))
        self.assertEqual(features(obs,r,cs[0])['rival_orientation_available'],1.)

if __name__=='__main__':unittest.main()
