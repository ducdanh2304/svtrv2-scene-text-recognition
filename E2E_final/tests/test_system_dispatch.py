from PIL import Image
from str_e2e.types import Prediction
from str_e2e.system import STRSystem

class Fake:
    def __init__(self, name):
        self.name=name
    def predict(self, image):
        return Prediction(text=self.name, confidence=.8, method=self.name, route="fake")

def test_motion_branch_is_matched():
    system = STRSystem({"mode": "motion", "assets": {}})
    system.backends["motion_b0"] = Fake("b0")
    system.backends["s2_aug"] = Fake("s2")
    base, method = system.predict_pair(Image.new("RGB", (90,32)), "motion")
    assert (base.text, method.text, method.route) == ("b0", "s2", "motion")

def test_artistic_fuse_branch():
    system = STRSystem({"mode": "artistic", "assets": {}})
    system.backends["r3_base"] = Fake("b0")
    system.backends["r3_tta"] = Fake("r3")
    system.backends["waterec"] = Fake("W")
    _, method = system.predict_pair(Image.new("RGB", (90,32)), "artistic")
    assert method.text == "W"
    assert method.details["delta"] == .15
