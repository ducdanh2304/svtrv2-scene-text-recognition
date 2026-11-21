"""Small CPU smoke test for the integration interface; no test labels needed."""
import argparse
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from inference.model import StageAdapters, load_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--openocr', type=Path, required=True)
    ap.add_argument('--models', type=Path, default=ROOT / 'models')
    args = ap.parse_args()
    b0 = load_pipeline('b0', args.models, args.openocr, device='cpu')
    for wh, expected in [((64, 64), (64, 64)), ((96, 48), (48, 96)),
                         ((112, 40), (40, 112)), ((128, 32), (32, 128))]:
        img = Image.new('RGB', wh, 'white')
        x = b0.preprocess(img)
        assert x.shape == (1, 3, *expected), (wh, x.shape)
        assert torch.isfinite(x).all()
        result = b0.predict_crop(img)
        assert isinstance(result['text'], str) and 0 <= result['confidence'] <= 1

    # A1+A2+A3 must have exactly the same output before its trained weights load.
    x = b0.preprocess(Image.new('RGB', (96, 32), 'white'))
    with torch.inference_mode():
        baseline = b0.model(x)
        attached = StageAdapters(b0.model).eval()
        identity = attached(x)
    assert torch.allclose(baseline, identity, atol=1e-6)
    assert sum(p.numel() for p in attached.adapters.parameters()) == 79648
    attached.close()
    for variant in ('s2_aug', 'a123'):
        needed = args.models / ('s2_aug_state.pt' if variant == 's2_aug' else 'adapter_a123_state.pt')
        if needed.exists():
            pipeline = load_pipeline(variant, args.models, args.openocr, device='cpu')
            assert isinstance(pipeline.predict_crop(Image.new('RGB', (96, 32), 'white'))['text'], str)
    print('Smoke PASS: shapes, finite output, strict B0 load, adapter identity')


if __name__ == '__main__':
    main()
