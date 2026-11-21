"""Predict one cropped word image with the exported Danh models."""
import argparse
import json
from pathlib import Path

from PIL import Image

from model import load_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['b0', 's2_aug', 'a123'], required=True)
    ap.add_argument('--image', type=Path, required=True, help='RGB word crop, not a full scene image')
    ap.add_argument('--openocr', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path, default=Path(__file__).parent / 'models')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    pipeline = load_pipeline(args.variant, args.model_dir, args.openocr, args.device)
    with Image.open(args.image) as image:
        result = pipeline.predict_crop(image)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
