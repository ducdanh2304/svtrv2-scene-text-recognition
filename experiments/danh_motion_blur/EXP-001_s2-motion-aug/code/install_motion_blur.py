"""Register the audited MotionBlurAug in a pinned OpenOCR checkout."""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--openocr', type=Path, required=True)
    args = ap.parse_args()
    preprocess = args.openocr / 'openrec' / 'preprocess'
    mapping = preprocess / '__init__.py'
    if not mapping.is_file():
        raise FileNotFoundError(mapping)
    source = Path(__file__).with_name('motion_blur_aug.py')
    shutil.copyfile(source, preprocess / source.name)
    text = mapping.read_text()
    entry = "    'MotionBlurAug': '.motion_blur_aug',\n"
    if entry not in text:
        marker = 'MODULE_MAPPING = {\n'
        if text.count(marker) != 1:
            raise ValueError('OpenOCR preprocess mapping changed; inspect before patching')
        mapping.write_text(text.replace(marker, marker + entry, 1))
    print('MotionBlurAug registered in', preprocess)


if __name__ == '__main__':
    main()
