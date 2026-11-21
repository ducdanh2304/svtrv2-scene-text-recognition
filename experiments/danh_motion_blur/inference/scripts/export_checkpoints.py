"""Export the exact audited Kaggle training checkpoints for inference.

Run only on checkpoints produced by your own trusted Kaggle notebook. The
training files contain optimizer state; the exported files contain weights only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import StageAdapters, load_pipeline  # noqa: E402

EXPECTED = {
    'official_release': '684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6',
    'bundled_b0': '3ec59b8b81ec84d2abf07b3156cb771949c7e255d7acd40563028b1414abc611',
    's2_latest': 'f7874600e9fe3cd666bccf7012a0b3a85f64f97d00fb1855c8085b7fddff39a8',
    'a123_latest': 'dc59b617ddc9d57454fcb64964b0a70bdc9bb4e844fb0eea9801e3c3f640c66f',
    'openocr_commit': '1ccfc6ee6161f7133b192e16af3a9265273997ba',
    'adapter_b0_init': '52f27066e3ab65232b450b16c579d415518f24e2d6b8df2a897bf9367ce03ca3',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def strip_prefix(state):
    return {k.removeprefix('module.'): v.detach().cpu() for k, v in state.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--official-release', type=Path, help='Optional: verify and re-export the official release; bundled B0 is otherwise used')
    ap.add_argument('--s2-latest', type=Path, required=True)
    ap.add_argument('--a123-latest', type=Path, required=True)
    ap.add_argument('--openocr', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=ROOT / 'models')
    args = ap.parse_args()
    source = {'s2_latest': args.s2_latest, 'a123_latest': args.a123_latest}
    if args.official_release is not None:
        source['official_release'] = args.official_release
    else:
        source['bundled_b0'] = ROOT / 'models' / 'b0_rctc_state.pt'
    for label, path in source.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha(path)
        if actual != EXPECTED[label]:
            raise ValueError(f'{label} SHA mismatch: {actual}; expected {EXPECTED[label]}')
    commit = subprocess.check_output(['git', '-C', str(args.openocr), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != EXPECTED['openocr_commit']:
        raise ValueError(f'OpenOCR commit {commit} differs from the audited commit')
    sys.path.insert(0, str(args.openocr.resolve()))
    from openrec.modeling import build_model
    from openrec.postprocess.ctc_postprocess import CTCLabelDecode

    config = yaml.safe_load((ROOT / 'configs' / 'inference.yml').read_text())
    post = CTCLabelDecode(character_dict_path=str(args.openocr / 'tools/utils/EN_symbol_dict.txt'))
    arch = config['Architecture']
    arch['Decoder']['out_channels'] = post.get_character_num()
    model = build_model(arch)

    if args.official_release is None:
        b0 = strip_prefix(torch.load(source['bundled_b0'], map_location='cpu', weights_only=True)['state_dict'])
    else:
        # The official release is GTC+RCTC. Keep encoder and RCTC weights only.
        official = torch.load(args.official_release, map_location='cpu', weights_only=False)
        raw = strip_prefix(official['state_dict'])
        b0 = {}
        for name, tensor in raw.items():
            if name.startswith('encoder.'):
                b0[name] = tensor
            elif name.startswith('decoder.ctc_decoder.'):
                b0['decoder.' + name[len('decoder.ctc_decoder.'):]] = tensor
            elif name.startswith('decoder.gtc_decoder.'):
                pass
            else:
                raise ValueError(f'Unexpected official checkpoint parameter {name}')
    model.load_state_dict(b0, strict=True)

    s2_raw = torch.load(args.s2_latest, map_location='cpu', weights_only=False)
    if int(s2_raw.get('epoch', -1)) != 12 or int(s2_raw.get('global_step', -1)) != 38304:
        raise ValueError('S2 checkpoint is not the audited final epoch 12 / step 38304')
    s2 = strip_prefix(s2_raw['state_dict'])
    model.load_state_dict(s2, strict=True)

    adapter_raw = torch.load(args.a123_latest, map_location='cpu', weights_only=False)
    if adapter_raw.get('placements') != '1+2+3' or int(adapter_raw.get('epoch', -1)) != 12:
        raise ValueError('Adapter checkpoint is not final A1+A2+A3 epoch 12')
    if adapter_raw.get('init_sha') != EXPECTED['adapter_b0_init']:
        raise ValueError('Adapter was trained from a different B0 initialization')
    model.load_state_dict(b0, strict=True)
    wrapper = StageAdapters(model)
    adapter = strip_prefix(adapter_raw['adapters'])
    wrapper.adapters.load_state_dict(adapter, strict=True)
    wrapper.close()

    args.output.mkdir(parents=True, exist_ok=True)
    files = {
        'b0_rctc': args.output / 'b0_rctc_state.pt',
        's2_aug': args.output / 's2_aug_state.pt',
        'adapter_a123': args.output / 'adapter_a123_state.pt',
    }
    torch.save({'state_dict': b0}, files['b0_rctc'])
    torch.save({'state_dict': s2}, files['s2_aug'])
    torch.save({'adapters': adapter}, files['adapter_a123'])

    # Verify the actual public inference entry point after export.
    from PIL import Image
    for variant in ('b0', 's2_aug', 'a123'):
        pipeline = load_pipeline(variant, args.output, args.openocr, device='cpu')
        result = pipeline.predict_crop(Image.new('RGB', (96, 32), 'white'))
        assert isinstance(result['text'], str) and 0 <= result['confidence'] <= 1
    manifest = {
        'openocr_commit': commit,
        'source_sha256': {label: sha(path) for label, path in source.items()},
        'export_sha256': {label: sha(path) for label, path in files.items()},
        'base_parameters': sum(x.numel() for x in model.parameters()),
        'adapter_parameters': sum(x.numel() for x in adapter.values()),
        'export_files': {label: path.name for label, path in files.items()},
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
