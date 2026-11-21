"""Inference interface for Danh's audited SVTRv2-T/RCTC checkpoints.

Input is an RGB word crop. Architecture comes from OpenOCR commit
1ccfc6ee6161f7133b192e16af3a9265273997ba. The residual adapter is
the exact module used in the A1+A2+A3 grid experiment.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from PIL import Image


class ResidualAdapter(nn.Module):
    """Post-stage identity-initialized adapter; preserves feature shape."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.norm = nn.GroupNorm(1, channels)
        self.dw = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2), groups=channels)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, sz: tuple[int, int]) -> torch.Tensor:
        sequence = x.ndim == 3
        if sequence:
            b, n, ch = x.shape
            assert n == int(sz[0]) * int(sz[1]), (x.shape, sz)
            y = x.transpose(1, 2).reshape(b, ch, int(sz[0]), int(sz[1]))
        else:
            assert x.ndim == 4 and tuple(x.shape[-2:]) == tuple(sz), (x.shape, sz)
            y = x
        delta = self.up(self.act(self.down(self.dw(self.norm(y)))))
        if sequence:
            delta = delta.flatten(2).transpose(1, 2)
        return x + delta


class StageAdapters(nn.Module):
    """Apply adapter after each of the three learned SVTRv2 stages."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict({
            '0': ResidualAdapter(128),
            '1': ResidualAdapter(256),
            '2': ResidualAdapter(256),
        })
        assert len(base.encoder.stages) == 4
        self._handles = [base.encoder.stages[i].register_forward_hook(self._hook(i)) for i in range(3)]

    def _hook(self, index: int):
        def apply(_module, _args, output):
            assert isinstance(output, tuple) and len(output) == 2
            feature, sz = output
            return self.adapters[str(index)](feature, sz), sz
        return apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _import_openocr(openocr_root: str | Path):
    root = Path(openocr_root).resolve()
    if not (root / 'openrec' / 'modeling' / '__init__.py').exists():
        raise FileNotFoundError(f'OpenOCR checkout missing at {root}')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from openrec.modeling import build_model
    from openrec.postprocess.ctc_postprocess import CTCLabelDecode
    return root, build_model, CTCLabelDecode


def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location='cpu', weights_only=True)
    state = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict) or not all(isinstance(x, torch.Tensor) for x in state.values()):
        raise ValueError(f'Expected inference state_dict in {path}')
    return {name.removeprefix('module.'): tensor for name, tensor in state.items()}


def load_pipeline(
    variant: str,
    model_dir: str | Path,
    openocr_root: str | Path,
    device: str | torch.device = 'cpu',
) -> 'OCRPipeline':
    """Load B0, S2_AUG or A1+A2+A3; strictly validate all parameter names."""
    import yaml

    if variant not in {'b0', 's2_aug', 'a123'}:
        raise ValueError('variant must be b0, s2_aug, or a123')
    root, build_model, Decoder = _import_openocr(openocr_root)
    cfg = yaml.safe_load((Path(__file__).parent / 'configs' / 'inference.yml').read_text())
    dictionary = root / 'tools' / 'utils' / 'EN_symbol_dict.txt'
    post = Decoder(character_dict_path=str(dictionary), use_space_char=False)
    arch = cfg['Architecture']
    arch['Decoder']['out_channels'] = post.get_character_num()
    base = build_model(arch)
    model_dir = Path(model_dir)
    base_file = 's2_aug_state.pt' if variant == 's2_aug' else 'b0_rctc_state.pt'
    base.load_state_dict(_load_state(model_dir / base_file), strict=True)
    if variant == 'a123':
        model = StageAdapters(base)
        adapter_file = model_dir / 'adapter_a123_state.pt'
        adapter_raw = torch.load(adapter_file, map_location='cpu', weights_only=True)
        adapter_state = adapter_raw.get('adapters', adapter_raw)
        model.adapters.load_state_dict(adapter_state, strict=True)
    else:
        model = base
    model.to(device).eval().requires_grad_(False)
    return OCRPipeline(model, post, torch.device(device), cfg['Resize'])


class OCRPipeline:
    def __init__(self, model: nn.Module, decoder, device: torch.device, resize_cfg: dict):
        self.model = model
        self.decoder = decoder
        self.device = device
        self.resize_cfg = resize_cfg

    def preprocess(self, crop: Image.Image) -> torch.Tensor:
        """Match OpenOCR RatioDataSetTVResize evaluation, padding=False."""
        from torchvision import transforms as T
        from torchvision.transforms import functional as F

        crop = crop.convert('RGB')
        width, height = crop.size
        if width <= 0 or height <= 0:
            raise ValueError('Empty image crop')
        ratio = int(np.clip(np.around(width / height), 1, self.resize_cfg['max_ratio']))
        if ratio <= 4:
            image_width, image_height = self.resize_cfg['base_shape'][ratio - 1]
        else:
            image_height = self.resize_cfg['base_h']
            image_width = image_height * ratio
        resized = F.resize(crop, (image_height, image_width), interpolation=T.InterpolationMode.BICUBIC)
        tensor = F.to_tensor(resized)
        tensor = F.normalize(tensor, [0.5] * 3, [0.5] * 3)
        return tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict_crop(self, crop: Image.Image) -> dict[str, float | str]:
        """Return text and mean CTC character confidence for one word crop."""
        x = self.preprocess(crop)
        probs = self.model(x)
        text, confidence = self.decoder(probs)[0]
        return {'text': text, 'confidence': float(confidence)}
