"""Frozen SVTRv2-T/RCTC with post-stage residual adapters; Kaggle worker.

The stage hooks replace (feature, spatial_size) with (adapter(feature), spatial_size).
The OpenOCR encoder and decoder remain frozen. This is an experiment designed for
this project, not a reproduction of an adapter paper.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ResidualAdapter(nn.Module):
    """Identity at initialization; 1x5 spatial depthwise + bottleneck mixing."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.norm = nn.GroupNorm(1, channels)
        self.dw = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2), groups=channels)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, sz):
        sequence = x.ndim == 3
        if sequence:
            b, n, c = x.shape
            assert n == int(sz[0]) * int(sz[1]), (x.shape, sz)
            y = x.transpose(1, 2).reshape(b, c, int(sz[0]), int(sz[1]))
        else:
            assert x.ndim == 4 and tuple(x.shape[-2:]) == tuple(sz), (x.shape, sz)
            y = x
        delta = self.up(self.act(self.down(self.dw(self.norm(y)))))
        if sequence:
            delta = delta.flatten(2).transpose(1, 2)
        return x + delta


class HookedOCR(nn.Module):
    def __init__(self, base, selected, channels):
        super().__init__()
        self.base = base
        self.selected = tuple(selected)
        self.adapters = nn.ModuleDict({str(i): ResidualAdapter(channels[i]) for i in selected})
        self.before = {}
        self.after = {}
        self.handles = []
        for i in range(3):
            self.handles.append(self.base.encoder.stages[i].register_forward_hook(self._hook(i)))

    def _hook(self, index):
        def apply(_module, _args, output):
            assert isinstance(output, tuple) and len(output) == 2
            x, sz = output
            self.before[index] = x.detach()
            if str(index) in self.adapters:
                x = self.adapters[str(index)](x, sz)
            self.after[index] = x.detach()
            return x, sz
        return apply

    def forward(self, x):
        self.before.clear()
        self.after.clear()
        return self.base(x)


def load_base(cfg, checkpoint, device):
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from openrec.losses import build_loss
    from openrec.metrics import build_metric
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    arch = copy.deepcopy(cfg["Architecture"])
    arch["Decoder"]["out_channels"] = post.get_character_num()
    base = build_model(arch)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw["state_dict"] if "state_dict" in raw else raw
    state = {k.removeprefix("module."): v for k, v in state.items()}
    base.load_state_dict(state, strict=True)
    base.to(device).eval().requires_grad_(False)
    assert len(base.encoder.stages) == 4 and not tuple(base.encoder.stages[3].parameters())
    return base, post, build_loss(cfg["Loss"]), build_metric(cfg["Metric"])


def norm(metric, value):
    if metric.ignore_space:
        value = value.replace(" ", "")
    if metric.is_filter:
        value = metric._normalize_text(value)
    if metric.is_lower:
        value = value.lower()
    return value


@torch.no_grad()
def evaluate(net, post, metric, cfg, data_dir, device, logger, save_path=None):
    from tools.data import build_dataloader
    c = copy.deepcopy(cfg)
    c["Eval"]["dataset"]["data_dir_list"] = [str(data_dir)]
    random.seed(991); np.random.seed(991); torch.manual_seed(991)
    loader = build_dataloader(c, "Eval", logger)
    net.eval(); metric.reset()
    targets, preds, correct = [], [], []
    for batch in loader:
        probs = net(batch[0].to(device))
        pred_batch, target_batch = post(probs, [x.numpy() for x in batch])
        metric((pred_batch, target_batch))
        for (p, _), (t, _) in zip(pred_batch, target_batch):
            preds.append(p); targets.append(t); correct.append(norm(metric, p) == norm(metric, t))
    acc = sum(correct) / len(correct)
    assert abs(acc - metric.get_metric()["acc"]) < 1e-4
    result = dict(acc=acc, n=len(targets), targets=targets, preds=preds, correct=correct)
    if save_path is not None:
        Path(save_path).write_text(json.dumps(result, ensure_ascii=False))
    return result


def construct(args, cfg, device):
    base, post, loss_fn, metric = load_base(cfg, args.init, device)
    channels = {0: 128, 1: 256, 2: 256}  # official Tiny config; checked below by a forward pass
    assert cfg["Architecture"]["Encoder"]["dims"] == [64, 128, 256]
    selected = [int(v) - 1 for v in args.placements.split("+")]
    assert selected == sorted(set(selected)) and all(i in (0, 1, 2) for i in selected)
    net = HookedOCR(base, selected, channels).to(device)
    return net, post, loss_fn, metric


def main():
    p = argparse.ArgumentParser()
    for key in ("openocr", "cfg", "init", "placements", "out", "val_dirs"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval_dirs", default="")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    import yaml
    sys.path.insert(0, str(Path(args.openocr).resolve()))
    os.chdir(args.openocr)
    from tools.data import build_dataloader
    from tools.utils.logging import get_logger
    logger = get_logger()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Training requires Kaggle GPU"
    cfg = yaml.safe_load(Path(args.cfg).read_text())
    net, post, loss_fn, metric = construct(args, cfg, device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    trainable = [v for v in net.adapters.parameters() if v.requires_grad]
    assert trainable and all(not x.requires_grad for x in net.base.parameters())
    train_loader = build_dataloader(cfg, "Train", logger)
    batch = next(iter(train_loader))
    x = batch[0].to(device)
    net.base.eval(); net.base.decoder.train()
    logits = net(x)
    assert logits.shape[0] == x.shape[0] and bool(torch.isfinite(logits).all())
    shapes = {}
    for i in range(3):
        y = net.before[i]
        ch = y.shape[1] if y.ndim == 4 else y.shape[-1]
        assert ch == {0: 128, 1: 256, 2: 256}[i], (i, y.shape)
        shapes[str(i + 1)] = list(y.shape)
    for i in net.selected:
        assert torch.allclose(net.before[i], net.after[i], atol=1e-6), i
    loss = loss_fn(logits.float(), [x, batch[1].to(device), batch[2].to(device)])["loss"]
    assert torch.isfinite(loss), loss
    loss.backward()
    assert any(v.grad is not None and torch.isfinite(v.grad).all() and bool((v.grad != 0).any()) for v in trainable)
    assert all(v.grad is None for v in net.base.parameters())
    smoke = dict(shapes=shapes, loss=float(loss.detach()), trainable=sum(v.numel() for v in trainable),
                 base_frozen=True, identity_at_init=True, gradient_ok=True, input_shape=list(x.shape))
    (out / "smoke.json").write_text(json.dumps(smoke, indent=2))
    print("SMOKE", smoke, flush=True)
    if args.smoke:
        return
    net.zero_grad(set_to_none=True)
    steps_per_epoch = len(train_loader)
    total = args.epochs * steps_per_epoch
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.05)
    warm = max(1, int(0.05 * total))
    def factor(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    latest = out / "latest.pth"
    start_epoch = 0; step = 0
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        assert saved["init_sha"] == sha256(args.init) and saved["cfg_sha"] == sha256(args.cfg)
        assert saved["placements"] == args.placements
        assert saved["epochs"] == args.epochs and saved["seed"] == args.seed
        net.adapters.load_state_dict(saved["adapters"], strict=True)
        opt.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        start_epoch = saved["epoch"]; step = saved["step"]
    with (out / "train_log.csv").open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "ctc", "lr"])
        if f.tell() == 0:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(args.seed * 1000 + epoch)
            net.base.eval(); net.base.decoder.train(); net.adapters.train()
            losses = []
            for batch in train_loader:
                xx = batch[0].to(device)
                pred = net(xx)
                val = loss_fn(pred.float(), [xx, batch[1].to(device), batch[2].to(device)])["loss"]
                assert torch.isfinite(val), (epoch, step, val)
                opt.zero_grad(set_to_none=True)
                val.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
                opt.step(); scheduler.step(); step += 1
                losses.append(float(val.detach()))
                if step % 200 == 0:
                    print(args.placements, epoch + 1, step, "loss", round(np.mean(losses[-200:]), 4), flush=True)
            writer.writerow(dict(epoch=epoch + 1, step=step, ctc=float(np.mean(losses)), lr=opt.param_groups[0]["lr"]))
            f.flush()
            torch.save(dict(adapters=net.adapters.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                            epoch=epoch + 1, step=step, epochs=args.epochs, seed=args.seed,
                            placements=args.placements, init_sha=sha256(args.init), cfg_sha=sha256(args.cfg)), latest)
    # Validation is separate from the 3000-image IIIT5K test; no test-based checkpoint selection.
    valid = {}
    for condition, data_dir in json.loads(Path(args.val_dirs).read_text()).items():
        valid[condition] = evaluate(net, post, metric, cfg, data_dir, device, logger)["acc"] * 100
    (out / "validation.json").write_text(json.dumps(valid, indent=2))
    if args.eval_dirs:
        for condition, data_dir in json.loads(Path(args.eval_dirs).read_text()).items():
            evaluate(net, post, metric, cfg, data_dir, device, logger, out / f"persample_{condition}.json")
    print("DONE", args.placements, args.epochs, valid, flush=True)


if __name__ == "__main__":
    main()
