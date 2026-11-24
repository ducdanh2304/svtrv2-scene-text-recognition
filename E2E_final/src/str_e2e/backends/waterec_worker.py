"""Dedicated WATERec-RS process: prevents OpenOCR module namespace collisions.

Use only trusted model/config files. Producer and consumer communicate newline JSON
and prefix model outputs with STR_JSON: to avoid confusion with OpenOCR logging.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PREFIX = "STR_JSON:"


def emit(obj):
    print(PREFIX + json.dumps(obj, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import os
    os.chdir(args.repo)
    import torch
    from tools.engine.config import Config
    from tools.infer_rec import OpenRecognizer
    cfg = Config(str(args.config.resolve())).cfg
    cfg["Global"]["device"] = args.device
    cfg["Global"]["use_amp"] = False
    cfg["Global"]["pretrained_model"] = str(args.checkpoint.resolve())
    cfg["Global"]["checkpoints"] = None
    recognizer = OpenRecognizer(config=cfg, backend="torch",
                                use_gpu="true" if args.device == "cuda" else "false")
    recognizer.model.eval().requires_grad_(False)
    # This is a trusted research checkpoint supplied by the user.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    actual = recognizer.model.state_dict()
    if set(checkpoint) != set(actual):
        raise RuntimeError("Official WATERec checkpoint keys do not match")
    for key, value in checkpoint.items():
        if value.shape != actual[key].shape or not torch.equal(value.cpu(), actual[key].cpu()):
            raise RuntimeError(f"WATERec strict checkpoint mismatch at {key}")
    emit({"status": "ready", "parameters": sum(p.numel() for p in recognizer.model.parameters())})
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("action") == "quit":
                emit({"status": "bye"})
                return
            if message.get("action") != "predict":
                raise ValueError("Unknown worker request")
            image_path = Path(message["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with torch.inference_mode():
                response = recognizer(img_path=str(image_path), batch_num=1)[0]
            emit({"status": "ok", "text": str(response["text"]),
                  "score": float(response["score"])})
        except Exception as exc:
            emit({"status": "error", "error": f"{type(exc).__name__}: {exc}"})

if __name__ == "__main__":
    main()
