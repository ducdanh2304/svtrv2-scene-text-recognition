"""Dedicated WATERec-RS process: prevents OpenOCR module namespace collisions.

Use only trusted model/config files. Producer and consumer communicate newline JSON
and prefix model outputs with STR_JSON: to avoid confusion with OpenOCR logging.
"""
from __future__ import annotations
import argparse
import copy
import inspect
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
    post_config = copy.deepcopy(cfg["PostProcess"])
    global_config = copy.deepcopy(cfg["Global"])
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
    from openrec.postprocess import build_post_process
    from str_e2e.backends.ar_scoring import ReplayAR
    post = build_post_process(post_config, global_config)
    # Save the *actual archived* decoder source for auditing even on a failed preflight.
    diagnostic = os.environ.get("CS406_AR_DIAGNOSTIC")
    if diagnostic:
        Path(diagnostic).write_text(inspect.getsource(type(recognizer.model.decoder)) + "\n\n" + inspect.getsource(type(post)), encoding="utf-8")
    scorer = ReplayAR(recognizer, post, global_config["max_text_length"])
    from str_e2e.backends.efficient_cache import ObservationCache,rgb_key
    from PIL import Image
    cache=ObservationCache()
    observed_path = None
    emit({"status": "ready", "parameters": sum(p.numel() for p in recognizer.model.parameters())})
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("action") == "quit":
                emit({"status": "bye"})
                return
            action = message.get("action")
            if action == "reset_cache":
                cache.clear(); observed_path=None
                emit({"status":"ok"}); continue
            if action == "predict":
                image_path = Path(message["image_path"])
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                key=message.get("cache_key"); hit=False
                if key is not None:
                    with Image.open(image_path) as im:
                        if rgb_key(im)!=key:raise RuntimeError("E2 RGB identity mismatch")
                    response=cache.restore(key,scorer); hit=response is not None
                else:response=None
                if response is None:
                    response=scorer.observe(image_path)
                    if key is not None:cache.store(key,scorer,response)
                observed_path = str(image_path)
                emit({"status": "ok", "text": str(response["text"]),
                      "score": float(response["score"]),"cache_hit":hit})
            elif action == "score":
                if observed_path is None or message.get("image_path") != observed_path:
                    raise RuntimeError("AR score requested without matching native observation")
                strings = message["strings"]
                if not isinstance(strings, list) or not all(isinstance(s,str) for s in strings):
                    raise ValueError("Expected list of raw strings")
                if not scorer.checked:
                    scorer.validate()
                reason = None
                if scorer.eos not in scorer.native_ids:
                    reason = "native_no_eos"
                elif any(len(s)>scorer.supported_len for s in strings):
                    reason = "candidate_exceeds_eos_capacity"
                if reason:
                    scores = None
                elif message.get("efficient_mode") in ("fast", "audit"):
                    from str_e2e.backends.efficient_ar import fast_score
                    scores = fast_score(scorer, strings, audit=message["efficient_mode"]=="audit")
                else:
                    scores = scorer.score(strings)
                emit({"status":"ok", "scores":scores,"reason":reason,
                      "supported_len":scorer.supported_len,"projection":scorer.head,
                      "prefix_preflight":scorer.checked})
            else:
                raise ValueError("Unknown worker request")
        except Exception as exc:
            emit({"status": "error", "error": f"{type(exc).__name__}: {exc}"})

if __name__ == "__main__":
    main()
