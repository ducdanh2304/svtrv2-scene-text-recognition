"""Persistent isolated WATERec subprocess for CPU/GPU inference."""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from PIL import Image
from ..paths import require_dir, require_file
from ..types import Prediction

PREFIX = "STR_JSON:"


class WATERecARBackend:
    def __init__(self, *, waterec_repo, waterec_checkpoint, waterec_config, device="cpu"):
        self.repo = require_dir(waterec_repo, "waterec_repo")
        self.ckpt = require_file(waterec_checkpoint, "waterec_checkpoint")
        self.config = require_file(waterec_config, "waterec_config")
        self.device = device
        self.tempdir = tempfile.TemporaryDirectory(prefix="str_waterec_")
        cmd = [sys.executable, "-u", "-m", "str_e2e.backends.waterec_ar_worker",
               "--repo", str(self.repo), "--checkpoint", str(self.ckpt),
               "--config", str(self.config), "--device", self.device]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            ready = self._next_json()
            if ready.get("status") != "ready":
                raise RuntimeError(f"WATERec worker failed to initialize: {ready}")
        except BaseException:
            self.close()
            raise

    def _next_json(self):
        logs = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("WATERec worker stopped; last logs: " + " | ".join(logs[-8:]))
            if line.startswith(PREFIX):
                return json.loads(line[len(PREFIX):])
            if line.strip():
                logs.append(line.strip())

    def predict(self, image: Image.Image) -> Prediction:
        start = time.perf_counter()
        path = Path(self.tempdir.name) / "request.png"
        image.convert("RGB").save(path, format="PNG")
        if self.proc.poll() is not None:
            raise RuntimeError("WATERec worker process exited")
        self.proc.stdin.write(json.dumps({"action": "predict", "image_path": str(path)}) + "\n")
        self.proc.stdin.flush()
        result = self._next_json()
        if result.get("status") != "ok":
            raise RuntimeError(f"WATERec inference error: {result}")
        return Prediction(text=result["text"], confidence=result["score"],
                          method="waterec_rs", route="artistic_component",
                          latency_ms=(time.perf_counter() - start) * 1000)

    def score(self, strings):
        path = Path(self.tempdir.name) / "request.png"
        if self.proc.poll() is not None:
            raise RuntimeError("WATERec AR worker process exited")
        self.proc.stdin.write(json.dumps({"action":"score","image_path":str(path),"strings":list(strings)}) + "\n")
        self.proc.stdin.flush()
        result = self._next_json()
        if result.get("status") != "ok":
            raise RuntimeError(f"WATERec AR likelihood error: {result}")
        return result

    def close(self):
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
                self.proc.wait()
        if getattr(self, "tempdir", None):
            self.tempdir.cleanup()
