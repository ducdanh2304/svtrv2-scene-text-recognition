"""GT-blind interfaces. The recognizer never sees the label or dataset name."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Prediction:
    text: str
    confidence: float | None = None
    method: str = ""
    route: str = ""
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "confidence": self.confidence, "method": self.method,
                "route": self.route, "latency_ms": self.latency_ms, "details": self.details}

@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str | None = None
    image_bytes: bytes | None = None
    gt: str | None = None

    def load_image(self):
        from PIL import Image
        import io
        if self.image_bytes is not None:
            with Image.open(io.BytesIO(self.image_bytes)) as image:
                return image.convert("RGB")
        if self.image_path is None:
            raise ValueError("Sample has no image bytes or path")
        with Image.open(self.image_path) as image:
            return image.convert("RGB")
