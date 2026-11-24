"""A simple, explicitly UNVALIDATED image-only router for an integrated prototype.

Routing NEVER reads labels, dataset names or sample IDs. Thresholds are not a
trained quality predictor and must not be presented as a published method gain.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageFilter


def laplacian_variance(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L").resize((128, 48)), dtype=np.float32)
    # 4-neighbour discrete Laplacian, no OpenCV requirement.
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
           - 4.0 * gray[1:-1, 1:-1])
    return float(np.var(lap))


def image_route(image: Image.Image, *, long_ratio_min=5.0,
                blur_laplacian_var_max=35.0, auto_short_default="artistic") -> tuple[str, dict]:
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError("Image has zero width or height")
    if auto_short_default not in {"artistic", "baseline"}:
        raise ValueError("auto_short_default must be artistic or baseline")
    aspect = w / h
    # Long horizontal text or unusually tall crops require SAGE geometry.
    if aspect >= long_ratio_min or aspect <= 0.8:
        return "longtext", {"aspect": aspect, "routing_reason": "geometry"}
    blur = laplacian_variance(image)
    if blur < blur_laplacian_var_max:
        return "motion", {"aspect": aspect, "laplacian_variance": blur,
                          "routing_reason": "heuristic_blur"}
    return auto_short_default, {"aspect": aspect, "laplacian_variance": blur,
                                "routing_reason": "default_short_text"}
