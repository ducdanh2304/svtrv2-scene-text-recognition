"""GT-blind features for selecting B0 or S2."""

import cv2
import numpy as np
from PIL import Image

from .router_v3_features import (
    FEATURE_NAMES as BASE_NAMES,
    image_features,
)

FEATURE_NAMES = list(BASE_NAMES) + [
    "original_laplacian_logvar",
    "original_grad_x_logenergy",
    "original_grad_y_logenergy",
    "original_grad_anisotropy",
]


def motion_features(image: Image.Image) -> np.ndarray:
    base = image_features(image)
    gray = np.asarray(
        image.convert("L"), dtype=np.uint8
    )

    if min(gray.shape) < 3:
        raise ValueError("Image too small")

    h, w = gray.shape

    if max(h, w) > 1024:
        scale = 1024.0 / max(h, w)
        gray = cv2.resize(
            gray,
            (
                max(3, round(w * scale)),
                max(3, round(h * scale)),
            ),
        )

    lap = cv2.Laplacian(
        gray, cv2.CV_32F, ksize=3
    )

    gx = cv2.Sobel(
        gray, cv2.CV_32F, 1, 0, ksize=3
    )

    gy = cv2.Sobel(
        gray, cv2.CV_32F, 0, 1, ksize=3
    )

    ex = float(np.mean(gx * gx))
    ey = float(np.mean(gy * gy))

    extra = np.asarray([
        np.log1p(float(lap.var())),
        np.log1p(ex),
        np.log1p(ey),
        (ex - ey) / (ex + ey + 1e-6),
    ], dtype=np.float32)

    result = np.concatenate([base, extra])

    if (
        result.shape != (len(FEATURE_NAMES),)
        or not np.isfinite(result).all()
    ):
        raise ValueError("Invalid motion features")

    return result
