import numpy as np
from PIL import Image

FEATURE_NAMES = [
    "log_aspect", "log_width", "log_height",
    "gray_mean", "gray_std", "gray_q10", "gray_q90",
    "dark_fraction", "bright_fraction",
    "edge_x_mean", "edge_y_mean", "edge_fraction",
    "laplacian_logvar", "gray_entropy",
    "rgb_channel_range", "saturation_mean",
]


def image_features(image: Image.Image):
    im = image.convert("RGB")
    w, h = im.size

    if min(w, h) < 1:
        raise ValueError("Invalid image dimensions")

    small = im.resize(
        (128, 48), Image.Resampling.BILINEAR
    )

    rgb = np.asarray(
        small, dtype=np.float32
    ) / 255.0

    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))

    lap = (
        gray[1:-1, :-2]
        + gray[1:-1, 2:]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        - 4 * gray[1:-1, 1:-1]
    )

    hist, _ = np.histogram(
        gray, bins=32, range=(0, 1)
    )

    p = hist.astype(np.float64)
    p /= max(1, p.sum())
    p = p[p > 0]

    sat = (
        rgb.max(axis=2)
        - rgb.min(axis=2)
    )

    edge_mag = np.zeros_like(gray)
    edge_mag[:, 1:] += gx
    edge_mag[1:, :] += gy

    channels = rgb.mean(axis=(0, 1))

    values = [
        np.log(max(w / h, 1e-6)),
        np.log1p(w),
        np.log1p(h),
        gray.mean(),
        gray.std(),
        np.quantile(gray, 0.1),
        np.quantile(gray, 0.9),
        (gray < 0.25).mean(),
        (gray > 0.75).mean(),
        gx.mean(),
        gy.mean(),
        (edge_mag > 0.16).mean(),
        np.log1p(lap.var() * 1e4),
        -(p * np.log(p)).sum(),
        channels.max() - channels.min(),
        sat.mean(),
    ]

    out = np.asarray(
        values, dtype=np.float32
    )

    if (
        out.shape != (len(FEATURE_NAMES),)
        or not np.isfinite(out).all()
    ):
        raise ValueError("Invalid V3 feature vector")

    return out
