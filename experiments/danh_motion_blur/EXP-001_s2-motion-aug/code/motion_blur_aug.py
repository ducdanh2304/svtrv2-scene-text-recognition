import math
import os
import time

import cv2
import numpy as np
from PIL import Image


class MotionBlurAug:
    # Làm mờ chuyển động tuyến tính cho ảnh train (PIL RGB). Độ dài kernel tỉ lệ với chiều cao ảnh
    # để sau khi resize về 32/64 px độ mờ nhìn giống nhau với mọi cỡ ảnh gốc.

    def __init__(self, p=0.5, len_frac=(0.1, 0.5), angle=(0.0, 180.0), min_len=3, **kwargs):
        self.p = float(p)
        self.len_frac = (float(len_frac[0]), float(len_frac[1]))
        self.angle = (float(angle[0]), float(angle[1]))
        self.min_len = int(min_len)
        self._rng = None
        self._pid = None

    def _get_rng(self):
        if self._rng is None or self._pid != os.getpid():        # mỗi worker của DataLoader có luồng ngẫu nhiên riêng
            self._pid = os.getpid()
            self._rng = np.random.default_rng([self._pid, time.time_ns() % (2**32)])
        return self._rng

    @staticmethod
    def kernel(length, angle_deg):
        k = max(3, int(round(length)))
        if k % 2 == 0:
            k += 1
        ker = np.zeros((k, k), np.float32)
        c = k // 2
        a = math.radians(angle_deg)
        dx, dy = math.cos(a) * (k - 1) / 2.0, math.sin(a) * (k - 1) / 2.0
        cv2.line(ker, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
        s = float(ker.sum())
        if s <= 0:
            ker[c, c] = 1.0
            s = 1.0
        return ker / s

    def blur(self, arr, length, angle_deg):
        return cv2.filter2D(arr, -1, self.kernel(length, angle_deg), borderType=cv2.BORDER_REPLICATE)

    def __call__(self, data):
        rng = self._get_rng()
        if rng.random() >= self.p:
            return data
        arr = np.asarray(data['image'])
        length = rng.uniform(*self.len_frac) * arr.shape[0]
        if length < self.min_len:
            return data
        data['image'] = Image.fromarray(self.blur(arr, length, rng.uniform(*self.angle)))
        return data
