"""JPEG-compression augmentation (training side).

Re-encodes the training crop through an in-memory JPEG round-trip at a
random quality, teaching robustness to the blocking/ringing artifacts of
transcoded surveillance streams. The measured exposure of the flagship
models (probe conditions `jpeg-q40`/`jpeg-q20`) is what this augmentation
targets. Applied at the PIL stage, after the photometric transforms —
real pipelines compress after capture, so lighting/blur precede
compression.
"""

import io
import random

from PIL import Image


class RandomJPEG:
    def __init__(self, probability, quality_range):
        self.probability = probability
        self.quality_range = tuple(quality_range)

    def __call__(self, img):
        if random.random() >= self.probability:
            return img
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
