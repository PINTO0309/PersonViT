"""Deterministic photometric shift conditions, shared between the style-shift
robustness probe (tools/eval_style_shift.py) and shift-aware validation during
training (SOLVER.VAL_SHIFT)."""

import torch
import torchvision.transforms as T


def _gain(g):
    return lambda x: x * g


def _contrast(c):
    return lambda x: (x - 0.5) * c + 0.5


def _channel_gain(r, g, b):
    def fn(x):
        return x * torch.tensor([r, g, b], dtype=x.dtype).view(3, 1, 1)
    return fn


def _gamma(y):
    return lambda x: x.clamp(min=1e-6) ** y


def _jpeg(quality):
    # deterministic JPEG round-trip: blocking/ringing artifacts of
    # transcoded surveillance streams (8x8 DCT blocks vs 16x16 ViT patches)
    def fn(x):
        from torchvision.io import decode_jpeg, encode_jpeg
        u8 = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
        return decode_jpeg(encode_jpeg(u8, quality=quality)).float() / 255.0
    return fn


# deterministic photometric shifts applied on the [0, 1] tensor
CONDITIONS = {
    'clean': None,
    'bright+30%': _gain(1.3),
    'dark-30%': _gain(0.7),
    'contrast-40%': _contrast(0.6),
    'contrast+40%': _contrast(1.4),
    'warm': _channel_gain(1.25, 1.0, 0.8),
    'cool': _channel_gain(0.8, 1.0, 1.25),
    'gamma0.6': _gamma(0.6),
    'gamma1.6': _gamma(1.6),
    'jpeg-q40': _jpeg(40),
    'jpeg-q20': _jpeg(20),
}


class StyleShift:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, tensor):
        if self.fn is None:
            return tensor
        return self.fn(tensor).clamp(0.0, 1.0)


def build_shift_val_transforms(cfg, condition):
    """Test-time transforms with the named shift inserted before Normalize."""
    if condition not in CONDITIONS:
        raise KeyError('unknown style-shift condition {!r} (choose from {})'.format(
            condition, ', '.join(CONDITIONS)))
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        StyleShift(CONDITIONS[condition]),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
