"""Build the occluder patch set for NPO augmentation (datasets/npo.py).

FED curates ~30 occluder patches (pillars, vehicles, umbrellas) by hand.
This tool automates an approximation: border strips of person crops are
background/occluder regions by construction (the person occupies the
center), so it samples training images deterministically and saves their
edge strips — vertical strips (full height, ~30% width) as portrait
occluders and top/bottom strips (~30% height, full width) as landscape
occluders.

Usage (from transreid_pytorch/):
    python tools/build_npo_patches.py [--root ./data/reid/train]
                                      [--out npo_patches] [--num 30]
"""

import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default='./data/reid/train')
    ap.add_argument('--out', default='npo_patches')
    ap.add_argument('--num', type=int, default=30,
                    help='total patches (2/3 portrait, 1/3 landscape)')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    images = sorted(glob.glob(os.path.join(args.root, '*.jpg')))
    if not images:
        raise FileNotFoundError('no training images under {}'.format(args.root))
    rng = random.Random(args.seed)
    picks = rng.sample(images, args.num)
    os.makedirs(args.out, exist_ok=True)

    n_portrait = args.num * 2 // 3
    for index, path in enumerate(picks):
        img = Image.open(path).convert('RGB')
        width, height = img.size
        if index < n_portrait:
            strip = int(width * 0.3)
            left = 0 if rng.random() < 0.5 else width - strip
            patch = img.crop((left, 0, left + strip, height))
            name = 'portrait_{:02d}.jpg'.format(index)
        else:
            strip = int(height * 0.3)
            top = 0 if rng.random() < 0.5 else height - strip
            patch = img.crop((0, top, width, top + strip))
            name = 'landscape_{:02d}.jpg'.format(index)
        patch.save(os.path.join(args.out, name), quality=95)
    print('wrote {} patches to {}'.format(args.num, args.out))


if __name__ == '__main__':
    main()
