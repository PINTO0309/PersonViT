"""NPO occlusion-paste augmentation (FED, arXiv:2112.08740, light version).

Pastes a realistic occluder patch flush against one edge of the training
image, simulating non-pedestrian occlusions (pillars, vehicles, bags) far
more faithfully than RandomErasing's uniform rectangles. This is the
augmentation-only variant of FED: no occlusion mask is produced and no
mask-supervised head is trained.

Patches come from tools/build_npo_patches.py (border strips of training
images — background regions by construction). Portrait-aspect patches are
resized to full image height and 1/4..1/2 of the width, pasted flush left
or right; landscape-aspect patches to full width and 1/4..1/2 of the
height, pasted flush top or bottom. A mild ColorJitter decorrelates the
patch from its source image.
"""

import glob
import os
import random

from PIL import Image
import torchvision.transforms as T


class RandomNPOPaste:
    def __init__(self, probability, patch_dir):
        paths = sorted(
            glob.glob(os.path.join(patch_dir, '*.jpg'))
            + glob.glob(os.path.join(patch_dir, '*.png')))
        if not paths:
            raise FileNotFoundError(
                'no NPO patches in {!r}; generate them with '
                'tools/build_npo_patches.py'.format(patch_dir))
        self.patches = [Image.open(p).convert('RGB') for p in paths]
        self.probability = probability
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __call__(self, img):
        if random.random() >= self.probability:
            return img
        patch = self.jitter(random.choice(self.patches))
        width, height = img.size
        portrait = patch.height >= patch.width
        if portrait:
            new_size = (random.randint(width // 4, width // 2), height)
            position = (0 if random.random() < 0.5 else width - new_size[0], 0)
        else:
            new_size = (width, random.randint(height // 4, height // 2))
            position = (0, 0 if random.random() < 0.5 else height - new_size[1])
        img = img.copy()
        img.paste(patch.resize(new_size, Image.BILINEAR), position)
        return img
