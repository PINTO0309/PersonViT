"""Build the unified `reid` dataset from the five source datasets under data/.

The five source datasets are fully merged into a single identity-disjoint
train / query / gallery split. Every generated file name is neutral and keeps
no trace of the source dataset names:

    data/reid/
    ├── train/    p{pid:05d}_d{dom:02d}_c{cam:03d}_{seq:06d}.{jpg|png}
    ├── query/
    └── gallery/

- `pid`  : global person id, contiguous. Train ids come first (0..N_train-1)
           so the loader can use them directly as classifier labels.
- `dom`  : anonymous domain id (d00..d04). Used by the domain-balanced
           sampler to keep every mini batch mixed across domains.
- `cam`  : global camera id (per-domain offset, 0-based). Used by the
           standard cross-camera evaluation protocol.
- `seq`  : global running index that guarantees unique file names.

Split policy (identity-disjoint, per domain, seeded):
- `--test-ratio` (default 0.15) of the identities of each domain are held
  out for evaluation; the rest go to train. Stratifying per domain keeps
  every domain represented in both splits.
- For each test identity: per camera with >= 2 images, one random image
  becomes a query and the rest go to the gallery; single-image cameras go
  to the gallery. Identities seen by a single camera contribute
  gallery-only distractors. If an identity spans >= 2 cameras but every
  camera has a single image, one image is promoted to query so the
  identity is still evaluated.
- Distractor images without a person identity are placed in the gallery
  under one shared reserved pid that never occurs in queries.

Images are hardlinked when possible (no extra disk usage, no reference to
the source path); `.tif` sources are converted to `.jpg`.

Usage:
    python tools/build_unified_dataset.py [--data-root data] [--seed 1234]
                                          [--test-ratio 0.15] [--force]
"""

import argparse
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # only needed for .tif conversion
    Image = None


# ---------------------------------------------------------------------------
# Source dataset adapters.
#
# Each adapter returns:
#   items       : list of (abs_path, local_pid, local_cam0)   real identities
#   distractors : list of (abs_path, local_cam0)              no identity
#   num_cams    : number of cameras of this source
#
# The order of ADAPTERS defines the anonymous domain id (d00, d01, ...).
# ---------------------------------------------------------------------------

_PID_CAM_RE = re.compile(r'^(-?\d+)_c(\d+)')


def _collect_flat(dirs):
    """Collect (path, pid, cam0) from Market/Duke/CUHK style flat dirs."""
    items, distractors = [], []
    for d in dirs:
        for f in sorted(Path(d).iterdir()):
            if f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            m = _PID_CAM_RE.match(f.name)
            if m is None:
                continue
            pid, cam = int(m.group(1)), int(m.group(2)) - 1
            if pid == -1:
                continue                      # junk images: dropped
            if pid == 0:
                distractors.append((str(f), cam))   # gallery-only distractors
            else:
                items.append((str(f), pid, cam))
    return items, distractors


def collect_cuhk(root):
    # The `detected` and `labeled` variants contain the same shots with
    # different bounding boxes; use the harder `detected` variant only to
    # avoid near-duplicate images.
    base = root / 'CUHK03-NP' / 'detected'
    items, distractors = _collect_flat(
        [base / 'bounding_box_train', base / 'bounding_box_test', base / 'query'])
    return items, distractors, 2


def collect_msmt(root):
    base = root / 'MSMT17_V1'
    items = []
    for split in ('train', 'test'):
        for pid_dir in sorted((base / split).iterdir()):
            if not pid_dir.is_dir():
                continue
            pid = int(pid_dir.name)
            for f in sorted(pid_dir.iterdir()):
                if f.suffix.lower() != '.jpg':
                    continue
                cam = int(f.name.split('_')[2]) - 1
                # pids restart from 0 in both `train` and `test`; offset the
                # test namespace so identities stay distinct.
                items.append((str(f), pid + (10000 if split == 'test' else 0), cam))
    return items, [], 15


def collect_market(root):
    base = root / 'Market-1501-v15.09.15'
    # gt_bbox/gt_query duplicate the same frames and are skipped.
    items, distractors = _collect_flat(
        [base / 'bounding_box_train', base / 'bounding_box_test', base / 'query'])
    return items, distractors, 6


def collect_occ_duke(root):
    base = root / 'Occluded-DukeMTMC'
    items, distractors = _collect_flat(
        [base / 'bounding_box_train', base / 'bounding_box_test', base / 'query'])
    return items, distractors, 8


def collect_occ_reid(root):
    base = root / 'Occluded_REID'
    items = []
    # No real camera labels; treat occluded / whole capture setups as two
    # pseudo cameras so the cross-camera protocol remains meaningful
    # (occluded queries are matched against whole-body gallery images).
    for cam, sub in enumerate(('occluded_body_images', 'whole_body_images')):
        for pid_dir in sorted((base / sub).iterdir()):
            if not pid_dir.is_dir():
                continue
            pid = int(pid_dir.name)
            for f in sorted(pid_dir.iterdir()):
                if f.suffix.lower() not in ('.tif', '.tiff', '.jpg', '.png', '.bmp'):
                    continue
                items.append((str(f), pid, cam))
    return items, [], 2


ADAPTERS = [collect_cuhk, collect_msmt, collect_market, collect_occ_duke, collect_occ_reid]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def place(src, dst_dir, name_stem):
    """Hardlink (or copy) `src` into `dst_dir`; convert .tif to .jpg."""
    ext = Path(src).suffix.lower()
    if ext in ('.tif', '.tiff'):
        if Image is None:
            sys.exit('Pillow is required to convert .tif images. Install it first.')
        dst = dst_dir / (name_stem + '.jpg')
        Image.open(src).convert('RGB').save(dst, quality=95)
        return
    dst = dst_dir / (name_stem + ext)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parent.parent / 'data'))
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--test-ratio', type=float, default=0.15)
    ap.add_argument('--force', action='store_true', help='rebuild even if data/reid exists')
    args = ap.parse_args()

    root = Path(args.data_root)
    out = root / 'reid'
    if out.exists():
        if not args.force:
            sys.exit(f'{out} already exists. Use --force to rebuild.')
        shutil.rmtree(out)
    for split in ('train', 'query', 'gallery'):
        (out / split).mkdir(parents=True)

    rng = random.Random(args.seed)

    # 1) Collect all sources ------------------------------------------------
    domains = []
    cam_offset = 0
    for dom, adapter in enumerate(ADAPTERS):
        items, distractors, num_cams = adapter(root)
        domains.append({'items': items, 'distractors': distractors,
                        'cam_offset': cam_offset, 'num_cams': num_cams})
        cam_offset += num_cams

    # 2) Identity-disjoint split per domain --------------------------------
    for dom, d in enumerate(domains):
        pids = sorted({pid for _, pid, _ in d['items']})
        n_test = max(1, round(len(pids) * args.test_ratio))
        test_pids = set(rng.sample(pids, n_test))
        d['train_pids'] = [p for p in pids if p not in test_pids]
        d['test_pids'] = sorted(test_pids)

    # 3) Assign global pids: train ids first (contiguous classifier labels)
    gpid = {}
    next_pid = 0
    for dom, d in enumerate(domains):
        for pid in d['train_pids']:
            gpid[(dom, pid)] = next_pid
            next_pid += 1
    num_train_pids = next_pid
    for dom, d in enumerate(domains):
        for pid in d['test_pids']:
            gpid[(dom, pid)] = next_pid
            next_pid += 1
    distractor_pid = next_pid  # single shared pid, never used by queries

    # 4) Write files --------------------------------------------------------
    seq = 0
    stats = defaultdict(lambda: defaultdict(int))

    def emit(split, src, pid, dom, cam):
        nonlocal seq
        place(src, out / split, f'p{pid:05d}_d{dom:02d}_c{cam:03d}_{seq:06d}')
        seq += 1
        stats[dom][split] += 1

    for dom, d in enumerate(domains):
        off = d['cam_offset']
        by_pid = defaultdict(list)
        for path, pid, cam in d['items']:
            by_pid[pid].append((path, cam + off))

        for pid in d['train_pids']:
            for path, cam in by_pid[pid]:
                emit('train', path, gpid[(dom, pid)], dom, cam)

        for pid in d['test_pids']:
            by_cam = defaultdict(list)
            for path, cam in by_pid[pid]:
                by_cam[cam].append(path)
            g = gpid[(dom, pid)]
            n_query = 0
            if len(by_cam) < 2:
                # single-camera identity: cross-camera matching is impossible,
                # keep the images as gallery distractors
                for cam, paths in by_cam.items():
                    for p in paths:
                        emit('gallery', p, g, dom, cam)
                continue
            promoted = False
            for cam in sorted(by_cam):
                paths = by_cam[cam][:]
                rng.shuffle(paths)
                if len(paths) >= 2:
                    emit('query', paths[0], g, dom, cam)
                    n_query += 1
                    rest = paths[1:]
                elif n_query == 0 and not promoted and cam == max(by_cam):
                    # every camera holds one image: promote the last one so
                    # the identity is still evaluated
                    emit('query', paths[0], g, dom, cam)
                    n_query += 1
                    promoted = True
                    rest = []
                else:
                    rest = paths
                for p in rest:
                    emit('gallery', p, g, dom, cam)

        for path, cam in d['distractors']:
            emit('gallery', path, distractor_pid, dom, cam + off)

    # 5) Report (stdout only; nothing dataset-identifying is persisted) ----
    print(f'\nUnified dataset written to {out}')
    print(f'  global train ids : {num_train_pids}')
    print(f'  global test ids  : {next_pid - num_train_pids} (+1 shared distractor pid)')
    print(f'  global cameras   : {cam_offset}')
    print('  domain | train_ids | train | query | gallery')
    for dom, d in enumerate(domains):
        print('  d{:02d}    | {:9d} | {:5d} | {:5d} | {:7d}'.format(
            dom, len(d['train_pids']), stats[dom]['train'],
            stats[dom]['query'], stats[dom]['gallery']))
    total = {s: sum(stats[dom][s] for dom in stats) for s in ('train', 'query', 'gallery')}
    print('  total  |           | {:5d} | {:5d} | {:7d}'.format(
        total['train'], total['query'], total['gallery']))


if __name__ == '__main__':
    main()
