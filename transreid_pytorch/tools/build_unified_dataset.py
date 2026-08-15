"""Build the unified `reid` dataset from five real and one optional synthetic domain.

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
        [--test-ratio 0.15] [--synthetic-root data/SyntheticReID33] [--force]
"""

import argparse
import atexit
import hashlib
import os
import random
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # only needed for .tif conversion
    Image = None

try:
    from synth_reid33_core import atomic_write_jsonl, read_jsonl, validate_synthetic_manifest
except ImportError:  # pragma: no cover - only absent when this file is imported unusually
    atomic_write_jsonl = read_jsonl = validate_synthetic_manifest = None


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


SYNTHETIC_EXPECTED = {'train': 16000, 'query': 400, 'gallery': 3600, 'total': 20000}
UNIFIED_WITH_SYNTH_EXPECTED = {'train': 191560, 'query': 5144, 'gallery': 33542}


def collect_synthetic(root):
    """Collect already-split, accepted SyntheticReID33 samples.

    Unlike real-domain adapters this never re-splits identities.  The returned
    records are ``(path, local_pid, local_cam0, explicit_split, sample_id)``.
    """
    if validate_synthetic_manifest is None:
        raise RuntimeError('SyntheticReID33 helpers are unavailable')
    validation = validate_synthetic_manifest(root, SYNTHETIC_EXPECTED)
    if not validation['valid']:
        detail = '\n  - '.join(validation['errors'][:30])
        raise ValueError(f'invalid SyntheticReID33 manifest:\n  - {detail}')
    items = []
    for row, path in validation['rows']:
        items.append((str(path), int(row['local_pid']), int(row['local_camera']),
                      str(row['split']), str(row['sample_id'])))
    return items, [], 33


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def place(src, dst_dir, name_stem):
    """Hardlink (or copy) `src` into `dst_dir`; convert .tif to .jpg."""
    ext = Path(src).suffix.lower()
    if ext in ('.tif', '.tiff', '.bmp'):
        if Image is None:
            sys.exit('Pillow is required to convert .tif images. Install it first.')
        dst = dst_dir / (name_stem + '.jpg')
        Image.open(src).convert('RGB').save(dst, quality=95)
        return
    # The REID loader intentionally supports .jpg/.png only.
    normalized_ext = '.jpg' if ext == '.jpeg' else ext
    dst = dst_dir / (name_stem + normalized_ext)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_unified_output(out, expected_counts=None, expected_cameras=None, check_sha=True):
    """Validate file names, split isolation, cameras and query positives."""
    pattern = re.compile(r'^p(\d{5})_d(\d{2})_c(\d{3})_(\d{6})\.(?:jpg|png)$')
    errors = []
    rows = {}
    seen_seq = set()
    seen_sha = {}
    for split in ('train', 'query', 'gallery'):
        rows[split] = []
        directory = out / split
        if not directory.is_dir():
            errors.append(f'missing split directory: {split}')
            continue
        for path in sorted(directory.iterdir()):
            match = pattern.match(path.name)
            if not match:
                errors.append(f'invalid file name: {path.name}')
                continue
            pid, dom, cam, seq = map(int, match.groups())
            if seq in seen_seq:
                errors.append(f'duplicate sequence id: {seq}')
            seen_seq.add(seq)
            if check_sha:
                digest = _file_sha256(path)
                if digest in seen_sha:
                    errors.append(f'duplicate SHA-256: {path.name} and {seen_sha[digest]}')
                else:
                    seen_sha[digest] = path.name
            rows[split].append((path, pid, dom, cam))
    counts = {split: len(values) for split, values in rows.items()}
    if expected_counts:
        for split, expected in expected_counts.items():
            if split in counts and counts[split] != expected:
                errors.append(f'{split}: got {counts[split]}, expected {expected}')
    train_pids = {pid for _, pid, _, _ in rows['train']}
    test_pids = {pid for split in ('query', 'gallery') for _, pid, _, _ in rows[split]}
    if train_pids & test_pids:
        errors.append('train and evaluation PID sets overlap')
    if train_pids and train_pids != set(range(len(train_pids))):
        errors.append('train PIDs are not contiguous from zero')
    cameras = {cam for split in rows.values() for _, _, _, cam in split}
    if expected_cameras is not None and cameras != set(range(expected_cameras)):
        errors.append(f'camera ids are not contiguous 0..{expected_cameras - 1}')
    galleries_by_pid = defaultdict(list)
    for _, pid, _, cam in rows['gallery']:
        galleries_by_pid[pid].append(cam)
    invalid_queries = 0
    for _, pid, _, cam in rows['query']:
        invalid_queries += not any(gallery_cam != cam for gallery_cam in galleries_by_pid[pid])
    if invalid_queries:
        errors.append(f'{invalid_queries} queries have no cross-camera gallery positive')
    return {
        'valid': not errors,
        'errors': errors,
        'counts': counts,
        'train_pids': len(train_pids),
        'test_pids': len(test_pids),
        'domains': len({dom for split in rows.values() for _, _, dom, _ in split}),
        'cameras': len(cameras),
        'query_valid_rate': 1.0 if not rows['query'] else 1.0 - invalid_queries / len(rows['query']),
    }


def validate_explicit_domain(out, domain=5, camera_start=33, expected=None,
                             expected_train_pids=400, expected_test_pids=100,
                             expected_images_per_pid=40, expected_cameras_per_pid=8,
                             expected_cross_camera_positives=32, expected_camera_count=33):
    """Apply the strict SyntheticReID33 acceptance checks after PID remapping."""
    expected = expected or SYNTHETIC_EXPECTED
    pattern = re.compile(r'^p(\d{5})_d(\d{2})_c(\d{3})_(\d{6})\.(?:jpg|png)$')
    rows = []
    for split in ('train', 'query', 'gallery'):
        for path in (out / split).iterdir():
            match = pattern.match(path.name)
            if match and int(match.group(2)) == domain:
                rows.append((split, int(match.group(1)), int(match.group(3))))
    errors = []
    counts = {split: sum(row[0] == split for row in rows) for split in ('train', 'query', 'gallery')}
    for split in ('train', 'query', 'gallery'):
        if counts[split] != expected[split]:
            errors.append(f'd{domain:02d} {split}: got {counts[split]}, expected {expected[split]}')
    cameras = {camera for _, _, camera in rows}
    if cameras != set(range(camera_start, camera_start + expected_camera_count)):
        errors.append(f'd{domain:02d} cameras are not {camera_start}..'
                      f'{camera_start + expected_camera_count - 1}')
    by_pid = defaultdict(list)
    for row in rows:
        by_pid[row[1]].append(row)
    train_pids = {pid for pid, items in by_pid.items() if any(row[0] == 'train' for row in items)}
    test_pids = set(by_pid) - train_pids
    if len(train_pids) != expected_train_pids or len(test_pids) != expected_test_pids:
        errors.append(f'd{domain:02d} must have {expected_train_pids} train and {expected_test_pids} test identities')
    for pid, items in by_pid.items():
        if len(items) != expected_images_per_pid or len({row[2] for row in items}) != expected_cameras_per_pid:
            errors.append(f'd{domain:02d} pid {pid}: expected {expected_images_per_pid} images and '
                          f'{expected_cameras_per_pid} cameras')
        galleries = [row for row in items if row[0] == 'gallery']
        for query in (row for row in items if row[0] == 'query'):
            positives = sum(gallery[2] != query[2] for gallery in galleries)
            if positives < expected_cross_camera_positives:
                errors.append(f'd{domain:02d} pid {pid}: query has only {positives} cross-camera positives')
    return {'valid': not errors, 'errors': errors, 'counts': counts,
            'pids': len(by_pid), 'cameras': len(cameras)}


def _atomic_replace_directory(staged, out):
    """Switch a validated staged tree into place and restore on swap failure."""
    backup = out.parent / f'.{out.name}.backup-{os.getpid()}'
    if backup.exists():
        raise RuntimeError(f'stale backup blocks atomic swap: {backup}')
    had_old = out.exists()
    if had_old:
        os.replace(out, backup)
    try:
        os.replace(staged, out)
    except BaseException:
        if had_old and backup.exists() and not out.exists():
            os.replace(backup, out)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _update_synthetic_global_pids(synthetic_root, pid_map):
    """Fill global_pid only after the unified tree has switched successfully."""
    if read_jsonl is None:
        return
    manifest = synthetic_root / 'manifest.jsonl'
    rows = read_jsonl(manifest)
    for row in rows:
        row['global_pid'] = pid_map[int(row['local_pid'])]
    atomic_write_jsonl(manifest, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parent.parent / 'data'))
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--test-ratio', type=float, default=0.15)
    ap.add_argument('--synthetic-root', type=Path,
                    help='accepted SyntheticReID33 root; preserves its explicit train/query/gallery split')
    ap.add_argument('--force', action='store_true', help='rebuild even if data/reid exists')
    ap.add_argument('--skip-sha-validation', action='store_true',
                    help='skip the expensive all-domain duplicate check (not recommended)')
    ap.add_argument('--allow-source-count-drift', action='store_true',
                    help='accept real-domain counts other than this repository baseline')
    args = ap.parse_args()

    root = Path(args.data_root)
    out = root / 'reid'
    if out.exists():
        if not args.force:
            sys.exit(f'{out} already exists. Use --force to rebuild.')
    staged = Path(tempfile.mkdtemp(prefix='.reid-building-', dir=root.resolve()))
    atexit.register(lambda: shutil.rmtree(staged) if staged.exists() else None)
    for split in ('train', 'query', 'gallery'):
        (staged / split).mkdir(parents=True)

    rng = random.Random(args.seed)

    # 1) Collect all sources ------------------------------------------------
    domains = []
    cam_offset = 0
    for dom, adapter in enumerate(ADAPTERS):
        items, distractors, num_cams = adapter(root)
        domains.append({'items': items, 'distractors': distractors,
                        'cam_offset': cam_offset, 'num_cams': num_cams,
                        'explicit': False})
        cam_offset += num_cams
    synthetic_root = args.synthetic_root.resolve() if args.synthetic_root else None
    if synthetic_root:
        items, distractors, num_cams = collect_synthetic(synthetic_root)
        domains.append({'items': items, 'distractors': distractors,
                        'cam_offset': cam_offset, 'num_cams': num_cams,
                        'explicit': True})
        cam_offset += num_cams

    # 2) Identity-disjoint split per domain --------------------------------
    for dom, d in enumerate(domains):
        if d['explicit']:
            train_pids = {pid for _, pid, _, split, _ in d['items'] if split == 'train'}
            test_pids = {pid for _, pid, _, split, _ in d['items'] if split in ('query', 'gallery')}
            if train_pids & test_pids:
                raise ValueError('synthetic train/test PID sets overlap')
            d['train_pids'] = sorted(train_pids)
            d['test_pids'] = sorted(test_pids)
        else:
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
        place(src, staged / split, f'p{pid:05d}_d{dom:02d}_c{cam:03d}_{seq:06d}')
        seq += 1
        stats[dom][split] += 1

    for dom, d in enumerate(domains):
        off = d['cam_offset']
        if d['explicit']:
            for path, pid, cam, split, _sample_id in sorted(d['items'], key=lambda item: item[4]):
                emit(split, path, gpid[(dom, pid)], dom, cam + off)
            continue
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

    # 5) Validate staging and switch atomically ----------------------------
    total = {s: sum(stats[dom][s] for dom in stats) for s in ('train', 'query', 'gallery')}
    if synthetic_root and not args.allow_source_count_drift and total != UNIFIED_WITH_SYNTH_EXPECTED:
        shutil.rmtree(staged)
        raise ValueError(f'unified counts {total} differ from locked expectation {UNIFIED_WITH_SYNTH_EXPECTED}')
    validation = validate_unified_output(
        staged, expected_counts=total, expected_cameras=cam_offset,
        check_sha=not args.skip_sha_validation)
    if synthetic_root:
        explicit_validation = validate_explicit_domain(
            staged, domain=len(domains) - 1, camera_start=domains[-1]['cam_offset'])
        validation['errors'].extend(explicit_validation['errors'])
        validation['valid'] = validation['valid'] and explicit_validation['valid']
    if not validation['valid']:
        detail = '\n  - '.join(validation['errors'][:50])
        shutil.rmtree(staged)
        raise ValueError(f'staged unified dataset failed validation:\n  - {detail}')
    _atomic_replace_directory(staged, out)
    if synthetic_root:
        synth_dom = len(domains) - 1
        synth_pid_map = {local_pid: global_pid for (dom, local_pid), global_pid in gpid.items()
                         if dom == synth_dom}
        _update_synthetic_global_pids(synthetic_root, synth_pid_map)

    # 6) Report ------------------------------------------------------------
    print(f'\nUnified dataset written to {out}')
    print(f'  global train ids : {num_train_pids}')
    print(f'  global test ids  : {next_pid - num_train_pids} (+1 shared distractor pid)')
    print(f'  global cameras   : {cam_offset}')
    print('  domain | train_ids | train | query | gallery')
    for dom, d in enumerate(domains):
        print('  d{:02d}    | {:9d} | {:5d} | {:5d} | {:7d}'.format(
            dom, len(d['train_pids']), stats[dom]['train'],
            stats[dom]['query'], stats[dom]['gallery']))
    print('  total  |           | {:5d} | {:5d} | {:7d}'.format(
        total['train'], total['query'], total['gallery']))
    print(f"  validator: query_valid_rate={validation['query_valid_rate']:.3f}, "
          f"domains={validation['domains']}, cameras={validation['cameras']}")


if __name__ == '__main__':
    main()
