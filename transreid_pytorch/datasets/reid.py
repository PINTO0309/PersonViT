import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class REID(BaseImageDataset):
    """Unified multi-domain ReID dataset built by tools/build_unified_dataset.py.

    Layout:
        reid/
        ├── train/    p{pid:05d}_d{dom:02d}_c{cam:03d}_{seq:06d}.{jpg|png}
        ├── query/
        └── gallery/

    pid is a global person id (train ids are contiguous from 0), dom is an
    anonymous domain id and cam is a global 0-based camera id. The domain id
    is exposed through the view/track slot of the sample tuple: the model
    ignores it unless SIE_VIEW is enabled, while the domain-balanced sampler
    reads it to keep batches mixed across domains.
    """
    dataset_dir = 'reid'

    _pattern = re.compile(r'p(\d+)_d(\d+)_c(\d+)_(\d+)')

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(REID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.pid_begin = pid_begin
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Unified reid dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError("'{}' is not available. Run "
                                   "tools/build_unified_dataset.py first.".format(d))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')) +
                           glob.glob(osp.join(dir_path, '*.png')))
        dataset = []
        pid_container = set()
        for img_path in img_paths:
            pid, _, _, _ = map(int, self._pattern.search(osp.basename(img_path)).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        for img_path in img_paths:
            pid, dom, camid, _ = map(int, self._pattern.search(osp.basename(img_path)).groups())
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, dom))
        return dataset
