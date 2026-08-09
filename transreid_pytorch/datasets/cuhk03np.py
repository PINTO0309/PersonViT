import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class CUHK03NP(BaseImageDataset):
    """CUHK03-NP (new protocol, detected variant) with the official
    767/700-identity split. Market-style flat directories:

        CUHK03-NP/detected/
        ├── bounding_box_train/
        ├── bounding_box_test/
        └── query/
    """
    dataset_dir = 'CUHK03-NP/detected'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(CUHK03NP, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.pid_begin = pid_begin
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> CUHK03-NP (detected) loaded")
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
                raise RuntimeError("'{}' is not available".format(d))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.png')) +
                           glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(osp.basename(img_path)).groups())
            if pid == -1:
                continue
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
