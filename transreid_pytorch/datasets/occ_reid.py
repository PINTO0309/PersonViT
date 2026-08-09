import glob
import os.path as osp

from .bases import BaseImageDataset


class OccludedREID(BaseImageDataset):
    """Occluded-REID with the standard evaluation protocol: the occluded
    captures are the queries and the whole-body captures form the gallery
    (treated as two pseudo cameras so cross-camera matching applies). The
    dataset defines no training split; `train` is empty and this dataset is
    evaluation-only.

        Occluded_REID/
        ├── occluded_body_images/<pid>/*.tif   -> query  (camid 0)
        └── whole_body_images/<pid>/*.tif      -> gallery (camid 1)
    """
    dataset_dir = 'Occluded_REID'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(OccludedREID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'whole_body_images')
        self.pid_begin = pid_begin
        self._check_before_run()

        train = []
        query = self._process_dir(self.query_dir, camid=0)
        gallery = self._process_dir(self.gallery_dir, camid=1)

        if verbose:
            print("=> Occluded-REID loaded (evaluation-only protocol)")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        for d in (self.dataset_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError("'{}' is not available".format(d))

    def _process_dir(self, dir_path, camid):
        dataset = []
        for pid_dir in sorted(glob.glob(osp.join(dir_path, '*'))):
            if not osp.isdir(pid_dir):
                continue
            pid = int(osp.basename(pid_dir))
            for img_path in sorted(
                    glob.glob(osp.join(pid_dir, '*.tif')) +
                    glob.glob(osp.join(pid_dir, '*.jpg')) +
                    glob.glob(osp.join(pid_dir, '*.png'))):
                dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
