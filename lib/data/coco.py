import os
import os.path as osp
import numpy as np

import torch.utils.data as data
import cv2
from pycocotools.coco import COCO

from lib.data.config import HOME
from lib.data.det_dataset import DetDataset

COCO_ROOT = osp.join(HOME, 'data/coco/')
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')

"""
coco minival urls
https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0
https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0
"""


def get_label_map(label_file):
    if not os.path.isfile(label_file):
        raise Exception("No coco label file")
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self, norm_box=True):
        self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))
        self.norm_box = norm_box

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                # origin box in xmin, ymin, width, height
                bbox = obj['bbox']
                bbox_ = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]

                label_idx = self.label_map[obj['category_id']] - 1  # TODO 0~79?!
                if self.norm_box:
                    final_box = list(np.array(bbox_) / scale)  # scale to [0, 1]
                else:
                    final_box = list(np.array(bbox_))
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                # TODO solve this problem
                print("in training coco: no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(DetDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        image_sets (tuple): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_sets=('train2014', 'valminusminival2014'), transform=None,
                 target_transform=COCOAnnotationTransform(), dataset_name='MS COCO'):
        # no same id in coco train and val
        super(COCODetection, self).__init__(root, image_sets, dataset_name,
                                               transform, target_transform)
        self.size_cum = np.array([])
        self.cocos = []
        self._setup()

    def _setup(self):
        # we need to deal with valminusminival and minival
        size_list = []
        for i, set_name in enumerate(self.image_sets):
            folder_name = 'val2014' if 'val' in set_name else 'train2014'
            self.cocos.append({'root': osp.join(self.data_root, IMAGES, folder_name),
                               'coco': COCO(osp.join(self.data_root, ANNOTATIONS,
                                                     INSTANCES_SET.format(set_name)))})
            # keys = self.cocos[i]['coco'].getImgIds()
            keys = list(self.cocos[i]['coco'].imgToAnns.keys())
            size_list.append(len(keys))  # only save images with ground truth
            self.ids += keys
        self.size_cum = np.cumsum(size_list)  # use cumsum to determine which dataset id in

    def _pre_process(self, index):
        img_id = self.ids[index]
        set_id = int(np.argmax(index < self.size_cum))  # not support index>size_cum[-1]
        target = self.cocos[set_id]['coco'].imgToAnns[img_id]  # list of annos
        f_n = self.cocos[set_id]['coco'].loadImgs(img_id)[0]['file_name']  # image name
        path = osp.join(self.cocos[set_id]['root'], f_n)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(osp.join(self.cocos[set_id]['root'], path))
        extra = img.shape
        return img, target, extra


def test_loader():
    # TODO: a strange bug: data loader hangs in cpu mode
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specified GPUs range
    # ('val2014', 'minival2014','valminusminival2014', 'train2014')
    dataset = COCODetection(COCO_ROOT, ('val2014',),
                               SSDAugmentation((300, 300), dataset_mean, use_base=True),
                               COCOAnnotationTransform())
    data_loader = data.DataLoader(dataset, batch_size=24, num_workers=24, shuffle=False,
                                  collate_fn=detection_collate, pin_memory=True)

    print(len(dataset))
    for i, (images, targets) in enumerate(data_loader):
        print(i)


def test_vis():
    dataset = COCODetection(COCO_ROOT, ('valminusminival2014',),
                               SSDAugmentation((300, 300), dataset_mean, use_base=True),
                               COCOAnnotationTransform())
    from lib.utils.visualize_utils import TBWriter
    from tensorboardX import SummaryWriter
    log_dir = './experiments/models/ssd_voc/test_vis_coco'
    writer = SummaryWriter(log_dir=log_dir)
    cfg = {'vis_list': [3, 4, 5, 6, 8]}
    tb_writer = TBWriter(writer, cfg)

    # import random
    # img_idx = random.randint(0, len(dataset)-1)
    # image, target = dataset.pull_item(img_idx, tb_writer)

    for img_idx in range(len(dataset)):
        if img_idx > 5:
            break
        tb_writer.cfg['steps'] = img_idx
        image, target = dataset.pull_item(img_idx, tb_writer)
        print(image.shape)


if __name__ == '__main__':
    from data import detection_collate
    from utils import SSDAugmentation

    type = 'test'
    dataset_mean = (104, 117, 123)
    test_loader()
    test_vis()
