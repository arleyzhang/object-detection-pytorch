"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import cv2
import torch.utils.data as data
import xml.etree.ElementTree as ET
from lib.data.config import *  # HOME, VARIANCE
from lib.data.det_dataset import DetDataset

# TODO move these global variable
# 20 classes altogether
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used download scripts in data/scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, norm_box=True, class_to_ind=None, keep_difficult=False):
        # TODO use label map
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult
        self.norm_box = norm_box

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1  # TODO one-based annotation?
                if self.norm_box:  # norm box using image height and width
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]  # TODO, from 0-19 ?!
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(DetDataset):
    """VOC Detection Dataset Object

     input is image, target is annotation

     Arguments:
         root (string): filepath to VOCdevkit folder.
         image_sets (list): imageset to use (eg. 'train', 'val', 'test')
         transform (callable, optional): transformation to perform on the
             input image
         target_transform (callable, optional): transformation to perform on the
             target `annotation`
             (eg: take in caption string, return tensor of word indices)
         dataset_name (string, optional): which dataset to load
             (default: 'VOC2007')
     """

    def __init__(self, root,
                 image_sets=(('2007', 'trainval'), ('2012', 'trainval')),
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        super(VOCDetection, self).__init__(root, image_sets, dataset_name, transform, target_transform)
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self._setup()

    def _setup(self):
        for (year, name) in self.image_sets:
            rootpath = osp.join(self.data_root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def _pre_process(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)  # Shape(H, W, C)
        extra = img.shape
        return img, target, extra


def test_loader():
    # TODO: a strange bug: data loader hangs in cpu mode
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specified GPUs range

    dataset = VOCDetection(VOC_ROOT, [('2007', 'test')],
                           SSDAugmentation((300, 300), dataset_mean, use_base=True),
                           VOCAnnotationTransform())
    data_loader = data.DataLoader(dataset, batch_size=32, num_workers=12, shuffle=False,
                                  collate_fn=detection_collate, pin_memory=True)
    for i, (images, targets, extra) in enumerate(data_loader):
        print(i)
        # print(targets)


# TODO make this an API
def test_vis():
    dataset = VOCDetection(VOC_ROOT, [('2007', 'test')],
                           SSDAugmentation((300, 500), dataset_mean),
                           VOCAnnotationTransform())
    from lib.utils.visualize_utils import TBWriter
    log_dir = './experiments/models/ssd_voc/test_pr'
    tb_writer = TBWriter(log_dir, {'vis_list': [3, 4, 5, 6, 8]})

    # import random
    # img_idx = random.randint(0, len(data_loader.dataset)-1)
    # image, target = dataset.pull_item(img_index, tb_writer)
    for img_idx in range(len(dataset)):
        if img_idx > 5:
            break
        tb_writer.cfg['steps'] = img_idx
        image, target, extra = dataset.pull_item(img_idx, tb_writer)
        print(image.shape)


if __name__ == '__main__':
    from lib.data import detection_collate
    from lib.utils import SSDAugmentation

    type = 'test'
    dataset_mean = (104, 117, 123)
    # test_loader()
    test_vis()
