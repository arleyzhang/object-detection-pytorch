"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import *   #HOME, VARIANCE
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
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

    def __init__(self, is_scale=True, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult
        self.is_scale = is_scale

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
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height and width
                if self.is_scale:
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    # VOCDetection(root=args.dataset_root, #root='../../data/VOCdevkit',
    #              transform=SSDAugmentation(cfg['min_dim'],    #cfg['min_dim']=300
    #              MEANS))
    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712', priors=None):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()   #record img_id
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        #if not priors is None:
        self.priors = priors
    
    def __getitem__(self, index):
        im, gt, h, w, loc_t, conf_t = self.pull_item(index)
        return im, gt, h, w, loc_t, conf_t

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()    #read xml
        #print (target, '================')
        img = cv2.imread(self._imgpath % img_id)    #Shape(H, W, C)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)   ##############bug if target==0
            #print (target.shape, '================')
            if target.size == 0:
                img, boxes, labels = self.transform(img)
            else:
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        if not self.priors is None:
            num_priors = self.priors.shape[0]
            loc_t = np.zeros([num_priors, 4])
            conf_t =  np.zeros([num_priors])
            truths = target[:, :-1]
            labels = target[:, -1]

            #match_timer = Timer()
            #match_timer.tic()
            match_ssd(0.5, truths, self.priors, VARIANCE, labels, loc_t, conf_t)
            #match_timer.toc()
            #print('debug----- match_time:', match_timer.diff)
            
            return torch.from_numpy(img).permute(2, 0, 1), target, height, width, loc_t, conf_t
        else:
            return torch.from_numpy(img).permute(2, 0, 1), target, height, width, -1, -1


def match_ssd(threshold, truths, priors, variances, labels, loc_t, conf_t):
    #print("debug match_ssd in voc0712.py----------------------")
    # jaccard index
    priors = point_form(priors)
    overlaps = np.zeros([len(truths), len(priors)])
    for gt_idx in range(len(truths)):
        for obj_idx in range(len(priors)):
            overlaps[gt_idx, obj_idx] = jaccard(truths[gt_idx], priors[obj_idx])

    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap = overlaps.max(1, keepdims=True)
    best_prior_idx = overlaps.argmax(1)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap = overlaps.max(0, keepdims=True)
    best_truth_idx = overlaps.argmax(0)

    # print('debug shape-----', best_prior_overlap.shape, best_prior_idx.shape, 
    #     best_truth_overlap.shape, best_truth_idx.shape)
    
    for idx in best_prior_idx:  #num_objects
        best_truth_overlap[:,idx] = 2
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.shape[0]):    #num_priors
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    for i, j in enumerate(best_truth_idx):
        conf_t[i] = labels[j] + 1         # Shape: [num_priors]-------------------->
    best_truth_overlap = np.squeeze(best_truth_overlap, axis=0)  #(8732, )
    #print(best_prior_overlap.shape, ' --------------')
    conf_t[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t = loc    # [num_priors,4] encoded offsets to learn
    
    pos_idx = conf_t > 0
    #print('\n  loc-----', matches[pos_idx], loc[pos_idx], priors[pos_idx])
    #conf_t = conf  # [num_priors] top class label for each prior

def point_form(boxes):
    return np.concatenate((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

def jaccard(b1, b2):
    b1_w = b1[2] - b1[0]
    b1_h = b1[3] - b1[1]
    b2_w = b2[2] - b2[0]
    b2_h = b2[3] - b2[1]
    x1 = np.max([b1[0], b2[0]])
    y1 = np.max([b1[1], b2[1]])
    x2 = np.min([b1[2], b2[2]]) 
    #[b1[0] + b1[2], b2[0] + b2[2]]    
    y2 = np.min([b1[3], b2[3]])
    w = np.max([0, x2 - x1])
    h = np.max([0, y2 - y1])
    iou_val = 0
    if w != 0 and h != 0:
        iou_val = float(w * h) / (b1_w * b1_h + b2_w * b2_h - w * h)
    return iou_val

def encode(matched, priors, variances):
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]    #matched_cxy - priors_cxy
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = np.log(g_wh + 1e-10) / variances[1]
    # return target for smooth_l1_loss
    return np.concatenate((g_cxcy, g_wh), 1)  # [num_priors,4]
