import sys, os
HOME = os.path.expanduser("~")
import copy
from data import *
import cv2

dataset_mean = (104, 117, 123)
dataset_root = os.path.join(HOME, 'data/coco/')

cfg = coco
dataset = COCODetection(root=dataset_root, image_set=['minival'],
                                transform=BaseTransform(300, dataset_mean),
                                target_transform=COCOAnnotationTransform(False))

im, gt, h, w, _, _ = dataset.pull_item(0)

#cv2.imwrite("../coco.jpg", dataset.pull_image(0))


print(gt.shape, '--hw ', h, w)
for i in range(gt.shape[0]):
    print(gt[i])

im, gt, h, w, _, _ = dataset.pull_item(0)

#cv2.imwrite("../coco.jpg", dataset.pull_image(0))


print(gt.shape, '--hw ', h, w)
for i in range(gt.shape[0]):
    print(gt[i])


def parse_rec(target_):
    """ Parse a COCO target """
    target = copy.deepcopy(target_)
    objects = []
    for obj in target:
        if 'bbox' in obj:
            rec = {}
            bbox = obj['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            rec['name'] = obj['category_id'] - 1
            rec['difficult'] = 0
            rec['bbox'] = [int(bbox[0]) - 1, int(bbox[1]) - 1, int(bbox[2]) - 1, int(bbox[3]) - 1]
            objects.append(rec)
    return objects

index = 0
imagename = dataset.data_set.image_info[index]["id"]
target = dataset.data_set.image_info[index]["annotations"]
#print(target)
print('name:', imagename)
print(parse_rec(target))

print(parse_rec(target))

