"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data

from data import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, BaseTransform, get_label_map
from data import COCO_CLASSES as labelmap
from data.config import *
from models.model_build import creat_model
from utils import *
from layers import *

import sys
import os
import time
import argparse
import numpy as np
import pickle
import copy
import cv2


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

#repul_ssd_VOC_180620_90000
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',default=None, type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='../../weights/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=COCO_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

###########################################
# test with trained_model
if args.trained_model is None:
    args.trained_model = '../../weights/ssd_coco_eval0710_395000.pth'
cfg=ssd_coco_vgg

#Annotations for crownd #Annotations_src for normal voc

devkit_path = args.voc_root
dataset_mean = (104, 117, 123)
set_type = 'minival' #test_full   #test_crowd

CUDA_VISIBLE_DEVICES="6"        #####################Specified GPUs range
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

print ('data_path:', devkit_path, 'test_type:', set_type, 'test_model:', args.trained_model,\
        'device_id:', CUDA_VISIBLE_DEVICES)

if not os.path.exists(args.save_folder):
    print(args.save_folder, "not exsit!!!")

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def parse_rec(target, label_map):
    """ Parse a COCO target """
    #target = copy.deepcopy(target_)
    objects = []
    for obj in target:
        if 'bbox' in obj:
            rec = {}
            bbox_ = obj['bbox']
            rec['name'] =  labelmap[label_map[obj['category_id']] - 1]
            rec['difficult'] = 0
            rec['bbox'] = [int(bbox_[0]) - 1, int(bbox_[1]) - 1, 
                            int(bbox_[2] + bbox_[0]) - 1, int(bbox_[3] + bbox_[1]) - 1]
            objects.append(rec)
    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} COCO results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):  #search all img ids
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                imagename = str(dataset.data_set.image_info[index]["id"])
                #print('test-----', imagename, type(imagename))
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(imagename, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1)) #img_id, label, box


def do_python_eval(output_dir, dataset, use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')#devkit_path=~/data/VOCdevkit/VOC2007/
    label_file = os.path.join(devkit_path, 'coco_labels.txt')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls) #VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        rec, prec, ap = voc_eval(filename, label_file, dataset, cls, cachedir,
                    ovthresh=0.5, use_07_metric=use_07_metric)#results, annoxml, test.txt, cls, cachedir
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):  ###???
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, label_file, dataset,classname, cachedir,
             ovthresh=0.5, use_07_metric=True):
    """
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
    detpath.format(classname) should produce the detection results file.
    imagesetfile: Text file containing the list of images, one image per line.(test.txt)
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
    (default True)
    """
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
  
    if os.path.isfile(cachefile):
         # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    if not os.path.isfile(cachefile):
        label_map = get_label_map(label_file)
        # load annots
        recs = {}
        for i, index in enumerate(dataset.ids):
            imagename = str(dataset.data_set.image_info[index]["id"])
            target = dataset.data_set.image_info[index]["annotations"]
            recs[imagename] = parse_rec(target, label_map)  #is ok
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(dataset.ids)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for index in dataset.ids:
        imagename = str(dataset.data_set.image_info[index]["id"])
        
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)  #detect flag,0:undetected 1:detected
        npos = npos + sum(~difficult)   #only think difficult=0
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets with this class
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:   #all predict results
        lines = f.readlines()
    if any(lines) == 1:
        # format: image_id, conf, box_coord
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]  #image name
        confidence = np.array([float(x[1]) for x in splitlines])    #conf
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  #bbox

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd): #
            R = class_recs[image_ids[d]]    #gt with this classname in this image
            bb = BB[d, :].astype(float)     #predict bbox
            ovmax = -np.inf #init min_val
            BBGT = R['bbox'].astype(float)  #gt box
            if BBGT.size > 0:   #if exist bbox
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih    #a^b
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)  #a v b
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:    #difficult bbox not in calc
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)  #cumulative sum
        tp = np.cumsum(tp)
        rec = tp / float(npos)  #recall
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)   #precision
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('{}/coco300_35k'.format(save_folder), set_type)  #set_type = 'test'
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w, _, _ = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)    #get time for this img

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                              copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir, dataset)

if __name__ == '__main__':
    # load net
    net, layer_dimensions = creat_model(phase='test', cfg=cfg, input_h = 300, input_w = 300)
    priorbox = PriorBox(cfg)
    priors = priorbox.forward(layer_dimensions) #<class 'torch.FloatTensor'>???????

    net.priors = Variable(priors, volatile=True)
    net.load_state_dict(torch.load(args.trained_model)['state_dict'])   #model is dict{}
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = COCODetection(args.voc_root, ['minival'],
                                BaseTransform(300, dataset_mean),
                                COCOAnnotationTransform(False))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = False
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)