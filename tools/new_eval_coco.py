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

from data import *
from models.model_build import creat_model
from utils import *


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
parser.add_argument('--batch_size', default=2, type=int,
                    help='Further restrict the batchsize')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Further restrict the batchsize')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset_root', default=COCO_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

###########################################
# test with trained_model
if args.trained_model is None:
    args.trained_model = '../../weights/ssd_coco_eval0710_120000.pth'


#Annotations for crownd #Annotations_src for normal voc

devkit_path = args.dataset_root
set_type = 'minival' #test_full   #test_crowd

CUDA_VISIBLE_DEVICES="2"        #####################Specified GPUs range
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

print ('data_path:', devkit_path, 'test_type:', set_type, 'test_model:', args.trained_model,\
        'device_id:', CUDA_VISIBLE_DEVICES)



if __name__ == '__main__':
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # load net
    cfg = ssd_coco_vgg
    net = creat_model(phase='train', cfg=cfg)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model)['state_dict'])   #model is dict{}
    
    if args.cuda:
        print('cuda----')
        net = torch.nn.DataParallel(net)
        net = net.cuda()
        cudnn.benchmark = False

    val_dataset = COCODetection(args.dataset_root, ['minival'],
                           BaseTransform(300, MEANS),
                           target_transform=COCOAnnotationTransform(False))

    val_loader = data.DataLoader(val_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)
                        
    eval_solver = EvalSolver(val_loader, len(val_dataset), args.save_folder, cfg)
    net.eval()
    mAP = eval_solver.validate(net)
    print("test_res: ", mAP)
