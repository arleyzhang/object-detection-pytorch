"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.autograd import Variable

from lib.data import *
from lib.models.model_factory import model_factory
from lib.utils import *
from lib.utils.utils import EvalVOC, EvalCOCO


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--dataset', default='voc', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--trained_model', default=None, type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./results/debug', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int,
                    help='cpu workers for data processing')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()


if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES = "0,1,2,3"
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for profile
    dataset_mean = (104, 117, 123)
    np.set_printoptions(precision=3, suppress=True, edgeitems=4)

    from lib.utils.visualize_utils import TBWriter
    log_dir = './experiments/models/ssd_voc/test_voc0712'
    tb_writer = TBWriter(log_dir, {'epoch': 50})

    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # cudnn.benchmark = False
        cudnn.deterministic = True
    if not args.cuda:
            torch.set_default_tensor_type('torch.FloatTensor')

    # configs of each dataset
    dataset_name = args.dataset
    if dataset_name == 'voc':
        if args.trained_model is None:
            args.trained_model = './results/ssd300_mAP_77.43_v2.pth'
        cfg = ssd_voc_vgg
        DataDetection = VOCDetection
        data_root = VOC_ROOT
        test_set_name = [('2007', 'test512')]
        anno_trans = VOCAnnotationTransform()
        Solver = EvalVOC
    elif dataset_name == 'coco':
        args.trained_model = './results/vgg16_ssd_coco_24.4.pth'
        cfg = ssd_coco_vgg
        DataDetection = COCODetection
        data_root = COCO_ROOT
        test_set_name = ('minival2014',)
        anno_trans = COCOAnnotationTransform()
        Solver = EvalCOCO
    else:
        raise Exception("Wrong dataset name {}".format(dataset_name))

    # load dataset and dataloader
    dataset = DataDetection(data_root, test_set_name,
                            SSDAugmentation(cfg['image_size'], dataset_mean, use_base=True),
                            anno_trans)  # TODO extra parameter
    loader = data.DataLoader(dataset, args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False, collate_fn=detection_collate,
                                 pin_memory=True)
    print('data points number', len(dataset))

    # load net
    net, priors = model_factory(phase='eval', cfg=cfg)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    if args.cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
        priors = Variable(priors.cuda(), volatile=True)
    else:
        priors = Variable(priors)
    net.eval()

    print('test_type:', test_set_name, 'test_model:', args.trained_model,
          'device_id:', CUDA_VISIBLE_DEVICES)
    tb_writer = None

    eval_solver = Solver(loader, cfg)
    eval_solver.validate(net, priors, tb_writer=tb_writer)

    # if dataset_name == 'voc':
    #     eval_voc(net, priors, cfg, tb_writer)
    # elif dataset_name == 'coco':
    #     eval_coco(net, priors, cfg, tb_writer)
