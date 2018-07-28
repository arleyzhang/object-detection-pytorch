"""Adapted from:
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import argparse
import os.path as osp

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.autograd import Variable

from lib.datasets import *
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
                    help='cpu workers for datasets processing')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if __name__ == '__main__':
    from lib.utils.config import cfg, merge_cfg_from_file
    from lib.utils.visualize_utils import TBWriter

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1,3,4,5'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for profile
    np.set_printoptions(precision=3, suppress=True, edgeitems=4)

    log_dir = osp.join(osp.join(cfg.LOG.ROOT_DIR, 'voc'))
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
            # args.trained_model = './weights/ssd_VOC_reference_10.pth'
            args.trained_model = './weights/zuwei/120000.pth'
            args.trained_model = './weights/maolei/ssd_voc_newbox_w2.1_0721_120000.pth'
            # args.trained_model = '/mnt/sdd1/zhicheng/workspace/torch/origin/ssd.pytorch/weights/ssd300_COCO_55000.pth'
        DataDetection = VOCDetection
        anno_trans = VOCAnnotationTransform()
        Solver = EvalVOC
    elif dataset_name == 'coco':
        cfg_path = osp.join(cfg.CFG_ROOT, 'coco.yml')
        merge_cfg_from_file(cfg_path)
        args.trained_model = './results/vgg16_ssd_coco_24.4.pth'
        DataDetection = COCODetection
        anno_trans = COCOAnnotationTransform()
        Solver = EvalCOCO
    else:
        raise Exception("Wrong dataset name {}".format(dataset_name))

    # load dataset and dataloader
    dataset = DataDetection(cfg.DATASET.DATASET_DIR, cfg.DATASET.TEST_SETS,
                            SSDAugmentation(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS, use_base=True),
                            anno_trans)
    loader = data.DataLoader(dataset, batch_size=cfg.DATASET.EVAL_BATCH_SIZE,
                             num_workers=cfg.DATASET.NUM_WORKERS, shuffle=False,
                             collate_fn=detection_collate, pin_memory=True)

    # load net
    net, priors, _ = model_factory(phase='eval', cfg=cfg)
    # net.load_state_dict(torch.load(args.trained_model))
    net.load_state_dict(torch.load(args.trained_model)['state_dict'])
    if args.cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
        priors = Variable(priors.cuda(), volatile=True)
    else:
        priors = Variable(priors)
    net.eval()

    print('test_type:', cfg.DATASET.TEST_SETS, 'test_model:', args.trained_model,
          'device_id:', cfg.CUDA_VISIBLE_DEVICES)

    tb_writer = None
    # dataset.ids = dataset.ids[:64]
    eval_solver = Solver(loader, cfg)
    eval_solver.validate(net, priors, tb_writer=tb_writer)
