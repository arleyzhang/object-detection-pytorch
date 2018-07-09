from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import *
import torch.utils.data as data
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='../../data/weights/repul_ssd_VOC_20000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    gt_filename = save_folder+'gt.txt'
    pd_filename = save_folder+'pred.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(gt_filename, mode='a') as f:
            f.write(img_id+' ')
            for box in annotation:
                f.write(' '.join(str(b) for b in box)+' ')
            f.write('\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                if pred_num == 0:
                    with open(pd_filename, mode='a') as f:
                        f.write(img_id+' ')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(pd_filename, mode='a') as f:
                    f.write(str(i-1) + ' ' + str(score) + ' ' +' '.join(str(c) for c in coords)+' ')                
                j += 1
        with open(pd_filename, mode='a') as f:
            f.write('\n')

def test_voc():
    # load net
    # num_classes = len(VOC_CLASSES) + 1 # +1 background
    cfg = voc
    net = build_ssd('test', cfg) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
Traceback (most recent call last):
  File "train.py", line 56, in <module>
    args = parser.parse_args()
  File "/home/maolei/anaconda2/envs/python35/lib/python3.5/argparse.py", line 1729, in parse_args
    self.error(msg % ' '.join(argv))
  File "/home/maolei/anaconda2/envs/python35/lib/python3.5/argparse.py", line 2383, in error
    self.print_usage(_sys.stderr)
  File "/home/maolei/anaconda2/envs/python35/lib/python3.5/argparse.py", line 2353, in print_usage
    self._print_message(self.format_usage(), file)
  File "/home/maolei/anaconda2/envs/python35/lib/python3.5/argparse.py", line 2319, in format_usage
    return formatter.format_help()
  File "/home/maolei/anaconda2/envs/python35/lib/python3.5/argparse.py", line 278, in format_help
    help = self._root_section.format_help()
  File "/home/maolei/anaconda2/envs/python35/lib/python3.5/argparse.py", line 208, in format_help
    func(*args)
  File "/home/maolei/anaconda2/envs/python35/lib/python3.5/argparse.py", line 316, in _format_usage
    action_usage = format(optionals + positionals, groups)
  File "/home/maolei/anaconda2/envs/python35/lib/python3.5/argparse.py", line 387, in _format_actions_usage
    start = actions.index(group._group_actions[0])
IndexError: list index out of range
