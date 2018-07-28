from lib.datasets import *
from lib.models.model_factory import model_factory
from lib.utils.augmentations import SSDAugmentation
from lib.layers.modules import MultiBoxLoss
import os
import os.path as osp
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
from lib.utils.visualize_utils import *
from lib.layers import *
import random

from lib.utils.config import cfg, merge_cfg_from_file
from lib.utils.visualize_utils import TBWriter


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')  # vgg16_reducedfc
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--tensorboard', default=False, type=str2bool,
                    help='Use tensorboard')
parser.add_argument('--save_folder', default='weights/reference/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--loss_type', default='ssd_loss', type=str,
                    help='ssd_loss or repul_loss')
parser.add_argument('--log_dir', default='./experiments/models/ssd_voc', type=str,
                    help='tensorboard log_dir')
args = parser.parse_args()

snapshot_prefix = 'ssd_VOC_reference_'
step_index = 0  # TODO need put in args ???

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
log_dir = osp.join(osp.join(cfg.LOG.ROOT_DIR, 'voc'))
tb_writer = TBWriter(log_dir, {'epoch': 50})

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):  # snapshot and save_model
    os.mkdir(args.save_folder)


def train():
    global step_index

    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        cudnn.benchmark = False
        # cudnn.deterministic = True
    if not args.cuda:
        torch.set_default_tensor_type('torch.FloatTensor')

    # configs of each dataset
    dataset_name = cfg.DATASET.NAME
    if dataset_name == 'VOC0712':
        DataDetection = VOCDetection
        anno_trans = VOCAnnotationTransform()
    elif dataset_name == 'COCO2014':
        cfg_path = osp.join(cfg.CFG_ROOT, 'coco.yml')
        merge_cfg_from_file(cfg_path)
        DataDetection = COCODetection
        anno_trans = COCOAnnotationTransform()
    else:
        raise Exception("Wrong dataset name {}".format(dataset_name))

    # load dataset and dataloader
    dataset = DataDetection(cfg.DATASET.DATASET_DIR,
                            cfg.DATASET.TRAIN_SETS,
                            SSDAugmentation(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS))
    data_loader = data.DataLoader(dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE,
                                  num_workers=cfg.DATASET.NUM_WORKERS,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    if args.tensorboard:
        pass

    ssd_net, priors, _ = model_factory(phase='eval', cfg=cfg)
    net = ssd_net  # net is for the parallel version of ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        priors = Variable(priors.cuda(), volatile=True)
    else:
        priors = Variable(priors)

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_iter = checkpoint['iteration']
        step_index = checkpoint['step_index']
        ssd_net.load_state_dict(checkpoint['state_dict'])
        raise Exception()
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    if args.cuda:
        net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=cfg.TRAIN.OPTIMIZER.LR,
                          momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
                          weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)

    criterion = MultiBoxLoss(cfg.MODEL.NUM_CLASSES, 0.5, True, 0, True, 3, 0.5,
                                False, args.cuda)

    if args.start_iter not in cfg.TRAIN.LR_SCHEDULER.STEPS and step_index != 0:  # consider 8w, 12w...
        adjust_learning_rate(optimizer, cfg.TRAIN.LR_SCHEDULER.GAMMA, step_index)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0

    epoch_size = len(dataset) // cfg.DATASET.TRAIN_BATCH_SIZE
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg.TRAIN.MAX_ITER):
        # tensorboard vis every epoch
        if args.tensorboard and iteration != 0 and (iteration % epoch_size == 0):
            # visualize_epoch(net, visualize_loader, priorbox, writer, epoch)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg.TRAIN.LR_SCHEDULER.STEPS:
            step_index += 1
            adjust_learning_rate(optimizer, cfg.TRAIN.LR_SCHEDULER.GAMMA, step_index)

        # load train data
        try:
            images, targets, extra = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets, extra = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        out1 = [out[0], out[1], priors]
        # backprop
        optimizer.zero_grad()

        if args.loss_type == 'ssd_loss':
            loss_l, loss_c = criterion(out1, targets)
            loss = loss_l + loss_c
        else:
            raise Exception()
        loss.backward()
        optimizer.step()

        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0), '  lr: ', optimizer.param_groups[0]['lr'])

            if args.loss_type == 'ssd_loss':
                print('Iteration ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]) + \
                      ' || conf_loss: %.4f' % (loss_c.data[0]) + ' || smoothl1 loss: %.4f ' % (loss_l.data[0]), \
                      end=' ')
            if args.tensorboard:
                # log for tensorboard
                # writer.add_scalar('Train/loc_loss', loss_l.data[0], iteration)
                # writer.add_scalar('Train/conf_loss', loss_c.data[0], iteration)
                pass

        # save model
        if iteration != args.start_iter and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            save_checkpoint({'iteration': iteration,
                             'step_index': step_index,
                             'state_dict': ssd_net.state_dict()},
                            args.save_folder,
                            snapshot_prefix + repr(iteration) + '.pth')
        if iteration == cfg.TRAIN.MAX_ITER - 1:  # save model in end of training
            print('Saving state, iter:', iteration)
            save_checkpoint({'iteration': iteration,
                             'step_index': step_index,
                             'state_dict': ssd_net.state_dict()},
                            args.save_folder,
                            snapshot_prefix + str(cfg.TRAIN.LR_SCHEDULER.STEPS) + '.pth')


def save_checkpoint(state, path, name):
    path_name = os.path.join(path, name)
    torch.save(state, path_name)


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def visualize_epoch(model, data_loader, priorbox, writer, epoch):
    model.eval()
    img_index = random.randint(0, len(data_loader.dataset) - 1)
    image, boxes, labels = data_loader.dataset.pull_aug(img_index)
    # get preproc
    transform = data_loader.dataset.transform
    transform.add_writer(writer, epoch)
    transform(image, boxes, labels)


if __name__ == '__main__':
    train()
