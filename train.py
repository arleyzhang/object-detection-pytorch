import argparse
import os
import os.path as osp
from distutils.dir_util import copy_tree
from shutil import copyfile

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from lib.datasets import dataset_factory
from lib.models import model_factory
from lib.utils import eval_solver_factory
from lib.layers.modules import MultiBoxLoss
from lib.utils.config import cfg
from lib.utils.utils import Timer, create_if_not_exist, setup_folder

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--cfg_name', default='ssd_vgg16_voc',
                    help='base name of config file')
parser.add_argument('--job_group', default='base', type=str,
                    help='Directory for saving checkpoint models')
parser.add_argument('--devices', default='0,1,2,3', type=str,
                    help='GPU to use')
parser.add_argument('--basenet', default='pretrain/vgg16_reducedfc.pth',
                    help='Pretrained base model')  # TODO config
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=1, type=int,
                    help='Resume training at this iter')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--tensorboard', default=True, type=bool,
                    help='Use tensorboard')
parser.add_argument('--loss_type', default='ssd_loss', type=str,
                    help='ssd_loss only now')
args = parser.parse_args()


def train():
    tb_writer, cfg_path, snapshot_dir, log_dir = setup_folder(args, cfg)
    step_index = 0

    train_loader = dataset_factory(phase='train', cfg=cfg)
    val_loader = dataset_factory(phase='eval', cfg=cfg)
    eval_solver = eval_solver_factory(val_loader, cfg)

    ssd_net, priors, _ = model_factory(phase='train', cfg=cfg)
    net = ssd_net  # net is the parallel version of ssd_net

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
    else:
        # pretained weights
        vgg_weights = torch.load(osp.join(cfg.GENERAL.WEIGHTS_ROOT, args.basenet))
        print('Loading base network...')
        ssd_net.base.load_state_dict(vgg_weights)

        # initialize newly added layers' weights with xavier method
        print('Initializing weights...')
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

    # continue training at 8w, 12w...
    if args.start_iter not in cfg.TRAIN.LR_SCHEDULER.STEPS and step_index != 0:
        adjust_learning_rate(optimizer, cfg.TRAIN.LR_SCHEDULER.GAMMA, step_index)

    net.train()
    epoch_size = len(train_loader.dataset) // cfg.DATASET.TRAIN_BATCH_SIZE
    num_epochs = (cfg.TRAIN.MAX_ITER + epoch_size - 1) // epoch_size
    print('Training SSD on:', train_loader.dataset.name)
    print('Using the specified args:')
    print(args)

    # timer
    t_ = {'network': Timer(), 'misc': Timer(), 'all': Timer(), 'eval': Timer()}
    t_['all'].tic()

    iteration = args.start_iter
    for epoch in range(num_epochs):
        tb_writer.cfg['epoch'] = epoch
        for images, targets, _ in train_loader:
            tb_writer.cfg['iteration'] = iteration
            t_['misc'].tic()
            if iteration in cfg.TRAIN.LR_SCHEDULER.STEPS:
                t_['misc'].tic()
                step_index += 1
                adjust_learning_rate(optimizer, cfg.TRAIN.LR_SCHEDULER.GAMMA, step_index)

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            # forward
            t_['network'].tic()
            out = net(images)
            out1 = [out[0], out[1], priors]

            # backward
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out1, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t_['network'].toc()

            # log
            if iteration % cfg.TRAIN.LOG_LOSS_ITER == 0:
                t_['misc'].toc()
                print('Iter ' + str(iteration) + ' || Loss: %.3f' % (loss.data[0]) +
                      '|| conf_loss: %.3f' % (loss_c.data[0]) + ' || loc loss: %.3f ' % (loss_l.data[0]), end=' ')
                print('Timer: %.3f sec.' % t_['misc'].diff, '  Lr: %.6f' % optimizer.param_groups[0]['lr'])
                if args.tensorboard:
                    phase = tb_writer.cfg['phase']
                    tb_writer.writer.add_scalar('{}/loc_loss'.format(phase), loss_l.data[0], iteration)
                    tb_writer.writer.add_scalar('{}/conf_loss'.format(phase), loss_c.data[0], iteration)
                    tb_writer.writer.add_scalar('{}/all_loss'.format(phase), loss.data[0], iteration)
                    tb_writer.writer.add_scalar('{}/time'.format(phase), t_['misc'].diff, iteration)

            # save model
            if iteration % cfg.TRAIN.SAVE_ITER == 0 and iteration != args.start_iter or \
                    iteration == cfg.TRAIN.MAX_ITER:
                print('Saving state, iter:', iteration)
                save_checkpoint({'iteration': iteration,
                                 'step_index': step_index,
                                 'state_dict': ssd_net.state_dict()},
                                snapshot_dir,
                                args.cfg_name + '_' + repr(iteration) + '.pth')

            # Eval
            if (iteration % cfg.TRAIN.EVAL_ITER == 0 and iteration != args.start_iter) or \
                    iteration == cfg.TRAIN.MAX_ITER:
                print('Start evaluation ......')
                tb_writer.cfg['phase'] = 'eval'
                t_['eval'].tic()
                net.eval()
                aps, mAPs = eval_solver.validate(net, priors, tb_writer=tb_writer)
                net.train()
                t_['eval'].toc()
                print('Iteration ' + str(iteration) + ' || mAP: %.3f' % mAPs[0] + ' ||eval_time: %.4f/%.4f' %
                      (t_['eval'].diff, t_['eval'].average_time))
                if cfg.DATASET.NAME == 'VOC0712':
                    tb_writer.writer.add_scalar('mAP/mAP@0.5', mAPs[0], iteration)
                else:
                    tb_writer.writer.add_scalar('mAP/mAP@0.5', mAPs[0], iteration)
                    tb_writer.writer.add_scalar('mAP/mAP@0.95', mAPs[1], iteration)
                tb_writer.cfg['phase'] = 'train'

            if iteration == cfg.TRAIN.MAX_ITER:
                break
            iteration += 1

    backup_jobs(cfg, cfg_path, log_dir)


def backup_jobs(cfg, cfg_path, log_dir):
    print('backing up cfg and log')
    out_dir = osp.join(cfg.GENERAL.HISTORY_ROOT, cfg.GENERAL.JOB_GROUP, args.cfg_name)
    if osp.exists(out_dir):
        out_name = args.cfg_name + '_n'
        print('\033[91m' + 'backup with new name {}'.format(out_name) + '\033[0m')
        out_dir = osp.join(cfg.GENERAL.HISTORY_ROOT, cfg.GENERAL.JOB_GROUP, out_name)
    create_if_not_exist(out_dir)
    cfg_name = args.cfg_name + '.yml'

    copyfile(cfg_path, osp.join(out_dir, cfg_name))
    copy_tree(log_dir, out_dir)


def save_checkpoint(state, path, name):
    path_name = os.path.join(path, name)
    torch.save(state, path_name)


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = cfg.TRAIN.OPTIMIZER.LR * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()
