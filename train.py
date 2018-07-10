# -*- coding:utf-8 -*-
from data import *
from layers.modules import MultiBoxLoss, MultiBoxLossSSD
from layers.functions import PriorBox

from utils import * #EvalSolver, Timer, SSDAugmentation
from models.model_build import creat_model

import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],    #################dataset type
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',    #######init weights， #ssd_VOC_180621_40000
                    help='Pretrained base model')   #vgg16_reducedfc
parser.add_argument('--batch_size', default=32, type=int,    ########################test is 8 default=32
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,    #####################resume from snapshot
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,    ############test lr
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='../../weights/',    ################ snapshot and pretrained path
                    help='Directory for saving checkpoint models')
parser.add_argument('--loss_type', default='ssd_loss', type=str,    ########## loss type
                    help='ssd_loss or repul_loss')
args = parser.parse_args()

###################################################  some configs need to update
snapshot_prefix = 'ssd_coco_eval0710_'
args.dataset = 'COCO'

cfg = ssd_coco_vgg

match_priors = False    # True: match in dataloader, False: match in loss
test_interval = 40000
snapshot = 5000
step_index = 0  #need put in args ???#ssd_voc_nocudnn0703_115000
run_break = -5
args.lr = 1e-3
args.num_workers = 8
#args.resume = '../../weights/ssd_coco_eval0707_35000.pth' #tmp2

cudnn_benchmark = False
torch.backends.cudnn.enabled = True   #cudnn switch

CUDA_VISIBLE_DEVICES="5,4,6,7"  #Specified GPUs range
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):    #snapshot and save_model
    os.mkdir(args.save_folder)


def train():
    global step_index, iteration, match_priors

    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
             args.dataset_root = COCO_ROOT
        
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
        val_dataset = COCODetection(args.dataset_root, ['minival'],
                           BaseTransform(300, MEANS),
                           target_transform=COCOAnnotationTransform(False))
    elif args.dataset == 'VOC':
        if match_priors:
            priorbox = PriorBox(cfg)
            match_priors = priorbox.forward().cpu().numpy()
        else:
            match_priors = None

        dataset = VOCDetection(args.dataset_root, [('2007', 'trainval'), ('2012', 'trainval')], #('2012', 'trainval')
                               transform=SSDAugmentation(cfg['min_dim'],MEANS),
                               priors = match_priors)
        val_dataset = VOCDetection(args.dataset_root, [('2007', 'test')],
                           BaseTransform(300, MEANS),
                           target_transform=VOCAnnotationTransform(False))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = creat_model('train', cfg)
    net = ssd_net
    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)    #device_ids=[0,1,2,3]
        cudnn.benchmark = cudnn_benchmark

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        #ssd_net.load_state_dict(torch.load(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_iter = checkpoint['iteration']
        step_index = checkpoint['step_index']
        ssd_net.load_state_dict(checkpoint['state_dict'])
        
        #load_filtered_state_dict(ssd_net, torch.load(args.resume))
    else:
        # init_model_file = init_new_model(args.save_folder, args.basenet)
        # ssd_net.load_state_dict(torch.load(init_model_file))
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)   #########################later vgg >> base

        print('Initializing extra weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
    
    if args.cuda:
        net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    
    ###################################################################### loss type
    if args.loss_type == 'repul_loss':
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                False, args.cuda)
    elif args.loss_type == 'ssd_loss':
        criterion = MultiBoxLossSSD(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                False, args.cuda)

    if step_index != 0:  #consider 8w, 12w...   #not iteration in cfg['lr_steps'] and 
        adjust_learning_rate(optimizer, args.gamma, step_index)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    repul_loss = 0
    epoch = 0
    print('Loading the dataset...')
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args, 'Gpus: ' + CUDA_VISIBLE_DEVICES)


    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    #collate_fn=detection_collate  is important
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    
    val_loader = data.DataLoader(val_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)
    #add eval_solver#######################################################
    eval_solver = EvalSolver(val_loader, len(val_dataset), args.save_folder, cfg)

    # create batch iterator
    #batch_iterator = iter(data_loader)

    net.train()
    iteration = args.start_iter
    #for iteration in range(args.start_iter, cfg['max_iter']):
    epoch_size = len(dataset) // args.batch_size    #517
    num_epochs = cfg['max_iter'] // epoch_size + 1  #232+1

    #timers
    timers = {'iter_time': Timer(), 'eval_time': Timer(), 'prepro_time': Timer()}
    timers['iter_time'].tic()
    timers['prepro_time'].tic()
    for epoch in range(num_epochs):
        for i, (images, targets, _, _, loc_t, conf_t) in enumerate(data_loader):
            timers['prepro_time'].toc()
            if images.size(0) < args.batch_size: continue
            
            # if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            #     update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
            #                     'append', epoch_size)
            #     # reset epoch loss counters
            #     loc_loss = 0
            #     conf_loss = 0
            #     repul_loss = 0
            #     epoch += 1
            if iteration in cfg['lr_steps'] and iteration != args.start_iter:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)
            
            #record time
            # timers['prepro_time'].tic()
            # # load train data
            # images, targets, _, _ = data
            # try:
            #     images, targets, _, _ = next(batch_iterator)
            # except StopIteration:
            #     batch_iterator = iter(data_loader)
            #     images, targets, _, _ = next(batch_iterator)
            # timers['prepro_time'].toc()

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            out = net(images)
            # backprop
            optimizer.zero_grad()
            
            if loc_t.ndim == 1:
                loc_t, conf_t = None, None
            else:
                loc_t = torch.from_numpy(loc_t)
                conf_t = torch.from_numpy(conf_t).long()
            if args.loss_type == 'ssd_loss':
                loss_l, loss_c = criterion(out, targets, loc_t, conf_t)    #######ssd loss        
                loss = loss_l + loss_c

            elif args.loss_type == 'repul_loss':
                loss_l, loss_l_repul, loss_c = criterion(out, targets)   #######repul loss
                loss = loss_l + loss_c + loss_l_repul
                repul_loss += loss_l_repul.data[0]

            loss.backward()
            optimizer.step()

            timers['iter_time'].toc()
            timers['iter_time'].tic()

            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]

            if iteration % 10 == 0:
                if args.loss_type == 'ssd_loss':
                    print('Iteration ' + repr(iteration) + ' || Loss: %.4f' % (loss.data[0]) +\
                    ' || conf_loss: %.4f' % (loss_c.data[0]) + ' || smoothl1 loss: %.4f' % (loss_l.data[0]),\
                    end=' ')
                elif args.loss_type == 'repul_loss':
                    print('Iteration ' + repr(iteration) + ' || Loss: %.4f' % (loss.data[0]) +\
                    ' || conf_loss: %.4f' % (loss_c.data[0]) + ' || smoothl1 loss: %.4f' % (loss_l.data[0]) +\
                    ' || repul_loss: %.4f' % (loss_l_repul.data[0]), end=' ')
                
                current_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print('timer: %.3f(%.3f)||%.3f(%.3f) s' %  (timers['iter_time'].diff, timers['iter_time'].average_time,\
                    timers['prepro_time'].diff, timers['prepro_time'].average_time), '|| sys_date:', current_date, '|| lr:',\
                    optimizer.param_groups[0]['lr'])
            
            if args.visdom:
                update_vis_plot(iteration, loss_l.data[0], loss_l_repul.data[0], loss_c.data[0], 
                                iter_plot, epoch_plot, 'append')

            #save model ssd_net.state_dict()? net.?
            if (iteration % snapshot == 0 and iteration != args.start_iter):
                print('Saving state, iter:', snapshot_prefix+repr(iteration) + '.pth')
                save_checkpoint({'iteration': iteration,
                                'step_index': step_index,
                                'state_dict': ssd_net.state_dict()},
                                args.save_folder,
                                snapshot_prefix+repr(iteration) + '.pth')
            #save model in end of training
            if iteration == cfg['max_iter'] - 1:
                save_checkpoint({'iteration': iteration,
                            'step_index': step_index,
                            'state_dict': ssd_net.state_dict()},
                            args.save_folder,
                            snapshot_prefix+repr(cfg['max_iter']) + '.pth')
                return 0
            
             ###eval model   ##################################
            if iteration % test_interval == 0 and iteration != args.start_iter:
                print('Start eval ......')
                net.eval()
                timers['eval_time'].tic()
                mAP = eval_solver.validate(net)
                timers['eval_time'].toc()
                print('Iteration ' + repr(iteration) + ' || mAP: %.3f' % (mAP) + ' ||eval_time: %.4f/%.4f' %
                    (timers['eval_time'].diff, timers['eval_time'].average_time))
                net.train()
            
            if iteration == run_break + args.start_iter: return 0
            iteration += 1
            timers['prepro_time'].tic()

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

def load_filtered_state_dict(model, snapshot=None):
    # By user apaszke from discuss.pytorch.org
    print('debug-----')
    model_dict = model.state_dict()
    # snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    # model_dict.update(snapshot)
    # model.load_state_dict(model_dict)
    for k in model_dict:
        print(k)
    print('////////////////')
    for k, v in snapshot.items():
        print(k)

def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
