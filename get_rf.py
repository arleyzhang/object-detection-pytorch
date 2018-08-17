import numpy as np
import torch

def get_receptive_filed(module):
    params = []
    for name, sub_module in module.named_modules():
        if (isinstance(sub_module, torch.nn.Conv2d) or
            isinstance(sub_module, torch.nn.MaxPool2d)) and \
                name.find('downsample') == -1:
            kernel_size = sub_module.kernel_size \
                if isinstance(sub_module.kernel_size, tuple) \
                else (sub_module.kernel_size, sub_module.kernel_size)
            dilation = sub_module.dilation \
                if isinstance(sub_module.dilation, tuple) \
                else (sub_module.dilation, sub_module.dilation)
            stride = sub_module.stride \
                if isinstance(sub_module.stride, tuple) \
                else (sub_module.stride, sub_module.stride)
            params.append(['{}({})'.format(name, sub_module.__class__.__name__),
                           kernel_size,
                           dilation, stride])

    for k in range(len(params)):
        rf = np.array((1, 1))
        for i in range(k + 1)[::-1]:
            if params[i][2] != (1, 1):
                effective_kernel = (
                    np.array(params[i][1]) - 1) * (np.array(params[i][2])) + 1
            else:
                effective_kernel = np.array(params[i][1])
            if i == k:
                params[k].append(tuple(effective_kernel))
            rf = (rf - 1) * (np.array(params[i][3])) + effective_kernel

        params[k].append(tuple(rf))
    for v in params:
        print('name: {}\t kernel: {}\t dilation: {}\t stride: {}\t effect kernel: {}\t effect rf: {}\t'.format(
            v[0], v[1], v[2], v[3], v[4], v[5]))


# example
if __name__ == '__main__':
    import argparse
    from lib.models import model_factory
    from lib.layers import *
    from lib.utils.config import cfg
    from lib.utils.utils import setup_folder

    def parse_args():
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

        return args

    args = parse_args()

    # create model
    # tb_writer, cfg_path, snapshot_dir, log_dir = setup_folder(args, cfg)
    # cfg.MODEL.BASE = 'drn_d_22'
    # cfg.MODEL.IMAGE_SIZE = (321,321)
    # cfg.MODEL.SSD_TYPE = 'FPN'
    args.cfg_name='ssd_drn22_voc_300_re'
    args.job_group='drn'
    tb_writer, cfg_path, snapshot_dir, log_dir = setup_folder(args, cfg)
    model, priors, _ = model_factory(phase='train', cfg=cfg, tb_writer=tb_writer)
    # print(model)
    model = model.cuda()
    size = (1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
    input = torch.autograd.Variable(torch.randn(size).cuda())

    print('The feature map size of the layers')
    get_receptive_filed(model)
