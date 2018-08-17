import torch
from lib.utils.flops_benchmark import add_flops_counting_methods


def get_flops(net, input_size):
    input_size = (1, 3, input_size[0], input_size[1])
    input = torch.randn(input_size)
    input = torch.autograd.Variable(input.cuda())

    net = add_flops_counting_methods(net)
    net = net.cuda().train()
    net.start_flops_count()

    _ = net(input)

    return net.compute_average_flops_cost()/1e9/2



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

    # cfg.MODEL.BASE='drn_d_22'
    # cfg.MODEL.IMAGE_SIZE=(321,321)
    # cfg.MODEL.SSD_TYPE = 'DRN_SSD'
    args.cfg_name='ssd_drn22_rfb_voc'
    args.job_group='rfb'
    tb_writer, cfg_path, snapshot_dir, log_dir = setup_folder(args, cfg)
    model, priors, _ = model_factory(phase='train', cfg=cfg, tb_writer=tb_writer)

    input_size = (cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])

    total_flops = get_flops(model, input_size)

    # For default vgg16 model, this shoud output 31.386288 G FLOPS
    print("The Model's Total FLOPS is : {:.6f} G FLOPS".format(total_flops))
