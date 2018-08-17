import torch
from lib.models import model_factory


def print_featuremap(self, input, output):
    print('{}: {}\t input: {}\t output: {}\t'.format(
        self.addition_name, self.__class__.__name__,
        input[0].size(), output.data.size()))


def print_featuremap_attri(self, input, output):
    print('{}: {}\t input: {}\t output: {}\t'
          'mean: {:.2e}\t var: {:.2e}\t norm: {:.2e}'.format(
              self.addition_name, self.__class__.__name__,
              input[0].size(), output.data.size(),
              output.data.mean(), output.data.var(), output.data.norm()))


def get_featuremap(module, input, forward_post_hook=print_featuremap,
                   forward_pre_hook=None, verbose=True):
    hook_list = []
    if verbose:
        for name, sub_module in module.named_modules():
            if not isinstance(sub_module, torch.nn.ModuleList) and \
                    not isinstance(sub_module, torch.nn.Sequential) and \
                    type(sub_module) in torch.nn.__dict__.values():
                sub_module.addition_name = name
                if forward_pre_hook:
                    hook_pre = sub_module.register_forward_pre_hook(
                        forward_pre_hook)
                hook_post = sub_module.register_forward_hook(
                    forward_post_hook)
                hook_list.extend([hook_pre, hook_post]
                                 if forward_pre_hook else [hook_post])
    else:
        for name, sub_module in module.named_children():
            sub_module.addition_name = name
            if forward_pre_hook:
                hook_pre = sub_module.register_forward_pre_hook(
                    forward_pre_hook)
            hook_post = sub_module.register_forward_hook(forward_post_hook)
            hook_list.extend([hook_pre, hook_post]
                             if forward_pre_hook else [hook_post])

    _ = module(input)

    for hook in hook_list:
        hook.remove()


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
    args.cfg_name='ssd_drn22_voc_321_media_v31'
    args.job_group='drn'
    tb_writer, cfg_path, snapshot_dir, log_dir = setup_folder(args, cfg)
    model, priors, _ = model_factory(phase='train', cfg=cfg, tb_writer=None)
    # print(model)
    model = model.cuda()
    size = (1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
    input = torch.autograd.Variable(torch.randn(size).cuda())

    print('The feature map size of the layers')
    get_featuremap(model, input, verbose=True)
