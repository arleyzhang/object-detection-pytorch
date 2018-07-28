import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.models import *
from lib.layers.functions.prior_box import PriorBoxSSD


bases_list = ['vgg16']
ssds_list = ['SSD', 'FSSD', 'FPN', 'SSD_COCO']
priors_list = ['PriorBoxSSD']


def create(n, lst, **kwargs):
    if n not in lst:
        raise Exception("unkown type {}, possible: {}".format(n, str(lst)))
    return eval('{}(**kwargs)'.format(n))


def model_factory(phase, cfg):
    prior = create(cfg.MODEL.PRIOR_TYPE, priors_list, cfg=cfg)
    cfg.MODEL.NUM_PRIOR = prior.num_priors
    base = create(cfg.MODEL.BASE, bases_list)
    model = create(cfg.MODEL.SSD_TYPE, ssds_list, phase=phase, cfg=cfg, base=base)
    layer_dims = get_layer_dims(model, cfg.MODEL.IMAGE_SIZE)
    priors = prior.forward(layer_dims)
    return model, priors, layer_dims


def get_layer_dims(model, image_size):
    def forward_hook(self, input, output):
        """input: type tuple, output: type Variable"""
        # print('{} forward\t input: {}\t output: {}\t output_norm: {}'.format(
        #     self.__class__.__name__, input[0].size(), output.datasets.size(), output.datasets.norm()))
        dims.append([input[0].size()[2], input[0].size()[3]])  # h, w

    dims = []
    handles = []
    for idx, layer in enumerate(model.loc.children()):  # loc...
        if isinstance(layer, nn.Conv2d):
            hook = layer.register_forward_hook(forward_hook)
            handles.append(hook)

    input_size = (1, 3, image_size[0], image_size[1])
    model(Variable(torch.randn(input_size)))
    [item.remove() for item in handles]
    return dims


if __name__ == '__main__':
    from lib.utils.config import cfg
    net, priors, layer_dims = model_factory(phase='train', cfg=cfg)
    print(net)
    print(priors)
    print(layer_dims)
