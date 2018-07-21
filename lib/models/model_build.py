import torch
from torch.autograd import Variable
from models import ssd, fssd, fpn
from models import vgg
from lib.layers.functions.prior_box import PriorBoxSSD

ssds_map = {
    'ssd': ssd,
    'fssd': fssd,
    'fpn': fpn,
}

networks_map = {
    'vgg16': vgg.vgg16,
}

priors_map = {
    'ssd': PriorBoxSSD,
}


def create_model(phase, cfg):
    prior = priors_map[cfg['prior_type']](cfg)
    cfg['num_priors'] = prior.num_priors
    base = networks_map[cfg['base_model']]()  # vgg.vgg16
    model = ssds_map[cfg['ssds_type']].build(phase, cfg, base)
    layer_dims = get_layer_dims(model, cfg['image_size'][0], cfg['image_size'][1])
    model.priors = prior.forward(layer_dims)
    # print('feature maps:', layer_dims)
    return model, layer_dims


def get_layer_dims(model, input_h, input_w):
    def forward_hook(self, input, output):
        """input: type tuple, output: type Variable"""
        # print('{} forward\t input: {}\t output: {}\t output_norm: {}'.format(
        #     self.__class__.__name__, input[0].size(), output.data.size(), output.data.norm()))
        dims.append([input[0].size()[2], input[0].size()[3]])  # h, w

    dims = []
    handles = []
    for idx, layer in enumerate(model.loc.children()):  # loc...
        if isinstance(layer, nn.Conv2d):
            hook = layer.register_forward_hook(forward_hook)
            handles.append(hook)

    input_size = (1, 3, input_h, input_w)
    model(Variable(torch.randn(input_size)))
    [item.remove() for item in handles]
    return dims


if __name__ == '__main__':
    from lib.data.config import ssd_voc_vgg as cfg
    import torch.nn as nn
    # net, layer_dims = create_model(phase='train', cfg=cfg)  # test ssd
    cfg['ssds_type'] = 'fpn'
    net, layer_dims = create_model(phase='train', cfg=cfg)  # test fpn
    # print(net)
    # print(layer_dims)
