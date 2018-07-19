import torch
import torch.nn as nn
from torch.autograd import Variable

from models import ssd, fssd, fpn

ssds_map = {
                'ssd': ssd.build_ssd,
                'fssd': fssd.build_fssd,
                'fpn': fpn.build_fpn,
            }

from models import vgg
networks_map = {
                    'vgg16': vgg.vgg16,
                }

def creat_model(phase, cfg, input_h = 300, input_w = 300):
    
    base = networks_map[cfg['base_model']]   #vgg.vgg16
    model = ssds_map[cfg['ssds_type']](phase, cfg, base)

    layer_dimension = get_layer_dimension(model, input_h, input_w)
    print('feature maps:', layer_dimension)
    
    return model, layer_dimension


def get_layer_dimension(model, input_h, input_w):
    def forward_hook(self, input, output):
        """
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        # input=torch.FloatTensor(input)
        # output=torch.FloatTensor(output)
        """
        # print('{} forward\t input: {}\t output: {}\t output_norm: {}'.format(
        #     self.__class__.__name__, input[0].size(), output.data.size(), output.data.norm()))
        dimensions.append([ input[0].size()[2], output.data.size()[3]])   #h, w
    
    dimensions = []
    layer_dimension = {}
    input_size = (1, 3, input_h, input_w)
    layer_id = 0
    for idx, layer in enumerate(model.loc.children()):  #loc...
        if isinstance(layer, nn.Conv2d):
            hook = layer.register_forward_hook(forward_hook)
            layer_dimension[layer_id] = hook #first record hook, this need remove in later
        layer_id += 1

    input = torch.randn(input_size)
    input = Variable(input)
    model(input)

    for idx, key in enumerate(layer_dimension.keys()):
        layer_dimension[key].remove()
        layer_dimension[key] = dimensions[idx]
    return layer_dimension


