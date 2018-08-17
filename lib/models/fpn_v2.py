import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.layers import *

import threading


class ThreadLocalData(threading.local):
    def __init__(self):
        self.list_data = []


class FPN(nn.Module):
    def __init__(self, phase, cfg, base):
        super(FPN, self).__init__()
        self.phase = phase
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.cfg = cfg
        self.priors = None
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.predict_source_output = ThreadLocalData()
        self.hook_list = list()

        # SSD network
        if isinstance(base, list):
            self.base = nn.ModuleList(base)
        else:
            self.base = base

        norm_layer_channel = get_layers(
            self, norm_layers[cfg.MODEL.BASE]).out_channels
        self.L2Norm = L2Norm(norm_layer_channel, 20)  # TODO automate this

        extras, features_ = add_extras(extras_config[cfg.MODEL.BASE])
        head = multibox(
            features_[1],
            cfg.MODEL.NUM_PRIOR,
            cfg.MODEL.NUM_CLASSES)

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.transforms = nn.ModuleList(features_[0])  # lateral layers 1x1
        self.pyramids = nn.ModuleList(features_[1])  # final output layers 3x3

        self.softmax = nn.Softmax(dim=-1)

        self.register_hook(predict_relu_source[cfg.MODEL.BASE])

        if self.phase == 'test':  # TODO add to config
            self.detect = DetectOut(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x, phase='train'):
        sources, transformed, pyramids, loc, conf = [list() for _ in range(5)]

        self.predict_source_output.list_data.clear()

        if isinstance(self.base, nn.ModuleList):
            # apply vgg up to fc7
            for k in range(len(self.base)):
                x = self.base[k](x)
        else:
            x = self.base(x)

        # apply extra layers and cache source layer outputs
        for k in range(len(self.extras)):
            x = self.extras[k](x)

        sources = self.predict_source_output.list_data

        s = self.L2Norm(sources[0])
        sources[0] = s

        assert len(self.transforms) == len(sources)

        for idx, func in enumerate(self.transforms):  # transforms layers
            transformed.append(func(sources[idx]))

        for idx, func in enumerate(self.pyramids):  # pyramids layers
            # TODO upsize is feat_dim
            upsize = (transformed[-1 - idx].size()[2],
                      transformed[-1 - idx].size()[3])
            size = None if idx == 0 else upsize  # need to upsample
            x = upsample_add(transformed[-1 - idx], transformed[-idx], size)
            pyramids.append(func(x))

        pyramids = pyramids[::-1]  # reverse

        # apply multibox head to pyramids layers
        for (x, l, c) in zip(pyramids, self.loc, self.conf):
            # print('debug-----', x.size())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            if self.priors is None:
                print('Test net init success!')
                return 0
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        elif phase == 'eval':
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
            )
        elif phase == 'train':
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors  # Shape: [2,num_priors*4] ????
            )
        else:
            raise Exception("Unknow phase {}".format(phase))
        return output

    def register_hook(self, predict_source):
        this_module = self

        def get_predict_output(self, input, output):
            this_module.predict_source_output.list_data.append(output)

        for name, sub_module in this_module.named_modules():
            if name in predict_source:
                sub_module.addition_name = name
                hook_handle = sub_module.register_forward_hook(
                    get_predict_output)
                this_module.hook_list.append(hook_handle)

    def delete_hook(self):
        for hook in self.hook_list:
            hook.remove()


def upsample_add(x, y, up_size):
    """Upsample and add two feature maps.
        Args:
          x: (Variable) lateral feature map.
          y: (Variable) top feature map to be upsampled.
          up_size: upsampled size
        Returns:
          (Variable) added feature map.(x + y*size)
    """
    if up_size is None:
        return x
    assert y.size(1) == x.size(1)  # same channel
    y = F.upsample(y, size=up_size, mode='bilinear')
    return x + y


# [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
def add_extras(layers_config):
    # Extra layers added to VGG for feature scaling
    extra_layers = []  # forward layers 3x3
    feature_transform_layers = []  # lateral layers 1x1
    pyramid_feature_layers = []  # final output layers 3x3

    in_d = None
    out_d = layers_config[1][-1]
    for layer, depth in zip(layers_config[0], layers_config[1]):
        if layer == 'S':
            extra_layers += [  # conv7_1 conv7_2 conv8_1 conv8_2
                nn.Conv2d(in_d, int(depth / 2), kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(depth / 2), depth,
                          kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)]
            in_d = depth
        # feature map dimension is 5 or 3 when input is 300
        elif layer == '':  # conv9 conv10
            extra_layers += [
                nn.Conv2d(in_d, int(depth / 2), kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(depth / 2), depth, kernel_size=3),
                nn.ReLU(inplace=True)]
            in_d = depth
        else:
            in_d = depth
        feature_transform_layers += [nn.Conv2d(depth, out_d, kernel_size=1)]
        pyramid_feature_layers += [
            nn.Conv2d(out_d, out_d, kernel_size=3, padding=1)]
    return extra_layers, (feature_transform_layers, pyramid_feature_layers)


def multibox(transforms, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    for k, v in enumerate(transforms):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return loc_layers, conf_layers


def get_layers(module, layer_names):
    layers = []
    for name, sub_module in module.named_modules():
        if name in layer_names:
            layers.append(sub_module)
    if len(layers) == 1:
        return layers[0]
    return layers


"""
[0]: add 'S' or '' can add extra layer
[1]: 'S' denote stride = 2

[14, 21, 33, 'S'], [256, 512, 1024, 512]  conv3-conv7
[21, 33, 'S', 'S', '', ''， ''], [512, 1024, 512, 256, 256, 256]
"""
extras_config = {  # conv4_3 relu, fc7 relu
    # feature map
    'vgg16': [[22, 34, 'S', 'S', '', ''], [512, 1024, 512, 256, 256, 256]],
    'drn_d_22': [['layer6.1', 'layer11', 'S', 'S', '', ''], [512, 512, 512, 256, 256, 256]]
}


predict_relu_source = {
    'vgg16': ['base.22', 'base.34',
              'extras.3', 'extras.7', 'extras.11', 'extras.15'],
    'drn_d_22': ['base.layer6.1.relu2', 'base.layer11.2',
                 'extras.3', 'extras.7', 'extras.11', 'extras.15']
}

norm_layers = {
    'vgg16': ['base.21'],
    'drn_d_22': ['base.layer6.1.conv2']
    # 'drn_d_22': ['base.layer5.1.conv2']
}
