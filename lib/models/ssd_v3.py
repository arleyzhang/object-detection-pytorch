import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.layers import *
import threading


class ThreadLocalData(threading.local):
    def __init__(self):
        self.list_data = []


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, cfg, base):
        super(SSD, self).__init__()
        if phase != "train" and phase != "eval":
            raise Exception(
                "ERROR: Input phase: {} not recognized".format(phase))
        self.phase = phase
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.cfg = cfg
        # self.priors = None
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.out = None
        self.predict_source_output = ThreadLocalData()
        self.hook_list = list()

        # SSD network
        if isinstance(base, list):
            self.base = nn.ModuleList(base)
        else:
            self.base = base
        # Layer learns to scale the l2 normalized features from conv4_3
        norm_layer_channel = get_layers(
            self, norm_layers[cfg.MODEL.BASE]).out_channels
        self.L2Norm = L2Norm(norm_layer_channel, 20)  # TODO automate this

        extras = add_extras(self, base_output_source[cfg.MODEL.BASE],
                            extras_config['ssd' + str(cfg.MODEL.IMAGE_SIZE[-1])])
        self.extras = nn.ModuleList(extras)

        head = multibox(self,
                        predict_conv_source['ssd' +
                                            str(cfg.MODEL.IMAGE_SIZE[-1])][cfg.MODEL.BASE],
                        cfg.MODEL.NUM_PRIOR, cfg.MODEL.NUM_CLASSES)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

        self.register_hook(predict_relu_source['ssd' +
                                               str(cfg.MODEL.IMAGE_SIZE[-1])][cfg.MODEL.BASE])
        # if self.phase == 'eval':  # TODO add to config
        #     self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x, phase='train'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        # sources = list()
        loc = list()
        conf = list()
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

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        if phase == 'eval':
            output = loc, self.softmax(conf)
        else:
            output = loc, conf
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


def get_layers(module, layer_names):
    layers = []
    for name, sub_module in module.named_modules():
        if name in layer_names:
            layers.append(sub_module)
    if len(layers) == 1:
        return layers[0]
    return layers


def add_extras(module, base_output_names, cfg, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    # in_channels = base[-2].out_channels  # TODO make this configurable
    base_output_layer = get_layers(module, base_output_names)
    in_channels = base_output_layer.out_channels

    stage_kernel = 1
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(stage_kernel, 3)[flag],
                                     stride=2, padding=1),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, v,
                                     kernel_size=(stage_kernel, 3)[flag]),
                           nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers


def multibox(module, predict_source_names, num_priors, num_classes):
    loc_layers = []
    conf_layers = []
    predict_source = get_layers(module, predict_source_names)
    for k, v in enumerate(predict_source):
        loc_layers += [nn.Conv2d(v.out_channels,
                                 num_priors[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels,
                                  num_priors[k] * num_classes, kernel_size=3, padding=1)]
    return loc_layers, conf_layers


predict_conv_source = {
    'ssd300': {
        'vgg16': ['base.21', 'base.33',
                  'extras.2', 'extras.6', 'extras.10', 'extras.14'],
        'drn_d_22': ['base.layer6.1.conv2', 'base.layer11.0',
                     'extras.2', 'extras.6', 'extras.10', 'extras.14']
        # 'drn_d_22': ['base.layer5.1.conv2', 'base.layer8.0',
        #              'extras.2', 'extras.6', 'extras.10', 'extras.14']
        # 'drn_d_22': ['base.layer6.1.conv2', 'base.layer11.0',
        #              'extras.2', 'extras.7', 'extras.11', 'extras.15']
    },
    'ssd512': {
        'vgg16': ['base.21', 'base.33',
                  'extras.2', 'extras.6', 'extras.10', 'extras.14', 'extras.18'],
        'drn_d_22': ['base.layer6.1.conv2', 'base.layer11.0',
                     'extras.2', 'extras.6', 'extras.10', 'extras.14', 'extras.18']
    }
}

predict_relu_source = {
    'ssd300': {
        'vgg16': ['base.22', 'base.34',
                  'extras.3', 'extras.7', 'extras.11', 'extras.15'],
        'drn_d_22': ['base.layer6.1.relu2', 'base.layer11.2',
                     'extras.3', 'extras.7', 'extras.11', 'extras.15']
        # 'drn_d_22': ['base.layer5.1.relu2', 'base.layer8.2',
        #              'extras.3', 'extras.7', 'extras.11', 'extras.15']
        # 'drn_d_22': ['base.layer6.1.relu2', 'base.layer11.2',
        #              'extras.3', 'extras.8', 'extras.12', 'extras.16']
    },
    'ssd512': {
        'vgg16': ['base.22', 'base.34',
                  'extras.3', 'extras.7', 'extras.11', 'extras.15', 'extras.19'],
        'drn_d_22': ['base.layer6.1.relu2', 'base.layer11.2',
                     'extras.3', 'extras.7', 'extras.11', 'extras.15', 'extras.19']
    }

}

base_output_source = {
    'vgg16': ['base.33'],
    'drn_d_22': ['base.layer11.0']
    # 'drn_d_22': ['base.layer8.0']
}

norm_layers = {
    'vgg16': ['base.21'],
    'drn_d_22': ['base.layer6.1.conv2']
    # 'drn_d_22': ['base.layer5.1.conv2']
}

extras_config = {
    'ssd300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    # 'ssd321': [256, 'S', 256, 256, 'M', 256, 256, 256, 256, 256],
    'ssd512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
