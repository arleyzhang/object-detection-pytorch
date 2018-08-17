import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from lib.layers import *


class DRN_SSD(nn.Module):
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

    # def __init__(self, phase, size, base, extras, head, num_classes, scale=1.0):
    def __init__(self, phase, cfg, base):
        super(DRN_SSD, self).__init__()
        if phase != "train" and phase != "eval":
            raise Exception("ERROR: Input phase: {} not recognized".format(phase))
        self.phase = phase
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.cfg = cfg
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.out = None

        # SSD network
        self.base = base
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(128, 20)

        head = multibox(predict_source, drn_channels, cfg.MODEL.NUM_PRIOR, cfg.MODEL.NUM_CLASSES)

        self.extras = None
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

        # if phase == 'eval':
        #     self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

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
        sources = list()
        loc = list()
        conf = list()

        x, y = self.base(x)

        s = self.L2Norm(y[3])
        sources.append(s)
        sources.extend(y[4:])

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


def multibox(predict_source, channels, num_priors, num_classes):
    loc_layers = []
    conf_layers = []
    drn_source = predict_source
    for k, v in enumerate(drn_source):
        loc_layers += [nn.Conv2d(channels[v - 1],
                                 num_priors[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(channels[v - 1],
                                  num_priors[k] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


# the index of the predict layer,such as:layer4,layer5,……,layer9
predict_source = [4, 5, 6, 7, 8, 9]

drn_channels = [16, 32, 64, 128, 256, 512, 512, 512, 512]
