import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
sys.path.insert(0, '../')
from layers import *
from models import *
from utils import Timer
from data.config import *



class FPN(nn.Module):
    #(self, base, extras, head, features, feature_layer, num_classes)
    #FSSD(phase, cfg, base_, extras_, head_, features_, extras[str(size)])
    def __init__(self, phase, cfg, base, extras, features_, head, feature_layer):
        super(FPN, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.cfg = cfg
        
        self.priors = None
        self.size = cfg['min_dim']

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)   #

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        
        self.feature_layer = feature_layer[0][0]
        self.transforms = nn.ModuleList(features_[0])
        self.pyramids = nn.ModuleList(features_[1])
       
        self.softmax = nn.Softmax(dim=-1)
        if self.phase == 'test':
            self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x, phase='train'):
        sources, transformed, pyramids, loc, conf = [list() for _ in range(5)]

        # apply bases layers and cache source layer outputs
        for k in range(len(self.vgg)):
            x = self.vgg[k](x)
            #if k in self.feature_layer:
            if k == 22:
                s = self.L2Norm(x)
                sources.append(s)
        sources.append(x)   #keep output of layer22 and layer34 

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:  #
                sources.append(x)
        assert len(self.transforms) == len(sources)

        for k, v in enumerate(self.transforms): #transforms layers
            transformed.append(v(sources[k]))
        
        for k, v in enumerate(self.pyramids): #pyramids layers
            upsize = (transformed[-1 - k].size()[2], transformed[-1 - k].size()[3])   #upsample size
            size = None if k == 0 else upsize   #need to upsample
            x = upsample_add(transformed[-1 - k], transformed[-k], size)
            # if self.priors is not None:
            #     print('upsample_add:\n', x[0][0], '\n', 'max:', torch.max(x))
            pyramids.append(v(x)) #
        
        # if self.priors is not None:
        #     assert 1==0
        pyramids = pyramids[::-1]   #reverse

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
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        elif phase == 'eval':
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors     # Shape: [2,num_priors*4] ????
            )
        return output

def upsample_add(x, y, up_size):
    '''Upsample and add two feature maps.
        Args:
          x: (Variable) lateral feature map.
          y: (Variable) top feature map to be upsampled.
          size: upsampled size
        Returns:
          (Variable) added feature map.(x + y*size)
    '''
    if up_size is None:
        return x
    
    assert y.size(1) == x.size(1)
    y = F.upsample(y, size=up_size, mode='bilinear')
    
    return x + y

#[256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
def add_extras(feature_layer, version='fssd'):
    # Extra layers added to VGG for feature scaling
    extra_layers = []
    feature_transform_layers = []
    pyramid_feature_layers = []

    in_channels = None
    '''#[21, 33, 'S', 'S', '', ''， ''], [512, 1024, 512, 256, 256, 256]
    feature_layer[0]:[[21, 33, S', 'S', '', ''， ''], [512, 1024, 512, 256, 256, 256]],  #concat layer
    '''
    feature_transform_channel = feature_layer[0][1][-1]
    for layer, depth in zip(feature_layer[0][0], feature_layer[0][1]):
        if layer == 'S':
            extra_layers += [   #conv7_1 conv7_2
                    nn.Conv2d(in_channels, int(depth/2), kernel_size=1),
                    nn.Conv2d(int(depth/2), depth, kernel_size=3, stride=2, padding=1)  ]
            in_channels = depth
        elif layer == '':   #if feature map dimension is 5 or 3
            extra_layers += [
                    nn.Conv2d(in_channels, int(depth/2), kernel_size=1),
                    nn.Conv2d(int(depth/2), depth, kernel_size=3)  ]
            in_channels = depth
        else:
            in_channels = depth
        feature_transform_layers += [nn.Conv2d(in_channels, feature_transform_channel, kernel_size=1)]
        pyramid_feature_layers += [nn.Conv2d(feature_transform_channel, feature_transform_channel, kernel_size=3, padding=1)]
    return extra_layers, (feature_transform_layers, pyramid_feature_layers)

def multibox(base, transforms, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    for k, v in enumerate(transforms):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return base, transforms, (loc_layers, conf_layers)

"""
[0]: add 'S' or '' can add extra layer
[1]: 'S' denote stride = 2

[14, 21, 33, 'S'], [256, 512, 1024, 512]  conv3-conv7
[21, 33, 'S', 'S', '', ''， ''], [512, 1024, 512, 256, 256, 256]
"""
extras = {#conv4_3 relu, fc7 relu
    '300': [[[22, 34, 'S', 'S', '', ''], [512, 1024, 512, 256, 256, 256]],  #feature map
            [['', 'S', 'S', 'S', '', ''], [512, 512, 256, 256, 256, 256]]], #not used
    '512': [],
}


def build_fpn(phase, cfg, base):
    size, num_classes = cfg['min_dim'], cfg['num_classes']
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    #[[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    number_box= [2*(len(aspect_ratios) + 1) if isinstance(aspect_ratios[0], int) 
                else (len(aspect_ratios) + 1) for aspect_ratios in cfg['aspect_ratios']]  
    
    base_ = base()
    extras_, features_ = add_extras(extras[str(size)])
    base_, _, head_ = multibox(base_,
                                features_[1],
                                number_box, num_classes)
  
    return FPN(phase, cfg, base_, extras_, features_, head_, extras[str(size)])

#test fpn net
def get_layer_dimension(model, module_list, input_h=300, input_w=300):
    def forward_hook(self, input, output):
        """
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        # input=torch.FloatTensor(input)
        # output=torch.FloatTensor(output)
        """
        print('{} forward\t input: {}\t output: {}\t output_norm: {}'.format(
            self.__class__.__name__, input[0].size(), output.data.size(), output.data.norm()))
        dimensions.append([ input[0].size()[2], output.data.size()[3]])   #h, w
    
    dimensions = []
    layer_dimension = {}
    input_size = (1, 3, input_h, input_w)
    layer_id = 0
    for module_ in module_list: 
        for idx, layer in enumerate(module_.children()):  #loc...
            if isinstance(layer, nn.Conv2d):
                hook = layer.register_forward_hook(forward_hook)
                layer_dimension[layer_id] = hook #first record hook, this need remove in later
            layer_id += 1

    input = torch.randn(input_size) * 255
    input = Variable(input)
    model(input)

    for idx, key in enumerate(layer_dimension.keys()):
        layer_dimension[key].remove()
        layer_dimension[key] = dimensions[idx]
    return layer_dimension

if __name__ == '__main__':

    model = build_fpn('train', fpn_voc_vgg, vgg.vgg16)
    get_layer_dimension(model, [model.pyramids, model.loc])   #test net #model.extras, model.transforms, model.pyramids,model.loc
