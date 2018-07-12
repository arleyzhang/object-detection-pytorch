from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['image_size'] #if len(cfg['image_size']) == 2 else [cfg['image_size'],cfg['image_size']]
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        
        #self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['dataset_name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self, layer_dimensions = None):
        mean = []
        if layer_dimensions is None:
            print('error: layer_dimensions is None')
        
        #print('XXXXXXXXXX', [t_ / self.steps[0] for t_ in self.image_size])
        for k, layer_id in enumerate(layer_dimensions.keys()):
            for i, j in product(range(layer_dimensions[layer_id][0]), 
                                    range(layer_dimensions[layer_id][1])):
                f_k = [t_ / self.steps[k] for t_ in self.image_size]
                # unit center x,y
                cx = (j + 0.5) / f_k[1]    #col
                cy = (i + 0.5) / f_k[0]    #row

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size[0]  #need to update
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size[0]))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
