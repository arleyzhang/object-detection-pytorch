from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
import numpy as np
import cv2


def vis(func):
    """tensorboard vis if has writer as input"""
    def wrapper(*args, **kw):
        return func(*args, **kw) if kw['writer'] is not None else None
    return wrapper


class PriorBoxBase(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBoxBase, self).__init__()
        self.image_size = cfg['image_size']
        self.steps = cfg['steps']
        self.cfg_list = []
        self.prior_cfg = {}
        self.clip = cfg['clip']
        self.variance = cfg['variance'] or [0.1, 0.2]
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        self.writer = None

    def setup(self, cfg):
        num_feat = len(self.steps)
        for item in self.cfg_list:
            if item not in cfg:
                raise Exception("wrong anchor config!")
            if len(cfg[item]) != num_feat and len(cfg[item]) != 0:
                raise Exception("config {} length does not match step length!".format(item))
            self.prior_cfg[item] = cfg[item]

    @property
    def num_priors(self):
        """allow prior num calculation before knowing feature map size"""
        assert self.prior_cfg is not {}
        return [int(len(self.create_prior(0, 0, k)) / 4) for k in range(len(self.steps))]

    def create_prior(self, cx, cy, k):
        raise NotImplementedError

    @vis
    def image_proc(self, image=None, writer=None):
        # TODO tesst with image
        if isinstance(image, type(None)):
            image = np.ones((self.image_size[1], self.image_size[0], 3))
        elif isinstance(image, str):
            image = cv2.imread(image, -1)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        return image

    @vis
    def prior_vis(self, anchor, image_ori, feat_idx, writer=None):
        # TODO add output path to the signature
        prior_num = self.num_priors[feat_idx]

        # transform coordinates
        scale = [self.image_size[1], self.image_size[0], self.image_size[1], self.image_size[0]]
        bboxs = np.array(anchor).reshape((-1, 4))
        box_centers = bboxs[:, :2] * scale[:2]  # [x, y]
        # bboxs: [xmin, ymin, xmax, ymax]
        bboxs = np.hstack((bboxs[:, :2] - bboxs[:, 2:4] / 2, bboxs[:, :2] + bboxs[:, 2:4] / 2)) * scale
        box_centers = box_centers.astype(np.int32)
        bboxs = bboxs.astype(np.int32)
        # visualize each anchor box on a feature map
        for prior_idx in range(prior_num):
            image = image_ori.copy()
            bboxs_ = bboxs[prior_idx::prior_num, :]
            box_centers_ = box_centers[4*prior_idx::prior_num, :]
            for archor, bbox in zip(box_centers_, bboxs_):
                cv2.circle(image, (archor[0], archor[1]), 1, (0, 0, 255), -1)
                if archor[0] == archor[1]:  # only show diagnal anchors
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            image = image[..., ::-1]
            writer.add_image('base/feature_map_{}_{}'.format(feat_idx, prior_idx), image, 2)

    def forward(self, layer_dims, writer=None, image=None):
        self.writer = writer
        priors = []
        image = self.image_proc(image=image, writer=writer)

        for k in range(len(layer_dims)):
            prior = []
            for i, j in product(range(layer_dims[k][0]), range(layer_dims[k][1])):
                steps_x = self.image_size[1] / self.steps[k]
                steps_y = self.image_size[0] / self.steps[k]
                cx = (j + 0.5) / steps_x  # unit center x,y
                cy = (i + 0.5) / steps_y
                prior += self.create_prior(cx, cy, k)
            priors += prior
            self.prior_vis(prior, image, k, writer=writer)

        output = torch.Tensor(priors).view(-1, 4)
        # TODO this clip is meanless, should clip on [xmin, ymin, xmax, ymax]
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class PriorBoxSSD(PriorBoxBase):
    def __init__(self, cfg):
        super(PriorBoxSSD, self).__init__(cfg)
        # self.image_size = cfg['image_size']
        self.cfg_list = ['min_sizes', 'max_sizes', 'aspect_ratios']
        self.flip = cfg['flip'] if 'flip' in cfg else True  # backward compatibility
        self.setup(cfg)

    def create_prior(self, cx, cy, k):
        # as the original paper do
        prior = []
        min_sizes = self.prior_cfg['min_sizes'][k]
        min_sizes = [min_sizes] if not isinstance(min_sizes, list) else min_sizes
        for ms in min_sizes:
            # min square
            s_i = ms / self.image_size[0]
            s_j = ms / self.image_size[1]
            prior += [cx, cy, s_j, s_i]
            # min max square
            if len(self.prior_cfg['max_sizes']) != 0:
                assert type(self.prior_cfg['max_sizes'][k]) is not list  # one max size per layer
                s_i_prime = sqrt(s_i * (self.prior_cfg['max_sizes'][k] / self.image_size[0]))
                s_j_prime = sqrt(s_j * (self.prior_cfg['max_sizes'][k] / self.image_size[1]))
                prior += [cx, cy, s_j_prime, s_i_prime]
            # rectangles by min and aspect ratio
            for ar in self.prior_cfg['aspect_ratios'][k]:
                prior += [cx, cy, s_j * sqrt(ar), s_i / sqrt(ar)]  # a vertical box
                if self.flip:
                    prior += [cx, cy, s_j / sqrt(ar), s_i * sqrt(ar)]
        return prior


PriorBox = PriorBoxSSD


def test_no_vis(cfg, writer):
    cfg = copy.deepcopy(cfg)
    cfg['feature_maps'] = [38, 19, 10, 5, 3, 1]
    cfg['min_sizes'] = [[30], [60], 111, 162, 213, 264]
    cfg['flip'] = True
    feat_dim = [list(a) for a in zip(cfg['feature_maps'], cfg['feature_maps'])]
    p = PriorBoxSSD(cfg)
    print(p.num_priors)
    p1 = p.forward(feat_dim)
    print(p1)


def test_filp(cfg, writer):
    cfg = copy.deepcopy(cfg)
    cfg['feature_maps'] = [38, 19, 10, 5, 3, 1]
    cfg['flip'] = True
    feat_dim = [list(a) for a in zip(cfg['feature_maps'], cfg['feature_maps'])]
    p = PriorBoxSSD(cfg)
    p1 = p.forward(feat_dim, writer=writer)

    cfg['flip'] = False
    cfg['aspect_ratios'] = [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
                            [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]]
    p = PriorBox(cfg)
    p2 = p.forward(feat_dim, writer=writer)
    # print(p2)
    assert (p2 - p1).sum() < 1e-8


def test_rectangle(cfg, writer):
    cfg = copy.deepcopy(cfg)
    cfg['feature_maps'] = [38, 19, 10, 5, 3, 1]
    cfg['min_sizes'] = [30, 60, 111, 162, 213, 264]
    cfg['flip'] = True
    # feat_dim = [list(a) for a in zip(cfg['feature_maps'], cfg['feature_maps'])]
    # cfg['image_size'] = [300, 300]
    # feat_dim = [list(a) for a in zip(cfg['feature_maps'], [item * 2 for item in cfg['feature_maps']])]
    # cfg['image_size'] = [300, 600]
    feat_dim = [list(a) for a in zip([item * 2 for item in cfg['feature_maps']], cfg['feature_maps'])]
    cfg['image_size'] = [600, 300]
    p = PriorBoxSSD(cfg)
    p1 = p.forward(feat_dim, writer=writer)
    print(p1.shape)


if __name__ == '__main__':
    import copy
    from lib.data.config import ssd_voc_vgg as cfg
    from tensorboardX import SummaryWriter
    log_dir = './experiments/models/ssd_voc/test'
    writer = SummaryWriter(log_dir=log_dir)

    test_no_vis(cfg, writer)
    test_filp(cfg, writer)
    test_rectangle(cfg, writer)
