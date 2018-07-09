# -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import VARIANCE
from ..box_utils import IoG, decode_new
import sys
import numpy as np
import math


class RepulsionLoss(nn.Module):

    def __init__(self, use_gpu=True, sigma=0.):
        super(RepulsionLoss, self).__init__()
        self.use_gpu = use_gpu
        self.variance = VARIANCE
        self.sigma = sigma
        
    # TODO 
    def smoothln(self, x, smooth=0.):    #need to test. test success
        #sigma = torch.tensor(sigma)

        ##### toch-0.3 not support torch.where()
        # sigma = torch.from_numpy(np.array([sigma])).float()
        # return torch.where(
        #     x <= sigma,    #condition   x.gt(smooth)
        #     -torch.log(1 - x),  #if condition is true
        #     ((x - sigma) / (1. - sigma)) - torch.log(1. - sigma)   #else
        # )
        pass

    #repul_loss(loc_p, loc_g, priors)
    def forward(self, loc_data, ground_data, prior_data):
        
        decoded_boxes = decode_new(loc_data, Variable(prior_data.data, requires_grad=False), self.variance)
        
        iog = IoG(ground_data, decoded_boxes)
        # sigma = 1
        # loss = torch.sum(-torch.log(1-iog+1e-10))  
        # sigma = 0
        # sigma = 1.
        # idx_ = iog <= sigma
        # iog[idx_] = -torch.log(1 - iog[idx_] + 1e-10)
        # idx_ = ~idx_
        # iog[idx_] = ((iog[idx_]  - sigma) / (1. - sigma + 1e-10)) - math.log(1. - sigma + 1e-10)

        loss_repgt = torch.sum(iog)
        #print ('loss_repgt', loss_repgt.size()) 
        return loss_repgt
