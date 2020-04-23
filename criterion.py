# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import torch.nn as nn

import numpy as np


class ADVLoss(nn.Module):
    def forward(self, predictions, target):
        mu, logsigma, mixture = predictions
        sigma = torch.exp(logsigma)
        cst = 1./np.sqrt(2. * np.pi)
        pdf = cst / sigma * torch.exp( -( (target - mu) ** 2 ) / (2. * (sigma ** 2)) )
        mixture_pdf = torch.sum( pdf * mixture, axis=1)
        loss = - torch.log(mixture_pdf)
        mean_loss = torch.mean(loss)
        return mean_loss


class GaussNLLLoss(nn.Module):
    def forward(self, input, target, logsigma):
        error = (input - target)
        error_sigma = error / torch.exp(logsigma)
        loss = logsigma + 0.5 * (error_sigma * error_sigma)
        mean_loss = torch.mean(loss)
        mse = torch.mean( error * error )
        return mean_loss, mse
