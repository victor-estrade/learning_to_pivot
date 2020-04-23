# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import torch.nn as nn


class GaussNLLLoss(nn.Module):
    def forward(self, input, target, logsigma):
        error = (input - target)
        error_sigma = error / torch.exp(logsigma)
        loss = logsigma + 0.5 * (error_sigma * error_sigma)
        mean_loss = torch.mean(loss)
        mse = torch.mean( error * error )
        return mean_loss, mse
