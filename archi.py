# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseArchi(nn.Module):
    def __init__(self, n_unit=20):
        super().__init__()
        self.name = "{}x{:d}".format(self.__class__.__name__, n_unit)


class F3Classifier(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=20):
        super().__init__(n_unit)
        self.fc_in  = nn.Linear(n_in, n_unit)
        self.fc1    = nn.Linear(n_unit, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x):
        x = self.fc_in(x)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.fc1.reset_parameters()
        self.fc_out.reset_parameters()


class F3Regressor(BaseArchi):
    def __init__(self, n_in=1, n_out=1, n_unit=20):
        super().__init__(n_unit)
        self.fc_in  = nn.Linear(n_in, n_unit)
        self.fc1    = nn.Linear(n_unit, n_unit)
        self.fc_out = nn.Linear(n_unit, n_out)

    def forward(self, x):
        x = self.fc_in(x)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.fc1.reset_parameters()
        self.fc_out.reset_parameters()


class F3GausianMixtureDensity(BaseArchi):
    def __init__(self, n_in=1, n_components=5, n_unit=20):
        super().__init__(n_unit)
        self.fc_in  = nn.Linear(n_in, n_unit)
        self.fc1    = nn.Linear(n_unit, n_unit)
        self.fc_mean = nn.Linear(n_unit, n_components)
        self.fc_log_sigma = nn.Linear(n_unit, n_components)
        self.fc_mixture = nn.Linear(n_unit, n_components)

    def forward(self, x):
        x = self.fc_in(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        mean = self.fc_mean(x)
        log_sigma = self.fc_log_sigma(x)
        mixture = self.fc_mixture(x)
        mixture = F.softmax(mixture, dim=1)
        return mean, log_sigma, mixture

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        self.fc1.reset_parameters()
        self.fc_out.reset_parameters()
        self.fc_out.reset_parameters()
        self.fc_out.reset_parameters()

