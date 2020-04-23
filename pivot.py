# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import json

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from itertools import islice
from minibatch import EpochShuffle
from minibatch import OneEpoch

from utils import to_torch
from utils import to_numpy


class Pivot():
    def __init__(self, net, adv_net, 
                net_criterion, adv_criterion, trade_off,
                net_optimizer, adv_optimizer, combined_optimizer,
                n_net_pre_training_steps=10, n_adv_pre_training_steps=10,
                n_steps=1000, n_recovery_steps=10,
                batch_size=20, rescale=True, cuda=False, verbose=0):
        self.net = net
        self.adv_net = adv_net
        self.net_criterion = net_criterion
        self.adv_criterion = adv_criterion
        self.trade_off = trade_off
        self.net_optimizer = net_optimizer
        self.adv_optimizer = adv_optimizer
        self.combined_optimizer = combined_optimizer
        self.n_net_pre_training_steps = n_net_pre_training_steps
        self.n_adv_pre_training_steps = n_adv_pre_training_steps
        self.n_steps = n_steps
        self.n_recovery_steps = n_recovery_steps
        self.batch_size = batch_size
        self.cuda = cuda
        self.verbose = verbose
        if rescale:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        self._reset_losses()
        self.cuda_flag = cuda
        if cuda:
            self.cuda()

    def cuda(self, device=None):
        self.net = self.net.cuda(device=device)
        self.adv_net = self.adv_net.cuda(device=device)
        self.net_criterion = self.net_criterion.cuda(device=device)
        self.adv_criterion = self.adv_criterion.cuda(device=device)

    def cpu(self):
        self.net = self.net.cpu()
        self.adv_net = self.adv_net.cpu()
        self.net_criterion = self.net_criterion.cpu()
        self.adv_criterion = self.adv_criterion.cpu()

    def get_losses(self):
        losses = dict(net_loss=self.net_loss
                    , adv_loss=self.adv_loss
                    , comb_loss=self.comb_loss
                    )
        return losses

    def reset(self):
        self._reset_losses()
        if self.rescale:
            self.rescale = StandardScaler()

    def _reset_losses(self):
        self.net_loss = []
        self.adv_loss = []
        self.comb_loss = []

    def fit(self, X, y, z, sample_weight=None):
        X = to_numpy(X)
        y = to_numpy(y)
        z = to_numpy(z)
        # Preprocessing
        X = self.scaler.fit_transform(X)
        # to cuda friendly types
        X = X.astype(np.float32)
        z = z.astype(np.float32)
        y = y.astype(np.int64)

        # Pre-training classifier
        net_generator = EpochShuffle(X, y, batch_size=self.batch_size)
        self._fit_net(net_generator, self.n_net_pre_training_steps)  # pre-training

        # Pre-training adversarial
        adv_generator = EpochShuffle(X, z, batch_size=self.batch_size)
        self._fit_adv_net(adv_generator, self.n_adv_pre_training_steps)  # pre-training
        
        # Training
        comb_generator = EpochShuffle(X, y, z, batch_size=self.batch_size)
        self._fit_combined(comb_generator, comb_generator, self.n_steps)
        return self

    def _fit_net(self, generator, n_steps):
        self.net.train()  # train mode
        for i, (X_batch, y_batch) in enumerate(islice(generator, n_steps)):
            X_batch = to_torch(X_batch, cuda=self.cuda_flag)
            y_batch = to_torch(y_batch, cuda=self.cuda_flag)
            self.net_optimizer.zero_grad()  # zero-out the gradients because they accumulate by default
            y_pred = self.net.forward(X_batch)
            loss = self.net_criterion(y_pred, y_batch)
            self.net_loss.append(loss.item())
            loss.backward()  # compute gradients
            self.net_optimizer.step()  # update params
        return self

    def _fit_adv_net(self, generator, n_steps):
        self.adv_net.train()  # train mode
        for i, (X_batch, z_batch) in enumerate(islice(generator, n_steps)):
            X_batch = to_torch(X_batch, cuda=self.cuda_flag)
            z_batch = to_torch(z_batch, cuda=self.cuda_flag).view(-1, 1)
            self.adv_optimizer.zero_grad()  # zero-out the gradients because they accumulate by default
            y_pred = self.net.forward(X_batch)
            z_pred = self.adv_net.forward(y_pred)
            loss = self.adv_criterion(z_pred, z_batch)
            # self.adv_loss.append(loss.item())
            loss.backward()  # compute gradients
            self.adv_optimizer.step()  # update params
        return self

    def _fit_recovery(self, generator, n_steps):
        self.adv_net.train()  # train mode
        for i, (X_batch, y_batch, z_batch) in enumerate(islice(generator, n_steps)):
            X_batch = to_torch(X_batch, cuda=self.cuda_flag)
            y_batch = to_torch(y_batch, cuda=self.cuda_flag)
            z_batch = to_torch(z_batch, cuda=self.cuda_flag).view(-1, 1)
            self.adv_optimizer.zero_grad()  # zero-out the gradients because they accumulate by default
            y_pred = self.net.forward(X_batch)
            z_pred = self.adv_net.forward(y_pred)
            # net_loss = self.net_criterion(y_pred, y_batch)
            adv_loss = self.adv_criterion(z_pred, z_batch)
            # loss = net_loss - (self.trade_off * adv_loss)
            # loss = - loss # gradient ascent
            loss = adv_loss
            # self.adv_loss.append(loss.item())
            loss.backward()  # compute gradients
            self.adv_optimizer.step()  # update params
        return self

    def _fit_combined(self, generator, recovery_generator, n_steps):
        self.net.train()  # train mode
        self.adv_net.train()  # train mode
        for i, (X_batch, y_batch, z_batch) in enumerate(islice(generator, n_steps)):
            X_batch = to_torch(X_batch, cuda=self.cuda_flag)
            y_batch = to_torch(y_batch, cuda=self.cuda_flag)
            z_batch = to_torch(z_batch, cuda=self.cuda_flag).view(-1, 1)
            self.combined_optimizer.zero_grad()  # zero-out the gradients because they accumulate by default
            y_pred = self.net.forward(X_batch)
            z_pred = self.adv_net.forward(y_pred)
            net_loss = self.net_criterion(y_pred, y_batch)
            adv_loss = self.adv_criterion(z_pred, z_batch)
            loss = net_loss - (self.trade_off * adv_loss)
            self.adv_loss.append(adv_loss.item())
            self.net_loss.append(adv_loss.item())
            self.comb_loss.append(loss.item())
            loss.backward()  # compute gradients
            self.combined_optimizer.step()  # update params
            # Adversarial recovery
            self._fit_recovery(recovery_generator, self.n_recovery_steps)
        return self

    def save(self, dir_path):
        path = os.path.join(dir_path, 'net_weights.pth')
        torch.save(self.net.state_dict(), path)

        path = os.path.join(dir_path, 'adv_net_weights.pth')
        torch.save(self.adv_net.state_dict(), path)

        path = os.path.join(dir_path, 'Scaler.pkl')
        joblib.dump(self.scaler, path)

        path = os.path.join(dir_path, 'losses.json')
        with open(path, 'w') as f:
            json.dump(self.get_losses(), f)
        return self

    def load(self, dir_path):
        path = os.path.join(dir_path, 'net_weights.pth')
        if self.cuda:
            self.dnet.load_state_dict(torch.load(path))
        else:
            self.dnet.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'adv_net_weights.pth')
        if self.cuda:
            self.rnet.load_state_dict(torch.load(path))
        else:
            self.rnet.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

        path = os.path.join(dir_path, 'Scaler.pkl')
        self.scaler = joblib.load(path)
        return self



class PivotClassifier(Pivot):

    def predict(self, X):
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        X = to_numpy(X)
        if self.scaler is not None :
            X = self.scaler.transform(X)
        proba = self._predict_proba(X)
        return proba

    def _predict_proba(self, X):
        y_proba = []
        self.net.eval()  # evaluation mode
        for X_batch in OneEpoch(X, batch_size=self.batch_size):
            X_batch = X_batch.astype(np.float32)
            with torch.no_grad():
                X_batch = to_torch(X_batch, cuda=self.cuda_flag)
                proba_batch = F.softmax(self.net.forward(X_batch), dim=1).cpu().data.numpy()
            y_proba.extend(proba_batch)
        y_proba = np.array(y_proba)
        return y_proba




class PivotRegressor(Pivot):
    # TODO

    def predict(self, X):
        X = to_numpy(X)
        if self.scaler is not None :
            X = self.scaler.transform(X)
        # TODO : extract regressed
        return None
