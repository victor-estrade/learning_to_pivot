#!/usr/bin/env python
# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from config import OUT_DIRECTORY

from toys import Generator
from visual import set_plot_config
set_plot_config()
from utils import set_logger
set_logger()

from pivot import PivotClassifier
from pivot import PivotBinaryClassifier
from archi import F3Classifier
from archi import F3GausianMixtureDensity

from criterion import ADVLoss

from evaluation import evaluate_classifier
from evaluation import evaluate_neural_net
from evaluation import evaluate_pivotal

SEED = 42
N_CV_ITER = 3
TRADE_OFF = 1.0
DIRECTORY = os.path.join(OUT_DIRECTORY, f"pivot_mdn-{TRADE_OFF}")

def main():
    print("hello world")
    os.makedirs(DIRECTORY, exist_ok=True)
    results = [run(i_cv) for i_cv in range(N_CV_ITER)]
    result_table = pd.DataFrame(results)
    result_table.to_csv(os.path.join(DIRECTORY, 'results.csv'))
    print(result_table)


def run(i_cv):
    N_SIG = 15000
    N_BKG = N_SIG
    N_SAMPLES = N_SIG + N_BKG
    mix = N_SIG / N_SAMPLES
    model_name = 'PivotClassifier'
    directory = os.path.join(DIRECTORY, f'cv_{i_cv}')
    os.makedirs(directory, exist_ok=True)
    print(f'running iter {i_cv}...')

    results = {'i_cv': i_cv}

    seed = SEED + 5 * i_cv
    train_seed = seed
    test_seed = seed + 1

    # Generate training data
    generator = Generator(train_seed)
    z_train = generator.sample_nuisance(N_SIG)
    X_train, y_train = generator.sample_event(z_train, mix, N_SAMPLES)
    z_train = np.concatenate((np.zeros(N_BKG), z_train), axis=0)

    # Define Pivot
    net = F3Classifier(n_in=2, n_out=1)
    adv_net = F3GausianMixtureDensity(n_in=1, n_components=5)
    # net_criterion = nn.CrossEntropyLoss()
    net_criterion = nn.BCEWithLogitsLoss()
    adv_criterion = ADVLoss()
    
    # ADAM
    # Reducing optimizer inertia with lower beta1 and beta2 help with density network
    net_optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.5, 0.9))
    adv_optimizer = optim.Adam(adv_net.parameters(), lr=1e-3, betas=(0.5, 0.9))
    # SGD
    # net_optimizer = optim.SGD(net.parameters(), lr=1e-3)
    # adv_optimizer = optim.SGD(adv_net.parameters(), lr=1e-3)

    # model = PivotClassifier(net, adv_net, net_criterion, adv_criterion, TRADE_OFF, net_optimizer, adv_optimizer,
    model = PivotBinaryClassifier(net, adv_net, net_criterion, adv_criterion, TRADE_OFF, net_optimizer, adv_optimizer,
                n_net_pre_training_steps=500, n_adv_pre_training_steps=3000,
                n_steps=2000, n_recovery_steps=20,
                batch_size=128, rescale=True, cuda=False, verbose=0)
    
    # Train Pivot
    model.fit(X_train, y_train, z_train)

    # Generate testing data
    generator = Generator(test_seed)
    z_test = generator.sample_nuisance(N_SIG)
    X_test, y_test = generator.sample_event(z_test, mix, N_SAMPLES)

    # Evaluation
    r = evaluate_neural_net(model, prefix='train', model_name=model_name, directory=directory)
    results.update(r)    

    evaluate_pivotal(model, generator, prefix='test', model_name=model_name, directory=directory)

    r = evaluate_classifier(model, X_train, y_train, prefix='train', model_name=model_name, directory=directory)
    results.update(r)
    r = evaluate_classifier(model, X_test, y_test, prefix='test', model_name=model_name, directory=directory)
    results.update(r)

    return results


if __name__ == '__main__':
    main()