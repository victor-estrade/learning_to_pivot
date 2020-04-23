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
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier

from config import OUT_DIRECTORY

from toys import Generator
from visual import set_plot_config
set_plot_config()
from utils import set_logger
set_logger()

from evaluation import evaluate_classifier
from evaluation import evaluate_pivotal

DIRECTORY = os.path.join(OUT_DIRECTORY, "gradient_boosting")
SEED = 42
N_CV_ITER = 3

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
    model_name = 'GradientBoosting'
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

    # Train classifier
    model = GradientBoostingClassifier(n_estimators=400, learning_rate=5e-2)
    model.fit(X_train, y_train)

    # Generate testing data
    generator = Generator(test_seed)
    z_test = generator.sample_nuisance(N_SIG)
    X_test, y_test = generator.sample_event(z_test, mix, N_SAMPLES)

    # Evaluation
    r = evaluate_classifier(model, X_train, y_train, prefix='train', model_name=model_name, directory=directory)
    results.update(r)
    r = evaluate_classifier(model, X_test, y_test, prefix='test', model_name=model_name, directory=directory)
    results.update(r)
    
    evaluate_pivotal(model, generator, prefix='test', model_name=model_name, directory=directory)

    return results


if __name__ == '__main__':
    main()