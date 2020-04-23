# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


import numpy as np

from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from visual import plot_test_distrib
from visual import plot_ROC
from visual import plot_losses
from visual import plot_pivotal_distrib
from visual import plot_decision_surface

from config import DEFAULT_DIR

def evaluate_classifier(model, X, y, w=None, prefix='test', suffix=''
                        , model_name='anonymous', directory=DEFAULT_DIR):
    results = {}
    y_proba = model.predict_proba(X)
    y_decision = y_proba[:, 1]
    y_predict = model.predict(X)
    accuracy = np.mean(y_predict == y)

    results[f'{prefix}_accuracy{suffix}'] = accuracy
    fname = f'{prefix}_distrib{suffix}.png'
    plot_test_distrib(y_proba, y, title=f"{prefix} decision distribution", directory=directory, fname=fname)

    fpr, tpr, thresholds = roc_curve(y, y_decision, pos_label=1)
    results[f"{prefix}_auc{suffix}"] = auc(fpr, tpr)
    fname = f'{prefix}_roc{suffix}.png'
    plot_ROC(fpr, tpr, title=f"{prefix} ROC {model_name}", directory=directory, fname=fname)

    return results


def evaluate_neural_net(model, prefix='', suffix='', model_name='anonymous', directory=DEFAULT_DIR):
    losses = model.get_losses()
    plot_losses(losses, title=model_name, directory=directory)
    for loss_name, loss_values in losses.items():
        plot_losses({loss_name: loss_values}, title=f"{loss_name} -- {model_name}", directory=directory, fname=f"{loss_name}.png")
    results = {loss_name: loss_values[-1] for loss_name, loss_values in losses.items() if loss_values}
    return results


def evaluate_pivotal(model, generator, prefix='', suffix='', model_name='anonymous', directory=DEFAULT_DIR):
    X_test, y_test = generator.sample_event(-1.0, 0.5, 1000)
    proba = model.predict_proba(X_test)
    decision_minus = proba[:, 1]

    X_test, y_test = generator.sample_event(0.0, 0.5, 1000)
    proba = model.predict_proba(X_test)
    decision = proba[:, 1]

    X_test, y_test = generator.sample_event(1.0, 0.5, 1000)
    proba = model.predict_proba(X_test)
    decision_plus = proba[:, 1]

    plot_pivotal_distrib(decision_minus, decision, decision_plus, 
        title=f"{prefix} decision distribution", directory=directory)

    xi = np.linspace(-1., 2., 100)
    yi = np.linspace(-1., 3, 100)
    xx, yy = np.meshgrid(xi, yi)
    proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    zi = proba[:, 1]
    zi = zi.reshape(xx.shape)

    plot_decision_surface(xi, yi, zi, directory=directory)


