# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEFAULT_DIR


def set_plot_config():
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("poster")

    mpl.rcParams['figure.figsize'] = [8.0, 6.0]
    mpl.rcParams['figure.dpi'] = 80
    mpl.rcParams['savefig.dpi'] = 100

    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 17
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 'large'
    mpl.rcParams['figure.titlesize'] = 'medium'
    mpl.rcParams['lines.markersize'] = np.sqrt(30)


def plot_test_distrib(y_proba, y_test, title="no title", 
                      directory=DEFAULT_DIR, fname='test_distrib.png', classes=('b', 's')):
    logger = logging.getLogger()
    # logger.info( 'Test accuracy = {} %'.format(100 * model.score(X_test, y_test)) )
    try:
        sns.distplot(y_proba[y_test==0, 1], label=classes[0])
        sns.distplot(y_proba[y_test==1, 1], label=classes[1])
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        logger.warning('Plot test distrib failed')
        logger.warning(str(e))


def plot_ROC(fpr, tpr, title="no title", directory=DEFAULT_DIR, fname='roc.png'):
    from sklearn.metrics import auc
    logger = logging.getLogger()
    try:
        plt.plot(fpr, tpr, label='AUC {}'.format(auc(fpr, tpr)))
        plt.title(title)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        logger.warning('Plot ROC failed')
        logger.warning(str(e))


def plot_losses(losses, title='no title', directory=DEFAULT_DIR, fname='losses.png'):
    logger = logging.getLogger()
    try:
        for name, values in losses.items():
            plt.plot(values, label=name)
        plt.title(title)
        plt.xlabel('# iter')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        logger.warning('Plot losses failed')
        logger.warning(str(e))


def plot_pivotal_distrib(decision_minus, decision_zero, decision_plus, title="no title", 
                      directory=DEFAULT_DIR, fname='decision_distrib.png'):
    logger = logging.getLogger()
    # logger.info( 'Test accuracy = {} %'.format(100 * model.score(X_test, y_test)) )
    try:
        plt.hist(decision_minus, bins=50, density=1, histtype="step", label='z=0')
        plt.hist(decision_zero, bins=50, density=1, histtype="step", label='z=1')
        plt.hist(decision_plus, bins=50, density=1, histtype="step", label='z=2')
        plt.ylim(0,4)
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        logger.warning('Plot pivotal distrib failed')
        logger.warning(str(e))


def plot_decision_surface(xi, yi, zi, directory=DEFAULT_DIR, fname="surface-plain.png"):
    logger = logging.getLogger()
    try:
        plt.contourf(xi, yi, zi, 20, cmap=plt.cm.viridis,
                          vmax=1.0, vmin=0.0)
        plt.colorbar()
        plt.scatter([0], [0], c="red", linewidths=0, label=r"$\mu_0$")
        plt.scatter([1], [0], c="blue", linewidths=0, label=r"$\mu_1|Z=z$")
        plt.scatter([1], [0+1], c="blue", linewidths=0)
        plt.scatter([1], [0+2], c="blue", linewidths=0)
        plt.text(1.1, 0-0.05, "$Z=-\sigma$", color="k")
        plt.text(1.1, 1-0.05, "$Z=0$", color="k")
        plt.text(1.1, 2-0.05, "$Z=+\sigma$", color="k")
        plt.xlim(-1, 2)
        plt.ylim(-1, 3)
        plt.xlabel('$x_0$')
        plt.ylabel('$x_1$')
        plt.legend(loc="upper left", scatterpoints=1)
        plt.savefig(os.path.join(directory, fname))
        plt.clf()
    except Exception as e:
        logger.warning('Plot pivotal distrib failed')
        logger.warning(str(e))
