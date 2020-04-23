#!/usr/bin/env python
# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from config import OUT_DIRECTORY

from toys import Generator

DIRECTORY = os.path.join(OUT_DIRECTORY, "explore")
SEED = None


def main():
    print("hello world")
    os.makedirs(DIRECTORY, exist_ok=True)
    N_SIG = 1000
    N_BKG = 1000
    N_SAMPLES = N_SIG + N_BKG
    mix = N_SIG / N_SAMPLES
    generator = Generator(SEED)

    # Without nuisance parameter
    z = 0
    X, label = generator.sample_event(z, mix, N_SAMPLES)

    df = pd.DataFrame(X, columns=["x_0","x_1"])
    df['label'] = label
    g = sns.PairGrid(df, vars=["x_0","x_1"], hue='label')
    g = g.map_upper(sns.scatterplot)
    g = g.map_diag(sns.kdeplot)
    g = g.map_lower(sns.kdeplot, n_levels=6)
    g.fig.suptitle('Without nuisance parameter (z=0)')
    g = g.add_legend()
    # g = g.map_offdiag(sns.kdeplot, n_levels=6)
    g.savefig(os.path.join(DIRECTORY, 'pairgrid_no_z.png'))
    plt.clf()


    # With random nuisance parameter
    z = generator.sample_nuisance(N_SIG)
    X, label = generator.sample_event(z, mix, N_SAMPLES)

    sns.distplot(z, label='z')
    plt.xlabel("z")
    plt.ylabel("distribution")
    plt.title("Distribution of nuisance parameter z")
    plt.legend()
    plt.savefig(os.path.join(DIRECTORY, 'z_distrib.png'))

    df = pd.DataFrame(X, columns=["x_0","x_1"])
    df['label'] = label
    g = sns.PairGrid(df, vars=["x_0","x_1"], hue='label')
    g = g.map_upper(sns.scatterplot)
    g = g.map_diag(sns.kdeplot)
    g = g.map_lower(sns.kdeplot, n_levels=6)
    g.fig.suptitle('With nuisance parameter ( z = normal(0, 1) )')
    g = g.add_legend()
    # g = g.map_offdiag(sns.kdeplot, n_levels=6)
    g.savefig(os.path.join(DIRECTORY, 'pairgrid_z_normal.png'))
    plt.clf()


    # With nuisance parameter z = 1
    z = 1.0
    X, label = generator.sample_event(z, mix, N_SAMPLES)

    df = pd.DataFrame(X, columns=["x_0","x_1"])
    df['label'] = label
    g = sns.PairGrid(df, vars=["x_0","x_1"], hue='label')
    g = g.map_upper(sns.scatterplot)
    g = g.map_diag(sns.kdeplot)
    g = g.map_lower(sns.kdeplot, n_levels=6)
    g.fig.suptitle('With nuisance parameter ( z = 1 )')
    g = g.add_legend()
    # g = g.map_offdiag(sns.kdeplot, n_levels=6)
    g.savefig(os.path.join(DIRECTORY, 'pairgrid_z_1.png'))
    plt.clf()





if __name__ == '__main__':
	main()