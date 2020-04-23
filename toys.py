# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
from scipy import stats

SEED = 42

class Generator():
    """
    Replicates the toy example in https://arxiv.org/abs/1611.01046
    """
    def __init__(self, seed=None):
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)

    def reset(self):
        self.random = np.random.RandomState(seed=self.seed)

    def get_bkg_mean(self):
        return np.array([0., 0.])

    def get_bkg_cov(self):
        return np.array([[1., -0.5], [-0.5, 1.]])

    def get_sig_mean(self, z):
        return np.array([1., 1.+z])

    def get_sig_cov(self):
        return np.eye(2)

    def get_nuisance_mean(self):
        return 0.0

    def get_nuisance_std(self):
        return 1.0

    def sample_nuisance(self, n_samples):
        z_mean = self.get_nuisance_mean()
        z_cov  = self.get_nuisance_std()
        z      = stats.norm.rvs(loc=z_mean, scale=z_cov, size=n_samples)
        return z

    def sample_event(self, z, mix, n_samples=1):
        n_sig = int(mix * n_samples)
        n_bkg = n_samples - n_sig
        x = self._generate_vars(z, n_bkg, n_sig)
        labels = self._generate_labels(n_bkg, n_sig)
        return x, labels

    def generate(self, z, mix, n_samples=1000):
        n_bkg = n_samples // 2
        n_sig = n_samples // 2
        X, y, w = self._generate(z, mix, n_bkg=n_bkg, n_sig=n_sig)
        return X, y, w

    def pdf(self, x, z, mix):
        """
        Computes the likelihood : p(x | z, mix)
        with:
        $$
            p(x | z, mix) = (1-mix) f_b(x) + mix f_s(x|z)
        $$

        $$
        f_b (x) = \mathcal N \left ( (x_0, x_1) | (0, 0) 
        \begin{bmatrix} 1 & -0.5 \\ -0.5 & 1 \end{bmatrix} \right )
        $$

        $$
        f_s (x|z) = \mathcal N \left ( (x_0, x_1) | (1, 1+z) 
        \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right )
        $$
        """
        sig_pdf = self._sig_pdf(x, z)
        bkg_pdf = self._bkg_pdf(x)
        pdf = mix * sig_pdf + (1-mix) * bkg_pdf
        return pdf

    def logpdf(self, x, z, mix):
        """
        Computes log likelihood : log p(x | z, mix)
        with:
        $$
            p(x | z, mix) = (1-mix) f_b(x) + mix f_s(x|z)
        $$

        $$
        f_b (x) = \mathcal N \left ( (x_0, x_1) | (0, 0) 
        \begin{bmatrix} 1 & -0.5 \\ -0.5 & 1 \end{bmatrix} \right )
        $$

        $$
        f_s (x|z) = \mathcal N \left ( (x_0, x_1) | (1, 1+z) 
        \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right )
        $$
        """
        pdf = self.pdf(x, z, mix)
        logpdf = np.log(pdf)
        return logpdf

    def nll(self, data, z, mix):
        """
        Computes the negative log likelihood of the data given mix and z.
        """
        nll = - self.logpdf(data, z, mix).sum()
        return nll

    def _generate(self, z, mix, n_bkg=1000, n_sig=50):
        """
        """
        X = self._generate_vars(z, n_bkg, n_sig)
        y = self._generate_labels(n_bkg, n_sig)
        w = self._generate_weights(mix, n_bkg, n_sig, self.n_expected_events)
        return X, y, w

    def _generate_sig(self, z, n_samples):
        sig_mean = self.get_sig_mean(0.)
        sig_cov  = self.get_sig_cov()
        x_sig    = stats.multivariate_normal.rvs(mean=sig_mean, cov=sig_cov, size=n_samples)
        x_sig[:, 1] += z 
        return x_sig

    def _generate_bkg(self, n_samples):
        bkg_mean = self.get_bkg_mean()
        bkg_cov  = self.get_bkg_cov()
        x_bkg    = stats.multivariate_normal.rvs(mean=bkg_mean, cov=bkg_cov, size=n_samples)
        return x_bkg

    def _generate_vars(self, z, n_bkg, n_sig):
        x_bkg = self._generate_bkg(n_bkg)
        x_sig = self._generate_sig(z, n_sig)
        x = np.concatenate([x_bkg, x_sig], axis=0)
        return x

    def _generate_labels(self, n_bkg, n_sig):
        y_b = np.zeros(n_bkg)
        y_s = np.ones(n_sig)
        y = np.concatenate([y_b, y_s], axis=0)
        return y

    def _generate_weights(self, mix, n_bkg, n_sig, n_expected_events):
        w_b = np.ones(n_bkg) * (1-mix) * n_expected_events/n_bkg
        w_s = np.ones(n_sig) * mix * n_expected_events/n_sig
        w = np.concatenate([w_b, w_s], axis=0)
        return w

    def _sig_pdf(self, x, z):
        sig_mean = self.get_sig_mean(z)
        sig_cov  = self.get_sig_cov()
        sig_pdf  = stats.multivariate_normal.rvs(x, mean=sig_mean, cov=sig_cov)
        return sig_pdf

    def _bkg_pdf(self, x):
        bkg_mean = self.get_bkg_mean()
        bkg_cov  = self.get_bkg_cov()
        bkg_pdf  = stats.multivariate_normal.rvs(x, mean=bkg_mean, cov=bkg_cov)
        return bkg_pdf

