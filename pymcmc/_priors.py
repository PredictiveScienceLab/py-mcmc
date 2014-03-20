"""
Some prior densities that are not GPy.

Author:
    Ilias Bilionis
"""


__all__ = ['Prior', 'GaussianPrior', 'LogGaussianPrior',
           'MultivariateGaussianPrior', 'GammaPrior',
           'InverseGammaPrior', 'UninformativeScalePrior']


import GPy.core.domains as domains
import GPy.core.priors as priors
Prior = priors.Prior
GaussianPrior = priors.Gaussian
LogGaussianPrior = priors.LogGaussian
MultivariateGaussianPrior = priors.MultivariateGaussian
GammaPrior = priors.Gamma
InverseGammaPrior = priors.inverse_gamma
import numpy as np


class UninformativeScalePrior(Prior):

    """
    An uninformative prior.
    """

    domain = domains.POSITIVE

    def lnpdf(self, x):
        """
        :param x:   The value of the parameter.
        :type x:    :class:`numpy.ndarray`
        :returns:   The logarithm of the probability.
        """
        return -np.log(x)

    def lnpdf_grad(self, x):
        """
        :param x:   The value of the parameter.
        :type x:    :class:`numpy.ndarray`
        :returns:   The gradient of the logarithm of the probability.
        """
        return -1. / x
