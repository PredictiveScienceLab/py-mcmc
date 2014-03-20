"""
Some prior densities that are not GPy.

Author:
    Ilias Bilionis
"""


__all__ = ['Prior', 'GaussianPrior', 'LogGaussianPrior',
           'MultivariateGaussianPrior', 'GammaPrior',
           'InverseGammaPrior', 'UninformativeScalePrior',
           'UninformativePrior']


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


class UninformativePrior(Prior):

    """
    An uninformative prior for bounded domains or the real line.
    The default is the real line.

    :param lower:   The lower bound of the distribution (It can be
                    ``-np.inf``).
    :type lower:    float
    :param upper:   The upper bound of the distribution (It can be
                    ``np.inf``).
    """

    domain = None

    def __init__(self, lower=-np.inf, upper=np.inf):
        """
        Initialize the object.
        """
        assert lower < upper
        if lower == -np.inf and upper == np.inf:
            self.domain = domains.REAL
            self.log_length = 0.
        elif lower == -np.inf or upper == np.inf:
            self.domain = domains.BOUNDED
            self.log_length = 0.
        else:
            self.domain = domains.BOUNDED
            self.log_length = np.log(upper - lower)
        self.lower = lower
        self.upper = upper

    def lnpdf(self, x):
        """
        :param x:   The value of the parameter.
        :type x:    :class:`numpy.ndarray`
        :returns:   The logarithm of the probability
        """
        if x < self.lower or x > self.upper:
            return -np.inf
        else:
            return self.log_length

    def lnpdf_grad(self, x):
        """
        :param x:   The value of the parameter.
        :type x:    :class:`numpy.ndarray`
        :returns:   The gradient of the logarithm of the probability.
        """
        return 0.
