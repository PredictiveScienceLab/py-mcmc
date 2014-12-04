"""
Some prior densities that are not GPy.

Author:
    Ilias Bilionis
"""


__all__ = ['Prior', 'GaussianPrior', 'LogGaussianPrior',
           'MultivariateGaussianPrior', 'GammaPrior',
           'InverseGammaPrior', 'UninformativeScalePrior',
           'UninformativePrior']


from GPy import priors
Prior = priors.Prior
GaussianPrior = priors.Gaussian
LogGaussianPrior = priors.LogGaussian
MultivariateGaussianPrior = priors.MultivariateGaussian
GammaPrior = priors.Gamma
InverseGammaPrior = priors.InverseGamma
import numpy as np


class UninformativeScalePrior(Prior):

    """
    An uninformative prior.
    """

    domain = priors._POSITIVE

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

    def __str__(self):
        """
        Return a string representation of the object.
        """
        return 'Uninformative Scale Prior: p(x) = 1/x, x > 0'


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
            self.domain = priors._REAL
            self.log_length = 0.
        elif lower == -np.inf or upper == np.inf:
            self.domain = priors._BOUNDED
            self.log_length = 0.
        else:
            self.domain = priors._BOUNDED
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

    def __str__(self):
        """
        Return a string representation of the object.
        """
        if (self.domain == priors._REAL or
            self.upper == np.inf or
            self.lower == -np.inf):
            return 'Uninformative Prior: p(x) = 1'
        else:
            return 'Uninformative Prior: p(x) = |D|*I_D(x)'

    def rvs(self, n):
        """
        Draw random samples from the probability density.

        It works only for BOUNDED domains.

        :param n:   The number of samples to draw.
        :type n:    int
        """
        if self.upper < np.inf and self.lower > -np.inf:
            return np.random.rand(n) * (self.upper - self.lower) + self.lower
        else:
            raise RuntimeError('Cannot draw samples from an improper '
                               ' probability distribution.')
