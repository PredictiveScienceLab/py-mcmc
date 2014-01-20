"""
Some prior densities that are not GPy.

Author:
    Ilias Bilionis
"""


__all__ = ['UninformativeScalePrior']


import GPy
import numpy as np


class UninformativeScalePrior(GPy.core.priors.Prior):

    domain = GPy.core.domains.POSITIVE

    def lnpdf(self, x):
        return -np.log(x)

    def lnpdf_grad(self, x):
        return -1. / x
