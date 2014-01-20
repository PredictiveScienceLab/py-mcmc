"""
This is a Metropolis Adjusted Langevin Algorithm (MALA) proposal.

Author:
    Ilias Bilionis
"""


__all__ = ['MALAProposal']


import numpy as np
from scipy.stats import norm
from . import GradProposal


class MALAProposal(GradProposal):

    """
    A MALA proposal.

    :param dt:      The time step. The larger you pick it, the bigger the steps
                    you make and the acceptance rate will go down.
    :type dt:       float
    :param name:    A name for the object.
    :type name:     str
    """

    def __init__(self, dt=1., name='MALA Proposal'):
        """
        Initialize the object.
        """
        self.dt = dt
        super(MALAProposal, self).__init__(name=name)

    def _sample(self, old_params, old_grad_params):
        return (old_params -
                0.5 * self.dt ** 2 * old_grad_params +
                self.dt * np.random.randn(old_params.shape[0]))

    def __call__(self, new_params, old_params, old_grad_params):
        return np.sum(norm.logpdf(new_params,
                                  old_params -
                                  0.5 * self.dt ** 2 * old_grad_params,
                                  self.dt ** 2))
