"""
A random walk proposal.

Author:
    Ilias Bilionis
"""


__all__ = ['RandomWalkProposal']


import numpy as np
from . import SimpleProposal


class RandomWalkProposal(SimpleProposal):

    """
    A random walk proposal.
    
    :param name:    A name for the object.
    :type name:     str
    :param cov:     A covariance matrix (must have the same dimensions
                    as the model we are going to use).
    :type cov:      2D numpy array
    :param scale:   The scale of the proposal.
    :type scale:    float
    """

    def __init__(self, cov=None, scale=1., name='Simple Proposal'):
        """
        Initialize the object.
        """
        if cov is None:
            cov = 1.
        self.cov = cov
        self.scale = scale
        super(SimpleProposal, self).__init__(name=name)

    def _eval_all(self, old_params):
        """
        Here it does not matter what log_p_old_cond_new and log_p_new_cond_old
        really are as soon as they are exactly the same. This is because the
        proposal is symmetric.
        """
        if isinstance(self.cov, float):
            self.cov = self.cov * np.eye(old_params.shape[0])
        new_params = np.random.multivariate_normal(old_params,
                                                   self.cov * self.scale ** 2)
        return new_params, 0., 0.
