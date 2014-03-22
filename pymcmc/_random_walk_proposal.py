"""
A random walk proposal.

Author:
    Ilias Bilionis
"""


__all__ = ['RandomWalkProposal']


import numpy as np
from . import SymmetricProposal
from . import SingleParameterTunableProposalConcept


class RandomWalkProposal(SymmetricProposal, SingleParameterTunableProposalConcept):

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

    def __init__(self, cov=None, scale=1., **kwargs):
        """
        Initialize the object.
        """
        if cov is None:
            cov = 1.
        self.cov = cov
        self.scale = scale
        if not kwargs.has_key('name'):
            kwargs['name'] = 'Random Walk Proposal'
        kwargs['param_name'] = 'scale'
        SymmetricProposal.__init__(self, **kwargs)
        SingleParameterTunableProposalConcept.__init__(self, **kwargs)

    def _sample(self, old_params):
        if isinstance(self.cov, float):
            self.cov = self.cov * np.eye(old_params.shape[0])
        new_params = np.random.multivariate_normal(old_params,
                                                   self.cov * self.scale ** 2)
        return new_params

    def __getstate__(self):
        """
        Get the state of the object.
        """
        state = SymmetricProposal.__getstate__(self)
        state['cov'] = self.cov
        state['scale'] = self.scale
        tuner_state = SingleParameterTunableProposalConcept.__getstate__()
        return dict(state.items() + tuner_state.items())

    def __setstate__(self, state):
        """
        Set the state of the object.
        """
        SymmetricProposal.__setstate__(self, state)
        self.cov = state['cov']
        self.scale = state['scale']
        SingleParameterTunableProposalConcept.__setstate__(state)
