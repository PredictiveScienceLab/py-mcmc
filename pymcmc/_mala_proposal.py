"""
This is a Metropolis Adjusted Langevin Algorithm (MALA) proposal.

Author:
    Ilias Bilionis
"""


__all__ = ['MALAProposal']


import numpy as np
from scipy.stats import norm
from . import GradProposal
from . import SingleParameterTunableProposalConcept


class MALAProposal(GradProposal, SingleParameterTunableProposalConcept):

    """
    A MALA proposal.

    :param dt:      The time step. The larger you pick it, the bigger the steps
                    you make and the acceptance rate will go down.
    :type dt:       float
    
    The rest of the keyword arguments is what you would find in:
        + :class:`pymcmc.GradProposal`
        + :class:`pymcmc.SingleParameterTunableProposal`

    """

    def __init__(self, dt=1., **kwargs):
        """
        Initialize the object.
        """
        self.dt = dt
        if not kwargs.has_key('name'):
            kwargs['name'] = 'MALA Proposal'
        kwargs['param_name'] = 'dt'
        GradProposal.__init__(self, **kwargs)
        SingleParameterTunableProposalConcept.__init__(self, **kwargs)

    def _sample(self, old_params, old_grad_params):
        return (old_params +
                0.5 * self.dt ** 2 * old_grad_params +
                self.dt * np.random.randn(old_params.shape[0]))

    def __call__(self, new_params, old_params, old_grad_params):
        return np.sum(norm.logpdf(new_params,
                                  old_params +
                                  0.5 * self.dt ** 2 * old_grad_params,
                                  self.dt ** 2))

    def __getstate__(self):
        state = GradProposal.__getstate__(self)
        state['dt'] = self.dt
        tuner_state = SingleParameterTunableProposalConcept.__getstate__(self)
        return dict(state.items() + tuner_state.items())

    def __setstate__(self, state):
        GradProposal.__setstate__(self, state)
        self.dt = state['dt']
        SingleParameterTunableProposalConcept.__setstate__(self, state['tuner'])
