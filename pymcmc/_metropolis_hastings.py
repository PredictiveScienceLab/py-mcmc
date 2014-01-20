"""
A simple implementation of the Metropolis-Hastings algorithm.

Author:
    Ilias Bilionis
"""


__all__ = ['MetropolisHastings']


from . import Model
from . import Proposal
from . import MALAProposal
import numpy as np
import math


class MetropolisHastings(object):

    """
    A simple implementation of the Metropolis-Hastings algorithm.

    :param model:   The model to sample from.
    :type model:    :class:`pymcmc.Model`

    """

    def __init__(self, model, proposal=MALAProposal()):
        """
        Initialize the object.
        """
        assert isinstance(model, Model)
        self.model = model
        assert isinstance(proposal, Proposal)
        self.proposal = proposal

    @property
    def acceptance_rate(self):
        """
        Get the acceptance rate.
        """
        return self.accepted / self.count

    def sample(self, num_samples, num_thin=1, num_burn=0, init_state=None):
        """
        Take samples from the target.

        :param num_samples:     The number of samples to take.
        :type num_samples:      int
        :param num_thin:        Record the samples every ``num_thin``.
        :type num_thin:         int
        :param num_burn:        Start collecting samples after ``num_burn``
                                samples have been burned.
        :type num_burn:         int
        :param init_state:      Set the initial state of the chain. If ``None``,
                                then the initial state of the model is used.
        :type init_state:       dict
        """
        if init_state is not None:
            self.model.__setstate__(init_state)
        self.accepted = 0.
        self.count = 0.
        all_log_p = []
        state = []
        for i in xrange(num_samples):
            new_state, log_p = self.proposal.propose(self.model)
            log_u = math.log(np.random.rand())
            if log_u <= log_p:
                self.model.__setstate__(new_state)
                self.accepted += 1
            self.count += 1
            if i > num_burn and i % num_thin == 0:
                print i + 1, self.model.log_p, self.acceptance_rate
                all_log_p.append(self.model.log_p)
                state.append(self.model.params)
        return np.array(all_log_p), np.array(state)
