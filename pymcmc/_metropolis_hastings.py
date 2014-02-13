"""
A simple implementation of the Metropolis-Hastings algorithm.

Author:
    Ilias Bilionis
"""


__all__ = ['MetropolisHastings']


from . import Model
from . import Proposal
from . import MALAProposal
from . import DataBase
import numpy as np
import math


class MetropolisHastings(object):

    """
    A simple implementation of the Metropolis-Hastings algorithm.

    :param model:       The model to sample from.
    :type model:        :class:`pymcmc.Model`
    :param proposal:    The MCMC proposal.
    :type proposal:     :class:`pymcmc.Proposal`
    :param db_filename: A filename to store the MCMC chains. If ``None``, then
                        nothing is saved.
    :type db_filename:  str
    """

    def __init__(self, model, proposal=MALAProposal(),
                 db_filename=None):
        """
        Initialize the object.
        """
        assert isinstance(model, Model)
        self.model = model
        assert isinstance(proposal, Proposal)
        self.proposal = proposal
        self.db_filename = db_filename
        if self.has_db:
            self.db = DataBase(db_filename, model.__getstate__(),
                               proposal.__getstate__())

    @property
    def has_db(self):
        """
        Return ``True`` if we are using a database, ``False`` otherwise.
        """
        return self.db_filename is not None

    @property
    def acceptance_rate(self):
        """
        Get the acceptance rate.
        """
        return self.accepted / self.count

    def sample(self, num_samples, num_thin=1, num_burn=0,
               init_model_state=None, init_proposal_state=None,
               verbose=False):
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
        if init_model_state is not None:
            self.model.__setstate__(init_model_state)
        if init_proposal_state is not None:
            self.proposal.__setstate__(init_proposal_state)
        self.accepted = 0.
        self.count = 0.
        if self.has_db:
            self.db.add_proposal(self.proposal.__getstate__())
            self.db.create_new_chain()
        for i in xrange(num_samples):
            new_state, log_p = self.proposal.propose(self.model)
            log_u = math.log(np.random.rand())
            if log_u <= log_p:
                self.model.__setstate__(new_state)
                self.accepted += 1
            self.count += 1
            if i > num_burn and i % num_thin == 0:
                if self.has_db:
                    self.db.add_chain_record(i + 1, self.accepted,
                                             self.model.__getstate__())
                if verbose:
                    print i + 1, self.model.log_p, self.acceptance_rate
