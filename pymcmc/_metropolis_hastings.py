"""
A simple implementation of the Metropolis-Hastings algorithm.

Author:
    Ilias Bilionis
"""


__all__ = ['MetropolisHastings']


from . import Model
from . import GPyModel
from . import Proposal
from . import TunableProposalConcept
from . import RandomWalkProposal
from . import MALAProposal
from . import DataBase
import GPy
import numpy as np
import math
import sys


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

    def __init__(self, model, proposal=None,
                 db_filename=None):
        """
        Initialize the object.
        """
        if isinstance(model, GPy.core.Model):
            model = GPyModel(model)
        else:
            assert isinstance(model, Model)
        self.model = model
        if proposal is None:
            try:
                y = model.grad_log_p
                proposal = MALAProposal()
            except:
                proposal = RandomWalkProposal()
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
               start_tuning_after=0, stop_tuning_after=None,
               tuning_frequency=1000,
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

        Tuning parameters:
        :param start_tuning_after:  Start tuning after this sample. If you do
                                    not want tuning, then all you have to do
                                    is set this equal to ``None``, or to
                                    ``num_samples``.
        :type start_tuning_after:   int
        :param stop_tuning_after:   Stop tuning after this sample. If ``None``, then
                                    we never stop tuning.
        :type stop_tuning_after:    int
        :param tuning_frequecny:    Tune every so many samples.
        :type param:                int
        """
        # Set the initial state of the model.
        if init_model_state is not None:
            self.model.__setstate__(init_model_state)
        if init_proposal_state is not None:
            self.proposal.__setstate__(init_proposal_state)
        # Check the tuning parameters
        start_tuning_after = (num_samples if start_tuning_after is None
                              else start_tuning_after)
        stop_tuning_after = (num_samples if stop_tuning_after is None
                             else stop_tuning_after)
        # Initialize counters
        self.accepted = 0.
        self.count = 0.
        # Initialize the database
        if self.has_db:
            self.db.add_proposal(self.proposal.__getstate__())
            self.db.create_new_chain()
        # Start sampling
        for i in xrange(num_samples):
            # MCMC Step
            new_state, log_p = self.proposal.propose(self.model)
            log_u = math.log(np.random.rand())
            if log_u <= log_p:
                self.model.__setstate__(new_state)
                self.accepted += 1
            self.count += 1
            # Output
            if i > num_burn and i % num_thin == 0:
                # To database
                if self.has_db:
                    self.db.add_chain_record(i + 1, self.accepted,
                                             self.model.__getstate__())
                # To user
                if verbose:
                    sys.stdout.write('sample ' + str(i + 1).zfill(len(str(num_samples)))
                                     + ' of ' + str(num_samples)
                                     + ', log_p: %.6f, acc. rate: %1.2f'
                                       % (self.model.log_p, self.acceptance_rate)
                                     + '\r')
                    sys.stdout.flush() 
            # Tuning
            if isinstance(self.proposal, TunableProposalConcept):
                if (i > 0 and
                    i >= start_tuning_after and
                    i % tuning_frequency == 0 and
                    i <= stop_tuning_after):
                    self.proposal.tune(self.acceptance_rate, verbose=verbose)

        if verbose:
            sys.stdout.write('\n')
