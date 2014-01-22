"""
A database to store the MCMC chains.

Author:
    Ilias Bilionis
"""


__all__ = ['DataBase']


import tables as pt
import numpy as np
import os
from datetime import datetime
from . import state_to_table_dtype, UnknownTypeException


class DataBase(object):

    """
    A database to store MCMC chains.

    :param filename:     The filename of the database.
    :type filename:      str
    :param model_state:        The model. This is needed so that we know exactly
                               how to store the state of a model.
    :type model_state:         dict
    :param proposal_state:     The MCMC proposal. This is needed so that we know
                               exactly what data are required for the proposal.
    :type proposal_state:      dict
    """

    def __init__(self, filename, model_state, proposal_state):
        """
        Initialize the object.
        """
        self.filename = filename
        self.ChainRecordDType = state_to_table_dtype(model_state)
        self.ChainRecordDType['step'] = pt.UInt32Col()
        self.ChainRecordDType['accepted'] = pt.UInt32Col()
        self.ChainRecordDType['proposal'] = pt.UInt16Col()
        self.ProposalRecordDType = state_to_table_dtype(proposal_state)
        self.ChainCounterDType = {'id': pt.UInt16Col(),
                                  'name': pt.StringCol(itemsize=32),
                                  'date': pt.StringCol(itemsize=26)
                                  }
        if os.path.exists(filename) and pt.is_pytables_file(filename):
            self.fd = pt.open_file(filename, mode='a')
        else:
            self.fd = pt.open_file(filename, mode='w')
            self.fd.create_group('/', 'mcmc',
                                 'Metropolis-Hastings Algorithm Data')
            self.proposals = self.fd.create_table('/mcmc', 'proposals',
                                                  self.ProposalRecordDType,
                                                  'MCMC Proposals')
            self.fd.create_table('/mcmc', 'chain_counter',
                                 self.ChainCounterDType, 'Chain Counter')
            self.fd.create_group('/mcmc', 'data', 'Collection of Chains')

    @property
    def proposals(self):
        return self.fd.root.mcmc.proposals

    @property
    def chain_counter(self):
        return self.fd.root.mcmc.chain_counter

    @property
    def proposal_id(self):
        """
        Get the id of the current proposal.
        """
        return self.proposals.nrows - 1

    def add_proposal(self, state):
        """
        Add a proposal to the database.
        """
        row = self.proposals.row
        for name in state.keys():
            row[name] = state[name]
        row.append()
        self.proposals.flush()

    def create_new_chain(self):
        """
        Create a new chain.
        """
        num_chains = self.chain_counter.nrows
        row = self.chain_counter.row
        row['id'] = num_chains
        row['name'] = 'chain_' + str(num_chains)
        row['date'] = str(datetime.now())
        row.append()
        self.chain_counter.flush()
        self.current_chain = self.fd.create_table('/mcmc/data',
                                                  'chain_' + str(num_chains),
                                                  self.ChainRecordDType,
                                                  'Chain Record ' + str(num_chains))

    def add_chain_record(self, step, accepted, state):
        """
        Add a chain record to the current state.
        """
        row = self.current_chain.row
        for name in state.keys():
            row[name] = state[name]
        row['step'] = step
        row['accepted'] = int(accepted)
        row['proposal'] = self.proposal_id
        row.append()
        self.current_chain.flush()
