"""
Experimenting in order to construct the database support for the MCMC chains.

Author:
    Ilias Bilionis
"""


import tables as tb
import numpy as np


if __name__ == '__main__':
    num_params = 10
    params = np.random.randn(num_params)
    filters = tb.Filters(complevel=9)
    fd = tb.open_file('test_db.h5', mode='a', filters=filters)
    fd.create_group('/', 'mcmc', 'Metropolis-Hastings Algorithm')
    # Data type for a single record in the chain
    single_record_dtype = {'step': tb.UInt16Col(),
                           'params': tb.Float32Col(shape=(num_params,)),
                           'proposal': tb.UInt16Col(),
                           'log_like': tb.Float32Col(),
                           'log_prior': tb.Float32Col(),
                           'grad_log_like': tb.Float32Col(shape=(num_params,)),
                           'grad_log_prior': tb.Float32Col(shape=(num_params,)),
                           'accepted': tb.UInt16Col()}
    table = fd.create_table('/mcmc', 'chain_000', single_record_dtype, 'Chain: 0')
    chain = table.row
    for i in xrange(1000):
        print i
        chain['step'] = i
        chain['params'] = np.random.randn(num_params)
        chain['proposal'] = 0
        chain['log_like'] = np.random.rand()
        chain['log_prior'] = np.random.rand()
        chain['grad_log_like'] = np.random.randn(num_params)
        chain['grad_log_prior'] = np.random.randn(num_params)
        chain['accepted'] = i % 2
        chain.append()
    table.flush()
    fd.close()
