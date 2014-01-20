"""
Unit tests for the GPyModel class.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.split(__file__)[0]))
import GPy
import pymcmc


if __name__ == '__main__':
    model = GPy.examples.regression.olympic_marathon_men(optimize=False, plot=False)
    mcmc_model = pymcmc.GPyModel(model, compute_grad=True)
    print str(mcmc_model)
    print mcmc_model.log_likelihood
    print mcmc_model.log_prior
    print mcmc_model.num_params
    print mcmc_model.params
    mcmc_model.params = mcmc_model.params
    print mcmc_model.param_names
    print mcmc_model.grad_log_likelihood
    print mcmc_model.grad_log_prior
    proposal = pymcmc.MALAProposal()
    print str(mcmc_model)
    new_state, log_p = proposal.propose(mcmc_model)
    print new_state
    print log_p
