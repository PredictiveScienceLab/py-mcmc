"""
Unit tests for the GPyModel class.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.split(__file__)[0]))
import GPy
import pymcmc as pm
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model = GPy.examples.regression.olympic_marathon_men(optimize=False, plot=False)
    noise_prior = pm.UninformativeScalePrior()
    variance_prior = pm.UninformativeScalePrior()
    length_scale_prior = pm.UninformativeScalePrior()
    model.set_prior('rbf_variance', variance_prior)
    model.set_prior('noise_variance', noise_prior)
    model.set_prior('rbf_lengthscale', length_scale_prior)
    print str(model)
    #model.optimize()
    #print str(model)
    mcmc_model = pm.GPyModel(model, compute_grad=True)
    mcmc = pm.MetropolisHastings(mcmc_model, db_filename='test_db.h5')
    model_state, prop_state = mcmc.db.get_states(-1, -1)
    mcmc.sample(1000, num_thin=100, init_model_state=model_state,
                      init_proposal_state=prop_state)
