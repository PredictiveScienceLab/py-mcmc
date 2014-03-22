"""
This demo demonstrates how to train a GPy model using the pymcmc module.

Author:
    Ilias Bilionis

Date:
    3/20/2014

"""

import GPy
import pymcmc as pm
import numpy as np
import matplotlib.pyplot as plt


# Construct a GPy Model (anything really..., here we are using a regression
# example)
model = GPy.examples.regression.olympic_marathon_men(optimize=False, plot=False)
# Look at the model before it is trained:
print 'Model before training:'
print str(model)
# Pick a proposal for MCMC (here we pick a Metropolized Langevin Proposal
proposal = pm.MALAProposal(dt=1.)
# Construct a Metropolis Hastings object
mcmc = pm.MetropolisHastings(model,                     # The model you want to train
                             proposal=proposal,         # The proposal you want to use
                             db_filename='demo_1_db.h5')# The HDF5 database to write the results
# Look at the model now: We have automatically added uninformative priors
# by looking at the constraints of the parameters
print 'Model after adding priors:'
print str(model)
# Now we can sample it:
mcmc.sample(100000,         # Number of MCMC steps
            num_thin=100,   # Number of steps to skip
            num_burn=1000,  # Number of steps to burn initially
            verbose=True)   # Be verbose or not
# Here is the model at the last MCMC step:
print 'Model after training:'
print str(model)
# Let's plot the results:
model.plot(plot_limits=(1850, 2050))
a = raw_input('press enter...')
