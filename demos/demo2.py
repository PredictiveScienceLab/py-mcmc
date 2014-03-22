"""
This demo demonstrates how to construct a GPy model with a mean function
and train it using the pymcmc module. This model is equivalent to Bayesian
linear regression.

Author:
    Ilias Bilionis

Date:
    3/20/2014

"""


import numpy as np
import GPy
import pymcmc as pm
import matplotlib.pyplot as plt


# Write a class that represents the mean you wish to use:
class PolynomialBasis(object):
    """
    A simple set of polynomials.

    :param degree:  The degree of the polynomials.
    :type degree:   int
    """

    def __init__(self, degree):
        """
        The constructor can do anything you want. The object should be
        constructed before doing anything with pymcmc in any case.
        Just make sure that inside the constructor you define the ``num_output``
        attribute whose value should be equal to the number of basis functions.
        """
        self.degree = degree
        self.num_output = degree + 1 # YOU HAVE TO DEFINE THIS ATTRIBUTE!

    def __call__(self, X):
        """
        Evaluate the basis functions at ``X``.

        Now, you should assume that ``X`` is a 2D numpy array of size
        ``num_points x input_dim``. If ``input_dim`` is 1, then you still need
        to consider it as a 2D array because this is the kind of data that GPy
        requires. If you want to make the function work also with 1D arrays if
        ``input_dim`` is one the use the trick below.

        The output of this function should be the design matrix. That is,
        it should be the matrix ``phi`` of dimensions
        ``num_points x num_output``. In otherwors, ``phi[i, j]`` should be
        the value of basis function ``phi_j`` at ``X[i, :]``.
        """
        if X.ndim == 1:
            X = X[:, None] # Trick for 1D arrays
        return np.hstack([X ** i for i in range(self.degree + 1)])


# Pick your degree
degree = 4
# Construct your basis
poly_basis = PolynomialBasis(degree)
# Let us generate some random data to play with
# The number of input dimensions
input_dim = 1
# The number of observations
num_points = 20
# The noise level we are going to add to the observations
noise = 0.1
# Observed inputs
X = np.random.rand(num_points, 1)
# We are going to generate the outputs from the space that is spanned by
# our basis functions and add some noise
# The weights of the basis functions:
weights = np.random.randn(poly_basis.num_output)
# The observations we make
Y = np.dot(poly_basis(X), weights) + noise * np.random.randn(num_points)
# The output need also be a 2D numpy array
Y = Y[:, None]
# Let's construct a GP model with just a mean and a diagonal covariance
# This is the mean (and at the same time the kernel)
mean = pm.mean_function(input_dim, poly_basis)
# Now, let's construct the model
model = GPy.models.GPRegression(X, Y, kernel=mean)
print 'Model before training:'
print str(model)
# You may just train the model by maximizing the likelihood:
model.optimize(messages=True)
print 'Trained model:'
print str(model)
# And just plot the predictions
model.plot(plot_limits=(0, 1))
# Let us also plot the full function
x = np.linspace(0, 1, 50)[:, None]
y = np.dot(poly_basis(x), weights)
plt.plot(x, y, 'r', linewidth=2)
plt.legend(['Mean of GP', '5\% percentile of GP', '95\% percentile of GP',
            'Observations', 'Real Underlying Function'], loc='best')
plt.title('Model trained by maximizing the likelihood')
plt.show()
a = raw_input('press enter to continue...')
# Or you might want to do it using MCMC:
new_model = GPy.models.GPRegression(X, Y, kernel=mean)
proposal = pm.MALAProposal(dt=1.)
mcmc = pm.MetropolisHastings(new_model, proposal=proposal)
mcmc.sample(10000, num_thin=100, num_burn=1000, verbose=True)
print 'Model trained with MCMC:'
print str(new_model)
# Plot everything for this too:
new_model.plot(plot_limits=(0, 1))
# Let us also plot the full function
x = np.linspace(0, 1, 50)[:, None]
y = np.dot(poly_basis(x), weights)
plt.plot(x, y, 'r', linewidth=2)
plt.legend(['Mean of GP', '5\% percentile of GP', '95\% percentile of GP',
            'Observations', 'Real Underlying Function'], loc='best')
plt.title('Model trained by MCMC')
plt.show()
a = raw_input('press enter to continue...')
