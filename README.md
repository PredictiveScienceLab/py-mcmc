A python module implementing some generic MCMC routines.
========================================================

The main purpose of this module is to serve as a simple MCMC framework for
generic models. Probably the most useful contribution at the moment, is that
it can be used to train Gaussian process (GP) models implemented in the 
[GPy package](http://sheffieldml.github.io/GPy/).


Features
--------
The code features the following things at the moment:
+ Fully object oriented. The models can be of any type as soon as they offer
  the right interface.
+ Random walk proposals.
+ Metropolis Adjusted Langevin Dynamics.
+ The MCMC chains are stored in fast [HDF5](http://www.hdfgroup.org/HDF5/)
  format using [PyTables](http://www.pytables.org/moin).
+ A mean function can be added to the (GP) models of the
[GPy package](http://sheffieldml.github.io/GPy/).


Related Packages
----------------
Probably, the most related package to what I am offering is the excellent
[PyMC](https://github.com/pymc-devs/pymc) code. The reason I have departed from
it is two-fold:
+ In the old versions (e.g.
[PyMC 2.3](http://pymc-devs.github.io/pymc/index.html)), could not find an easy
way to implement Metropolis Adjusted Langevin Dynamics. This was unfortunate
because it is one of the most powerful sampling methods when derivatives are
available.
+ In the new version (e.g. 
 [PyMC 3](http://nbviewer.ipython.org/github/pymc-devs/pymc/blob/master/pymc/examples/tutorial.ipynb),
 which is based on [Theano](http://www.deeplearning.net/software/theano/)
 schemes with derivatives can be easily implemented but there are several issues
 when one tries to deal with existing models. In particular, it is not possible
 at the moment to deal in an easy way with a model that is not directly implemented
 using Theano (e.g. if it calls an external library or runs a complicated program).
 This is a tremendous limitation when it comes to solving realistic inverse
 problems. In addition, it is not easy to exploit the Gaussian process
 functionality of GPy in order to train these models with MCMC.

Therefore, the purpose of this package is to fill the gap between PyMC 2.3
and PyMC 3. When the programers of PyMC 3 fix the afforementioned problem, then
the MCMC part of this code will become obsolete.


Notes
-----
I have deliberately not adding any tuning capabilities in this library. This is
because tuning will be done (in the near future) using the Sequential Monte
Carlo (SMC) component.

Ilias Bilionis,
January, 2014
Mathematics and Computer Science Division,
Argonne National Laboratory
