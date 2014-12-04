#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension

setup (name = 'pymcmc',
       version = '0.0a1',
       author      = 'Ilias Bilionis',
       author_email = 'ibilion@purdue.edu',
       license = 'LGPL',
       description = 'A python module implementing some generic MCMC routines',
       long_description = """\
The main purpose of this module is to serve as a simple MCMC framework for
generic models. Probably the most useful contribution at the moment, is that
it can be used to train Gaussian process (GP) models implemented in the 
[GPy package](http://sheffieldml.github.io/GPy/).
""",
       packages = ['pymcmc'],
       classifiers = ['Development Status :: 3 - Alpha',
                      'Intended Audience :: Science/Research',
                      'Topic :: Scientific/Engineering :: Mathematics'
                      'Programming Language :: Python :: 2',
                      'Programming Language :: Python :: 2.6',
                      'Programming Language :: Python :: 2.7'],
        keywords = 'Markov-Chain-Monte-Carlo MCMC Metrpolis-Adjusted-Langevin-Dynamics MALA GPy',
        install_requires = ['GPy>=0.6.0']
       )
