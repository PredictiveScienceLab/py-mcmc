#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from setuptools import setup

with open('README.md', 'r') as fd:
    long_description = fd.read()

setup (name = 'pymcmc',
       version = '0.0a1',
       author      = 'Ilias Bilionis',
       author_email = 'ibilion@purdue.edu',
       license = 'LGPL',
       description = 'A python module implementing some generic MCMC routines',
       long_description = long_description,
       packages = ['pymcmc'],
       package_dir={'pymcmc': 'pymcmc'},
       py_modules = ['pymcmc.__init__'],
       classifiers = ['Development Status :: 3 - Alpha',
                      'Intended Audience :: Science/Research',
                      'Topic :: Scientific/Engineering :: Mathematics',
                      'Programming Language :: Python :: 2',
                      'Programming Language :: Python :: 2.6',
                      'Programming Language :: Python :: 2.7'],
        keywords = 'Markov-Chain-Monte-Carlo MCMC Metrpolis-Adjusted-Langevin-Dynamics MALA GPy',
        install_requires = ['GPy>=0.6.0'],
        url = 'https://github.com/ebilionis/pymcmc',
        download_url = 'https://github.com/ebilionis/pymcmc/tarball/0.0a1'
       )
