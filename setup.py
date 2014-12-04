#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension

setup (name = 'pymcmc',
       version = '0.6',
       author      = "Ilias Bilionis",
       description = """Simple MCMC Python module""",
       packages = ['pymcmc']
       )
