"""
A generic Python module for MCMC.

Author:
    Ilias Bilionis
"""

__all__ = ['Model', 'GPyModel', 'Proposal']


from _model import Model
from _GPyModel import GPyModel
from _proposal import Proposal
