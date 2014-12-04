"""
Automatically assign uninformative priors to a GPYModel.

Author:
    Ilias Bilionis

Date:
    3/20/2014
"""


__all__ = ['assign_priors_to_gpy_model']


import GPy
from GPy import priors
POSITIVE = priors._POSITIVE
REAL = priors._REAL
import itertools
from . import UninformativeScalePrior
from . import UninformativePrior


def assign_priors_to_gpy_model(model):
    """
    Automatically assign uninformative priors to a GPYModel.

    It only assigns priors to variables that do not have one already.

    The assignmnent of priors is done as follows:
    + If a parameter is constrained to be POSITIVE, then it is assigned a
      :class:`pymcmc.UninformativeScalePrior`.
    + If a parameter is constrained to be REAL, then it is assigned a
      :class:`pymcmc.UninformativePrior` with infinite bounds.
    + If a parameter is constrained to be REAL, then it is assigned a
      :class:`pymcmc.UninformativePrior` with (semi)-finite bounds.

    :param model:   The GPy model you want to assign uninformative priors to.
                    Upon exit, the model will contain priors.
    """
    assert isinstance(model, GPy.core.Model)
    param_names = model._get_param_names()
    if model.priors is None:
        model.priors = [None] * len(param_names)
    # Loop over the parameters of the model
    for mp in model.flattened_parameters:
        print mp._constraints_str
        if mp._constraints_str[0] == '+ve':
            mp.set_prior(UninformativeScalePrior())
        else:
            print 'there'
            mp.set_prior(UninformativePrior())
