"""
Automatically assign uninformative priors to a GPYModel.

Author:
    Ilias Bilionis

Date:
    3/20/2014
"""


__all__ = ['assign_priors_to_gpy_model']


import GPy
from GPy.core.domains import POSITIVE, NEGATIVE, REAL, BOUNDED
import itertools
from . import UninformativeScalePrior
from . import UninformativePrior


def assign_priors_to_gpy_model(model):
    """
    Automatically assign uninformative priors to a GPYModel.

    The assignmnent of priors is done as follows:
    + If a parameter is constrained to be POSITIVE, then it is assigned a
      :class:`pymcmc.UninformativeScalePrior`.
    + If a parameter is constrained to be REAL, then it is assigned a
      :class:`pymcmc.UninformativePrior` with infinite bounds.
    + If a parameter is constrained to be BOUNDED, then it is assigned a
      :class:`pymcmc.UninformativePrior` with (semi)-finite bounds.

    :param model:   The GPy model you want to assign uninformative priors to.
                    Upon exit, the model will contain priors.
    """
    assert isinstance(model, GPy.core.Model)
    param_names = model._get_param_names()
    # Loop over the constrains
    for idx, c in itertools.izip(model.constrained_indices, model.constraints):
        # Find the prior corresponding to constrain c
        if c.domain == POSITIVE:
            prior = UninformativeScalePrior()
        elif c.domain == BOUNDED:
            # This case requires a fix because of a bug in
            # GPy.core.model.py:113
            # They should allow for priors defined on a BOUNDED domain.
            # TODO: Notify them so that they can change it.
            # For now this case is done separately:
            prior = UninformativePrior(lower=c.lower, upper=c.upper)
            # START OF FIX FOR GPY BUG:
            for i in idx:
                model.priors[i] = prior
            continue
            # END OF FIX FOR GPY BUG
        for i in idx:
            model.set_prior(param_names[i], prior)
    # Any unconstrained indices should receive an unbounded UninformativePrior
    unbounded_uninformative_prior = UninformativePrior()
    for i in range(len(param_names)):
        if model.priors[i] is None:
            model.set_prior(param_names[i], unbounded_uninformative_prior)
