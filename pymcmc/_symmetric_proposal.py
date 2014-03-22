"""
The base class for symmetric MCMC proposals.

Author:
    Ilias Bilionis
"""


__all__ = ['SymmetricProposal']


from . import SimpleProposal


class SymmetricProposal(SimpleProposal):

    """
    The base class for all symmetric proposals.

    """

    def __init__(self, **kwargs):
        """
        Initialize the object.
        """
        super(SymmetricProposal, self).__init__(**kwargs)

    def __call__(self, new_params, old_params):
        """
        Since the proposal is symmetric, it does not really matter what this
        probability is as soon as it is symetric with respect to ``new_params``
        and ``old_params``.
        """
        return 0.
