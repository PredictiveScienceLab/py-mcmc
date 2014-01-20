"""
A simple MCMC proposal is one that is based only on the current state of the
model, e.g. a Random Walk proposal.

Author:
    Ilias Bilionis
"""


__all__ = ['SimpleProposal']


from . import Proposal


class SimpleProposal(Proposal):

    """
    The base class for simple proposals (proposals based only on the current
    parameters of a model.
    
    :param name:    A name for the object.
    :type name:     str
    """

    def __init__(self, name='Simple Proposal'):
        """
        Initialize the object.
        """
        super(SimpleProposal, self).__init__(name=name)

    def _do_propose(self, model):
        """
        Do the actual proposal and change the state of the model to contain
        the new parameters.
        """
        old_params = model.params
        (new_params,
         log_p_new_cond_old,
         log_p_old_cond_new) = self._eval_all(old_params)
        model.params = new_params
        return log_p_old_cond_new - log_p_new_cond_old

    def _eval_all(self, old_params):
        """
        The user needs to implement this function.

        :param old_params:  The old parameters of the model.
        :returns:           The tuple:
                            (new_params, log_p_new_cond_old, log_p_old_cond_new)
                            where:
                            + new_params are the new parameters
                            + log_p_new_cond_old is the probability of moving to the
                              new parameters given the old
                            + log_p_old_cond_new is the probability of moving to the
                              old parameters given the new
        """
        raise NotImplementedError('Implement this.')
