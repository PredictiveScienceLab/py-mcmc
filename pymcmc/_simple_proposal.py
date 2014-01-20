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

        :param model:   The model.
        :returns:       The ratio of the forward and backward moves.

        Upon return, the model shall contain the new parameters.
        """
        old_params = model.params
        new_params = self._sample(old_params)
        log_p_new_cond_old = self(new_params, old_params)
        log_p_old_cond_new = self(old_params, new_params)
        model.params = new_params
        return log_p_old_cond_new - log_p_new_cond_old

    def _sample(self, old_params):
        """
        Sample the proposal given the ``old_params``.

        :param old_params:  The old parameters of the model
        :returns:           The new parameters of the model.
        """
        raise NotImplementedError('Implement this.')

    def __call__(self, new_params, old_params):
        """
        Evaluate the proposal at the new parameters given the old parameters.
        :param new_params:      The new parameters.
        :param old_params:      The old parameters. We are assuming that we
                                are conditioning on them.
        """
        raise NotImplementedError('Implement this.')
