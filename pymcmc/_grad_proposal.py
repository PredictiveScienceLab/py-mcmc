"""
A proposal for MCMC that depends on the gradients.

Author:
    Ilias Bilionis
"""


__all__ = ['GradProposal']


from . import Proposal


class GradProposal(Proposal):

    """
    The base class for proposals that depend on the gradient of the target
    probability distribution with respect to the parameters.

    The keyword arguments are the same as in :class:`pymcmc.Proposal`.
    """

    def __init__(self, **kwargs):
        """
        Initialize the object.
        """
        if not kwargs.has_key('name'):
            kwargs['name'] = 'Grad Proposal'
        super(GradProposal, self).__init__(**kwargs)

    def _do_propose(self, model):
        """
        Do the actual proposal and change the state of the model to contain
        the new parameters.

        :param model:   The model.
        :returns:       The ratio of the forward and backward moves.

        Upon return, the model shall contain the new parameters.
        """
        old_params = model.params
        old_grad_params = model.grad_log_p
        new_params = self._sample(old_params, old_grad_params)
        log_p_new_cond_old = self(new_params, old_params, old_grad_params)
        model.params = new_params
        new_grad_params = model.grad_log_p
        log_p_old_cond_new = self(old_params, new_params, new_grad_params)
        return log_p_old_cond_new - log_p_new_cond_old

    def _sample(self, old_params, old_grad_params):
        """
        Sample the proposal given the ``old_params`` and the gradient of the
        target probability with respect to them.

        :param old_params:      The old parameters of the model.
        :param old_grad_params: The gradient of the target probability with
                                respect to the parameters.
        """
        raise NotImplementedError('Implement this.')

    def __call__(self, new_params, old_params, old_grad_params):
        """
        Evaluate the proposal at the new parameters given the old parameters.
        :param new_params:      The new parameters.
        :param old_params:      The old parameters. We are assuming that we
                                are conditioning on them.
        :param old_grad_params: The gradient of the target probability with
                                respect to the parameters.
        """
        raise NotImplementedError('Implement this.')
