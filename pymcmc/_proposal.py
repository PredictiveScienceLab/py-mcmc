"""
Defines the base proposal class for doing MCMC.

Author:
    Ilias Bilionis
"""


__all__ = ['Proposal']


class Proposal(object):

    """
    The base class of all MCMC proposals.

    :param name:    A name for the object.
    :type name:     str
    """

    def __init__(self, name='Proposal'):
        """
        Initialize the object.
        """
        self.__name__ = name

    def __str__(self):
        """
        Return a string representation of the object.
        """
        return 'Name:\t' + self.__name__

    def __getstate__(self):
        """
        Get the state of the object.
        """
        return {}

    def __setstate__(self, state):
        """
        Set the state of the object.
        """
        pass

    def propose(self, model):
        """
        Propose a move.

        :param model:       The model.
        :returns:           A tuple of the following form:
                            (new_state, log_p)
                            where:
                                + ``new_state`` is a new state for the model
                                + ``log_p`` is the probability for acceptance

        The model shall remain the same after a call to this method. That is,
        no matter what happens to it, its parameters should remain the same.
        """
        old_state = model.__getstate__()
        old_log_like = model.log_likelihood
        old_log_prior = model.log_prior
        log_a2 = self._do_propose(model)
        new_state = model.__getstate__()
        new_log_like = model.log_likelihood
        new_log_prior = model.log_prior
        log_a1 = (new_log_like - old_log_like) + (new_log_prior - old_log_prior)
        model.__setstate__(old_state)
        return new_state, log_a1 + log_a2

    def _do_propose(self, model):
        """
        Actually propose a move.

        :param model:   The model.

        This needs to be reimplemented by the deriving classes.
        Here, it is assumed that the model is left to the new state.
        """
        raise NotImplementedError('Implement this.')
