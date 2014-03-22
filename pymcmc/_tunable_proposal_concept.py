"""
A tunable proposal concept.

Author:
    Ilias Bilionis

Date:
    3/22/2014

"""


__all__ = ['TunableProposalConcept']


class TunableProposalConcept(object):

    """
    The tunable proposal concept should be inherited by any proposal that can
    tune itself. It is not a proper proposal, because it should be inherited
    in addition to a proper proposal class.
    """

    def __init__(self, **kwargs):
        """
        Initialize the object.
        """
        pass

    def tune(self, ac, **kwargs):
        """
        Tune the proposal.

        :param ac:      The acceptance rate.
        :type ac:       float
        :param kwargs:  Any other parameters that are required to tune the
                        proposal.
        """
        raise NotImplementedError('Implement this.')

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
