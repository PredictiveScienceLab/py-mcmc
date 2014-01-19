"""
Defines the functionality of a model class.

All models should simply inherit from this class.

Author:
    Ilias Bilionis

"""


def Model(object):

    """
    A generic model class.

    :param name:    A name for the model.
    :type name:     str
    """

    def __init__(self, name='The model class'):
        """
        Initialize the object.
        """
        self.__name__ = name

    def __getstate__(self):
        """
        Get the state of the model.

        This shoud return the state of the model. That is, return everything
        is necessary to avoid redundant computations. It is also used in oder
        to pickle the object or send it over the network.
        """
        raise NotImplementedError('Implement this.')

    def __setstate__(self, state):
        """
        Set the state of the model.

        This is supposed to take the return value of the
        :method:`Model.__getstate__`.
        """
        raise NotImplementedError('Implement this.')

    @property
    def log_likelihood(self):
        """
        Return the log likelihood of the model at the current state.
        """
        raise NotImplementedError('Implement this.')

    @property
    def log_prior(self):
        """
        Retutnr the log prior of the model at the current state.
        """
        raise NotImplementedError('Implement this.')
