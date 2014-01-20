"""
Defines the functionality of a model class.

All models should simply inherit from this class.

Author:
    Ilias Bilionis

"""


__all__ = ['Model']


class Model(object):

    """
    A generic model class.

    :param name:    A name for the model.
    :type name:     str
    """

    def __init__(self, name='Pymcmc Model'):
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

    @property
    def num_params(self):
        """
        Return the number of parameters.
        """
        raise NotImplementedError('Implement this.')

    @property
    def params(self):
        """
        Set/Get the parameters.
        """
        raise NotImplementedError('Implement this.')

    @params.setter
    def params(self, value):
        raise NotImplementedError('Implement this.')

    @property
    def param_names(self):
        """
        Return a list containing the names of the parameters.
        """
        raise NotImplementedError('Implement this.')

    @property
    def grad_log_likelihood(self):
        """
        Return the gradient of the log likelihood.
        """
        raise NotImplementedError('Implement this.')

    @property
    def grad_log_prior(self):
        """
        Return the gradient of the log prior.
        """
        raise NotImplementedError('Implement this.')

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = 'Model name:\t' + self.__name__ + '\n'
        s += 'num_param:\t' + str(self.num_params) + '\n'
        s += 'param names:\t' + str(self.param_names)
        s += str(self._state)
        return s
