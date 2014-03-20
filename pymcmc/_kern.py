"""
Implements various usefull kernels for GPy.

Author:
    Ilias Bilionis

Date:
    3/20/2014

"""


__all__ = ['MeanFunctionKernpart', 'mean_function']


from GPy.kern import kern
from GPy.kern import Kernpart
import numpy as np


class MeanFunctionKernpart(Kernpart):

    """
    A kernel representing a mean function.

    :param input_dim:   The number of input dimensions.
    :type input_dim:    int
    :param basis:       The basis. It should be a class that implements
                        ``__call__(X)`` where X is a 2D numpy array with
                        ``X.shape[1] == input_dim`` columns and returns the
                        design matrix. In addition, it should at least have
                        the attribute ``num_output`` which should store the
                        number of basis functions.
    :type basis:        any type that satisfies the consept of a basis function
    :param variance:    The strength of the kernel. The smaller it is, the
                        less important its contribution.
    :type variance:     float
    :param weights:     The weight of each basis function. The closer it is to
                        zero, the less important the corresponding basis
                        function.
    :type weights:      :class:`numpy.ndarray`
    """

    # The basis functions
    _basis = None

    # The variance of the kernel
    _variance = None

    # The weights of the kernel
    _weights = None

    @property
    def basis(self):
        """
        Get the basis functions.
        """
        return self._basis

    @property
    def num_basis(self):
        """
        Get the number of basis functions.
        """
        return self.basis.num_output

    @property
    def num_params(self):
        """
        Get the number of parameters of the object.
        """
        return self.num_basis + 1

    @property
    def variance(self):
        """
        Get the variance of the kernel.
        """
        return self._variance

    @property
    def weights(self):
        """
        Get the weights of the kernel.
        """
        return self._weights

    def __init__(self, input_dim, basis, variance=1., weights=None):
        """
        Initialize the object.
        """
        self.input_dim = int(input_dim)
        if not hasattr(basis, '__call__'):
            raise TypeError('The basis functions must implement the '
                            '\'__call__()\' method. This method should '
                            ' the basis functions given a 2D dimensional numpy'
                            ' numpy array of \'num_points x input_dim\''
                            ' dimensions.')
        if not hasattr(basis, 'num_output'):
            raise TypeError('The basis functions must have an attribute '
                            ' \'num_output\' which should store the number of'
                            ' basis functions it contains.')
        self._basis = basis
        if weights is not None:
            assert weights.shape == (self.num_basis, )
        else:
            weights = np.ones(self.num_basis)
        self.name = 'mean_function'
        self._set_params(np.hstack([variance, weights]))

    def _get_params(self):
        """
        Get an array containing the parameters of the kernel.

        :returns:   An 1D numpy array containing the parameters (variance,
                    weights).
        """
        return np.hstack([self.variance, self.weights])

    def _set_params(self, x):
        """
        Set the parameters from a 1D numpy array ``x``. The first element
        should be the variance and the rest elements should correspond to the
        weights of the kernel.
        """
        assert x.size == self.num_params, 'Illegal number of parameters.'
        self._variance = x[0]
        self._weights = x[1:]

    def _get_param_names(self):
        """
        Get a list containing the names of the parameters in the order they
        appear when you call :meth:`MeanFunctionKernel._get_params()`.
        """
        if self.num_params == 2:
            return ['variance', 'weight']
        else:
            return (['variance'] +
                    ['weights_%d' % i for i in range(self.num_basis)])

    def K(self, X, X2, target):
        """
        Evaluate the covariance (or cross covariance matrix) at ``X`` and ``X2``.
        If ``X2`` is ``None`` the covariance matrix is computed. The result is
        added to ``target``.
        """
        # Compute the design matrix
        # TODO: This should happen only once if X does not change!
        phi_X = self.basis(X)
        phi_X2 = phi_X if X2 is None or X2 is X else self.basis(X2)
        target += (self.variance *
                   np.einsum('ij,j,kj', phi_X, self.weights, phi_X2))

    def Kdiag(self, X, target):
        """
        Evaluate only the diagonal part of the covariance matrix at ``X`` and
        add it to ``target``.
        """
        phi_X = self.basis(X)
        target += (self.variance *
                   np.einsum('ij,j,ij->i', phi_X, self.weights, phi_X))

    def dK_dtheta(self, dL_dK, X, X2, target):
        """
        Evaluate the contribution of this kernel to the derivative of the
        likelihood with respect to each hyper-parameter. Now, ``dL_dK`` is the
        partial derivative of the likelihood with respect to each component of
        this covariance matrix (of shape ``X.shape[0] x X.shape[0]`` if ``X2``
        is ``None`` and ``X.shape[0] x X2.shape[0]``, otherwise. This function
        adds to ``target`` the final value of the derivative of the likelihood
        with respect to each hyper-parameter, i.e.
        ``target[i] += dL_dK[s, t] * (dK[s, t] / dtheta[i])`` in which
        summation over ``s`` and ``t`` is implied.
        """
        phi_X = self.basis(X)
        phi_X2 = phi_X if X2 is None or X2 is X else self.basis(X2)
        target[0] += np.einsum('ij,ik,k,jk', dL_dK, phi_X, self.weights, phi_X2)
        target[1] += (self.variance *
                      np.einsum('ij,ik,jk->k', dL_dK, phi_X, phi_X2))


def mean_function(input_dim, basis, variance=1., weights=None):
    """
    Construct a kernel representing a mean function.

    :param input_dim:   The number of inputs.
    :type input_dim:    int
    :param basis:       The basis. It should be a class that implements
                        ``__call__(X)`` where X is a 2D numpy array with
                        ``X.shape[1] == input_dim`` columns and returns the
                        design matrix. In addition, it should at least have
                        the attribute ``num_output`` which should store the
                        number of basis functions.
    :type basis:        any type that satisfies the consept of a basis function
    :param variance:    The strength of the kernel. The smaller it is, the
                        less important its contribution.
    :type variance:     float
    :param weights:     The weight of each basis function. The closer it is to
                        zero, the less important the corresponding basis
                        function.
    :type weights:      :class:`numpy.ndarray`
    """
    part = MeanFunctionKernpart(input_dim, basis, variance=variance,
                                weights=weights)
    return kern(input_dim, [part])
