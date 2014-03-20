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
    :param kappa:       The weight of each basis function. The closer it is to
                        zero, the less important the corresponding basis
                        function.
    :type kappa:        :class:`numpy.ndarray`
    :param ARD:         If ``ARD`` is ``True``, then the mean function behaves
                        like a Relevance Vector Machine (RVM). If it is
                        ``False`` then it is equivalent to a common mean
                        function.
    :type ARD:          bool
    """

    # The basis functions
    _basis = None

    # The variance of the kernel
    _variance = None

    # The kappa of the kernel
    _kappa = None

    # Automatic Relevance Determination
    _ARD = None

    # The number of parameters of the kernel
    _num_params = None

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
        return self._num_params

    @property
    def ARD(self):
        """
        Get the ARD flag.
        """
        return self._ARD

    @property
    def variance(self):
        """
        Get the variance of the kernel.
        """
        return self._variance

    @property
    def kappa(self):
        """
        Get the kappa of the kernel.
        """
        return self._kappa

    def __init__(self, input_dim, basis, variance=1., kappa=None, ARD=False,
                 parametrize_variance=False):
        """
        Initialize the object.
        """
        self.input_dim = int(input_dim)
        self._ARD = ARD
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
        if kappa is not None:
            assert kappa.shape == (self.num_basis, )
        else:
            kappa = np.ones(self.num_basis)
        self._variance = variance
        self._kappa = kappa
        self.name = 'mean_function'
        self.parametrize_variance = parametrize_variance
        num_params = 0
        params = []
        if self.parametrize_variance:
            num_params += 1
            params.append(variance)
        if self.ARD:
            num_params += self.num_basis
            params.append(kappa)
        self._num_params = num_params
        if not len(params) == 0:
            self._set_params(np.hstack(params))

    def _get_params(self):
        """
        Get an array containing the parameters of the kernel.

        :returns:   An 1D numpy array containing the parameters (variance,
                    kappa).
        """
        if self.num_params == 0:
            return []
        params = []
        if self.parametrize_variance:
            params.append(self.variance)
        if self.ARD:
            params.append(self.kappa)
        return np.hstack(params)

    def _set_params(self, x):
        """
        Set the parameters from a 1D numpy array ``x``. The first element
        should be the variance and the rest elements should correspond to the
        kappa of the kernel.
        """
        assert x.size == self.num_params, 'Illegal number of parameters.'
        i = 0
        if self.parametrize_variance:
            self._variance = x[0]
            i = 1
        if self.ARD:
            self._kappa = x[i:]

    def _get_param_names(self):
        """
        Get a list containing the names of the parameters in the order they
        appear when you call :meth:`MeanFunctionKernel._get_params()`.
        """
        names = []
        if self.parametrize_variance:
            names.append('variance')
        if self.ARD:
            names += ['kappa_%d' % i for i in range(self.num_basis)]
        return names

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
                   np.einsum('ij,j,kj', phi_X, self.kappa, phi_X2))

    def Kdiag(self, X, target):
        """
        Evaluate only the diagonal part of the covariance matrix at ``X`` and
        add it to ``target``.
        """
        phi_X = self.basis(X)
        target += (self.variance *
                   np.einsum('ij,j,ij->i', phi_X, self.kappa, phi_X))

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
        i = 0
        if self.parametrize_variance:
            target[0] += np.einsum('ij,ik,k,jk', dL_dK, phi_X, self.kappa, phi_X2)
            i = 1
        if self.ARD:
            target[i:] += (self.variance *
                          np.einsum('ij,ik,jk->k', dL_dK, phi_X, phi_X2))


def mean_function(input_dim, basis, variance=1., kappa=None, ARD=False,
                  parametrize_variance=False):
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
    :param kappa:     The weight of each basis function. The closer it is to
                        zero, the less important the corresponding basis
                        function.
    :type kappa:      :class:`numpy.ndarray`
    :param ARD:         If ``ARD`` is ``True``, then the mean function behaves
                        like a Relevance Vector Machine (RVM). If it is
                        ``False`` then it is equivalent to a common mean
                        function.
    :type ARD:          bool
    """
    part = MeanFunctionKernpart(input_dim, basis, variance=variance,
                                kappa=kappa, ARD=ARD,
                                parametrize_variance=parametrize_variance)
    return kern(input_dim, [part])
