"""
Implements various usefull kernels for GPy.

Author:
    Ilias Bilionis

Date:
    3/20/2014

"""


__all__ = ['MeanFunction']


from GPy.kern import Kern
from GPy.core.parameterization import Param
from GPy.core.parameterization.transformations import Logexp
from GPy.util.caching import Cache_this
import numpy as np


class MeanFunction(Kern):

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

    def __init__(self, input_dim, basis, variance=None, ARD=False,
                 active_dims=None, name='mean', useGP=False):
        """
        Initialize the object.
        """
        super(MeanFunction, self).__init__(input_dim, active_dims, name,
                                           useGP=useGP)
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
        self._num_params = basis.num_output
        if not ARD:
            if variance is None:
                variance = np.ones(1)
            else:
                variance = np.asarray(variance)
                assert variance.size == 1, 'Only 1 variance needed for a non-ARD kernel'
        else:
            if variance is not None:
                variance = np.asarray(variance)
                assert variance.size in [1, self.num_params], 'Bad number of variances'
                if variance.size != self.num_params:
                    variance = np.ones(self.num_params) * variance
            else:
                variance = np.ones(self.num_params)
        self.variance = Param('variance', variance, Logexp())
        self.link_parameters(self.variance)

    @Cache_this(limit=5, ignore_args=())
    def K(self, X, X2=None):
        """
        Evaluate the covariance (or cross covariance matrix) at ``X`` and ``X2``.
        If ``X2`` is ``None`` the covariance matrix is computed. The result is
        added to ``target``.
        """
        # Compute the design matrix
        # TODO: This should happen only once if X does not change!
        phi_X = self.basis(X)
        phi_X2 = phi_X if X2 is None or X2 is X else self.basis(X2)
        return np.einsum('ij,j,kj', phi_X, self.variance, phi_X2)

    @Cache_this(limit=5, ignore_args=())
    def Kdiag(self, X):
        """
        Evaluate only the diagonal part of the covariance matrix at ``X`` and
        add it to ``target``.
        """
        phi_X = self.basis(X)
        return np.einsum('ij,j,ij->i', phi_X, self.variance, phi_X)

    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        phi_X = self.basis(X)
        phi_X2 = phi_X if X2 is None or X2 is X else self.basis(X2)
        i = 0
        self.variance.gradient = np.einsum('ij,ik,jk->k', dL_dK, phi_X, phi_X2)