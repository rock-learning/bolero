import numpy as np
from sklearn.utils import check_array
from sklearn.metrics.pairwise import euclidean_distances
from .validation import check_random_state
from ._ranking_svm import optimize


MACHINE_EPSILON = np.finfo(np.float).eps ** 2


class RankingSVM(object):
    """Ranking Support Vector Machine.

    A trained ranking SVM model will predict rank-preserving values for each
    sample.

    The constraint violation cost will be computed according to

    .. math::

        C_i = 10^{C_{base}} (N_{training} - i)^{C_{pow}},
        \\text{ for} i = 1, \\ldots, N_{training}

    so that top-ranked samples have higher violation costs.

    Parameters
    ----------
    n_iter : int
        Number of training iterations

    c_base : float, optional (default: 6)
        Base for constraint violation cost

    c_pow : float, optional (default: 2)
        Exponent for constraint violation cost

    c_sigma : float, optional (default: 1)
        The sigma of the RBF kernel will be set to c_sigma times the average
        distance of training samples

    random_state : optional, int
        Seed for the random number generator
    """
    def __init__(self, n_iter=-1, epsilon=1.0, c_base=6.0, c_pow=2.0,
                 c_sigma=1.0, random_state=None):
        self.n_iter = n_iter
        self.c_base = c_base
        self.c_pow = c_pow
        self.c_sigma = c_sigma
        self.random_state = random_state

    def fit(self, X):
        """Fit ranking SVM.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data, sorted, highest rank first
        """
        self.n_samples, self.n_features = X.shape
        self.n_alpha = self.n_samples - 1
        self.X = X

        if self.n_samples < 2:
            raise ValueError("Expected at least 2 training samples, got %d"
                             % self.n_samples)

        random_state = check_random_state(self.random_state)
        n_iter = self.n_iter
        if n_iter < 0:
            n_iter = int(50000 * np.sqrt(self.n_features))

        K = euclidean_distances(self.X, squared=True)

        # Average distance between training data
        sigma = np.sqrt(K).sum() / ((self.n_samples - 1) * self.n_samples)
        sigma *= self.c_sigma
        self.denom = -np.maximum(2.0 * sigma ** 2, MACHINE_EPSILON)

        K /= self.denom
        np.exp(K, K)

        # Constraint violation cost
        Ci = np.linspace(self.n_alpha, 1, self.n_alpha) ** self.c_pow
        Ci *= 10 ** self.c_base

        # Optimize alpha parameters
        self.alpha = optimize(Ci, K, 1.0, n_iter, random_state)

        return self

    def predict(self, X):
        """Predict ranking values for new data.

        Parameters
        ----------
        X : array, shape (n_test, n_features)
            Test data

        Returns
        -------
        y : array, shape (n_test,)
            Ranking values
        """
        n_features = X.shape[1]

        if self.n_features != n_features:
            raise ValueError("Expected %d dimensions, got %d"
                             % (self.n_features, n_features))

        K = euclidean_distances(self.X, X, squared=True)
        K /= self.denom
        np.exp(K, K)

        return np.sum(self.alpha[:, np.newaxis] * (K[:-1] - K[1:]), axis=0)
