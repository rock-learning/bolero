# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>


from abc import ABCMeta, abstractmethod
import numpy as np
from ..representation.context_transformations import CONTEXT_TRANSFORMATIONS
from ..utils.scaling import Scaling, NoScaling
from ..utils.validation import check_random_state
from ..utils.dependency import compatible_version


class UpperLevelPolicy(object):
    """Upper-level policy interface."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, context=None, explore=True):
        """Evaluates policy.

        Samples weight vector from distribution if explore is true, otherwise
        return the distribution's mean.

        Parameters
        ----------
        context: array-like, shape (n_context_dims,), optional (default: None)
            Context vector (ignored by non-contextual policies)

        explore: bool
            if true, weight vector is sampled from distribution. otherwise the
            distribution's mean is returned
        """

    @abstractmethod
    def fit(self, X, Y, weights, context_transform=True):
        """Trains policy by weighted maximum likelihood.

        .. note:: This call changes this policy (self)

        Parameters
        ----------
        X: array-like, shape (n_samples, n_context_dims)
            2d array of context vectors

        Y: array-like, shape (n_samples, n_params)
            2d array of parameter vectors

        weights: array-like, (n_samples,)
            weights of individual samples (weight vectors)
        """

    def transform_context(self, context):
        """Transform context based on internal context transformation. """
        return context  # no transformation as default


class BoundedScalingPolicy(UpperLevelPolicy):
    """Combines a scaling operation, an upper-level policy, and applies limits.

    Parameters
    ----------
    upper_level_policy : UpperLevelPolicy
        Upper level policy

    scaling : Scaling
        Parameter scaling for the output of the upper-level policy, can be
        "auto". In this case we will use a scaling with a covariance based
        on the range of the boundaries. The standard deviation will be half
        of the parameter range for each component. Hence, the parameter
        'bounds' must not be None. If scaling is "none", no scaling is
        performed.

    bounds : array-like, shape (n_params, 2), optional (default: None)
        Upper and lower boundaries for each parameter
    """
    def __init__(self, upper_level_policy, scaling, bounds=None):
        self.upper_level_policy = upper_level_policy

        if scaling == "auto":
            if bounds is None:
                raise ValueError("scaling='auto' requires boundaries")
            else:
                covariance_diag = (bounds[:, 1] - bounds[:, 0]) ** 2 / 4.0
                scaling = Scaling(covariance=covariance_diag,
                                  compute_inverse=True)
        elif scaling == "none" or scaling is None:
            scaling = NoScaling()

        self.scaling = scaling
        self.bounds = bounds

    def transform_context(self, context):
        """Transform context based on internal context transformation.

        Parameters
        ----------
        context: array-like, (n_context_dims,)
            context vector

        Returns
        -------
        context_features: array-like, (self.n_features,)
            the features obtained by the context transformation
        """
        return self.upper_level_policy.transform_context(context)

    @property
    def W(self):
        return self.upper_level_policy.W

    @W.setter
    def W(self, W):
        self.upper_level_policy.W = W

    def __call__(self, context, explore=True):
        """Evaluates policy for given context.

        Samples weight vector from distribution if explore is true, otherwise
        return the distribution's mean (which depends on the context).

        Parameters
        ----------
        context: array-like, (n_context_dims,)
            context vector

        explore: bool
            if true, weight vector is sampled from distribution. otherwise the
            distribution's mean is returned

        Returns
        -------
        params: array, shape (n_params,)
            Parameters
        """
        params = self.upper_level_policy(context, explore)
        params = self.scaling.scale(params)
        if self.bounds is not None:
            np.clip(params, self.bounds[:, 0], self.bounds[:, 1], out=params)
        return params

    def fit(self, X, Y, weights=None, context_transform=True):
        """Trains policy by weighted maximum likelihood.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_context_dims)
            Context vectors

        Y: array-like, shape (n_samples, n_params)
            Low-level policy parameter vectors

        weights: array-like, shape (n_samples,)
            Weights of individual samples (should depend on the obtained
            reward)
        """
        Y = self.scaling.inv_scale(Y.T).T
        self.upper_level_policy.fit(X, Y, weights, context_transform)


class ConstantGaussianPolicy(UpperLevelPolicy):
    """Gaussian policy with constant mean.

    Upper-level policy, which samples weight vectors for lower-level policies
    (like DMPs) from a Gaussian distribution

    .. math::

        \pi(u) = \mathcal{N}(u|mu, \Sigma)

    with constant mean mu and covariance Sigma. Thus, contextual information
    cannot be taken into account

    See [Deisenroth2013]_, page 131 for details.

    Parameters
    ----------
    n_weights : int
        dimensionality of weight vector of lower-level policy

    covariance : string ("full" or "diag")
        whether full or diagonal covariance is learned

    mean : array-like, shape (num_samples)
        initial mean of policy

    covariance_scale : float
        the covariance is initialized to numpy.eye(n_weights) *
        covariance_scale

    random_state : optional, int
        Seed for the random number generator.
    """
    def __init__(self, n_weights, covariance="full", mean=None,
                 covariance_scale=1.0, random_state=None):
        self.n_weights = n_weights
        self.covariance = covariance
        self.random_state = check_random_state(random_state)

        self.mean = mean
        if self.mean is None:
            self.mean = np.ones(n_weights)
        self.Sigma = np.eye(n_weights) * covariance_scale

    def __call__(self, context=None, explore=True):
        """Evaluates policy.

        Samples weight vector from distribution if explore is true, otherwise
        return the distribution's mean.

        Parameters
        ----------
        context : array-like, (n_context_dims,)
            context vector (ignored by this policy, defaults to None)

        explore : bool
            if true, weight vector is sampled from distribution. otherwise the
            distribution's mean is returned

        Returns
        -------
        parameter_vector: array-like, (n_weights,)
            the selected parameters
        """
        # Note: Context is ignored
        if not explore:
            return self.mean
        else:
            # Sample from normal distribution
            return self.random_state.multivariate_normal(
                mean=self.mean, cov=self.Sigma, size=1)[0]

    def fit(self, X, Y, weights, *_):
        """Trains policy by weighted maximum likelihood.

        Parameters
        ----------
        X : ignored

        Y : array-like, shape (num_samples, n_weights)
            2d array of parameter vectors

        weights : array-like, (num_samples,)
            weights of individual samples (weight vectors)
        """
        # Avoid that all but one weights become 0
        weights[weights == 0] = np.finfo(np.float).eps

        self.mean = (weights * Y.T).sum(axis=1) / weights.sum()

        # Estimate covariance matrix (either full or diagonal)
        Z = (weights.sum() ** 2 - (weights ** 2).sum()) / weights.sum()
        if self.covariance == 'full':
            nominator = np.zeros_like(self.Sigma)
            for i in range(Y.shape[0]):
                temp = Y[i] - self.mean
                nominator += weights[i] * np.outer(temp, temp)
            self.Sigma = nominator / (1e-10 + Z)
        elif self.covariance == 'diag':
            nominator = np.zeros(self.Sigma.shape[0])
            for i in range(Y.shape[0]):
                nominator += weights[i] * (Y[i] - self.mean) ** 2
            self.Sigma = np.diag(nominator / (1e-10 + Z))

        if not np.isfinite(self.Sigma).all():
            raise ValueError("Computed non-finite covariance matrix.")

    def probabilities(self, Y):
        """Probabilities of parameter vectors Y

        Parameters
        ----------
        Y : array-like, shape (num_samples, n_weights)
            2d array of parameter vectors

        Returns
        ----------
        resp : array-like, shape (num_samples)
            the probabilities of the samples under this policy

        """
        if not compatible_version("scipy", ">= 0.14"):
            raise ImportError(
                "SciPy >= 0.14 is required for "
                "'scipy.stats.multivariate_normal'.")
        from scipy.stats import multivariate_normal
        return multivariate_normal(mean=self.mean, cov=self.Sigma).pdf(Y)


class ContextTransformationPolicy(UpperLevelPolicy):
    """ A wrapper class around a policy which transform contexts.

    Parameters
    ----------
    PolicyClass: subclass of UpperLevelPolicy
        The class of the actual policy, which will be constructed internally.
        All calls are delegated to this class after context transformation.

    n_params : int
        dimensionality of weight vector of lower-level policy

    n_context_dims : int
        dimensionality of context vector

    context_transformation : string or callable
        (Nonlinear) transformation for the context.
    """
    def __init__(self, PolicyClass, n_params, n_context_dims,
                 context_transformation, *args, **kwargs):
        self.context_transformation = context_transformation
        if self.context_transformation is None:
            self.ct = CONTEXT_TRANSFORMATIONS["affine"]
        elif (isinstance(self.context_transformation, basestring) and
              self.context_transformation in CONTEXT_TRANSFORMATIONS):
            self.ct = CONTEXT_TRANSFORMATIONS[self.context_transformation]
        else:
            self.ct = self.context_transformation

        # Determine dimensionality of context feature vector
        self.n_features = self.transform_context(
            np.zeros(n_context_dims)).shape[0]

        # Create actual policy class to which all calls will be delegated after
        # context transformation
        self.policy = PolicyClass(n_params, self.n_features, *args, **kwargs)

    @property
    def W(self):
        return self.policy.W

    @W.setter
    def W(self, W):
        self.policy.W = W

    def transform_context(self, context):
        """Transform context based on internal context transformation.

        Parameters
        ----------
        context: array-like, (n_context_dims,)
            context vector

        Returns
        ----------
        context_features: array-like, (self.n_features,)
            the features obtained by the context transformation

        """
        return self.ct(context)

    def __call__(self, context, explore=True):
        """Evaluates policy for given context.

        Samples weight vector from distribution if explore is true, otherwise
        return the distribution's mean (which depends on the context).

        Parameters
        ----------
        context: array-like, (n_context_dims,)
            context vector

        explore: bool
            if true, weight vector is sampled from distribution. otherwise the
            distribution's mean is returned

        Returns
        -------
        params: array, shape (n_params,)
            Parameters
        """
        context_features = self.transform_context(context)
        return self.policy(context_features, explore)

    def fit(self, X, Y, weights=None, context_transform=True):
        """Trains policy by weighted maximum likelihood.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_context_dims)
            Context vectors

        Y: array-like, shape (n_samples, n_params)
            Low-level policy parameter vectors

        weights: array-like, shape (n_samples,)
            Weights of individual samples (should depend on the obtained
            reward)

        context_transform: bool
            If true, the contexts (stored in X) will be transformed to context
            features prior to actual fitting.
        """
        if context_transform:
            # Perform context transformation
            X = np.array([self.transform_context(X[i]) for i in range(X.shape[0])])
        self.policy.fit(X, Y, weights)


class LinearGaussianPolicy(UpperLevelPolicy):
    """Gaussian policy with mean that is linear in context.

    Upper-level policy, which samples weight vectors for lower-level policies
    (like DMPs) from a Gaussian distribution

    .. math::

        \pi(u|x) = \mathcal{N}(u|W^T x, \Sigma).

    The distribution's mean depends linearly on the context x via the matrix W
    but the covariance Sigma is context independent.

    See [Deisenroth2013]_, page 131 for details.

    Parameters
    ----------
    n_params : int
        dimensionality of weight vector of lower-level policy

    n_context_dims : int
        dimensionality of context vector

    mean : array-like, shape (n_params,), optional (default: None)
        initial mean of policy. Note: This mean is overwritten in the first
        learning step.

    covariance_scale : float, optional (default: 1)
        The covariance is initialized to np.eye(n_params) * covariance_scale.

    gamma : float, optional (default: 0)
        regularization parameter for weighted maximum likelihood estimation
        of W.

    random_state : int, optional (default: None)
        Seed for the random number generator.
    """
    def __init__(self, n_params, n_context_dims, mean=None,
                 covariance_scale=1.0, gamma=0.0, random_state=None):
        self.n_params = n_params
        self.n_context_dims = n_context_dims
        self.mean = mean
        self.covariance_scale = covariance_scale
        self.gamma = gamma
        self.random_state = check_random_state(random_state)

        # Create weight matrix and covariance matrix Sigma
        self.W = np.zeros((self.n_params, self.n_context_dims))
        if self.mean is not None:
            # It is assumed that the last dimension of the context is a
            # constant bias dimension
            self.W[:, -1] = self.mean
        self.Sigma = np.eye(self.n_params) * self.covariance_scale

    def __call__(self, context, explore=True):
        """Evaluates policy for given context.

        Samples weight vector from distribution if explore is true, otherwise
        return the distribution's mean (which depends on the context).

        Parameters
        ----------
        context: array-like, (n_context_dims,)
            context vector

        explore: bool
            if true, weight vector is sampled from distribution. otherwise the
            distribution's mean is returned
        """
        if explore:
            return self.random_state.multivariate_normal(
                self.W.dot(context), self.Sigma, size=[1])[0]
        else:
            return self.W.dot(context)

    def fit(self, X, Y, weights, context_transform=True):
        """Trains policy by weighted maximum likelihood.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_context_dims)
            Context vectors

        Y: array-like, shape (n_samples, n_params)
            Low-level policy parameter vectors

        weights: array-like, shape (n_samples,)
            Weights of individual samples (should depend on the obtained
            reward)
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        weights = np.asarray(weights)

        # Avoid that all but one weights become 0
        weights[weights == 0] = np.finfo(np.float).eps
        Z = (weights.sum() ** 2 - (weights ** 2).sum()) / weights.sum()

        nominator = np.zeros_like(self.Sigma)
        for i in range(Y.shape[0]):
            temp = Y[i] - self.W.dot(X[i])
            nominator += weights[i] * np.outer(temp, temp)
        self.Sigma = nominator / Z

        if not np.isfinite(self.Sigma).all():
            raise ValueError("Computed non-finite covariance matrix.")

        D = np.diag(weights)
        self.W = np.linalg.pinv(X.T.dot(D).dot(X) + np.eye(X.shape[1]) *
                                self.gamma).dot(X.T).dot(D).dot(Y).T
