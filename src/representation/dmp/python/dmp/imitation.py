"""Imitation learning for DMP-based representations."""

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>
#          Arne Boeckmann <arneboe@informatik.uni-bremen.de>

import numpy as np
import warnings
from .dmp_cpp import DMP
from .rigid_body_dmp_cpp import RbDMP


def imitate_dmp(dmp, X, Xd=None, Xdd=None, alpha=0.0, set_weights=True):
    """Imitate DMP from demonstrations.

    Parameters
    ----------
    dmp : MovementPrimitive
        The DMP whose parameters should be adapted so that the
        demonstrations will be imitated. Note that the current weights
        will be replaced if 'set_weights' is true.

    X : array, shape (n_task_dims, n_steps, n_demos)
        The demonstrated trajectories to be imitated.

    Xd : array, shape (n_task_dims, n_steps, n_demos), optional
        Velocities of the demonstrated trajectories.

    Xdd : array, shape (n_task_dims, n_steps, n_demos), optional
        Accelerations of the demonstrated trajectories.

    alpha : float >= 0, optional (default: 0)
        The ridge parameter of linear regression. Small positive values of
        alpha improve the conditioning of the problem and reduce the
        variance of the estimates.

    set_weights : bool, optional (default: True)
        Set weights of the DMP after learning.

    Returns
    -------
    weights_mean : array, shape (n_task_dims, n_weights)
        Weights of imitating DMP

    weights_variance : array, shape (n_task_dims, n_weights)
        Variance of the best weight estimate
    """
    F = _determine_forces(dmp, X, Xd, Xdd)
    mean, variance = _learn_dmp_weights(dmp, F, alpha)
    if set_weights:
        dmp.set_weights(mean)
    return mean, variance


def _determine_forces(dmp, X, Xd=None, Xdd=None):
    """Reconstruct forces of a DMP to obtain the trajectories X."""
    return [dmp.determine_forces(X[:, :, i], Xd, Xdd)
            for i in range(X.shape[2])]


def _learn_dmp_weights(dmp, F, alpha=0.0):
    """Learn model of typical forces based on sample forces.

    Parameters
    ----------
    dmp : DMP
        Dynamical Movement Primitive

    F : list of array-like, shape (n_demos, n_task_dims, n_steps)
        Required forces determined from the demonstrations

    alpha : float >= 0, optional (default: 0)
        The ridge parameter of linear regression. Small positive values of
        alpha improve the conditioning of the problem and reduce the
        variance of the estimates.

    Returns
    -------
    weights_mean : array, shape (n_task_dims,)
        Weights of imitating DMP

    weights_variance : array, shape (n_task_dims,)
        Variance of the best weight estimate
    """
    from sklearn.linear_model import Ridge

    F = np.asarray(F)
    n_demos, n_task_dims, _ = F.shape

    # Create design matrix X
    X = np.array([dmp.get_activations(s, normalized=True)
                  for s in dmp.get_phases()])

    # Multiply with phase S
    X = X * dmp.get_phases()[:, np.newaxis]

    # Estimate variance of weights:
    # Fit weights for each trajectory individually and compute then the
    # variance of the weights
    sample_weights = []
    for sample_index in range(n_demos):
        # Train linear regression model (a separate one for each dimension)
        sample_weights.append([])
        for dim_index in range(n_task_dims):
            lr = Ridge(alpha=alpha, fit_intercept=False)
            lr.fit(X, F[sample_index][dim_index])
            sample_weights[-1].extend(list(lr.coef_))
    weights_variance = np.array(sample_weights).var(axis=0)

    # Fit weights for ->all<- trajectories jointly and use this as weight mean
    X = np.vstack([X for _ in range(n_demos)])
    y = np.hstack(F)
    assert n_task_dims == y.shape[0]
    weights = []
    for dim_index in range(n_task_dims):
        lr = Ridge(alpha=alpha, fit_intercept=False)
        lr.fit(X, y[dim_index])
        weights.extend(list(lr.coef_))

    return np.array(weights), weights_variance


class DMPImitator(object):
    """Learning by imitation of DMPs based on supplied reference trajectories.

    .. note::

        All demonstrations must have the same length.

    Parameters
    ----------
    alpha : float >= 0, optional (default: 0)
        The ridge parameter of linear regression. Small positive values of
        alpha improve the conditioning of the problem and reduce the
        variance of the estimates.

    set_weights : bool, optional (default: True)
        Set weights of the DMP after learning.
    """
    def __init__(self, alpha=0.0, set_weights=True):
        self.alpha = alpha
        self.set_weights = set_weights

    def imitate(self, dmp, X, Xd=None, Xdd=None):
        """Imitate demonstrated trajectories by adapting DMP parameters.

        This method implicitely assumes that start and goal state are
        identical in all demonstrations.

        Parameters
        ----------
        dmp : MovementPrimitive
            The DMP whose parameters should be adapted so that the
            demonstrations will be imitated. Note that the current weights
            will be replaced.

        X : array, shape (n_task_dims, n_steps, n_demos)
            The demonstrated trajectories to be imitated.

        Xd : array, shape (n_task_dims, n_steps, n_demos), optional
            Velocities of the demonstrated trajectories.

        Xdd : array, shape (n_task_dims, n_steps, n_demos), optional
            Accelerations of the demonstrated trajectories.

        Returns
        -------
        weights_mean : array, shape = (n_task_dims, n_weights)
            Weights of imitating DMP.

        weights_variance : array, shape = [n_task_dims, n_weights]
            Variance of the best weight estimate.
        """
        weights_mean, weights_variance = imitate_dmp(
            dmp, X, Xd, Xdd, self.alpha, set_weights=self.set_weights)
        return weights_mean, weights_variance


class DMPImitatorSchaal(object):
    """Fit demonstrated trajectories based on Schaal's DMP implementation."""
    def imitate(self, dmp, T, Td=None, Tdd=None):
        task_space_dimensions = T.shape[0]
        num_features = dmp.get_num_features()
        num_demonstrations = T.shape[2]
        weights = np.zeros((task_space_dimensions, num_features))

        for d in range(num_demonstrations):
            F = dmp.determine_forces(T[:, :, d], Td, Tdd)
            # Fit each task space dimension individually
            psi = np.array([dmp.get_activations(s, normalized=False)
                            for s in dmp.get_phases()])
            X = dmp.get_phases()
            for i in range(task_space_dimensions):
                sx2 = np.sum(np.array([X ** 2 for _ in
                                       range(num_features)]).T * psi, 0)
                sxtd = np.sum(np.array([(X * F[i]).flatten() for _ in
                                        range(num_features)]).T * psi, 0)
                weights[i] += sxtd / (sx2 + 1e-10)

        return weights / num_demonstrations, np.ones_like(weights)
