import numpy as np


# Make product functions pickable

def multiply(a, b):
    return np.multiply(a, b)


def dot(a, b):
    return np.dot(a, b)


class Scaling(object):
    """Combines multiple scaling and preconditioning variables.

    Parameters
    ----------
    variance : float, optional (default: None)
        A scalar scaling factor or the statistical variance of a variable.

    covariance : array-like, shape = (n_params,) or (n_params, n_params),
            optional (default: None)
        Either a diagonal or a full covariance matrix. A full covariance
        can contain information about the correlation of variables.

    compute_inverse : boolean
        Flag to indicate whether we need an inversion of the scaling.
    """
    def __init__(self, variance=None, covariance=None, compute_inverse=False):
        self.variance = variance
        self.covariance = covariance
        self.compute_inverse = compute_inverse
        self._combine()

    def _combine(self):
        """Combine variance and covariance to a single scaling variable."""
        self.scaling_ = 1.0
        if self.compute_inverse:
            self.inv_scaling_ = 1.0
        self.product_ = multiply

        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance)
            if self.covariance.ndim == 1:
                self.scaling_ = self.covariance
                if self.compute_inverse:
                    self.inv_scaling_ = 1.0 / self.covariance
            elif self.covariance.ndim == 2:
                self.scaling_ = self.covariance
                if self.compute_inverse:
                    self.inv_scaling_ = np.linalg.inv(self.covariance)
                self.product_ = dot
            else:
                raise ValueError("Covariance matrix must have either "
                                 "1 or 2 dimensions but has %d"
                                 % self.covariance.ndim)

        if self.variance is not None:
            self.scaling_ *= self.variance
            if self.compute_inverse:
                self.inv_scaling_ /= self.variance

    def scale(self, params):
        """Scale variables.

        Transform from search space of the optimizer to parameter space.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            Parameters.

        Returns
        -------
        scaled_params : array-like, shape = (n_params,)
            Scaled parameters.
        """
        return self.product_(self.scaling_, params)

    def inv_scale(self, scaled_params):
        """Inverse scaling.

        Transform from parameter space to the search space of the optimizer.

        Parameters
        ----------
        scaled_params : array-like, shape = (n_params,)
            Scaled parameters.

        Returns
        -------
        params : array-like, shape = (n_params,)
            Parameters.
        """
        if not self.compute_inverse:
            raise ValueError("Inverse scaling is not computed!")

        return self.product_(self.inv_scaling_, scaled_params)


class NoScaling(object):
    """Scaler which does not change scaling."""

    def scale(self, params):
        """Scale variables.

        Transform from search space of the optimizer to parameter space.

        Parameters
        ----------
        params : array-like, shape = (n_params,)
            Parameters.

        Returns
        -------
        scaled_params : array-like, shape = (n_params,)
            Scaled parameters.
        """
        return params

    def inv_scale(self, scaled_params):
        """Inverse scaling.

        Transform from parameter space to the search space of the optimizer.

        Parameters
        ----------
        scaled_params : array-like, shape = (n_params,)
            Scaled parameters.

        Returns
        -------
        params : array-like, shape = (n_params,)
            Parameters.
        """
        return scaled_params
