"""External mathematical dependencies."""
import numpy as np
import scipy
from .dependency import compatible_version


if compatible_version("scipy", ">= 0.12"):
    # You can use this function from current scipy implementations
    from scipy.misc import logsumexp
else:
    def logsumexp(a, axis=None, b=None):
        a = np.asarray(a)
        if axis is None:
            a = a.ravel()
        else:
            a = np.rollaxis(a, axis)
        a_max = a.max(axis=0)
        if b is not None:
            b = np.asarray(b)
            if axis is None:
                b = b.ravel()
            else:
                b = np.rollaxis(b, axis)
            out = np.log(np.sum(b * np.exp(a - a_max), axis=0))
        else:
            out = np.log(np.sum(np.exp(a - a_max), axis=0))
        out += a_max
        return out
