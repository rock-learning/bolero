import numpy as np
import numbers


def is_array_like(obj):
    """Check if obj is array-like."""
    return hasattr(obj, "__len__")


def is_scalar(obj):
    """Check if obj is a scalar value."""
    return isinstance(obj, numbers.Number)


def accumulate_feedbacks(feedbacks):
    """Accumulate feedbacks to one value."""
    if isinstance(feedbacks, memoryview):
        tmp = np.ndarray(len(feedbacks))
        tmp[:] = feedbacks[:]
        return np.sum(tmp)
    else:
        return np.sum(feedbacks)


try:
    from sklearn.utils import check_random_state
except:
    def check_random_state(seed):
        """Turn seed into a np.random.RandomState instance."""
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                         ' instance' % seed)
