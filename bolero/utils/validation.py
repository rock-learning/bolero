import numpy as np
import numbers


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


def check_feedback(feedback, compute_sum=False, check_inf=True, check_nan=True):
    """Check feedbacks.

    Parameters
    ----------
    feedback : float or array-like, shape (n_feedbacks,)
        Feedbacks, rewards, or fitness values

    compute_sum : bool, optional (default: False)
        Return the sum of feedbacks (e.g. the sum of rewards is called the
        return)

    check_inf : bool, optional (default: True)
        Raise ValueError if feedback contains 'inf'

    check_nan : bool, optional (default: True)
        Raise ValueError if feedback contains 'nan'
    """
    if check_inf and np.any(np.isinf(feedback)):
        raise ValueError("Received illegal feedback (inf). Check your environment!")

    if check_nan and np.any(np.isnan(feedback)):
        raise ValueError("Received illegal feedback (NaN). Check your environment!")

    if compute_sum:
        return np.sum(feedback)
    else:
        return np.asarray(feedback)
