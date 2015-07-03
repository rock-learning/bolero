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


def check_feedback(feedback, compute_sum=False):
    finite = np.isfinite(feedback)
    if not np.all(finite):
        raise ValueError("Received illegal feedback. Check your environment!")

    if compute_sum:
        return np.sum(feedback)
    else:
        return feedback
