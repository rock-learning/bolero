import warnings
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


def assert_warns(warning_class, func, *args, **kw):
    """Test that a certain warning occurs.
    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.
    func : callable
        Callable object to trigger warnings.
    *args : the positional arguments to `func`.
    **kw : the keyword arguments to `func`
    Returns
    -------
    result : the return value of `func`
    """
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        result = func(*args, **kw)
        if hasattr(np, 'FutureWarning'):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w
                 if e.category is not np.VisibleDeprecationWarning]

        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s"
                                 % func.__name__)

        found = any(warning.category is warning_class for warning in w)
        if not found:
            raise AssertionError("%s did not give warning: %s( is %s)"
                                 % (func.__name__, warning_class, w))
    return result


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


def check_context(context):
    """Check context vector.

    Parameters
    ----------
    context : array-like, shape (n_context_dims,)
        Context vector

    Returns
    -------
    context : array, shape (n_context_dims,)
        Context vector
    """
    if context is None:
        raise ValueError("Context is None.")
    return np.asarray(context)
