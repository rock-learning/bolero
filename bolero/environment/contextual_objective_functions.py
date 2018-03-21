"""Contextual function optimization benchmarks."""


# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>


from abc import ABCMeta, abstractmethod
from ..utils import check_random_state
from .objective_functions import Sphere, Rastrigin
from .environment import ContextualEnvironment
import numpy as np


class ContextualObjectiveFunction(object):
    """Base of contextual objective functions.

    Parameters
    ----------
    random_state : RandomState or int
        Random number generator or seed

    n_dims : int
        Number of dimensions

    n_context_dims : int
        Number of context dimensions
    """
    __metaclass__ = ABCMeta

    def __init__(self, random_state, n_dims, n_context_dims):
        self.random_state = check_random_state(random_state)
        self.n_dims = n_dims
        self.n_context_dims = n_context_dims

    @abstractmethod
    def feedback(self, x, s):
        """Evaluate function at a given point.

        Parameters
        ----------
        x : array-like, shape (n_dims,)
            Parameters

        s : array-like, shape (n_context_dims,)
            Context

        Returns
        -------
        f : float
            Function value
        """


class ConstantContextualSphere(ContextualObjectiveFunction):
    """Sphere function with f_opt depending linearly on the context."""
    def __init__(self, random_state, n_dims, n_context_dims):
        super(ConstantContextualSphere, self).__init__(
            random_state, n_dims, n_context_dims)
        self.sphere = Sphere(self.random_state, self.n_dims)
        self.w = self.random_state.randn(n_context_dims)

    def feedback(self, x, _):
        return self.sphere.feedback(x)

    def x_opt(self, _):
        return self.sphere.x_opt

    def f_opt(self, _):
        return self.sphere.f_opt


class LinearContextualSphere(ContextualObjectiveFunction):
    """Sphere function with f_opt depending linearly on the context."""
    def __init__(self, random_state, n_dims, n_context_dims):
        super(LinearContextualSphere, self).__init__(
            random_state, n_dims, n_context_dims)
        self.sphere = Sphere(self.random_state, self.n_dims)
        self.w = self.random_state.randn(n_context_dims)

    def feedback(self, x, s):
        return self.w.dot(s) + self.sphere.feedback(x)

    def x_opt(self, _):
        return self.sphere.x_opt

    def f_opt(self, s):
        return self.w.dot(s) + self.sphere.f_opt


class QuadraticContextualSphere(ContextualObjectiveFunction):
    """Sphere function with x_opt depending quadratic on the context."""
    def __init__(self, random_state, n_dims, n_context_dims,
                 respect_bounds=False):
        super(QuadraticContextualSphere, self).__init__(
            random_state, n_dims, n_context_dims)
        self.sphere = Sphere(self.random_state, self.n_dims)
        self.V = self.random_state.randn(n_context_dims, n_context_dims)
        self.V *= 0.1 / n_context_dims
        self.W = self.random_state.randn(n_dims, n_context_dims, n_context_dims)
        self.W *= 1.0 / n_context_dims
        self.respect_bounds = respect_bounds
        self._scale_W()

    def _scale_W(self):
        scaled = True
        while scaled:
            scaled = False
            s_min = -5 * np.ones(self.n_context_dims)
            x_opt_abs = np.abs(self.x_opt(s_min))
            if np.any(x_opt_abs == 5):
                self.W /= 2.0
                scaled = True

            s_max = 5 * np.ones(self.n_context_dims)
            x_opt_abs = np.abs(self.x_opt(s_max))
            if np.any(x_opt_abs == 5):
                self.W /= 2.0
                scaled = True

    def _x_offset(self, s):
        x_opt_offset = np.empty(self.n_dims)
        for d in range(self.n_dims):
            x_opt_offset[d] = s.dot(self.W[d]).dot(s)
        return x_opt_offset

    def _f_offset(self, s):
        return s.dot(self.V).dot(s)

    def feedback(self, x, s):
        return (self.sphere.feedback(x, x_opt_offset=self._x_offset(s)) +
                self._f_offset(s))

    def x_opt(self, s):
        x = self.sphere.x_opt + self._x_offset(s)
        if self.respect_bounds:
            x = np.clip(x, -5 * np.ones_like(x), 5 * np.ones_like(x))
        return x

    def f_opt(self, s):
        if self.respect_bounds:
            return self.feedback(self.x_opt(s), s)
        else:
            return self.sphere.f_opt + self._f_offset(s)


class QuadraticContextualRastrigin(ContextualObjectiveFunction):
    """Rastrigin function with x_opt depending quadratic on the context."""
    def __init__(self, random_state, n_dims, n_context_dims,
                 respect_bounds=False):
        super(QuadraticContextualRastrigin, self).__init__(
            random_state, n_dims, n_context_dims)
        self.rastrigin = Rastrigin(self.random_state, self.n_dims)
        self.V = self.random_state.randn(n_context_dims, n_context_dims)
        self.V *= 0.1 / n_context_dims
        self.W = self.random_state.randn(n_dims, n_context_dims, n_context_dims)
        self.W *= 1.0 / n_context_dims
        self.respect_bounds = respect_bounds
        self._scale_W()

    def _scale_W(self):
        scaled = True
        while scaled:
            scaled = False
            s_min = -5 * np.ones(self.n_context_dims)
            x_opt_abs = np.abs(self.x_opt(s_min))
            if np.any(x_opt_abs == 5):
                self.W /= 2.0
                scaled = True

            s_max = 5 * np.ones(self.n_context_dims)
            x_opt_abs = np.abs(self.x_opt(s_max))
            if np.any(x_opt_abs == 5):
                self.W /= 2.0
                scaled = True

    def _x_offset(self, s):
        x_opt_offset = np.empty(self.n_dims)
        for d in range(self.n_dims):
            x_opt_offset[d] = s.dot(self.W[d]).dot(s)
        return x_opt_offset

    def _f_offset(self, s):
        return s.dot(self.V).dot(s)

    def feedback(self, x, s):
        return (self.rastrigin.feedback(x, x_opt_offset=self._x_offset(s)) +
                self._f_offset(s))

    def x_opt(self, s):
        x = self.rastrigin.x_opt + self._x_offset(s)
        if self.respect_bounds:
            x = np.clip(x, -5 * np.ones_like(x), 5 * np.ones_like(x))
        return x

    def f_opt(self, s):
        if self.respect_bounds:
            return self.feedback(self.x_opt(s), s)
        else:
            return self.rastrigin.f_opt + self._f_offset(s)


CONTEXTUAL_FUNCTIONS = {
    "LinearContextualSphere": LinearContextualSphere,
    "QuadraticContextualSphere": QuadraticContextualSphere,
    "QuadraticContextualRastrigin": QuadraticContextualRastrigin,
    }


class ContextualObjectiveFunction(ContextualEnvironment):
    """Artificial contextual benchmark function.

    Parameters
    ----------
    name : string, optional (default: 'LinearContextualSphere')
        Name of the objective function

    n_params : int, optional (default: 1)
        Number of dimensions

    n_context_dims : int, optional (default: 1)
        Number of context dimensions

    random_state : RandomState or int, optional (default: None)
        Random number generator or seed

    kwargs : optional (default: {})
        Additional keyword arguments for objective function
    """
    def __init__(self, name="LinearContextualSphere", n_params=1,
                 n_context_dims=1, random_state=None, **kwargs):
        self.name = name
        self.n_params = n_params
        self.n_context_dims = n_context_dims
        self.random_state = random_state
        self.kwargs = kwargs

    def init(self):
        if self.n_params <= 0:
            raise ValueError("Number of parameters (%d) must be > 0"
                             % self.n_params)
        if self.n_context_dims <= 0:
            raise ValueError("Number of context dimensions (%d) must be > 0"
                             % self.n_context_dims)
        if not self.name in CONTEXTUAL_FUNCTIONS:
            raise ValueError("Unknown function '%s' requested, select one of "
                             "%s instead"
                             % (self.name, CONTEXTUAL_FUNCTIONS.keys()))

        self.random_state = check_random_state(self.random_state)
        self.objective = CONTEXTUAL_FUNCTIONS[self.name](
            self.random_state, self.n_params, self.n_context_dims,
            **self.kwargs)

        self.params = np.empty(self.n_params)
        self.context = np.zeros(self.n_context_dims)
        self.f = np.nan

    def reset(self):
        self.done = False

    def get_num_inputs(self):
        return self.n_params

    def get_num_outputs(self):
        return 0

    def get_outputs(self, _):
        pass

    def set_inputs(self, values):
        self.params = values
        self.done = True

    def step_action(self):
        self.f = self.objective.feedback(self.params, self.context)

    def is_evaluation_done(self):
        return self.done

    def get_feedback(self):
        return np.array([self.f])

    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return False

    def request_context(self, context=None):
        """Request that a specific context is used.

        Parameters
        ----------
        context : array-like, shape (n_context_dims,), optional (default=None)
            The requested context that shall be used in the next rollout.
            Defaults to None. In that case, the environment selects the next
            context.

        Returns
        -------
        context : array-like, shape (n_context_dims,)
            The actual context used in the next rollout. This is either the
            requested context or selected by the environment.
        """
        if context is None:
            self.context = 10 * self.random_state.rand(self.n_context_dims) - 5
        else:
            self.context = context
        return self.context

    def get_num_context_dims(self):
        """Returns the number of context dimensions."""
        return self.n_context_dims

    def get_maximum_feedback(self, context):
        """Returns the maximum feedback obtainable in given context."""
        return self.objective.f_opt(context)
