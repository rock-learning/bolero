# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
from scipy.stats import uniform
from scipy.optimize import fmin_l_bfgs_b
from bolero.utils.validation import check_random_state
from .environment import ContextualEnvironment


class Catapult(ContextualEnvironment):
    """Catapult environment, a benchmark for contextual policy search.

    In this benchmark problem, the agent controls a catapult which shoots onto
    specific target positions (the contexts) on a one-dimensional surface. The
    agent sets the parameters of the shot (velocity and angle of the
    catapult), and this environment simulates the shoot. The actual position
    where the object hits the ground is not communicated to the agent. Instead
    the agent is told only the cost of this specific trial, where cost is
    defined as
    cost = -abs(hit_position - target_position) - velocity_penalty * v,
    where target_position is the respective context, v ist the velocity of the
    shoot and velocity_penalty is configurable. Thus, this environment defines
    a contextual policy search problem.

    .. seealso::
        Bruno Castro da Silva, George Konidaris, Andrew Barto, "Active Learning
        of Parameterized Skills", ICML 2014

    Parameters
    ----------
    segments : sequence of tuples or int, optional (default: 10)
        Definition of the surface onto which the catapult throws. If an
        integer is passed, a surface consisting of the given number of segments
        is created randomly. Alternatively, the segments can be explicitly
        provided as a sequence of pairs, e.g.
        [(0.0, 0.0), (2.0, 1.0), (10.0, 1.0)].
        Each element of the sequence defines one point of the surface.
        It is assumed that the elements of the sequence are ordered
        according to their first component. The surface is created by
        linearly connecting all points of the segments sequence.

    catapult_pos : array-like, shape (2,), optional (default: (0, 0))
        The x and y positions at which the catapult is placed.

    velocity_penality : float>=0, optional (default: 0.1)
        A factor which controls how strongly large velocities are penalized in
        the cost function. Larger values correspond to stronger penalties.

    context_distribution : distribution from scipy.stats, optional (default: None)
        The distribution from which the contexts (goal positions for the
        catapult) are drawn. If None is given, a uniform distribution on the
        interval [2, 10] based on the random_state is used.

    context_interval : tuple, optional (default: (2, 10))
        Interval of the target position

    random_state : RandomState or int, optional (default: None)
        Random number generator or seed

    verbose : int, optional (default: 0)
        Verbosity level
    """
    def __init__(self, segments=10, catapult_pos=np.zeros(2),
                 velocity_penalty=0.1, context_distribution=None,
                 context_interval=(2, 10), random_state=None, verbose=0):
        self.segments = segments
        self.catapult_pos = catapult_pos
        self.velocity_penalty = velocity_penalty
        self.context_distribution = context_distribution
        self.context_interval = context_interval
        self.random_state = random_state
        self.verbose = verbose

    def init(self):
        self.random_state = check_random_state(self.random_state)
        if hasattr(self.segments, "__iter__"):
            self.segments = np.asarray(self.segments, dtype=np.float)
        else:
            self.segments = self._generate_segments(self.segments)
        self._compute_surface()

        # Remember the maximum feedback obtainable in a context (the baseline)
        self.max_feedback_cache = {}
        self.params = np.zeros(2)
        self.context = np.array([0.5])

    def _generate_segments(self, n_segments, n_superpositions=5):
        # Assume that the actual surface is a superposition of sinusoid
        # functions from which sample n_segments points and connect those
        # linearly

        # Generate sinusoids of the form -5 * sin(a * x + b)
        a = np.logspace(0, 0.5, n_superpositions)
        b = (0.25 * self.random_state.rand(n_superpositions) - 0.125) * np.pi

        # Generate x and y components of segments
        x = np.hstack((np.sort(self.random_state.rand(n_segments) * 8.0)))
        y = (-5 * np.sin(a * x[:, None] + b)).mean(axis=1)

        # Start at (0, 0)
        x[0] = y[0] = 0
        # Planar segment at the end which is long enough to avoid shooting
        # over the border
        x[-1] = 100.0
        y[-1] = y[-2]

        return np.vstack((x, y)).T

    def _compute_surface(self):
        """Determine coefficients of the linear segments of the surface."""
        self.coefficients = np.empty((self.segments.shape[0] - 1, 2))
        self.coefficients[:, 1] = (
            (self.segments[1:, 1] - self.segments[:-1, 1]) /
            (self.segments[1:, 0] - self.segments[:-1, 0]))
        self.coefficients[:, 0] = (self.segments[1:, 1] -
                                   self.segments[1:, 0] *
                                   self.coefficients[:, 1])

    def reset(self):
        """Reset the catapult environment."""
        self.evaluation_done = False

    def get_num_inputs(self):
        """Get number of inputs (desired state)."""
        return 2

    def get_num_outputs(self):
        """Get number of outputs (actual state)."""
        return 0

    def get_outputs(self, values):
        """Get current outputs."""

    def set_inputs(self, values):
        """Set desired velocity and angle."""
        self.params[:] = values

    def get_feedback(self, noisy=False):
        """The reward of the last roll-out."""
        return np.array([self.reward])

    def step_action(self):
        """Execute step perfectly."""
        self.reward = self._compute_reward(self.params, self.context)
        if self.verbose >= 1:
            print("[Catapult] Shooting with v = %g, angle = %g" %
                  tuple(self.params))
            print("[Catapult] Reward: %g" % self.reward)
        self.evaluation_done = True

    def is_evaluation_done(self):
        """Test if time is over."""
        return self.evaluation_done

    def is_behavior_learning_done(self):
        return False

    def get_num_context_dims(self):
        """Get number of context dimensions."""
        return 1

    def request_context(self, context=None):
        """ Request that a specific context is used.

        Parameters
        ----------
        context : ndarray, default=None
            The requested context that shall be used in the next rollout.
            Defaults to None. In that case, the environment selects the next
            context

        Returns
        -------
        context: ndarray
            The actual context used in the next rollout. This environment
            accepts all external requests.
        """
        if context is None:
            context = self._normalize_context(self._sample_new_context())
        self.context = context
        return self.context

    def _normalize_context(self, context):
        context_range = self.context_interval[1] - self.context_interval[0]
        return (context - self.context_interval[0]) / context_range

    def _denormalize_context(self, context):
        context_range = self.context_interval[1] - self.context_interval[0]
        return context * context_range + self.context_interval[0]

    def get_maximum_feedback(self, context):
        """Returns the maximal feedback obtainable in given context."""
        c = tuple(list(context))
        if c not in self.max_feedback_cache:
            self.max_feedback_cache[c] = -np.inf
            for _ in range(10):
                x0 = [uniform.rvs(5.0, 5.0), uniform.rvs(0.0, np.pi / 2)]
                result = fmin_l_bfgs_b(
                    lambda x: -self._compute_reward(x, context),
                    x0, approx_grad=True,
                    bounds=[(5.0, 10.0), (0.0, np.pi / 2)])
                if -result[1] > self.max_feedback_cache[c]:
                    self.max_feedback_cache[c] = -result[1]
        return self.max_feedback_cache[c]

    def _sample_new_context(self):
        if self.context_distribution is None:
            return self.random_state.uniform(*self.context_interval, size=1)
        else:
            return self.context_distribution.rvs(1)

    def _compute_reward(self, params, context):
        context = self._denormalize_context(context)
        v, theta = params
        # Enforce boundaries for velocity and angle
        v = np.clip(v, 5.0, 10.0)
        theta = np.clip(theta, 0.0, np.pi / 2.0)

        hit = self._shoot(v, theta)
        if self.verbose >= 1:
            print("[Catapult] Hit the ground at %g (target: %g)"
                  % (hit, self._denormalize_context(self.context[0])))
        return -np.abs(hit - context) - self.velocity_penalty * v

    def _shoot(self, v, theta):
        a, b, c = self._trajectory_params(v, theta)
        return self._intersect(a, b, c)

    def _trajectory_params(self, v, theta):
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        return (-9.8 / (2 * v ** 2 * np.cos(theta) ** 2),
                vy / vx + 9.8 * self.catapult_pos[0] / vx ** 2,
                self.catapult_pos[1] - vy * self.catapult_pos[0] / vx -
                0.5 * 9.8 * self.catapult_pos[0] ** 2 / vx ** 2)

    def _intersect(self, a, b, c):
        p = (b - self.coefficients[:, 1]) / a
        q = (c - self.coefficients[:, 0]) / a
        x = -p / 2 + np.sqrt(np.maximum(p ** 2 / 4 - q, 0.0))
        intersections = np.where(np.logical_and(self.segments[:-1, 0] <= x,
                                                x <= self.segments[1:, 0]))
        if len(intersections[0]) == 0:
            raise Exception("Could not intersect trajectory and surface. The "
                            "ball did not hit the ground. Extend the surface!")
        return x[intersections].min()

    def plot(self, ax, v=10.0):
        x = np.linspace(0, 10, 1000)
        ys = np.interp(x, self.segments[:, 0], self.segments[:, 1])

        for theta in np.linspace(0.0, np.pi / 2, 25):
            a, b, c = self._trajectory_params(v=v, theta=theta)
            y = a * x ** 2 + b * x + c
            select = np.where(np.logical_and.accumulate(y >= ys))
            ax.plot(x[select], y[select])
            x0 = self._intersect(a, b, c)
            ax.plot([x0], [a * x0 ** 2 + b * x0 + c], 'ko')

        ax.plot(self.segments[:, 0], self.segments[:, 1], 'k')
        ax.set_xlim(-0.1, 10)
        ax.set_ylim(-1, 10)
