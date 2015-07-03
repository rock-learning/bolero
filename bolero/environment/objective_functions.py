"""Function optimization benchmarks.

Function optimization benchmarks
================================

These functions are similar to the ones found at BBOB (Real-Parameter
Black-Box Optimization Benchmarking). Code and documentation are based on the
software. See the following links for more details:

* http://coco.gforge.inria.fr/doku.php
* http://coco.gforge.inria.fr/doku.php?id=downloads
* http://coco.lri.fr/downloads/download13.09/bbobdocfunctions.pdf

Test Functions
==============

Search Domain
-------------

All functions are defined and can be evaluated over :math:`R^D`, while the
actual search domain is given as :math:`[-5,5]^D`. On some functions a
penalty boundary handling is applied (see :func:`f_pen`).

Location of the optimum
-----------------------

The optimum is denoted as :math:`x_{opt}` and :math:`f_{opt} = f(x_{opt})`.
All functions have their global optimum in :math:`[-5,5]^D`. The majority of
functions has the global optimum in :math:`[-4,4]^D` and for many of them
x_opt is drawn uniformly from this compact. The value is rounded after two
decimal places and clipped to :math:`[-1000, 1000]`. In the function
definitions a transformed variable vector z is often used instead of the
argument x. The vector z has its optimum in :math:`z_{opt} = 0`, if not
stated otherwise.

Linear Transformations
----------------------

Linear transformations of the search space are applied to derive
non-separable functions from separable ones and to control the
conditioning of the function. We use multiplication with orthogonal
(rotation) matrices :math:`Q, R`. Orthogonal matrices are generated
from standard normally distributed entries by Gram-Schmidt
orthonormalization. Columns and rows of an orthogonal matrix form an
orthonormal basis.

Non-Linear Transformations and Symmetry Breaking
------------------------------------------------

In order to make relatively simple, but well understood functions less
regular, on some functions non-linear transformations are applied in x-
or f-space. Both transformations :math:`T_{osc}: R^n \\rightarrow R^n, n
\in \{1, D\}`, and :math:`T_{asy}: R^D \\rightarrow R^D` are defined
coordinate-wise (see :func:`T_asy` and :func:`T_osc`). They are smooth
and have, coordinate-wise, a strictly positive derivative. :math:`T_{osc}`
is oscillating about the identity, where the oscillation is scale invariant
w.r.t. the origin. :math:`T_{asy}` is the identity for negative values. When
:math:`T_{asy}` is applied, a portion of :math:`1 / 2^D` of the search space
remains untransformed.

Function Properties
===================

Ill-Conditioning
----------------

Ill-conditioning is a typical challenge in real-parameter optimization and,
besides multimodality, probably the most common one. Conditioning of a
function can be rigorously formalized in the case of convex quadratic
functions, :math:`\\frac{1}{2} x^THx` where H is a symmetric positive
definit matrix, as the condition number of the Hessian matrix H. Since
contour lines associated to a convex quadratic function are ellipsoids,
the condition number corresponds to the square root of the ratio between
the largest axis of the ellipsoid and the shortest  axis. For more general
functions, conditioning loosely refers to the square of the ratio between
the largest direction and smallest of a contour line. The testbed contains
ill-conditioned functions with a typical condtioning of :math:`10^6`. We
believe this is a realistic requirement, while we have seen practical
problems with condtioning as large as :math:`10^{10}`.

Regularity
----------

Functions from simple formulas are often highly regular. We have used a
non-linear transformation, :math:`T_{osc}`, in order to introduce small,
smooth but clearly visible irregularities. Furthermore, the testbed contains
a few highly irregular functions.

Separability
------------

In general, separable functions pose an essentially different search
problem to solve, because the search process can be reduced to D
one-dimensional search procedures. Consequently, non-separable problems
must be considered much more difficult and most benchmark functions are
designed being non-separable. The typical well-established technique to
generate non-separable functions from separable ones is the application
of a rotation matrix R.

Symmetry
--------

Stochastic search procedures often rely on Gaussian distributions to
generate new solutions and it has been argued that symmetric benchmark
functions could be in favor of these operators. To avoid a bias in favor
of highly symmetric operators we have used a symmetry breaking
transformation, :math:`T_{asy}`. We have also included some highly asymmetric
functions.

Target function value to reach
------------------------------

The typical target function value for all functions is :math:`f_{opt} -
10^{-8}`. On many functions a value of :math:`f_{opt} - 1` is not very
difficult to reach, but the difficulty versus function value is not uniform
for all functions. These properties are not intrinsic, that is
:math:`f_{opt} - 10^{-8}` is not intrinsically "very good". The value mainly
reflects a scalar multiplier in the function definition.
"""

# Based on BBOB code (copyright 2011 The BBOB Team members)
# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>


from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..utils import check_random_state
from .environment import Environment


def generate_x_opt(random_state, n_dims):
    """Generate random optimum parameters.

    For many functions, :math:`x_opt` is drawn randomly from [-4, 4]. Each
    component of :math:`x_opt` is guaranteed to be less or greater than 0
    and they rounded to 4 decimal digits.

    Parameters
    ----------
    random_state : RandomState
        Random number generator

    n_dims : int
        Number of dimensions

    Returns
    -------
    x_opt : array-like, shape (n_dims,)
        Optimum parameter vector
    """
    x_opt = np.round(random_state.uniform(-4.0, 4.0, (n_dims,)), 4)
    x_opt[x_opt == 0] = -1e-5
    return x_opt


def generate_f_opt(random_state):
    """Generate random optimal function value.

    The value for :math:`f_{opt}` is drawn from a Cauchy distributed random
    variable, with zero median and with roughly 50% of the values between
    -100 and 100. The value is rounded after two decimal places and clipped
    to :math:`[-1000, 1000]`.

    Parameters
    ----------
    random_state : RandomState
        Random number generator
    """
    r = random_state.randn(2)
    return np.clip(np.round(100 * r[0] / r[1], 2), -1000.0, 1000.0)


def T_osc(f):
    """Introduce small, smooth, but clearly visible irregularities.

    .. math::

        T_{osc}(x) = \\text{sign}(x)\\exp(\\hat{x} + 0.049(\\sin(c_1\\hat{x}) +
                                          \\sin(c_2\\hat{x})))\\\\
        \\hat{x} = \\begin{cases}
                       \\log{|x|} & \\text{if } x \\neq 0\\\\
                       0 & \\text{otherwise}
                   \\end{cases},
        c_1 = \\begin{cases}
                  10 & \\text{if } x > 0\\\\
                  5.5 & \\text{otherwise}
              \\end{cases},
        c_2 = \\begin{cases}
                  7.9 & \\text{if } x > 0\\\\
                  3.1 & \\text{otherwise}
              \\end{cases}

    Parameters
    ----------
    f : array-like, shape (n_dims,) or float
        Input

    Returns
    -------
    g : array-like, shape (n_dims,) or float
        Output
    """
    is_scalar = np.isscalar(f)
    f = np.atleast_1d(f)
    g = f.copy()

    idx = f > 0
    g[idx] = np.log(f[idx]) / 0.1
    g[idx] = np.exp(g[idx] + 0.49 * (np.sin(g[idx]) +
                                     np.sin(0.79 * g[idx]))) ** 0.1

    idx = f < 0
    g[idx] = np.log(-f[idx]) / 0.1
    g[idx] = -np.exp(g[idx] + 0.49 * (np.sin(0.55 * g[idx]) +
                                      np.sin(0.31 * g[idx]))) ** 0.1

    if is_scalar:
        g = float(g)
    return g


def T_asy(x, beta):
    """Symmetry breaking transformation.

    .. math::

        T_{asy}^{\\beta} (x_i) =
            \\begin{cases}x_i^{
                1 + \\beta \\frac{i-1}{D-1}\sqrt{x_i}} & \\text{if } x_i > 0\\\\
                x_i & \\text{otherwise}
            \\end{cases}

    Parameters
    ----------
    x : array-like, shape (n_dims,)
        Input

    Returns
    -------
    z : array-like, shape (n_dims,)
        Output
    """
    exponent = np.linspace(0, beta, len(x))
    idx = np.where(x > 0)
    z = x.copy()
    z[idx] **= 1.0 + exponent[idx] * np.sqrt(x[idx])
    return z


def Lambda(n_dims, alpha):
    """Diagonal matrix with exponentially increasing values.

    .. math::

        \\lambda_{ii} = \\alpha^{0.5 \\frac{i-1}{D-1}}

    Parameters
    ----------
    n_dims : int
        Number of dimensions

    Returns
    -------
    Lambda : array-like, shape (n_dims,)
        Diagonal part of the matrix
    """
    return alpha ** np.linspace(0.0, 0.5, n_dims)


def generate_rotation(random_state, n_dims):
    """Returns an orthogonal basis.

    The rotation is used in several ways and in combination with
    non-linear transformations. Search space rotation invariant
    algorithms are not expected to be invariant under this rotation.

    Parameters
    ----------
    random_state : RandomState
        Random number generator

    n_dims : int
        Number of dimensions

    Returns
    -------
    B : array-like, shape (n_dims,)
        Orthogonal basis
    """
    B = random_state.randn(n_dims, n_dims)
    for i in range(n_dims):
        for j in range(i):
            B[i] -= B[i].dot(B[j]) * B[j]
        B[i] /= np.linalg.norm(B[i])
    return B


def f_pen(x):
    """Penalty outside of the boundaries.

    .. math::

        f_{pen} = \\sum_{i=1}^D \\max (0, |x_i| - 5)^2

    Parameters
    ----------
    x : array-like, shape (n_dims,)
        Input

    Returns
    -------
    f_pen : float
        Penalty
    """
    return np.sum(np.maximum(0.0, np.abs(x) - 5.0) ** 2)


def linear_transformation(random_state, n_dims, alpha, return_rotations=False):
    """Generate a linear transformation.

    The linear transformation has the form

    .. math::

        Q \Lambda^\\alpha

    Parameters
    ----------
    random_state : RandomState
        Random number generator

    n_dims : int
        Number of dimensions

    alpha : float
        See :func:`Lambda`

    return_rotations : bool, optional (default: False)
    """
    Q = generate_rotation(random_state, n_dims)
    R = generate_rotation(random_state, n_dims)
    lin_trans = Q.dot(Lambda(n_dims, alpha)[:, np.newaxis] * R)
    if return_rotations:
        return lin_trans, Q, R
    else:
        return lin_trans


def rastrigin(x, penalize_norm=True):
    """Original Rastrigin function.

    .. math::

        f(x) = 10 \\left(D - \\sum_{i=1}^D \\cos ( 2 \\pi x_i) \\right) + ||x||^2

    Parameters
    ----------
    x : array-like, shape (n_dims,)
        Input

    penalize_norm : bool (default: True)
        Add squared norm term to result

    Returns
    -------
    f : float
        Function value
    """
    f = 10.0 * (x.shape[0] - np.sum(np.cos(2.0 * np.pi * x)))
    if penalize_norm:
        f += x.dot(x)
    return f


def rosenbrock(x):
    """Original Rosenbrock function.

    .. math::

        f(x) = \sum_{i=1}^{D-1} 100 (x_i^2 - x_{i+1})^2 + (x_i - 1)^2

    Parameters
    ----------
    x : array-like, shape (n_dims,)
        Input

    Returns
    -------
    f : float
        Function value
    """
    return np.sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)


class ObjectiveFunction(object):
    """Base of objective functions for black-box optimization.

    The default constructor of a objective function will initialize the optimal
    function value :math:`f_{opt}` according to :func:`generate_f_opt` and
    the optimal parameter vector :math:`x_{opt}` according to
    :func:`generate_x_opt`.

    Parameters
    ----------
    random_state : RandomState or int
        Random number generator or seed

    n_dims : int
        Number of dimensions

    Attributes
    ----------
    `x_opt` : array-like, shape (n_dims,)
        Optimum parameter vector

    `f_opt` : float
        Maximum function value
    """
    __metaclass__ = ABCMeta

    def __init__(self, random_state, n_dims):
        self.random_state = check_random_state(random_state)
        self.n_dims = n_dims
        self.x_opt = generate_x_opt(self.random_state, self.n_dims)
        self.f_opt = generate_f_opt(self.random_state)

    @abstractmethod
    def feedback(self, x):
        """Evaluate function at a given point.

        Parameters
        ----------
        x : array-like, shape (n_dims,)
            Parameters

        Returns
        -------
        f : float
            Function value
        """


class Sphere(ObjectiveFunction):
    """Sphere function.

    Properties

    * unimodal
    * highly symmetric, in particular rotationally invariant, scal invariant

    Information gained from this function

    * What is the optimal convergence rate of an algorithm?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("Sphere")
    """
    def __init__(self, random_state, n_dims):
        super(Sphere, self).__init__(random_state, n_dims)

    def feedback(self, x, x_opt_offset=None):
        x_opt = self.x_opt
        if x_opt_offset is not None:
            x_opt = x_opt + x_opt_offset
        return -(np.linalg.norm(x - x_opt) ** 2) + self.f_opt


class Ellipsoidal(ObjectiveFunction):
    """Separable ellipsoid with monotone transformation.

    Properties

    * unimodal
    * conditioning is about :math:`10^6`

    Information gained from this function

    * In comparison to Sphere: Is symmetry exploited?
    * In comparison to EllipsoidalRotated: Is separability exploited?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("Ellipsoidal")
    """
    def __init__(self, random_state, n_dims, condition=1e6):
        super(Ellipsoidal, self).__init__(random_state, n_dims)
        self.condition = condition
        self.scales = self.condition ** np.linspace(0, 1, self.n_dims)

    def feedback(self, x):
        z = T_osc(x - self.x_opt)
        return -np.dot(self.scales, z ** 2) + self.f_opt


class Rastrigin(ObjectiveFunction):
    """Rastrigin with monotone transformation separable condition 10.

    The implementation is based on :func:`rastrigin`.

    Properties

    Highly multimodal function with a comparatively regular structure for the
    placement of the optima. The transformations T_asy and T_osc alleviate the
    symmetry and regularity of the original Rastrigin function

    * roughly :math:`10^6` local optima
    * conditioning is about 10

    Information gained from this function

    * In comparison to Ellipsoidal: What is the effect of multimodality?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("Rastrigin")
    """
    def __init__(self, random_state, n_dims, condition=10.0):
        super(Rastrigin, self).__init__(random_state, n_dims)
        self.condition = condition
        self.Lambda = Lambda(self.n_dims, self.condition)

    def feedback(self, x, x_opt_offset=None):
        x_opt = self.x_opt
        if x_opt_offset is not None:
            x_opt = x_opt + x_opt_offset
        z = self.Lambda * T_asy(T_osc(x - x_opt), 0.2)
        return -rastrigin(z) + self.f_opt


class BuecheRastrigin(ObjectiveFunction):
    """Skew Rastrigin-Bueche, condition 10, skew-"condition" 100.

    The implementation is based on :func:`rastrigin`.

    Properties

    Highly multimodal function with a structured but highly asymmetric
    placement of the optima. Constructed as a deceptive function for
    symmetrically distributed search operators.

    * roughly :math:`10^D` local optima, conditioning is about 10, skew factor
      is about 10 in x-space and 100 in f-space

    Information gained from this function

    * In comparison to Rastrigin: What is the effect of asymmetry?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("BuecheRastrigin")
    """
    def __init__(self, random_state, n_dims, condition=10.0):
        super(BuecheRastrigin, self).__init__(random_state, n_dims)
        self.condition = condition
        self.Lambda = Lambda(self.n_dims, self.condition)

    def feedback(self, x):
        z = T_osc(x - self.x_opt)
        tmp_z = z[::2]
        tmp_z[tmp_z > 0] *= 10.0
        z *= self.Lambda
        return -(rastrigin(z) + 100.0 * f_pen(x)) + self.f_opt


class LinearSlope(ObjectiveFunction):
    """Linear slope.

    Properties

    Purely linear function testing whether the search can go outside the
    initial convex hull of solutions right into the domain boundary.

    * x_opt is on the domain boundary

    Information gained from this function

    * Can the search go outside the initial convex hull of solutions into the
      domain boundary? Can the step size be increased accordingly?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("LinearSlope")
    """
    def __init__(self, random_state, n_dims):
        super(LinearSlope, self).__init__(random_state, n_dims)
        self.x_opt = 5 * np.sign(self.x_opt)
        self.scales = np.sign(self.x_opt) * Lambda(self.n_dims, 100)
        self.offset = 5 * np.sum(np.abs(self.scales))

    def feedback(self, x):
        z = x.copy()
        idx = self.x_opt * x > 25
        z[idx] = np.sign(z[idx])
        return -(self.offset - np.dot(self.scales, z)) + self.f_opt


class AttractiveSector(ObjectiveFunction):
    """Attractive sector function.

    Properties

    Only a "hypercone" (with angular base area) with a volume of roughly
    :math:`0.5^D` yields low function values. The optimum is located at the
    tip of this cone. This function can be deceptive for cumulative step size
    adaptation.

    * highly asymmetric
    * unimodal

    Information gained from this function

    * In comparison to Sphere: What is the effect of a highly asymmetric
      landscape?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("AttractiveSector")
    """
    def __init__(self, random_state, n_dims, alpha=100.0):
        super(AttractiveSector, self).__init__(random_state, n_dims)
        self.alpha = alpha
        self.lin_trans = linear_transformation(self.random_state, self.n_dims,
                                               self.alpha)

    def feedback(self, x):
        z = np.dot(self.lin_trans, x - self.x_opt)
        z[z * self.x_opt > 0] *= self.alpha
        return -T_osc(z.dot(z)) ** 0.9 + self.f_opt


class StepEllipsoidal(ObjectiveFunction):
    """Step ellipsoidal function.

    Properties

    The function consists of many plateaus of different sizes. Apart from a
    small area close to the global optimum, the gradient is zero almost
    everywhere.

    * unimodal, non-separable, conditioning is about 100

    Information gained from this function

    * Does the search get stuck on plateaus?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("StepEllipsoidal")
    """
    def __init__(self, random_state, n_dims, condition=100.0, alpha=10.0):
        super(StepEllipsoidal, self).__init__(random_state, n_dims)
        self.condition = condition
        self.alpha = alpha
        R = generate_rotation(self.random_state, self.n_dims)
        self.lin_trans = Lambda(self.n_dims, self.condition / 10)[:, np.newaxis] * R
        self.Q = generate_rotation(self.random_state, self.n_dims)
        self.scales = self.condition ** np.linspace(0, 1, self.n_dims)

    def feedback(self, x):
        z_hat = np.dot(self.lin_trans, x - self.x_opt)
        z_sim = np.empty(self.n_dims)
        idx = z_hat > 0.5
        z_sim[idx] = np.round(z_hat[idx])
        idx = np.negative(idx)
        z_sim[idx] = np.round(self.alpha * z_hat[idx]) / self.alpha
        z = self.Q.dot(z_sim)
        return -(0.1 * np.maximum(1e-4 * np.abs(z_hat[0]),
                                  np.dot(self.scales, z ** 2)) +
                 100.0 * f_pen(z)) + self.f_opt


class Rosenbrock(ObjectiveFunction):
    """Rosenbrock function.

    The implementation is based on :func:`rosenbrock`.

    Properties

    So-called banana function due to its 2D contour lines as a bent ridge (or
    valley). In the beginning, the prominent first term of the function
    definition attracts to the point z=0. Then, a long bending valley needs to
    be followed to reach the global optimum. The ridge changes its orientation
    D-1 times.

    * partial separable (tri-band structure)
    * in larger dimensions the function has a local optimum with an attraction
      volume of about 25%

    Information gained from this function

    * Can the search follow a long path with D-1 changes in the direction?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("Rosenbrock")
    """
    def __init__(self, random_state, n_dims):
        super(Rosenbrock, self).__init__(random_state, n_dims)
        self.scale = np.maximum(1, np.sqrt(self.n_dims) / 8)

    def feedback(self, x, x_opt_offset=None):
        x_opt = self.x_opt
        if x_opt_offset is not None:
            x_opt = x_opt + x_opt_offset
        z = self.scale * (x - self.x_opt) + 1
        return -rosenbrock(z) + self.f_opt


class RosenbrockRotated(ObjectiveFunction):
    """Rotated Rosenbrock function.

    The implementation is based on :func:`rosenbrock`.

    Properties

    Rotated version of the Rosenbrock function

    Information gained from this function

    * In comparison to Rosenbrock: Can the search follow a long path with D-1
      changes in the direction without exploiting partial separability?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("RosenbrockRotated")
    """
    def __init__(self, random_state, n_dims):
        super(RosenbrockRotated, self).__init__(random_state, n_dims)
        scale = np.maximum(1, np.sqrt(self.n_dims) / 8)
        self.R = scale * generate_rotation(self.random_state, self.n_dims)
        self.x_opt = np.dot(0.5 * np.ones(self.n_dims), self.R.T) / scale ** 2

    def feedback(self, x):
        z = x.dot(self.R) + 0.5
        return -rosenbrock(z) + self.f_opt


class EllipsoidalRotated(ObjectiveFunction):
    """Rotated ellipsoid with monotone transformation.

    Properties

    Globally quadratic ill-conditioned function with smooth local
    irregularities, non-separable counterpart to Ellipsoidal

    * unimodal
    * conditioning is about :math:`10^6`

    Information gained from this function

    * In comparison to Ellipsoidal: What is the effect of rotation
      (non-separability)?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("EllipsoidalRotated")
    """
    def __init__(self, random_state, n_dims, condition=1e6):
        super(EllipsoidalRotated, self).__init__(random_state, n_dims)
        self.condition = condition
        self.scales = self.condition ** np.linspace(0, 1, self.n_dims)
        self.R = generate_rotation(self.random_state, self.n_dims)

    def feedback(self, x):
        z = T_osc(np.dot(self.R, x - self.x_opt))
        f = np.dot(self.scales, z ** 2)
        return -f + self.f_opt


class Discus(ObjectiveFunction):
    """Discus (tablet) with monotone transformation.

    Properties

    Globally quadratic function with irregularities. A single direction
    in search space is a thousand times more sensitive than all others.

    * conditioning is about :math:`10^6`

    Information gained from this function

    * In comparison to EllipsoidalRotated: What is the effect of constraints?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("Discus")
    """
    def __init__(self, random_state, n_dims, condition=1e6):
        super(Discus, self).__init__(random_state, n_dims)
        self.condition = condition
        self.R = generate_rotation(self.random_state, self.n_dims)

    def feedback(self, x):
        z = T_osc((x - self.x_opt).dot(self.R))
        return -((self.condition - 1) * z[0] ** 2 + z.dot(z)) + self.f_opt


class BentCigar(ObjectiveFunction):
    """Bent cigar with asymmetric space distortion.

    Properties

    A ridge defined as :math:`\\sum_{i=2}^D z_i^2 = 0` needs to be followed.
    The ridge is smooth but very narrow. Due to :math:`T_{asy}^{0.5}` the overall
    shape deviates remarkably from being quadratic.

    * conditioning is about :math:`10^6`
    * rotated
    * unimodal

    Information gained from this function

    * Can the search continuously change its search direction?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("BentCigar")
    """
    def __init__(self, random_state, n_dims, condition=1e6, beta=0.5):
        super(BentCigar, self).__init__(random_state, n_dims)
        self.condition = condition
        self.beta = beta
        self.R = generate_rotation(self.random_state, self.n_dims)

    def feedback(self, x):
        z = T_asy((x - self.x_opt).dot(self.R), self.beta).dot(self.R)
        return -(self.condition * z.dot(z) +
                 (1 - self.condition) * z[0] ** 2) + self.f_opt


class SharpRidge(ObjectiveFunction):
    """Sharp ridge.

    Properties

    As for the previous function, a ridge defined as :math:`\\sum_{i=2}^D
    z_i^2 = 0` needs to be followed. The ridge is shape (non-differential)
    and the gradient remains constant, when the ridge is approached from a
    given point. Approaching the ridge is initially effective, but becomes
    ineffective close to the ridge where the ridge needs to be followed
    in :math:`z_1`-direction to its optimum. The necessary change in "search
    behavior" close to the ridge is difficult to diagnose, because the
    gradient towards the ridge does not flatten out.

    Information gained from this function

    * In comparison to BentCigar: What is the effect of non-smooth,
      non-differentiable ridge?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("SharpRidge")
    """
    def __init__(self, random_state, n_dims):
        super(SharpRidge, self).__init__(random_state, n_dims)
        self.lin_trans = linear_transformation(self.random_state, self.n_dims,
                                               10)

    def feedback(self, x):
        z = np.dot(self.lin_trans, x - self.x_opt)
        return -(z[0] ** 2 + 100.0 * np.sqrt(np.sum(z[1:] ** 2))) + self.f_opt


class DifferentPowers(ObjectiveFunction):
    """Abstract Sum of different powers.

    Properties

    Due to the different exponents the sensitivies of the :math:`z_i`-variables
    become more and more different when approching the optimum.

    * unimodal
    * small solution volume
    * rotated

    Information gained from this function

    * In comparison to EllipsoidalRotated: What is the effect of missing
      self-similarity?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("DifferentPowers")
    """
    def __init__(self, random_state, n_dims):
        super(DifferentPowers, self).__init__(random_state, n_dims)
        self.R = generate_rotation(self.random_state, self.n_dims)

    def feedback(self, x):
        z = np.dot(self.R, x - self.x_opt)
        return -(np.sum(np.abs(z) ** np.linspace(2, 6, self.n_dims))) + self.f_opt


class RastriginRotated(ObjectiveFunction):
    """Rastrigin with asymmetric non-linear distortion.

    The implementation is based on :func:`rastrigin`.

    Properties

    Prototypical highly multimodal function which has originally a very
    regular and symmetric structure for the placement of the optima. The
    transformations T_asy and T_osc alleviate the symmetry and regularity
    of the original Rastrigin function.

    * non-separable less regular counterpart of Rastrigin
    * roughly 10^D local optima
    * conditioning is about 10
    * global amplitude large compared to local amplitudes

    Information gained from this function

    * In comparison to Rastrigin: What is the effect of non-separability
      for a highly multimodal function?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("RastriginRotated")
    """
    def __init__(self, random_state, n_dims):
        super(RastriginRotated, self).__init__(random_state, n_dims)
        self.lin_trans, _, self.R = linear_transformation(
            self.random_state, self.n_dims, 10, return_rotations=True)

    def feedback(self, x):
        z = np.dot(self.lin_trans, T_asy(T_osc(np.dot(self.R, x - self.x_opt)),
                                         0.2))
        return -rastrigin(z) + self.f_opt


class Weierstrass(ObjectiveFunction):
    """Weierstrass.

    Properties

    Highly rugged and moderately repetitive landscape, where the global
    optimum is not unique.

    * the term :math:`\sum_k 0.5^k cos(2 \\pi 3^k...)` introduces the
      ruggedness, where lower frequencies have a larger weight :math:`0.5^k`
    * rotated
    * locally irregular
    * non-unique global optimum

    Information gained from this function

    * In comparison to SchaffersF7: Does ruggedness or a repetitive landscape
      deter the search behavior.

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("Weierstrass")
    """
    def __init__(self, random_state, n_dims):
        super(Weierstrass, self).__init__(random_state, n_dims)
        self.lin_trans, self.R, _ = linear_transformation(
            self.random_state, self.n_dims, 0.01, return_rotations=True)
        self.k_range = np.arange(12)
        self.f0 = np.sum(0.5 ** self.k_range * np.cos(np.pi * 3 ** self.k_range))

    def feedback(self, x):
        z = np.dot(self.lin_trans, T_osc(np.dot(self.R, x - self.x_opt)))
        return -(10.0 * (np.sum([0.5 ** self.k_range *
                                 np.cos(2 * np.pi * 3 ** self.k_range *
                                        (zi + 0.5)) / self.n_dims
                                 for zi in z]) - self.f0) ** 3 +
                 10.0 / self.n_dims * f_pen(x)) + self.f_opt


class SchaffersF7(ObjectiveFunction):
    """Schaffers F7 with asymmetric non-linear transformation.

    Properties

    A highly multimodal function where frequency and amplitude of the
    modulation vary.

    * asymmetric
    * rotated
    * conditioning is low

    Information gained from this function

    * In comparison to RastriginRotated: What is the effect of multimodality
      on a less regular function?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("SchaffersF7")
    """
    def __init__(self, random_state, n_dims, condition=10.0):
        super(SchaffersF7, self).__init__(random_state, n_dims)
        self.condition = condition
        self.R = generate_rotation(self.random_state, self.n_dims)
        self.Q = generate_rotation(self.random_state, self.n_dims)
        self.Lambda = Lambda(self.n_dims, self.condition)

    def feedback(self, x):
        z = np.diag(self.Lambda).dot(self.Q).dot(T_asy(self.R.dot(x - self.x_opt), 0.5))
        s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        f = np.mean(np.sqrt(s) + np.sqrt(s) * np.sin(50 * s ** 0.2) ** 2) ** 2
        return -(f + 10.0 * f_pen(x)) + self.f_opt


class SchaffersF7Ill(SchaffersF7):
    """Schaffers F7 with asymmetric non-linear transformation, ill-conditioned.

    Properties

    Moderately ill-conditioned counterpart to SchaffersF7.

    * conditioning is about 1000

    Information gained from this function

    * In comparison to SchaffersF7: What is the effect of ill-conditioning?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("SchaffersF7Ill")
    """
    def __init__(self, random_state, n_dims):
        super(SchaffersF7Ill, self).__init__(random_state, n_dims, 1000.0)


class CompositeGriewankRosenbrockF8F2(ObjectiveFunction):
    """F8F2 sum of Griewank-Rosenbrock 2-D blocks.

    The implementation is based on :func:`rosenbrock`.

    Properties

    Resembling the Rosenbrock function in a highly multimodal way.

    Information gained from this function

    * In comparison to RosenbrockRotated: What is the effect of high
      signal-to-noise ratio?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("CompositeGriewankRosenbrockF8F2")
    """
    def __init__(self, random_state, n_dims):
        super(CompositeGriewankRosenbrockF8F2, self).__init__(
            random_state, n_dims)
        self.scale = np.maximum(1.0, np.sqrt(self.n_dims) / 8.0)
        self.R = generate_rotation(self.random_state, self.n_dims)
        self.x_opt = np.dot(0.5 / self.scale * np.ones(self.n_dims), self.R)

    def feedback(self, x):
        z = self.scale * np.dot(self.R, x) + 0.5
        s = rosenbrock(z)
        f = 10.0 + 10.0 * np.mean(s / 4000.0 - np.cos(s))
        return -f + self.f_opt


class Schwefel(ObjectiveFunction):
    """Schwefel with tridiagonal variable transformation.

    Properties

    The most prominent 2^D minima are located comparatively close to the
    corners of the unpenalized search area.

    * the penalization is essential, as otherwise more and better optima
      occur further away from the search space origin
    * diagonal structure
    * partial separable
    * combinatorial problem
    * two search regimes

    Information gained from this function

    * In comparison to e.g. SchaffersF7: What is the effect of a weak
      global structure?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("Schwefel")
    """
    def __init__(self, random_state, n_dims):
        super(Schwefel, self).__init__(random_state, n_dims)
        self.x_signs = np.sign(self.x_opt)
        self.x_opt = 0.5 * 4.2096874633 * self.x_signs
        self.Lambda = Lambda(self.n_dims, 10.0)

    def feedback(self, x):
        z_hat = 2 * self.x_signs * x
        z_hat[1:] += 0.25 * (z_hat[:-1] - 2 * self.x_opt[:-1])
        z = 100.0 * (self.Lambda * (z_hat - 2 * self.x_opt) + 2 * self.x_opt)
        fpen = 100.0 * f_pen(z / 100.0)
        f = 4.189828872724339 - 0.01 * np.mean(z * np.sin(np.sqrt(np.abs(z))))
        return -(f + fpen) + self.f_opt


class GallaghersGaussian101mePeaks(ObjectiveFunction):
    """Gallagher with 101 Gaussian peaks, condition 30, one global rotation.

    Properties

    * 101 optima with position and height being unrelated and rnadomly chosen
      (different for each instantiation of the function).
    * the conditioning around the global optimum is about 30

    Information gained from this function

    * Is the search effective without any global structure?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("GallaghersGaussian101mePeaks")
    """
    def __init__(self, random_state, n_dims, conditioning=1000.0 ** 0.5, n_peaks=101):
        # TODO how did they generate different rotations for different peaks?
        super(GallaghersGaussian101mePeaks, self).__init__(random_state, n_dims)
        self.alpha_base = conditioning ** 2
        self.n_peaks = n_peaks
        self.w = np.hstack([10.0, 1.1 + np.linspace(0, 8, self.n_peaks - 1)])
        self.R = generate_rotation(self.random_state, self.n_dims)
        self.alpha = np.empty(self.n_peaks)
        self.alpha[0] = 1000.0
        self.alpha[1:] = self.alpha_base ** np.linspace(0, 2, self.n_peaks - 1)
        self.random_state.shuffle(self.alpha)
        self.C = np.empty((self.n_peaks, self.n_dims))
        for i in range(self.n_peaks):
            self.C[i] = (Lambda(self.n_dims, self.alpha[i]) /
                         self.alpha[i] ** 0.25)
        self.Y = np.empty((self.n_peaks, self.n_dims))
        self.Y[0] = self.random_state.uniform(-5, 5, (self.n_dims,))
        self.Y[1:] = self.random_state.uniform(-4, 4, (self.n_peaks - 1,
                                                       self.n_dims))
        self.x_opt = self.Y[0]

    def feedback(self, x):
        D = (x - self.Y).dot(self.R.T)
        p = np.exp(-0.5 / self.n_dims *
                   np.array([D[i].T.dot(np.diag(self.C[i])).dot(D[i])
                             for i in range(self.n_peaks)]))
        return -(T_osc(10.0 - np.max(self.w * p)) ** 2 + f_pen(x)) + self.f_opt


class GallaghersGaussian21hiPeaks(GallaghersGaussian101mePeaks):
    """Gallagher with 21 Gaussian peaks, condition 1000, one global rotation.

    Properties

    * 21 optima with position and height being unrelated and randomly chosen
      (different for each instantiation of the function).
    * the conditioning around the global optimum is about 1000

    Information gained from this function

    * In comparison to GallaghersGaussian101mePeaks: What is the effect of
      higher condition?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("GallaghersGaussian21hiPeaks")
    """
    def __init__(self, random_state, n_dims):
        super(GallaghersGaussian21hiPeaks, self).__init__(
              random_state, n_dims, 1000.0, 21)


class Katsuura(ObjectiveFunction):
    """Katsuura function.

    Properties

    Focus on global search behavior.

    * highly rugged
    * highly repetitive
    * more than 10^D global optima

    Information gained from this function

    * What is the effect of regular local structure on the global search?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("Katsuura")
    """
    def __init__(self, random_state, n_dims):
        super(Katsuura, self).__init__(random_state, n_dims)
        R = generate_rotation(self.random_state, self.n_dims)
        Q = generate_rotation(self.random_state, self.n_dims)
        self.lin_trans = Q.dot(Lambda(self.n_dims, 100)[:, np.newaxis] * R)
        self.scale = 10.0 / self.n_dims ** 2

    def feedback(self, x):
        z = self.lin_trans.dot(x - self.x_opt)
        return -(self.scale * np.prod(
            1 + np.arange(1, self.n_dims + 1) *
            np.sum([np.abs(2 ** j * z - np.round(2 ** j * z)) / 2 ** j
                    for j in range(1, 33)])) ** (10.0 / self.n_dims ** 1.2) -
            self.scale + f_pen(x)) + self.f_opt


class LunacekBiRastrigin(ObjectiveFunction):
    """Lunacek bi-Rastrigin.

    The implementation is based on :func:`rastrigin`.

    Properties

    * highly multimodal function
    * two funnels around mu_0 1+- and -mu_1 1+- being superimposed by the
      cosine.
    * the funnel of the local optimum at -mu_1 has roughly 70% of the search
      space volume within [-5, 5]^D

    Presumably different approaches need to be used for "selecting the funnel"
    and for search the highly multimodal function "within" the funnel. The
    function was constructed to be deceptive for some evolutionary algorithms
    with large population size.

    Information gained from this function

    * Can the search behavior be local on the global scale but global on the
      local scale?

    .. plot::

        from contextual_optimization.objective_functions import plot_function
        plot_function("LunacekBiRastrigin")
    """
    def __init__(self, random_state, n_dims):
        super(LunacekBiRastrigin, self).__init__(random_state, n_dims)
        R = generate_rotation(self.random_state, self.n_dims)
        Q = generate_rotation(self.random_state, self.n_dims)
        self.lin_trans = Q.dot(Lambda(self.n_dims, 100)[:, np.newaxis] * R)
        self.mu_0 = 2.5
        self.d = 1.0
        self.s = 1 - 1.0 / (2 * np.sqrt(self.n_dims + 20) - 8.2)
        self.mu_1 = -np.sqrt((self.mu_0 ** 2 - self.d) / self.s)
        self.x_opt = 0.5 * np.sign(self.x_opt) * self.mu_0

    def feedback(self, x):
        x_hat = 2 * np.sign(self.x_opt) * x
        z = self.lin_trans.dot(x_hat - self.mu_0)
        d_mu_0 = x_hat - self.mu_0
        d_mu_1 = x_hat - self.mu_1
        return -(np.minimum(d_mu_0.dot(d_mu_0), self.d * self.n_dims +
                            self.s * d_mu_1.dot(d_mu_1)) +
                 rastrigin(z, penalize_norm=False) + 1e4 * f_pen(x)) + self.f_opt


SEPARABLE_FUNCTIONS = [
    "Sphere",
    "Ellipsoidal",
    "Rastrigin",
    "BuecheRastrigin",
    "LinearSlope",
    ]


MODERATE_FUNCTIONS = [
    "AttractiveSector",
    "StepEllipsoidal",
    "Rosenbrock",
    "RosenbrockRotated",
    ]


ILL_CONDITIONED_FUNCTIONS = [
    "EllipsoidalRotated",
    "Discus",
    "BentCigar",
    "SharpRidge",
    "DifferentPowers",
    ]


MULTIMODAL_STRUCTURED_FUNCTIONS = [
    "RastriginRotated",
    "Weierstrass",
    "SchaffersF7",
    "SchaffersF7Ill",
    "CompositeGriewankRosenbrockF8F2",
    ]


MULTIMODAL_WEAKLY_STRUCTURED_FUNCTIONS = [
    "Schwefel",
    "GallaghersGaussian101mePeaks",
    "GallaghersGaussian21hiPeaks",
    "Katsuura",
    "LunacekBiRastrigin",
    ]


NON_SMOOTH_FUNCTIONS = [
    "StepEllipsoidal",
    "Weierstrass",
    "Katsuura",
    ]


UNIMODAL_FUNCTIONS = [
    "Sphere",
    "Ellipsoidal",
    "LinearSlope",
    "AttractiveSector",
    "StepEllipsoidal",
    "Rosenbrock",
    "RosenbrockRotated",
    "EllipsoidalRotated",
    "Discus",
    "BentCigar",
    "SharpRidge",
    "DifferentPowers",
    ]


FUNCTIONS = {
    "Sphere": Sphere,
    "Ellipsoidal": Ellipsoidal,
    "Rastrigin": Rastrigin,
    "BuecheRastrigin": BuecheRastrigin,
    "LinearSlope": LinearSlope,
    "AttractiveSector": AttractiveSector,
    "StepEllipsoidal": StepEllipsoidal,
    "Rosenbrock": Rosenbrock,
    "RosenbrockRotated": RosenbrockRotated,
    "EllipsoidalRotated": EllipsoidalRotated,
    "Discus": Discus,
    "BentCigar": BentCigar,
    "SharpRidge": SharpRidge,
    "DifferentPowers": DifferentPowers,
    "RastriginRotated": RastriginRotated,
    "Weierstrass": Weierstrass,
    "SchaffersF7": SchaffersF7,
    "SchaffersF7Ill": SchaffersF7Ill,
    "CompositeGriewankRosenbrockF8F2": CompositeGriewankRosenbrockF8F2,
    "Schwefel": Schwefel,
    "GallaghersGaussian101mePeaks": GallaghersGaussian101mePeaks,
    "GallaghersGaussian21hiPeaks": GallaghersGaussian21hiPeaks,
    "Katsuura": Katsuura,
    "LunacekBiRastrigin": LunacekBiRastrigin,
    }


def plot_function(name, random_state=None, fig=None, contour=False):
    """Plot function values of area [-5, 5] x [-5, 5].

    This will show two plots, one with linear scale and one with logarithmic
    scale.

    Parameters
    ----------
    name : string
        Function name

    random_state : RandomState or int
        Random number generator or seed

    fig : Figure, optional (default: None)
        Matplotlib figure

    contour : bool, optional (default: False)
        Plot function as contour plot (in 3D otherwise)

    Returns
    -------
    ax_lin : Axis
        Subplot with linear scale

    ax_log : Axis
        Subplot with logarithmic scale

    f : ObjectiveFunction
        Objective function
    """
    if name not in FUNCTIONS:
        raise ValueError("Could not find function %r, available functions are "
                         "%r" % (name, FUNCTIONS.keys()))

    f = FUNCTIONS[name](random_state, 2)
    X = np.arange(-5, 5, 0.2)
    Y = np.arange(-5, 5, 0.2)
    X, Y = np.meshgrid(X, Y)
    F = np.ndarray(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            F[i, j] = f.feedback(np.array([X[i, j], Y[i, j]]))

    if fig is None:
        fig = plt.figure(figsize=(12, 5))

    if contour:
        ax_lin = fig.add_subplot(121)
        ax_lin.set_title(name)
        _plot_function_lin_contour(f, ax_lin, X, Y, F)

        ax_log = fig.add_subplot(122)
        _plot_function_log_contour(f, ax_log, X, Y, F)
    else:
        ax_lin = fig.add_subplot(121, projection="3d")
        ax_lin.set_title(name)
        _plot_function_lin_3d(f, ax_lin, X, Y, F)

        ax_log = fig.add_subplot(122, projection="3d")
        _plot_function_log_3d(f, ax_log, X, Y, F)

    if show_now:
        plt.show()

    return ax_lin, ax_log, f


def _plot_function_lin_contour(f, ax, X, Y, F):
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.contourf(X, Y, F, rstride=1, cstride=1, cmap=plt.cm.jet)


def _plot_function_log_contour(f, ax, X, Y, F):
    F = -np.log(-(F - f.f_opt) + 0.1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.contourf(X, Y, F, rstride=1, cstride=1, cmap=plt.cm.jet)


def _plot_function_lin_3d(f, ax, X, Y, F):
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")
    ax.plot_surface(X, Y, F, rstride=1, cstride=1, cmap=plt.cm.jet, lw=0)


def _plot_function_log_3d(f, ax, X, Y, F):
    F = -np.log(-(F - f.f_opt) + 0.1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$-\log(-(f - f_{opt}) + 0.1)$")
    ax.plot_surface(X, Y, F, rstride=1, cstride=1, cmap=plt.cm.jet, lw=0)


class ObjectiveFunction(Environment):
    """Artificial benchmark function.

    Parameters
    ----------
    name : string, optional (default: 'Sphere')
        Name of the objective function

    n_params : int, optional (default: 2)
        Number of dimensions

    random_state : RandomState or int, optional (default: None)
        Random number generator or seed
    """
    def __init__(self, name="Sphere", n_params=2, random_state=None):
        self.name = name
        self.n_params = n_params
        self.random_state = random_state

    def init(self):
        if self.n_params <= 0:
            raise ValueError("Number of parameters (%d) must be > 0"
                             % self.n_params)
        if not self.name in FUNCTIONS:
            raise ValueError("Unknown function '%s' requested, select one of "
                             "%s instead" % (self.name, FUNCTIONS.keys()))

        self.objective = FUNCTIONS[self.name](self.random_state, self.n_params)

        self.params = np.empty(self.n_params)
        self.f = np.nan

    def reset(self):
        self.done = False

    def get_num_inputs(self):
        return self.n_params

    def get_num_outputs(self):
        return 0

    def get_outputs(self, _):
        self.done = True

    def set_inputs(self, values):
        self.params[:] = values[:]

    def step_action(self):
        self.f = self.objective.feedback(self.params)

    def is_evaluation_done(self):
        return self.done

    def get_feedback(self):
        return np.array([self.f])

    def is_behavior_learning_done(self):
        return False

    def get_maximum_feedback(self):
        return self.objective.f_opt

    def plot(self, fig):
        plot_function(self.name, self.random_state, fig)
