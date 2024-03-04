#cython.wraparound=False
#cython.boundscheck=False
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
cimport cython


cdef double MACHINE_EPSILON = np.finfo(float).eps ** 2


def optimize(np.ndarray[np.float_t, ndim=1] Ci,
             np.ndarray[np.float_t, ndim=2] K, double epsilon, int n_iter,
             object random_state):
    cdef int n_samples = K.shape[0]
    cdef np.ndarray[np.float_t, ndim=1] alpha = Ci * (
        0.95 + 0.05 * random_state.rand(n_samples - 1))
    coptimize(n_samples, Ci, epsilon, n_iter, K, alpha)
    return alpha


@cython.cdivision(True)
cdef void coptimize(
        int n_samples, np.ndarray[np.float_t, ndim=1] Ci, double epsilon,
        int n_iter, np.ndarray[np.float_t, ndim=2] K,
        np.ndarray[np.float_t, ndim=1] alpha):
    cdef int n_alpha = n_samples - 1
    cdef np.ndarray[np.float_t, ndim=2] dKij = np.empty((n_alpha, n_alpha))
    cdef np.ndarray[np.float_t, ndim=1] sum_alpha_dKij = np.empty(n_alpha)
    cdef np.ndarray[np.float_t, ndim=2] div_dKij = np.empty((n_alpha, n_alpha))

    cdef int i, j
    for i in range(n_alpha):
        dKij[i] = K[i, :-1] - K[i, 1:] - K[i + 1, :-1] + K[i + 1, 1:]

    cdef double sum_alpha
    for i in range(n_alpha):
        sum_alpha = 0
        for j in range(n_alpha):
            sum_alpha += alpha[j] * dKij[i, j]
        sum_alpha_dKij[i] = -(sum_alpha - epsilon) / max(dKij[i, i], MACHINE_EPSILON)

    for i in range(n_alpha):
        for j in range(n_alpha):
            div_dKij[i, j] = dKij[i, j] / max(dKij[j, j], MACHINE_EPSILON)

    cdef int it
    cdef double new_alpha, delta_alpha, dL
    for it in range(n_iter):
        i = it % n_alpha
        new_alpha = alpha[i] + sum_alpha_dKij[i]
        if new_alpha > Ci[i]:
            new_alpha = Ci[i]
        if new_alpha < 0:
            new_alpha = 0
        delta_alpha = new_alpha - alpha[i]

        dL = delta_alpha * dKij[i, i] * (sum_alpha_dKij[i] - 0.5 * delta_alpha)

        if dL > 0:
            for j in range(n_alpha):
                sum_alpha_dKij[j] -= delta_alpha * div_dKij[i, j]
            alpha[i] = new_alpha
