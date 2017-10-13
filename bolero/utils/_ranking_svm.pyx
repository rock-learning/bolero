import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
cimport cython


cdef double MACHINE_EPSILON = np.finfo(np.float).eps ** 2


def optimize(np.ndarray[np.float_t, ndim=1] Ci,
             np.ndarray[np.float_t, ndim=2] K, double epsilon, int n_iter,
             object random_state):
    cdef int n_train = K.shape[0]
    cdef double* p_Ci = &Ci[0]
    cdef double* p_K = &K[0, 0]
    # Initialize alphas randomly
    cdef np.ndarray[np.float_t, ndim=1] alpha = Ci * (0.95 + 0.05 * random_state.rand(n_train - 1))
    cdef double* p_alpha = &alpha[0]
    coptimize(n_train, p_Ci, epsilon, n_iter, p_K, p_alpha)
    return alpha


@cython.cdivision(True)
cdef void coptimize(int n_train, double* p_Ci, double epsilon, int n_iter,
                    double* p_K, double* p_alpha):
    cdef int n_alpha = n_train - 1
    cdef double old_alpha, new_alpha, delta_alpha, sumAlpha, dL
    cdef int i, i1, j
    cdef double* p_dKij = <double*> malloc(n_alpha * n_alpha * sizeof(double))
    cdef double* p_sumAlphaDKij = <double*> malloc(n_alpha * sizeof(double))
    cdef double* p_div_dKij = <double*> malloc(n_alpha * n_alpha * sizeof(double))

    for i in range(n_alpha):
        for j in range(n_alpha):
            p_dKij[i * n_alpha + j] = (p_K[i * n_train + j] -
                                       p_K[i * n_train + (j + 1)] -
                                       p_K[(i + 1) * n_train + j] +
                                       p_K[(i + 1) * n_train + (j + 1)])

    for i in range(n_alpha):
        sumAlpha = 0
        for j in range(n_alpha):
            sumAlpha += p_alpha[j] * p_dKij[i * n_alpha + j]
        p_sumAlphaDKij[i] = (-(sumAlpha - epsilon) /
                             max(p_dKij[i * n_alpha + i], MACHINE_EPSILON))

    for i in range(n_alpha):
        for j in range(n_alpha):
            p_div_dKij[i * n_alpha + j] = (p_dKij[i * n_alpha + j] /
                                           max(p_dKij[j * n_alpha + j],
                                               MACHINE_EPSILON))

    for i in range(n_iter):
        i1 = i % n_alpha
        old_alpha = p_alpha[i1]
        new_alpha = old_alpha + p_sumAlphaDKij[i1]
        if new_alpha > p_Ci[i1]:
            new_alpha = p_Ci[i1]
        if new_alpha < 0:
            new_alpha = 0
        delta_alpha = new_alpha - old_alpha

        dL = delta_alpha * p_dKij[i1 * n_alpha + i1] * (p_sumAlphaDKij[i1] -
                                                        0.5 * delta_alpha)

        if dL > 0:
            for j in range(n_alpha):
                p_sumAlphaDKij[j] -= delta_alpha * p_div_dKij[i1 * n_alpha + j]

            p_alpha[i1] = new_alpha

    free(p_dKij)
    free(p_sumAlphaDKij)
    free(p_div_dKij)
