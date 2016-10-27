import numpy as np
import dmp


n_features = 9
T = np.linspace(0, 1, 101)
Y = np.hstack((T[:, np.newaxis], np.cos(2 * np.pi * T)[:, np.newaxis]))
alpha = 25.0


def imitate(T, Y, n_features, alpha):
    widths = np.empty(n_features)
    centers = np.empty(n_features)
    weights = np.empty(n_features * 2)
    dmp.initialize_rbf(widths, centers, 1.0, 0.8, 25.0 / 3.0)
    dmp.imitate(T, Y.ravel(), weights, widths, centers, 1e-10, alpha, alpha / 4.0, alpha / 3.0, False)
    return widths, centers, weights


def execute(T, Y, weights, widths, centers, alpha):
    last_t = T[0]

    last_y = Y[0].copy()
    last_yd = np.zeros(2)
    last_ydd = np.zeros(2)

    y = Y[0].copy()
    yd = np.zeros(2)
    ydd = np.zeros(2)

    g = Y[-1].copy()
    gd = np.zeros(2)
    gdd = np.zeros(2)

    y0 = Y[0].copy()
    y0d = np.zeros(2)
    y0dd = np.zeros(2)

    for t in np.linspace(T[0], T[-1], T.shape[0]):
        dmp.dmp_step(
            last_t, t,
            last_y, last_yd, last_ydd,
            y, yd, ydd,
            g, gd, gdd,
            y0, y0d, y0dd,
            T[-1], T[0],
            weights,
            widths,
            centers,
            False,
            alpha, alpha / 4.0, alpha / 3.0,
            0.001
        )
        last_t = t
        last_y[:] = y
        last_yd[:] = yd
        last_ydd[:] = ydd


# %run test/benchmark.py
# widths, centers, weights = imitate(T, Y, 50, alpha)
# %timeit execute(T, Y, weights, widths, centers, alpha)
# %prun execute(T, Y, weights, widths, centers, alpha)
