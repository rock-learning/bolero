import numpy as np
import dmp
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises_regexp, assert_less, assert_almost_equal


def test_compute_gradient():
    n_steps = 101
    T = np.linspace(0, 1, n_steps)
    P = np.empty((n_steps, 2))
    P[:, 0] = np.cos(2 + np.pi * T)
    P[:, 1] = np.sin(2 + np.pi * T)

    P = P.ravel()
    V = np.empty_like(P)
    dmp.compute_gradient(P, V, T, True)
    P = P.reshape(n_steps, 2)
    V = V.reshape(n_steps, 2)

    P_derivative = np.vstack((np.zeros(2), np.diff(P, axis=0) / np.diff(T)[:, np.newaxis]))
    assert_array_almost_equal(V, P_derivative)

    V_integral = np.vstack(((P[0],), np.cumsum(V[1:] * np.diff(T)[:, np.newaxis], axis=0) + P[0]))
    assert_array_almost_equal(P, V_integral)


def test_compute_quaternion_gradient():
    n_steps = 101
    T = np.linspace(0, 1, n_steps)
    R = np.empty((n_steps, 4))
    for i in range(n_steps):
        angle = T[i] * np.pi
        R[i] = np.array([np.cos(angle / 2.0),
                         np.sqrt(0.5) * np.sin(angle / 2.0),
                         np.sqrt(0.5) * np.sin(angle / 2.0),
                         0.0])
    R = R.ravel()
    V = np.empty((n_steps, 3)).ravel()
    dmp.compute_quaternion_gradient(R, V, T, True)
    R = R.reshape(n_steps, 4)
    V = V.reshape(n_steps, 3)

    V_integral = np.empty((n_steps, 4))
    V_integral[0] = R[0]
    for i in range(1, n_steps):
        d = 0.5 * (T[i] - T[i - 1]) * V[i]
        n = np.linalg.norm(d)
        vec = np.sin(n) * d / n
        q = np.hstack(((np.cos(n),), vec))
        r = np.empty(4)
        V_integral[i, 0] = q[0] * V_integral[i - 1, 0] - np.dot(q[1:], V_integral[i - 1, 1:])
        V_integral[i, 1:] = q[0] * V_integral[i - 1, 1:] + V_integral[i - 1, 0] * q[1:] + np.cross(q[1:], V_integral[i - 1, 1:])

    assert_array_almost_equal(R, V_integral)


def test_calculate_alpha_invalid_final_phase():
    assert_raises_regexp(
        ValueError, "phase must be > 0", dmp.calculate_alpha, 0.0, 0.446, 0.0)


def test_calculate_alpha():
    alpha = dmp.calculate_alpha(0.01, 0.446, 0.0)
    assert_almost_equal(4.5814764780043, alpha)


def test_initialize_rbf_too_few_weights():
    n_features = 1
    widths = np.empty(n_features)
    centers = np.empty(n_features)
    assert_raises_regexp(
        ValueError, "number of weights .* must be > 1",
        dmp.initialize_rbf, widths, centers, 1.0, 0.0, 0.8, 25.0 / 3.0)


def test_initialize_rbf_invalid_times():
    widths = np.empty(10)
    centers = np.empty(10)
    assert_raises_regexp(
        ValueError, "Goal must be chronologically after start",
        dmp.initialize_rbf, widths, centers, 0.0, 1.0, 0.8, 25.0 / 3.0)


def test_initialize_rbf_backward_compatibility():
    widths = np.empty(10)
    centers = np.empty(10)
    dmp.initialize_rbf(widths, centers, 0.446, 0.0, 0.8, 4.5814764780043)
    assert_array_almost_equal(
        widths,
        np.array([
            1.39105769528669, 3.87070066903258, 10.7704545397461,
            29.969429545615, 83.3917179609352, 232.042408878407,
            645.671786535403, 1796.62010036397, 4999.20215246208,
            4999.20215246208])
        )
    assert_array_almost_equal(
        centers,
        np.array([
            1, 0.59948425031894, 0.359381366380461, 0.215443469003187,
            0.129154966501487, 0.0774263682681117, 0.0464158883361271,
            0.0278255940220708, 0.0166810053720003, 0.00999999999999978])
        )


def test_imitate_ill_conditioning():
    n_features = 101
    widths = np.empty(n_features)
    centers = np.empty(n_features)
    dmp.initialize_rbf(widths, centers, 1.0, 0.0, 0.8, 25.0 / 3.0)
    T = np.linspace(0, 1, 101)
    Y = np.hstack((T[:, np.newaxis], np.cos(2 * np.pi * T)[:, np.newaxis]))
    weights = np.empty(n_features * 2)
    alpha = 25.0
    assert_raises_regexp(
        ValueError, "must be >= 0",
        dmp.imitate, T, Y.ravel(), weights, widths, centers, -1.0, alpha,
        alpha / 4.0, alpha / 3.0, False)
    assert_raises_regexp(
        ValueError, "instable",
        dmp.imitate, T, Y.ravel(), weights, widths, centers, 0.0, alpha,
        alpha / 4.0, alpha / 3.0, False)


def test_step():
    last_y = np.array([0.0])
    last_yd = np.array([0.0])
    last_ydd = np.array([0.0])

    y = np.empty([1])
    yd = np.empty([1])
    ydd = np.empty([1])

    g = np.array([1.0])
    gd = np.array([0.0])
    gdd = np.array([0.0])

    n_weights = 10
    weights = np.zeros(n_weights)
    execution_time = 1.0
    alpha = 25.0

    widths = np.empty(n_weights)
    centers = np.empty(n_weights)
    dmp.initialize_rbf(widths, centers, execution_time, 0.0, 0.8, alpha / 3.0)

    last_t = 0.0
    # Execute DMP longer than expected duration
    for t in np.linspace(0.0, 1.5 * execution_time, 151):
        dmp.dmp_step(
            last_t, t,
            last_y, last_yd, last_ydd,
            y, yd, ydd,
            g, gd, gdd,
            np.array([0.0]), np.array([0.0]), np.array([0.0]),
            execution_time, 0.0,
            weights,
            widths,
            centers,
            alpha, alpha / 4.0, alpha / 3.0,
            0.001
        )
        last_t = t
        last_y[:] = y
        last_yd[:] = yd
        last_ydd[:] = ydd

    assert_array_almost_equal(y, g, decimal=6)
    assert_array_almost_equal(yd, gd, decimal=5)
    assert_array_almost_equal(ydd, gdd, decimal=4)


def test_imitate():
    T = np.linspace(0, 2, 101)
    n_features = 9
    widths = np.empty(n_features)
    centers = np.empty(n_features)
    dmp.initialize_rbf(widths, centers, T[-1], T[0], 0.8, 25.0 / 3.0)
    Y = np.hstack((T[:, np.newaxis], np.cos(np.pi * T)[:, np.newaxis]))
    weights = np.empty(n_features * 2)
    alpha = 25.0
    dmp.imitate(T, Y.ravel(), weights, widths, centers, 1e-10, alpha, alpha / 4.0, alpha / 3.0, False)

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

    Y_replay = []
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
            alpha, alpha / 4.0, alpha / 3.0,
            0.001
        )
        last_t = t
        last_y[:] = y
        last_yd[:] = yd
        last_ydd[:] = ydd
        Y_replay.append(y.copy())

    Y_replay = np.asarray(Y_replay)

    distances = np.array([np.linalg.norm(y - y_replay)
                          for y, y_replay in zip(Y, Y_replay)])
    assert_less(distances.max(), 0.032)
    assert_less(distances.min(), 1e-10)
    assert_less(sorted(distances)[len(distances) // 2], 0.02)
    assert_less(np.mean(distances), 0.02)


def test_quaternion_step():
    last_r = np.array([0.0, 1.0, 0.0, 0.0])
    last_rd = np.array([0.0, 0.0, 0.0])
    last_rdd = np.array([0.0, 0.0, 0.0])

    r = np.array([0.0, 1.0, 0.0, 0.0])
    rd = np.array([0.0, 0.0, 0.0])
    rdd = np.array([0.0, 0.0, 0.0])

    g = np.array([0.0, 0.0, 7.07106781e-01, 7.07106781e-01])
    gd = np.array([0.0, 0.0, 0.0])
    gdd = np.array([0.0, 0.0, 0.0])

    r0 = np.array([0.0, 1.0, 0.0, 0.0])
    r0d = np.array([0.0, 0.0, 0.0])
    r0dd = np.array([0.0, 0.0, 0.0])

    n_features = 10
    weights = np.zeros(3 * n_features)
    execution_time = 1.0
    alpha = 25.0

    widths = np.empty(n_features)
    centers = np.empty(n_features)
    dmp.initialize_rbf(widths, centers, execution_time, 0.0, 0.8, alpha / 3.0)

    T = np.linspace(0.0, 1.0 * execution_time, 1001)

    last_t = 0.0
    R_replay = []
    for t in T:
        dmp.quaternion_dmp_step(
            last_t, t,
            last_r, last_rd, last_rdd,
            r, rd, rdd,
            g, gd, gdd,
            r0, r0d, r0dd,
            execution_time, 0.0,
            weights,
            widths,
            centers,
            alpha, alpha / 4.0, alpha / 3.0,
            0.001
        )
        last_t = t
        last_r[:] = r
        last_rd[:] = rd
        last_rdd[:] = rdd
        R_replay.append(r.copy())

    R_replay = np.asarray(R_replay)

    assert_array_almost_equal(r, g, decimal=4)
    assert_array_almost_equal(rd, gd, decimal=3)
    assert_array_almost_equal(rdd, gdd, decimal=2)


def test_quaternion_imitate():
    T = np.linspace(0, 2, 101)
    n_features = 20
    alpha = 25.0
    widths = np.empty(n_features)
    centers = np.empty(n_features)
    dmp.initialize_rbf(widths, centers, T[-1], T[0], 0.8, alpha / 3.0)
    R = np.empty((T.shape[0], 4))
    for i in range(T.shape[0]):
        angle = T[i] * np.pi
        R[i] = np.array([np.cos(angle / 2.0),
                         np.sqrt(0.5) * np.sin(angle / 2.0),
                         np.sqrt(0.5) * np.sin(angle / 2.0),
                         0.0])
    weights = np.empty(3 * n_features)
    dmp.quaternion_imitate(
        T, R.ravel(), weights, widths, centers, 1e-10,
        alpha, alpha / 4.0, alpha / 3.0, False)

    last_t = T[0]

    last_r = R[0].copy()
    last_rd = np.zeros(3)
    last_rdd = np.zeros(3)

    r = R[0].copy()
    rd = np.zeros(3)
    rdd = np.zeros(3)

    g = R[-1].copy()
    gd = np.zeros(3)
    gdd = np.zeros(3)

    r0 = R[0].copy()
    r0d = np.zeros(3)
    r0dd = np.zeros(3)

    R_replay = []
    for t in T:
        dmp.quaternion_dmp_step(
            last_t, t,
            last_r, last_rd, last_rdd,
            r, rd, rdd,
            g, gd, gdd,
            r0, r0d, r0dd,
            T[-1], T[0],
            weights,
            widths,
            centers,
            alpha, alpha / 4.0, alpha / 3.0,
            0.001
        )
        last_t = t
        last_r[:] = r
        last_rd[:] = rd
        last_rdd[:] = rdd
        R_replay.append(r.copy())

    R_replay = np.asarray(R_replay)

    distances = np.array([np.linalg.norm(r - r_replay)
                          for r, r_replay in zip(R, R_replay)])
    assert_less(distances.max(), 0.043)
    assert_less(distances.min(), 1e-10)
    assert_less(sorted(distances)[len(distances) // 2], 0.02)
    assert_less(np.mean(distances), 0.02)
