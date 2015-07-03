"""
================================
Influence of DMP meta-parameters
================================

Demonstrate the influence of DMP meta-parameters.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from bolero.datasets import make_minimum_jerk
from dmp import DMP, imitate_dmp


def dmp_to_trajectory(dmp, x0, g, gd, execution_time):
    x, xd, xdd = np.copy(x0), np.zeros_like(x0), np.zeros_like(x0)
    X, Xd, Xdd = [], [], []

    dmp.set_metaparameters(["execution_time"], [execution_time])
    dmp.reset()
    while dmp.can_step():
        dmp.execute_step(x, xd, xdd, x0, g, gd)
        X.append(x.copy())
        Xd.append(xd.copy())
        Xdd.append(xdd.copy())

    return np.array(X), np.array(Xd), np.array(Xdd)


x0 = np.zeros(2)
g = np.ones(2)
dt = 0.001
dmp = DMP(execution_time=1.0, dt=dt, n_features=10)
demo_X, demo_Xd, demo_Xdd = make_minimum_jerk(x0, g, 1.0, 0.001)
imitate_dmp(dmp, demo_X, demo_Xd, demo_Xdd, set_weights=True)

plt.figure()
plt.subplots_adjust(wspace=0.3, hspace=0.6)

for gx in np.linspace(0.5, 1.5, 6):
    g_new = np.array([gx, 1.0])
    X, Xd, Xdd = dmp_to_trajectory(dmp, x0, g_new, None, 1.0)

    ax = plt.subplot(321)
    ax.set_title("Goal adaption")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.plot(X[:, 0], X[:, 1])

    ax = plt.subplot(322)
    ax.set_title("Velocity profile")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$v$")
    ax.plot(np.linspace(0, 1, X.shape[0]),
            np.sqrt(Xd[:, 0] ** 2 + Xd[:, 1] ** 2))

for gxd in np.linspace(-1.5, 1.5, 6):
    gd = np.array([gxd, 0.0])
    X, Xd, Xdd = dmp_to_trajectory(dmp, x0, g, gd, 1.0)

    ax = plt.subplot(323)
    ax.set_title("Goal velocity adaption")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.plot(X[:, 0], X[:, 1])

    ax = plt.subplot(324)
    ax.set_title("Velocity profile")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$v$")
    ax.plot(np.linspace(0, 1, X.shape[0]),
            np.sqrt(Xd[:, 0] ** 2 + Xd[:, 1] ** 2))

gd = np.array([0.5, 0.0])
for t in np.linspace(0.5, 2.5, 6):
    X, Xd, Xdd = dmp_to_trajectory(dmp, x0, g, gd, t)

    ax = plt.subplot(325)
    ax.set_title("Execution time adaption")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.plot(X[:, 0], X[:, 1])

    ax = plt.subplot(326)
    ax.set_title("Velocity profile")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$v$")
    ax.plot(np.linspace(0, t, X.shape[0]),
            np.sqrt(Xd[:, 0] ** 2 + Xd[:, 1] ** 2))

plt.show()
