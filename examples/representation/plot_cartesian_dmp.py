"""
===================
Cartesian Space DMP
===================

In a Cartesian Space DMP, the rotation are represented by quaternions.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bolero.representation import CartesianDMPBehavior


def matrix_from_quaternion(q):
    w, x, y, z = q
    x2 = 2.0 * x * x
    y2 = 2.0 * y * y
    z2 = 2.0 * z * z
    xy = 2.0 * x * y
    xz = 2.0 * x * z
    yz = 2.0 * y * z
    xw = 2.0 * x * w
    yw = 2.0 * y * w
    zw = 2.0 * z * w

    R = np.array([[1.0 - y2 - z2,       xy - zw,       xz + yw],
                  [      xy + zw, 1.0 - x2 - z2,       yz - xw],
                  [      xz - yw,       yz + xw, 1.0 - x2 - y2]])

    return R


def plot_pose(ax, x, s=1.0, **kwargs):
    p = x[:3]
    R = matrix_from_quaternion(x[3:])
    for d, c in enumerate(["r", "g", "b"]):
        ax.plot([p[0], p[0] + s * R[0, d]],
                [p[1], p[1] + s * R[1, d]],
                [p[2], p[2] + s * R[2, d]], color=c, **kwargs)

    return ax


dmp = CartesianDMPBehavior(dt=0.001)
dmp.init(7, 7)

random_state = np.random.RandomState(0)
q0 = random_state.randn(4)
q0 /= np.linalg.norm(q0)
qg = random_state.randn(4)
qg /= np.linalg.norm(qg)

dmp.set_meta_parameters(["x0", "g", "q0", "qg"],
                        [np.zeros(3), np.ones(3), q0, qg])
X = dmp.trajectory()

ax = plt.subplot(111, projection="3d", aspect="equal")
plt.setp(ax, xlabel="X", ylabel="Y", zlabel="Z")
for x in X[50:-50:50]:
    plot_pose(ax, x, s=0.3, alpha=0.3)
plot_pose(ax, X[0], s=0.5)
plot_pose(ax, X[-1], s=0.5)
plt.show()
