"""
====================
Catapult Environment
====================

Illustration of the catapult environment for example surface.
"""
print(__doc__)

import matplotlib.pyplot as plt
from bolero.environment import Catapult

catapult = Catapult([(0, 0), (2.0, -0.5), (3.0, 0.5), (4, 0.25), (5, 2.5),
                     (7, 0), (10, 0.5), (15, 0)])
catapult.init()
plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
catapult.plot(ax)
plt.xlabel("Target")
plt.ylim(-1, 6)
plt.title("Trajectories for different angles under maximal velocity (v=10)")
plt.show()
