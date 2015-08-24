__author__ = 'eidonfiloi'

import time
import numpy as np
import matplotlib.pyplot as plt

plt.axis([-1, 1, -1, 1])
plt.ion()
plt.show()

for i in range(1000):

    data = np.random.rand(500, 500) * 2.0 - 1.0
    plt.scatter(data[:, 0], data[:, 1])
    plt.draw()
    time.sleep(0.05)
    plt.cla()
