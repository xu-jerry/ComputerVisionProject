import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = 1.0 / (1 + np.e **(-x))
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")

plt.show()
