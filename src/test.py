import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 2*np.pi, 0.01)
y = np.sin(x)


def foo(param=None):
    param = 25 if param is None else param
    return param


plt.plot(x, y)

print(foo(10))

plt.show()
