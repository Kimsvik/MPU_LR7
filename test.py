import numpy as np

T = 50
mgn = 2

def f(t):
    return 2 * np.sin(2 * np.pi * t / T) + mgn / 5 * np.sin(20 * np.pi * t / T) + 2 / 20 * np.sin(50 * np.pi * t / T)


x = np.arange(0, 512)

for i in range(512):
    print(f(x[i]))