import random

import numpy as np
from matplotlib import pyplot as plt


def f_sin(t):
    return 1 * np.sin(2 * np.pi * t / 50)

def f_log(t):
    return np.log2(t)

def f_exp(t):
    return np.cos(2 * np.pi * t / 50)

def f_tg(t):
    return np.arctan(t)

x = np.arange(0, 512)

file = open("C:/Users/Сергей/PycharmProjects/MPU_LR7/dataset3.txt", "w")


for i in range(128):
    file.write(str(f_sin(i) + random.normalvariate(0, 0.2)) + '\n')
for i in range(129, 256):
    file.write(str(f_log(i - 128) + random.normalvariate(0, 0.2)) + '\n')
for i in range(257, 384):
    file.write(str(f_exp(i - 256) + random.normalvariate(0, 0.2)) + '\n')
for i in range(385, 516):
    file.write(str(f_tg(i-384) + random.normalvariate(0, 0.2)) + '\n')

file.close()