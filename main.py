import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from math import *

fq = 50
T = 1 / fq
mgn = 2
b_size = 256
d_size = 2
max_l = np.log2(256)


def size(n):
    counter = 0
    while n != 1:
        n = n // 2
        counter = counter + 1
    return counter


def get_ave_values(xvalues, yvalues, n = 5):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    print(yarr)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    print('-------------')
    print(yarr)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    print('-------------')
    print(yarr_reshaped)
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave


def f(t):
    return mgn * np.sin(2 * np.pi * t / T) + mgn/5 * np.sin(20 * np.pi * t / T) + mgn/20 * np.sin(50 * np.pi * t / T)


class wavelet:
    def __init__(self, func):
        self.f = func
        self.len = len(self.f)
        self.deep = 5


    def haara(self, max_deep = -1, clean = 1):

        def A(n):
            a = np.zeros((n, n))
            for i in range(0, n // 2):
                a[i, i * 2] = 1
                a[i, i * 2 + 1] = 1
                a[i + n // 2, i * 2] = 1
                a[i + n // 2, i * 2 + 1] = -1
            return a/2

        def A_t(n):
            a = np.zeros((n, n))
            for i in range(0, n // 2):
                a[i * 2, i] = 1
                a[i * 2 + 1, i] = 1
                a[i * 2, i + n // 2] = clean
                a[i * 2 + 1, i + n // 2] = -clean
            print(a)
            return a/2

        n = len(self.f)

        plt.subplot(max_deep + 2, 2, 1)
        plt.plot(self.f)
        plt.grid()

        plt.subplot(max_deep + 2, 2, 2)
        plt.plot(self.f)

        count = 3

        def haara_merge(a, deep, counter):
            len_a = len(a)
            if (np.log2(n / len_a) == max_deep) or (len(a) == 2):
                return a
            else:
                new = np.dot(A(len_a), a)
                new_a = np.split(new, 2)[0]
                new_d = np.split(new, 2)[1]
                plt.subplot(deep + 2, 2, counter, ylabel = '{}lvl'.format(round(np.log2(n / len_a)+1)))
                plt.plot(new_a, 'tab:orange')
                plt.grid()
                counter = counter + 1
                plt.subplot(deep + 2, 2, counter)
                plt.plot(new_d, 'tab:green')
                plt.grid()
                counter = counter + 1
                print()
                new_a = haara_merge(new_a, deep, counter)
                print(new_a)
                # return np.dot(2*np.transpose(A(len(a))), np.transpose(new))
                return np.dot(2 * A_t(len(a)), np.transpose(new))

        res = haara_merge(self.f, max_deep, count)

        plt.subplot(max_deep + 2, 2, 2)
        plt.plot(res, 'tab:red')
        plt.grid()
        plt.show()

        time = np.arange(0, n)
        plt.plot(time, self.f, label='Исходный сигнал')
        plt.plot(time, res, label='Обработанный сигнал')
        plt.ylabel('Amplitude', fontsize=16)
        plt.xlabel('time', fontsize=16)
        plt.title('Signals', fontsize=16)
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

        return res

    def dobeshi(self, max_deep = -1, clean = 1):

        def A(n):
            a = np.zeros((n, n))
            for i in range(0, n // 2):
                a[i, i * 2] = (1 + sqrt(3)) / (4 * sqrt(2))
                a[i, i * 2 + 1] = (3 + sqrt(3)) / (4 * sqrt(2))
                a[i, (i * 2 + 2) % n] = (3 - sqrt(3)) / (4 * sqrt(2))
                a[i, (i * 2 + 3) % n] = (1 - sqrt(3)) / (4 * sqrt(2))
                a[i + n // 2, i * 2] = -(1 - sqrt(3)) / (4 * sqrt(2))*clean
                a[i + n // 2, i * 2 + 1] = (3 - sqrt(3)) / (4 * sqrt(2))*clean
                a[i + n // 2, (i * 2 + 2) % n] = -(3 + sqrt(3)) / (4 * sqrt(2))*clean
                a[i + n // 2, (i * 2 + 3) % n] = (1 + sqrt(3)) / (4 * sqrt(2))*clean
            return a/sqrt(2)

        n = len(self.f)

        plt.subplot(max_deep + 2, 2, 1)
        plt.plot(self.f)
        plt.grid()

        plt.subplot(max_deep + 2, 2, 2)
        plt.plot(self.f)

        count = 3

        def dobeshi_merge(a, deep, counter):
            len_a = len(a)
            if (np.log2(n / len_a) == max_deep) or (len(a) == 2):
                return a
            else:
                new = np.dot(A(len_a), a)
                new_a = np.split(new, 2)[0]
                new_d = np.split(new, 2)[1]
                plt.subplot(deep + 2, 2, counter, ylabel = '{}lvl'.format(round(np.log2(n / len_a)+1)))
                plt.plot(new_a, 'tab:orange')
                plt.grid()
                counter = counter + 1
                plt.subplot(deep + 2, 2, counter)
                plt.plot(new_d, 'tab:green')
                plt.grid()
                counter = counter + 1
                print()
                new_a = dobeshi_merge(new_a, deep, counter)
                return np.dot(2 * np.transpose(A(len(a))), np.transpose(new))

        res = dobeshi_merge(self.f, max_deep, count)

        plt.subplot(max_deep + 2, 2, 2)
        plt.plot(res, 'tab:red')
        plt.grid()
        plt.show()

        time = np.arange(0, n)
        plt.plot(time, self.f, label='Исходный сигнал')
        plt.plot(time, res, label='Обработанный сигнал')
        plt.ylabel('Amplitude', fontsize=16)
        plt.xlabel('time', fontsize=16)
        plt.title('Signals', fontsize=16)
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

        return res

dataset = "C:/Users/Сергей/PycharmProjects/MPU_LR7/dataset1.txt"
df_nino = pd.read_table(dataset)
N = 2 ** size(df_nino.shape[0])
t0=0
dt=0.25
time = np.arange(0, N) * dt + t0
signal = df_nino.values.squeeze()

time1 = np.arange(0, 0.01 * 2 ** 10, 0.01)
signal1 = f(time1)

plt.figure()
plt.title('Функция')
y = np.array(signal[0:N])


wl = wavelet(y)


haara = wl.haara(5, 0)
dobeshi = wl.dobeshi(5, 0.5)

