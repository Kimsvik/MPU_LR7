from numpy import *
from scipy import *
import pandas as pd
from pylab import *
import matplotlib as plt
import pywt


def get_ave_values(xvalues, yvalues, n = 5):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = array(xvalues)
    yarr = array(yvalues)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    x_ave = xarr_reshaped[:,0]
    y_ave = nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave

def plot_signal_plus_average(time, signal, average_over = 5):

    fig, ax = subplots(figsize=(15, 3))
    time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='Сигнал')
    ax.plot(time_ave, signal_ave, label = 'Скользящее среднее сигнала (n={})'.format(5))
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Амплитуда сигнала', fontsize=18)
    ax.set_title('Сигнал + Скользящее среднее сигнала', fontsize=18)
    ax.set_xlabel('Время', fontsize=18)
    ax.legend()
    show()


def plot_wavelet(time, signal, scales,
                 waveletname='cmor1.0-0.4',
                 cmap=plt.cm.seismic,
                 title='Вейвлет-преобразование(Спектр мощности) сигнала',
                 ylabel='Период (год)',
                 xlabel='Время'):
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3]
    contourlevels = log2(levels)

    fig, ax = subplots(figsize=(15, 10))
    im = ax.contourf(time, log2(period), log2(power), contourlevels, extend='both', cmap=cmap)

    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    yticks = 2 ** arange(np.ceil(log2(period.min())), ceil(log2(period.max())))
    ax.set_yticks(log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    show()


fq = 50
T = 1 / fq
mgn = 2
b_size = 256
d_size = 2
max_l = np.log2(256)


def f(t):
    return mgn * np.sin(2 * np.pi * t / T) + mgn/5 * np.sin(20 * np.pi * t / T) + mgn/20 * np.sin(50 * np.pi * t / T)

x = np.arange(0, d_size * T, T / b_size)
y = f(x)
plot_signal_plus_average(x, y, 7)
scales = arange(1, d_size * T, T / b_size)
plot_wavelet(x, y, scales)