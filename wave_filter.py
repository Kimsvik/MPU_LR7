from pywt import wavedec
import pywt
from pylab import *
from numpy import *

fq = 50
T = 1/fq
mgn = 2
b_size = 256


x = np.arange(0, 2*T, T/b_size)
y = mgn * np.sin(2 * np.pi * x / T)
st='sym5'
coeffs = wavedec(y, st, level=1)
coefsa=pywt.downcoef('a', y, 'db20', mode='symmetric', level=1)
print(coefsa)
coefsd=pywt.downcoef('d', y, 'db20', mode='symmetric', level=1)
print(coefsd)
subplot(2, 1, 1)
plot(coeffs[0],'b',linewidth=2, label='cA,level-5')
grid()
legend(loc='best')
subplot(2, 1, 2)
plot(coeffs[1],'r',linewidth=2, label='cD,level-5')
grid()
legend(loc='best')
show()