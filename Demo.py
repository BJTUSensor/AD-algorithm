import numpy as np
from scipy.optimize import leastsq
import scipy.io as sio
import matplotlib.pyplot as plt
from Anisodiff2D import anisodiff2D

def lorentz(p, x):
    return p[0] / ((x - p[1]) ** 2 + p[2])


def errorfunc(p, x, z):
    return z - lorentz(p, x)


def lorentz_fit(x, y):
    p3 = ((max(x) - min(x)) / 10) ** 2
    p2 = (max(x) + min(x)) / 2
    p1 = max(y) * p3
    p0 = np.array([p1, p2, p3], dtype=np.float)  # Initial guess
    solp, ier = leastsq(errorfunc, p0, args=(x, y), maxfev=50000)
    return solp[1]


data = sio.loadmat('C:\\Users\\Lightning\\Desktop\\data.mat')
BGS1 = np.array(data['data_G']).T

BGS_new = anisodiff2D(BGS1, 0.5, 10, 1/7, 1)

#lorentz-fit
fre=np.linspace(10.750,10.950,200)  #frequency
lortz_data = []
for each in BGS_new:
    lortz_fit = lorentz_fit(fre, each)
    lortz_data.append(lortz_fit)
BFS_lortz = np.array(lortz_data)

#plot
x=np.linspace(1,2549,2550)
plt.plot(x,BFS_lortz)
plt.show()

