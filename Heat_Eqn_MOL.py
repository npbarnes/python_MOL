import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from time_steppers import quasilinear_trapezoid

def build_matrix(dx, size):
    A = sp.diags([1/dx**2,-2/dx**2,1/dx**2], [-1,0,1], shape=(size+2,size+2), format='lil')
    A[0,0] = 0.0
    A[0,1] = 0.0
    A[-1,-1] = 0.0
    A[-1,-2] = 0.0

    return A.tocsr()

if __name__ == '__main__':
    domain_size = 100
    dt = .001
    dx = .1

    q0 = np.zeros(domain_size+2, dtype='d')
    q0[0] = 100.0
    A = build_matrix(dx, domain_size)

    fd = quasilinear_trapezoid(0, dt, q0, A)
    fd.step(1000)
    plt.plot(fd.q)
    fd.step(1000)
    plt.plot(fd.q)
    fd.step(1000)
    plt.plot(fd.q)
    plt.show()
