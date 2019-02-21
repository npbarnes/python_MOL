import numpy as np
import scipy.sparse as sp
from time_steppers import quasilinear_trapezoid

def build_matrix(dx, size):
    A = sp.diags([1/dx**2,-2/dx**2,1/dx**2], [-1,0,1], shape=(size+2,size+2), format='lil')
    A[0,0] = 0.0
    A[0,1] = 0.0
    A[-1,-1] = 0.0
    A[-1,-2] = 0.0

    return A.tocsr()

def benchmark():
    domain_size = 10000
    dt = .001
    dx = .1

    q0 = np.zeros(domain_size+2, dtype='d')
    q0[0] = 100.0
    A = build_matrix(dx, domain_size)
    Adense = A.toarray()

    fd = quasilinear_trapezoid(0, dt, q0, A) # noqa: F841
    fd_dense = quasilinear_trapezoid(0,dt,q0,Adense) # noqa: F841

    import timeit
    setup = 'from __main__ import fd'
    time = timeit.timeit('fd.step(1)', setup=setup, number=3)
    print(time)
    setup_dense = 'from __main__ import fd_dense'
    time_dense = timeit.timeit('fd_dense.step(1)', setup=setup_dense, number=3)
    print(time_dense)
    print(time_dense/time)

def plot_heat():
    import matplotlib.pyplot as plt
    domain_size = 100
    dt = .01
    dx = .1

    q0 = np.zeros(domain_size+2, dtype='d')
    q0[0] = 100.0
    A = build_matrix(dx, domain_size)

    fd = quasilinear_trapezoid(0, dt, q0, A)

    plt.plot(fd.q, label=f'time = {round(fd.t)}')
    for i in range(3):
        fd.stepBy(1)
        plt.plot(fd.q, label=f'time = {round(fd.t)}')

    plt.legend()
    plt.show()

plot_heat()
