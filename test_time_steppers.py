import math
import numpy as np
import scipy.sparse as sp
from parameterized import parameterized_class
from time_steppers import quasilinear_forward_euler, quasilinear_backward_euler, quasilinear_trapezoid

class Simple_ODE:
    dt_init = 0.0001
    q_init = 1.0
    A = np.asarray(1)

    @staticmethod
    def exact(t):
        return math.exp(t)

class Independent_ODEs:
    dt_init = 0.0001
    q_init = np.array([1.0,1.0])
    A = np.array([[2.,0.],[0.,1.]])

    @classmethod
    def exact(cls, t):
        A = cls.A
        if sp.issparse(A):
            A = A.toarray()
        return cls.q_init*np.exp(np.diag(A)*t)

class Dependent_ODEs:
    dt_init = 0.0001
    q_init = np.array([1.0,1.0])
    A = np.array([[0.,2.],[1.,0.]])

    @staticmethod
    def exact(t):
        exp = math.exp
        r2 = math.sqrt(2)
        r22 = 2*math.sqrt(2)
        return np.array([
            0.5*exp(-r2*t)*(exp(r22*t)+1) + (exp(-r2*t)*(exp(r22*t)-1))/r2,
            (exp(-r2*t)*(exp(r22*t)-1))/r22 + 0.5*exp(-r2*t)*(exp(r22*t)+1)
        ])

def make_sparse(ODE):
    class Sparse_ODE(ODE):
        A = sp.csr_matrix(ODE.A)
    Sparse_ODE.__name__ = 'Sparse_' + ODE.__name__
    return Sparse_ODE

ODEs = [Simple_ODE, Independent_ODEs, Dependent_ODEs]
ODEs += [make_sparse(ODE) for ODE in ODEs]
algorithms = [quasilinear_forward_euler, quasilinear_backward_euler, quasilinear_trapezoid]

@parameterized_class(('TestName', 'ODE', 'algorithm', 'num_timesteps',), [
    (ODE.__name__ + '_' + alg.__name__ + '_' + str(steps).replace('.', '_'),
        ODE, alg, steps) for ODE in ODEs # noqa: E131
                         for alg in algorithms
                         for steps in [1,10.2]
])
class Test_ODE:
    def setup_class(self):
        self.final_t = self.num_timesteps*self.ODE.dt_init
        self.stepper = self.algorithm(0, self.ODE.dt_init, self.ODE.q_init, self.ODE.A)
        self.stepper.stepUntil(self.final_t)

    def test_t(self):
        assert np.isclose(self.stepper.t, self.final_t)

    def test_q(self):
        assert np.allclose(self.stepper.q, self.ODE.exact(self.stepper.t))
