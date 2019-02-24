import math
import numpy as np
import scipy.sparse as sp
from parameterized import parameterized_class, parameterized
from time_steppers import quasilinear_forward_euler, quasilinear_backward_euler, quasilinear_trapezoid

def custom_testname_func(func, num, params):
    """Generates a reasonable testname for parameterized function tests"""
    return "%s_%s_%s" % (
        func.__name__, int(num),
        parameterized.to_safe_name('_'.join((params.args[0].__name__, params.args[1].__name__)))
    )

class Simple_ODE:
    dt_init = 0.001
    q_init = 1.0
    A = np.asarray(1)

    @staticmethod
    def exact(t):
        return math.exp(t)

class Simple_t_Dependent_ODE:
    dt_init = 0.001
    q_init = 1.0

    @staticmethod
    def A(t, q=None):
        return np.asarray(t)

    @staticmethod
    def exact(t):
        return math.exp(t**2/2)

class Independent_ODEs:
    dt_init = 0.001
    q_init = np.array([1.0,1.0])
    A = np.array([[2.,0.],[0.,1.]])

    @classmethod
    def exact(cls, t):
        A = cls.A
        if sp.issparse(A):
            A = A.toarray()
        return cls.q_init*np.exp(np.diag(A)*t)

class Dependent_ODEs:
    dt_init = 0.001
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
ODEs.append(Simple_t_Dependent_ODE)
explicit_algorithms = [quasilinear_forward_euler]
implicit_algorithms = [quasilinear_backward_euler, quasilinear_trapezoid]
algorithms = explicit_algorithms + implicit_algorithms
expected_convergences = [1, 1, 2]

@parameterized_class(('TestName', 'ODE', 'algorithm', 'num_timesteps',), [
    (ODE.__name__ + '_' + alg.__name__ + '_' + str(steps).replace('.', '_'),
        ODE, alg, steps) for ODE in ODEs # noqa: E131
                         for alg in algorithms
                         for steps in [1,10.2]
])
class Test_Solution:
    """Tests whether we are getting close to the right answer or not"""
    def setup_class(self):
        self.final_t = self.num_timesteps*self.ODE.dt_init
        self.stepper = self.algorithm(0, self.ODE.dt_init, self.ODE.q_init, self.ODE.A)
        self.stepper.stepUntil(self.final_t)

    def test_t(self):
        """Expected to be close to fairly high accuracy"""
        assert np.isclose(self.stepper.t, self.final_t)

    def test_q(self):
        """Expected to be close, but only within a few digits"""
        assert np.allclose(self.stepper.q, self.ODE.exact(self.stepper.t), rtol=1e-3, atol=1e-5)

@parameterized.expand([
    (ODE, alg, rate) for ODE in ODEs # noqa: E131
                     for alg,rate in zip(algorithms, expected_convergences)
], testcase_func_name=custom_testname_func)
def test_convergence(ODE, alg, expected_rate):
    """Tests whether the rate of convergence is close enough to what's expected"""
    final_t = 5*ODE.dt_init
    dts = [ODE.dt_init/2**i for i in range(4)]
    steppers = [alg(0, dt, ODE.q_init, ODE.A) for dt in dts]

    for s in steppers:
        s.stepUntil(final_t)

    errs = [np.linalg.norm(s.q - ODE.exact(s.t), ord=np.inf) for s in steppers]

    p, logM = np.polyfit(np.log10(dts), np.log10(errs), 1)

    # This does not need to be especially close. Being within a digit or two
    # is enough to demonstrate convergence.
    assert np.isclose(p, expected_rate, rtol=1e-2, atol=0)

@parameterized.expand([
    (ODE, alg) for ODE in ODEs # noqa: E131
                     for alg in implicit_algorithms
], testcase_func_name=custom_testname_func)
def test_sparsity_detection(ODE, alg):
    """Implicit algorithms will detect whether or not A is sparse
    and build an identity matrix that matches. This may fail.
    """
    stepper = alg(0, ODE.dt_init, ODE.q_init, ODE.A)
    assert ODE.__name__.startswith('Sparse') == sp.issparse(stepper.I)
