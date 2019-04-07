import pytest
import math
import numpy as np
import scipy.sparse as sp
from time_steppers import quasilinear_forward_euler, quasilinear_backward_euler, quasilinear_trapezoid

class Simple_ODE:
    def __init__(self, array):
        self.dt_init = 0.001
        self.q_init = 1.0
        self.A = array(1)

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

ODEs = [Simple_ODE, Independent_ODEs, Dependent_ODEs]

@pytest.fixture(scope='module', params=[np.array, sp.csr_matrix], ids=['dense', 'sparse'])
def array(request):
    return request.param

@pytest.fixture(scope='module', params=ODEs)
def ODE(request, array):
    return request.param(array)

# ODEs += [make_sparse(ODE) for ODE in ODEs]
# ODEs.append(Simple_t_Dependent_ODE)
explicit_algorithms = [quasilinear_forward_euler]
implicit_algorithms = [quasilinear_backward_euler, quasilinear_trapezoid]
algorithms = explicit_algorithms + implicit_algorithms
# expected_convergences = [1, 1, 2]

@pytest.fixture(scope='module', params=algorithms)
def algorithm(request):
    return request.param

@pytest.fixture(scope='class', params=[1, 10.2])
def solution(request, algorithm, ODE):
    """Build a timestepper object, take a number of steps, and equip it with
    the expected exact solution.
    """
    final_t = request.param*ODE.dt_init

    stepper = algorithm(0, ODE.dt_init, ODE.q_init, ODE.A)
    stepper.stepUntil(final_t)

    stepper.expected = {'t':final_t, 'exact_q':ODE.exact(final_t)}
    return stepper

class Test_Solution:
    def test_t(self, solution):
        """Expected to be close to fairly high accuracy"""
        assert np.isclose(solution.t, solution.expected['t'])

    def test_q(self, solution):
        """Expected to be close, but only within a few digits"""
        assert np.allclose(solution.q, solution.expected['exact_q'], rtol=1e-3, atol=1e-5)

# @parameterized.expand([
#     (ODE, alg, rate) for ODE in ODEs # noqa: E131
#                      for alg,rate in zip(algorithms, expected_convergences)
# ], name_func=name_func)
# def test_convergence(ODE, alg, expected_rate):
#     """Tests whether the rate of convergence is close enough to what's expected"""
#     final_t = 5*ODE.dt_init
#     dts = [ODE.dt_init/2**i for i in range(4)]
#     steppers = [alg(0, dt, ODE.q_init, ODE.A) for dt in dts]

#     for s in steppers:
#         s.stepUntil(final_t)

#     errs = [np.linalg.norm(s.q - ODE.exact(s.t), ord=np.inf) for s in steppers]

#     p, logM = np.polyfit(np.log10(dts), np.log10(errs), 1)

#     # This does not need to be especially close. Being within a digit or two
#     # is enough to demonstrate convergence.
#     assert np.isclose(p, expected_rate, rtol=1e-2, atol=0)

# @parameterized.expand([
#     (ODE, alg) for ODE in ODEs # noqa: E131
#                      for alg in implicit_algorithms
# ], name_func=name_func)
# def test_sparsity_detection(ODE, alg):
#     """Implicit algorithms will detect whether or not A is sparse
#     and build an identity matrix that matches. This may fail.
#     """
#     stepper = alg(0, ODE.dt_init, ODE.q_init, ODE.A)
#     assert ODE.__name__.startswith('Sparse') == sp.issparse(stepper.I)
