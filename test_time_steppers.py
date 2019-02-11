import pytest
import math
import numpy as np
from parameterized import parameterized_class
from time_steppers import linear_forward_euler, linear_backward_euler

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
        return cls.q_init*np.exp(np.diag(cls.A)*t)

ODEs = [Simple_ODE, Independent_ODEs]
algorithms = [linear_forward_euler, linear_backward_euler]

@parameterized_class(('TestName', 'ODE', 'algorithm', 'num_timesteps',), [
    (ODE.__name__ + '_' + alg.__name__ + '_' + str(steps).replace('.', '_'),
    ODE, alg, steps) for ODE in ODEs
                      for alg in algorithms
                      for steps in [1,10.2]
])
class Test_ODE:#pylint: disable=no-member
    def setup_class(self):
        self.final_t = self.num_timesteps*self.ODE.dt_init
        self.stepper = self.algorithm(0, self.ODE.dt_init, self.ODE.q_init, self.ODE.A)
        self.stepper.stepUntil(self.final_t)

    def test_t(self):
        assert np.isclose(self.stepper.t, self.final_t)

    def test_q(self):
        assert np.allclose(self.stepper.q, self.ODE.exact(self.stepper.t))