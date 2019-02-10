import pytest
import math
import numpy as np
from time_steppers import linear_forward_euler

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

def make_TestCase(ODE, algorithm, final):
    class ODE_TestCase(ODE):
        final_t = final
        def setup_class(self):
            self.stepper = algorithm(0, self.dt_init, self.q_init, self.A)
            self.stepper.stepUntil(final)
    
        def test_t(self):
            assert np.isclose(self.stepper.t, self.final_t)

        def test_q(self):
            assert np.allclose(self.stepper.q, self.exact(self.stepper.t))
        

    return ODE_TestCase

class Test_linear_forward_euler_simple(make_TestCase(Simple_ODE, linear_forward_euler, 10.2*Simple_ODE.dt_init)):
    pass

