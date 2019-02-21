#!/usr/bin/env python3
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

class time_stepper:
    """This is the base class for classes that solve equations in the form
    dq/dt = F(t,q)
    for vector q and function F
    Specific solvers must provide a step function and possibly a setup function.
    It's possible to use this class for a method of lines PDE solver.
    """

    def __init__(self, t0, dt, q0, *args, **kwargs):
        """Arguments
        t0: Starting time
        dt: Timestep
        q0: Initial condition
        args, kwargs: Extra data needed for solvers. Passed on to setup function
        """
        self.t = t0
        self.dt = dt
        if isinstance(q0, np.ndarray):
            self.q0 = q0
        else:
            self.q0 = np.array([q0])

        self.q = np.copy(self.q0)
        self.dim = self.q.shape[0]

        self.setup(*args, **kwargs)

    def setup(self, *args, **kwargs):
        pass

    def _step(self):
        """Updates self.q by one step of size self.dt"""
        raise NotImplementedError

    def step(self, N=1):
        for _ in range(N):
            self._step()
            self.t += self.dt

    def stepBy(self, time):
        if time < 0:
            raise ValueError('Must step forward in time')

        tmp_dt = self.dt
        numsteps = math.ceil(time/self.dt)
        self.dt = time/numsteps
        self.step(numsteps)
        self.dt = tmp_dt

    def stepUntil(self, time):
        if time < self.t:
            raise ValueError('Time must be in the future')

        delta = time - self.t
        self.stepBy(delta)

class quasilinear_time_stepper(time_stepper):
    """A base class for quasilinear systems. I.e. systems where
    F(t,q) = A(t,q) q
    for some matrix valued function A. It also supports A being a constant matrix.
    """
    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, A):
        if callable(A):
            self._A = A
        else:
            self._A = lambda *ignore: A

    def setup(self, A):
        self.A = A

class quasilinear_forward_euler(quasilinear_time_stepper):
    """The matrix A may be a function of time and state"""
    def _step(self):
        self.q = self.q + self.dt*self.A(self.t, self.q).dot(self.q)

class implicit_quasilinear_time_stepper(quasilinear_time_stepper):
    """The matrix A may be a function of time, but not state since that could require
    a nonlinear solver."""
    def setup(self, A):
        self.A = A
        # We designate a solver based on weather A is sparse or not.
        # We also construct an identity matrix since having one is handy for
        # many implicit algorithms.
        if sp.issparse(self.A(0)):
            self.I = sp.identity(self.dim) # noqa: E741
            self.solve = spsolve
        else:
            self.I = np.eye(self.dim) # noqa: E741
            self.solve = np.linalg.solve

class quasilinear_backward_euler(implicit_quasilinear_time_stepper):
    def _step(self):
        self.q = self.solve(self.I - self.dt*self.A(self.t+self.dt), self.q)

class quasilinear_trapezoid(implicit_quasilinear_time_stepper):
    def _step(self):
        self.q = self.solve(self.I - 0.5*self.dt*self.A(self.t+self.dt), self.q + 0.5*self.dt*self.A(self.t).dot(self.q))
