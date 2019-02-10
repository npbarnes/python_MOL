#!/usr/bin/env python3
import math
import numpy as np

class time_stepper:
    """This is the abstract superclass for classes that solve
    dq/dt = A q
    for vector q and matrix A
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
        self.q0 = np.asarray(q0)

        self.q = np.copy(q0)
        try:
            self.dim = len(self.q)
        except TypeError:
            self.dim = 1

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

class linear_time_stepper(time_stepper):
    def setup(self, A):
        self.A = A

class linear_forward_euler(linear_time_stepper):
    def _step(self):
        self.q = self.q + self.dt*self.A.dot(self.q)

class lienar_backward_euler_save_inv(linear_time_stepper):
    def setup(self, A):
        """Warning: this won't work right if dt changes"""
        super().__init__(A)
        self.inv = np.linalg.inv(np.eye(self.dim) - self.dt*self.A)

    def _step(self):
        self.q = self.inv.dot(self.q)

class linear_backward_euler_saved_matrix(linear_time_stepper):
    def setup(self, A):
        """Warning: this won't work right if dt changes"""
        super().__init__(A)
        self.I_dtA = np.eye(self.dim) - self.dt*self.A

    def _step(self):
        self.q = np.linalg.solve(self.I_dtA, self.q)

class linear_backward_euler(linear_time_stepper):
    def setup(self, A):
        super().__init__(A)
        self.I = np.eye(self.dim)
    
    def _step(self):
        self.q = np.linalg.solve(self.I - self.dt*self.A, self.q)