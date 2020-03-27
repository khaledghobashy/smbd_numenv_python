
# Standard library imports.
import sys
import time

# Third party imports.
import numpy as np
import scipy as sc
import pandas as pd
import numba

#from scipy.sparse.linalg import spsolve

# Local imports.
from .base import abstract_solver, solve, progress_bar

class kds_solver(abstract_solver):
    
    def __init__(self, model):
        super().__init__(model)
        if self.model.n != self.model.nc:
            raise ValueError('Model is not fully constrained.')
    
    def solve(self, run_id):
        
        t0 = time.perf_counter()
        
        time_array = self.time_array
        dt = self.step_size
        t_end = time_array[-1]
        
        A = self._eval_jac_eq()

        lu, p = self._factorize_jacobian(A)

        vel_rhs = self._eval_vel_eq()
        v0 = sc.linalg.lu_solve((lu, p), -vel_rhs)

        self._set_gen_velocities(v0)
        
        acc_rhs = self._eval_acc_eq()
        qdd = sc.linalg.lu_solve((lu, p), -acc_rhs)

        lamda = self._eval_lagrange_multipliers(qdd, (lu, p))
        
        self._vel_history[0] = v0
        self._acc_history[0] = qdd
        self._lgr_history[0] = lamda
        
        print('\nRunning System Kinematic Analysis:')
        bar_length = len(time_array)
        self.i = i = 0
        t = 0
        while t < t_end:
            self.i = i
            t = time_array[i+1]

            progress_bar(bar_length, i+1, t0, t+dt)
            self._set_time(t)

            guess = self._q + self._qd*dt + 0.5*self._qdd*(dt**2)
            
            self._solve_constraints(guess)

            A = self._jac
            lu, p = self._factorize_jacobian(A)
            
            vel_rhs = self._eval_vel_eq()
            vi = sc.linalg.lu_solve((lu, p), -vel_rhs)
            self._set_gen_velocities(vi)

            acc_rhs = self._eval_acc_eq()
            qdd = sc.linalg.lu_solve((lu, p), -acc_rhs)
            self._set_gen_accelerations(qdd)
            
            lamda = self._eval_lagrange_multipliers(qdd, (lu, p))

            self._pos_history[i+1] = self._q.copy()
            self._vel_history[i+1] = self._qd.copy()
            self._acc_history[i+1] = self._qdd.copy()
            self._lgr_history[i+1] = lamda

            i += 1
        
        print('\n')
        self._creat_results_dataframes()    
    
    
    def _eval_lagrange_multipliers(self, qdd, jac):        
        applied_forces = self._eval_frc_eq()
        mass_matrix = self._eval_mass_eq()
        inertia_forces = mass_matrix.dot(qdd)
        rhs = applied_forces - inertia_forces
        lamda = sc.linalg.lu_solve(jac, rhs, 1)
        return lamda
