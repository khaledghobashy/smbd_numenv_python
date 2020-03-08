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
        
        A = self._eval_jac_eq()
        vel_rhs = self._eval_vel_eq()
        v0 = solve(A, -vel_rhs)
        self._set_gen_velocities(v0)
        self._vel_history[0] = v0
        
        acc_rhs = self._eval_acc_eq()
        self._acc_history[0] = solve(A, -acc_rhs)
        
        print('\nRunning System Kinematic Analysis:')
        bar_length = len(time_array)-1
        for i,t in enumerate(time_array[1:]):
            progress_bar(bar_length, i, t0, t)
            self._set_time(t)

            g =   self._pos_history[i] \
                + self._vel_history[i]*dt \
                + 0.5*self._acc_history[i]*(dt**2)
            
            self._newton_raphson(g)
            self._pos_history[i+1] = self._pos
            A = self._eval_jac_eq()
            
            vel_rhs = self._eval_vel_eq()
            vi = solve(A,-vel_rhs)
            self._set_gen_velocities(vi)
            self._vel_history[i+1] = vi

            acc_rhs = self._eval_acc_eq()
            self._acc_history[i+1] = solve(A,-acc_rhs)
        
        print('\n')
        self._creat_results_dataframes()    
    
    
    def _eval_lagrange_multipliers(self, i):
        self._set_gen_coordinates(self._pos_history[i])
        self._set_gen_velocities(self._vel_history[i])
        self._set_gen_accelerations(self._acc_history[i])
        
        applied_forces = self._eval_frc_eq()
        mass_matrix = self._eval_mass_eq()
        jac = self._eval_jac_eq()
        qdd = self._acc_history[i]
        inertia_forces = mass_matrix.dot(qdd)
        rhs = applied_forces - inertia_forces
        lamda = solve(jac.T, rhs)
        
        return lamda
