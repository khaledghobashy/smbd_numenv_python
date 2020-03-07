
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
from ..math_funcs.numba_funcs import matrix_assembler
from .abstract import abstract_solver, solve, progress_bar
from .integrators import (Explicit_RK4, Explicit_RK23, 
                          Explicit_RK45, Explicit_RK2, 
                          BDF)

class dds_solver(abstract_solver):
    
    def __init__(self, model):
        super().__init__(model)
        if self.model.n == self.model.nc:
            raise ValueError('Model is fully constrained.')
        
    
    def solve(self, run_id):
        t0 = time.perf_counter()
        
        time_array = self.time_array
        dt = self.step_size
        bar_length = len(time_array)-1
        
        print('\nStarting System Dynamic Analysis:')
        
        self._extract_independent_coordinates()
        print('Estimated DOF : %s'%(len(self.independent_cord),))
        print('Estimated Independent Coordinates : %s'%(self.independent_cord,))
        
        pos_t0 = self._pos_history[0]
        vel_t0 = self._vel_history[0]
        
        self._newton_raphson(pos_t0)

        M, J, Qt, Qd = self._eval_augmented_matricies(pos_t0, vel_t0)
        acc_t0, lamda_t0 = self._solve_augmented_system(M, J, Qt, Qd)        
        self._acc_history[0] = acc_t0
        self._lgr_history[0] = lamda_t0

        v  = self.get_indpenednt_q(pos_t0)
        vd = self.get_indpenednt_q(vel_t0)
        state_vector = np.concatenate([v, vd])

        self.integrator = Explicit_RK23(self.SSODE, state_vector, 0, time_array[-1], dt)
        
        print('\nRunning System Dynamic Analysis:')
        i = 0
        while i != bar_length:
            progress_bar(bar_length, i, t0)
            t = time_array[i+1]
            self._extract_independent_coordinates(self._jac[:-self.dof])
            self._set_time(t)
            self._solve_time_step(t, i, dt)
            i += 1            
        print('\n')
        self._creat_results_dataframes()

    def _solve_time_step(self, t, i, dt):
        
        q   = self._pos_history[i]
        qd  = self._vel_history[i]
        qdd = self._acc_history[i]
        
        v   = self.get_indpenednt_q(q)
        vd  = self.get_indpenednt_q(qd)
        vdd = self.get_indpenednt_q(qdd)

        state_vector = np.concatenate([v, vd])
        current_derivative = np.concatenate([vd, vdd])
        
        self.integrator.step(state_vector, i, current_derivative)
        soln = self.integrator.y

        y1 = soln[:self.dof]
        y2 = soln[self.dof:]

        guess = q + (qd * dt) + (0.5 * qdd * (dt**2))

        for c in range(self.dof): 
            guess[np.argmax(self.independent_cols[:, c]), 0] = y1[c,0]
        
        self._newton_raphson(guess)
        A  = self._jac
        qi = self._pos
        
        vel_rhs = self._eval_vel_eq(y2)
        vi = solve(A, -vel_rhs)
        
        self._set_gen_coordinates(qi)
        self._set_gen_velocities(vi)
        
        J  = A[:-self.dof]
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        Qd = self._eval_acc_eq()
        acc_ti, lamda_ti = self._solve_augmented_system(M, J, Qt, Qd)
        
        self._pos_history[i+1] = qi
        self._vel_history[i+1] = vi
        self._acc_history[i+1] = acc_ti
        self._lgr_history[i+1] = lamda_ti
        
#    @profile
    def SSODE(self, state_vector, t, i):
          
        self._set_time(t)
        
        y1 = state_vector[:self.dof]
        y2 = state_vector[self.dof:]
        
        dt = self.step_size
        
        guess = self._pos_history[i] \
              + self._vel_history[i]*dt \
              + 0.5*self._acc_history[i]*(dt**2)

        for c in range(self.dof): 
            guess[np.argmax(self.independent_cols[:, c]), 0] = y1[c,0]
                        
        self._newton_raphson(guess)
        self._set_gen_coordinates(self._pos)
                
        vel_rhs = self._eval_vel_eq(y2)
        vi = solve(self._jac, -vel_rhs)
        self._set_gen_velocities(vi)

        J  = self._jac[:-self.dof]
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        Qd = self._eval_acc_eq()
        acc_ti, lamda_ti = self._solve_augmented_system(M, J, Qt, Qd)
        
        y3 = self.get_indpenednt_q(acc_ti)
        
        rhs_vector = np.concatenate([y2, y3])
        
        return rhs_vector
        

    def _extract_independent_coordinates(self, jacobian=None):
        A = super()._eval_jac_eq() if jacobian is None else jacobian
        rows, cols = A.shape
        permutaion_mat = sc.linalg.lu(A.T)[0]
        independent_cols = permutaion_mat[:, rows:]
        self.dof = dof = independent_cols.shape[1]
        independent_cord = [self._coordinates_indicies[np.argmax(independent_cols[:,i])] for i in range(dof) ]
        self.permutaion_mat  = permutaion_mat.T
        self.independent_cols = independent_cols
        self.independent_cord = independent_cord
    
    
    def _eval_augmented_matricies(self, q , qd):
        self._set_gen_coordinates(q)
        self._set_gen_velocities(qd)
        J  = super()._eval_jac_eq()
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        Qd = self._eval_acc_eq()
        return M, J, Qt, Qd
    
        
    def _solve_augmented_system(self, M, J, Qt, Qd):
        
        z = np.zeros((self.model.nc, self.model.nc))

        u = np.concatenate([M, J.T], axis=1)
        l = np.concatenate([J, z], axis=1)
        
        A = np.concatenate([u, l], axis=0)
#        A = sc.sparse.coo_matrix(A)
        
        b = np.concatenate([Qt, -Qd])
        x = solve(A, b)
        n = self.model.n
        accelerations = x[:n]
        lamda = x[n:]
        return accelerations, lamda
    
    def _eval_pos_eq(self):
        A = super()._eval_pos_eq()
        Z = np.zeros((self.dof, 1))
        A = np.concatenate([A, Z])
        return A

    def _eval_vel_eq(self,ind_vel_i):
        A = super()._eval_vel_eq()
        V = np.array(ind_vel_i).reshape((self.dof, 1))
        A = np.concatenate([A, -V])
        return A
    
    def _eval_jac_eq(self):
        A = np.concatenate([super()._eval_jac_eq(), self.independent_cols.T])
        return A
    
            
    def get_indpenednt_q(self, q):
        # Boolean matrix (ndof x n)
        P  = self.independent_cols.T
        qv = P@q
        return qv
        
    def _eval_lagrange_multipliers(self, i):
        self._set_gen_coordinates(self._pos_history[i])
        return self._lgr_history[i]
        
