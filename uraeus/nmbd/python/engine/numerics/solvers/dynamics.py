
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
from .integrators import (Explicit_RK45, Explicit_RK23, 
                          Explicit_RK4, Explicit_RK2)

class dds_solver(abstract_solver):
    
    def __init__(self, model):
        # Calling the parent class constructor to initialize the model
        super().__init__(model)
        
        # Checking the model dimensionality for redundancy
        if self.model.n == self.model.nc:
            msg = 'Model is fully constrained! Cannot perform dynamic'\
                  'analysis on such model.'
            raise ValueError(msg)
        
    
    def solve(self, run_id):
        # getting CPU start time to calculate CPU simulation time
        t0 = time.perf_counter()
        
        time_array = self.time_array
        dt = self.step_size

        # Getting the length of the time grid.
        bar_length = len(time_array)-1
        
        print('\nStarting System Dynamic Analysis:')
        
        # Extracting the system independent coordinates based
        # on the constraint jacobian structure
        print('Extracting the system independent coordinates ...')
        self._extract_independent_coordinates()
        print('Estimated DOF : %s'%(len(self.independent_cord),))
        print('Estimated Independent Coordinates : %s'%(self.independent_cord,))
        
        # Getting states initial conditions
        pos_t0 = self._pos_history[0]
        vel_t0 = self._vel_history[0]
        
        # Solving the constraints equations based on the given initial 
        # configuration.
        self._newton_raphson(pos_t0)

        # Evaluating the coefficient matrices at the initial
        # configuration
        M, J, Qt, Qd = self._eval_augmented_matricies(pos_t0, vel_t0)
        
        # Solving the DAE system for the accelerations and lagrange 
        # multipliers.
        acc_t0, lamda_t0 = self._solve_augmented_system(M, J, Qt, Qd)        
        
        # Storing the initial accelerations and lagrange multipliers.
        self._acc_history[0] = acc_t0
        self._lgr_history[0] = lamda_t0

        # Extracting the independent coordinates from the full system
        # coordinates based on the latest coordinates partiotioning
        v  = self.get_indpenednt_q(pos_t0)
        vd = self.get_indpenednt_q(vel_t0)

        # Constructing the initial state vector
        y0 = np.concatenate([v, vd])

        # Initializing the integrator (time-stepper)
        self.integrator = Explicit_RK23(self.SSODE, y0, 0, time_array[-1], dt)
        
        # Starting the simulation main loop
        print('\nRunning System Dynamic Analysis:')
        i = 0
        while i != bar_length:
            # Updating the progress bar
            progress_bar(bar_length, i, t0)
            t = time_array[i+1]

            # Re-Partition the system coordinates based on the latest
            # evaluation of the constraints jacobian
            self._partition_system_coordinates(self._jac[:-self.dof])
            
            # Solving the current time step
            self._solve_time_step(t, i)

            # Updating the iteration count
            i += 1
              
        print('\n')
        # Constructing pandas dataframes to hold the simulation results
        self._creat_results_dataframes()


    def _solve_time_step(self, t, i):
        
        # Getting the last system states
        q   = self._pos_history[i]
        qd  = self._vel_history[i]
        qdd = self._acc_history[i]
        
        # Extracting the independent coordinates from the full system
        # coordinates based on the latest coordinates partiotioning
        v   = self.get_indpenednt_q(q)
        vd  = self.get_indpenednt_q(qd)
        vdd = self.get_indpenednt_q(qdd)
        
        # Constructing the independent state vector
        state_vector = np.concatenate([v, vd])
        # Constructing the independent state vector derivative
        current_derivative = np.concatenate([vd, vdd])
        
        # Stepping an integration step
        self.integrator.step(state_vector, i, current_derivative)
        
        # Storing the latset computed states in their corresponding
        # containers. These are evaluaed inside the SSODE method
        self._pos_history[i+1] = self._q_new
        self._vel_history[i+1] = self._qd_new
        self._acc_history[i+1] = self._qdd_new
        self._lgr_history[i+1] = self._lgr_new
        
#    @profile
    def SSODE(self, state_vector, t, i, h):
        
        # Setting the current time to the model instance
        self._set_time(t)
        
        # Decomposing the state_vector into position and velocity
        # vectors
        y1 = state_vector[:self.dof]
        y2 = state_vector[self.dof:]
        
        # Getting the last system states
        q   = self._pos_history[i]
        qd  = self._vel_history[i]
        qdd = self._acc_history[i]
        
        # Evaluating an initial guess for the position level
        # coordinates for the solution of the constraints equations
        guess = q + (qd * h) + (0.5 * qdd * (h**2))
        for c in range(self.dof): 
            guess[np.argmax(self.independent_cols[:, c]), 0] = y1[c,0]
        
        # Solving the constraints equations based on the given guessed 
        # configuration.
        self._newton_raphson(guess)
        q_i = self._pos
        # Setting the current system generalized coordinates
        self._set_gen_coordinates(q_i)
        
        # Evaluating the rhs of the velocity constraint equations
        vel_rhs = self._eval_vel_eq(y2)
        # Solving for the system generalized velocities
        qd_i = solve(self._jac, -vel_rhs)
        # Setting the current system generalized coordinates
        self._set_gen_velocities(qd_i)

        # Evaluating the system coefficient matrices
        J  = self._jac[:-self.dof]
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        Qd = self._eval_acc_eq()

        # Solving the DAE system for the accelerations and lagrange 
        # multipliers.
        qdd_i, lamda_i = self._solve_augmented_system(M, J, Qt, Qd)
        
        # Extracting the independent accelerations based on the 
        # latest coordinates partiotioning
        y3 = self.get_indpenednt_q(qdd_i)
        
        # Constructing the rhs dy/dt vector
        rhs_vector = np.concatenate([y2, y3])

        # Storing the latest computed states in private variables
        # for usage in the main _solve_time_step method
        self._q_new   = q_i
        self._qd_new  = qd_i
        self._qdd_new = qdd_i
        self._lgr_new = lamda_i
        
        return rhs_vector
        

    def _extract_independent_coordinates(self, jacobian=None):
        self._partition_system_coordinates(jacobian)
        self.dof = dof = self.independent_cols.shape[1]
        independent_cord = [self._coordinates_indicies[np.argmax(self.independent_cols[:,i])] for i in range(dof) ]
        self.independent_cord = independent_cord

    def _partition_system_coordinates(self, jacobian=None):
        A = super()._eval_jac_eq() if jacobian is None else jacobian
        rows, cols = A.shape
        permutaion_mat = sc.linalg.lu(A.T)[0]
        self.independent_cols = permutaion_mat[:, rows:]
        self.permutaion_mat   = permutaion_mat.T

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
        #V = np.array(ind_vel_i).reshape((self.dof, 1))
        V = ind_vel_i
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
        
