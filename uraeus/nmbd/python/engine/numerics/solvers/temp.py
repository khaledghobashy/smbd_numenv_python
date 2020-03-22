# Standard library imports.
import time

# Third party imports.
import numpy as np
import scipy as sc

# Local imports.
from .base import abstract_solver, solve, progress_bar
from .integrators import BDF

################################################################
################################################################

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
        
        pos_t0 = self._pos_history[0]
        vel_t0 = self._vel_history[0]
        
        self._newton_raphson(pos_t0)

        M, J, Qt, Qd = self._eval_augmented_matricies(pos_t0, vel_t0)
        acc_t0, lamda_t0 = self._solve_augmented_system(M, J, Qt, Qd)        
        self._acc_history[0] = acc_t0
        self._lgr_history[0] = lamda_t0

        self.integrator = BDF(self.SSODE, 0, 0, time_array[-1], dt)
        
        print('\nRunning System Dynamic Analysis:')
        i = 0
        while i != bar_length:
            t = time_array[i+1]
            progress_bar(bar_length, i, t0, t)
            self._set_time(t)
            self._extract_independent_coordinates(self._jac[:-self.dof])
            self._solve_time_step(t, i, dt)
            i += 1            
        print('\n')
        self._creat_results_dataframes()

    def _solve_time_step(self, t, i, dt):
        
        q   = self._pos_history[i]
        qd  = self._vel_history[i]
        qdd = self._acc_history[i]
        lamda = self._lgr_history[i]

        fi = np.concatenate([qdd, lamda])

        soln = self.integrator.step([self._pos_history, self._vel_history], i, fi)
        
        qn, qdn, fn = soln
        qddn = fn[:self.model.n]
        lamda_n = fn[self.model.n:]

        #self._newton_raphson(qn)
        
        self._pos_history[i+1] = qn
        self._vel_history[i+1] = qdn
        self._acc_history[i+1] = qddn
        self._lgr_history[i+1] = lamda_n
        
#    @profile
    def SSODE(self, fi, q, qd, t, beta0, h):
        self._set_time(t)

        acc = fi[:self.model.n]
        lamda = fi[self.model.n:]

        v  = self.get_indpenednt_q(q)
        vd = self.get_indpenednt_q(qd)

        guess = q + (qd * h) + (0.5 * acc * (h**2))
        for c in range(self.dof): 
            guess[np.argmax(self.independent_cols[:, c]), 0] = v[c,0]

        self._newton_raphson(guess)
        A  = self._jac
        q_mod = self._pos
        
        vel_rhs = self._eval_vel_eq(vd)
        qd_mod = solve(A, -vel_rhs)

        self._set_gen_coordinates(q_mod)
        self._set_gen_velocities(qd_mod)

        # 3) evaluating the mass matrix and force vector
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        J  = self._jac[:-self.dof,:]
        #Qd = self._eval_acc_eq()
        A = self.eval_coefficient_matrix(M, J)
        A = np.eye(A.shape[0]) - h*beta0*A

        constraints = super()._eval_pos_eq()

        eq1 = M@acc + J.T@lamda - Qt
        eq2 = constraints

        b = fi - h*beta0 * np.concatenate([eq1, eq2])
        print('Norm b = %s'%np.linalg.norm(b))
        print('Norm eq1 = %s'%np.linalg.norm(eq1))
        print('Norm eq2 = %s'%np.linalg.norm(eq2))
        return A, b
        

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
        A = self.eval_coefficient_matrix(M, J)
        
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

    def eval_coefficient_matrix(self, M, J):
        z = np.zeros((self.model.nc, self.model.nc))
        u = np.concatenate([M, J.T], axis=1)
        l = np.concatenate([J, z], axis=1)
        A = np.concatenate([u, l], axis=0)
        return A

################################################################

class dds_solver4(abstract_solver):
    
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
        
        pos_t0 = self._pos_history[0]
        vel_t0 = self._vel_history[0]
    
        self.factorize_constraints_jacobian()
        self._extract_independent_coordinates()

        self._newton_raphson(pos_t0)

        M, J, Qt, Qd = self._eval_augmented_matricies(pos_t0, vel_t0)
        acc_t0, lamda_t0 = self._solve_augmented_system(M, J, Qt, Qd)        
        self._acc_history[0] = acc_t0
        self._lgr_history[0] = lamda_t0
        
        print('\nRunning System Dynamic Analysis:')
        i = 0
        while i != bar_length:
            t = time_array[i+1]
            progress_bar(bar_length, i, t0, t)
            self._set_time(t)
            self.factorize_constraints_jacobian()
            self._solve_time_step(t, dt, i)
            i += 1            
        print('\n')
        self._creat_results_dataframes()

    def _solve_time_step(self, t, h, i):
        print('SOLVING TIME STEP (%s)'%i)
        print('======================')

        qdd0 = self._acc_history[i]
        vdd0 = self.get_indpenednt_q(qdd0)

        q, qd, qdd, lamda = self.step(vdd0, t, h, i)
        
        self._pos_history[i+1] = q
        self._vel_history[i+1] = qd
        self._acc_history[i+1] = qdd
        self._lgr_history[i+1] = lamda

        print('FINISHED TIME STEP (%s)'%i)
        print('=======================\n')


    def factorize_constraints_jacobian(self, jacobian=None):
        A = super()._eval_jac_eq() if jacobian is None else jacobian
        P, L, U = sc.linalg.lu(A.T)
        self.permutaion_mat  = P.T
    
    
    def step(self, vdd, t, h, i):
        #print('Entring Step Calculation')
        soln = self.integrator_rhs(vdd, t, h, i)
        residual, q, qd, qdd, lamda, M_hat = soln
        delta = solve(M_hat, -residual)
        norm = np.linalg.norm(delta)
        itr = 0
        while norm >1e-3:
            print('Itr : %s'%itr)
            print('ResNorm = %s' %np.linalg.norm(residual))
            print('DelNorm = %s\n' %np.linalg.norm(delta))
            vdd = vdd + delta
            soln = self.integrator_rhs(vdd, t, h, i)
            residual, q, qd, qdd, lamda, _ = soln
            delta = solve(M_hat, -residual)
            norm = np.linalg.norm(delta)
            if itr > 50:
                print("Integration Iterations exceded \n")
                raise ValueError("Integration Iterations exceded \n")
                break
            itr += 1
        return q, qd, qdd, lamda
        

    def integrator_rhs(self, vdd, t, h, i):
        print('Entring Integrator RHS')

        self._set_time(t)

        # setting some common used values
        dof = self.dof
        P = self.permutaion_mat

        # Getting last time step states
        q0   = self._pos_history[i]
        qd0  = self._vel_history[i]
        qdd0 = self._acc_history[i]

        # Getting independent states using premutation matrix
        v0   = self.get_indpenednt_q(q0)
        vd0  = self.get_indpenednt_q(qd0)
        vdd0 = self.get_indpenednt_q(qdd0)

        # getting the new (n+1) independent position (v)
        v_bar = v0 + (h * vd0) + ((h**2/4) * vdd0)
        v = v_bar + (h**2/4) * vdd

        # getting the new (n+1) independent velocity (vd)
        vd_bar = vd0 + (h * vdd0)
        vd = vd_bar + (h/2) * vdd

        # Evaluating matrices for the RHS expression
        # 1) getting dependent coordinates u from v
#        guess = q0 #+ (qd0 * h) + (0.5* qdd0 *(h**2))
#        for c in range(self.dof): 
#            guess[np.argmax(self.independent_cols[:, c]), 0] = v[c,0]
#        self._newton_raphson(guess)
#        self._set_gen_coordinates(self._pos)
#        q = self._pos
#        qp = P@q
#        u = qp[:-dof, :]

        self._set_gen_coordinates(q0)
        J = super()._eval_jac_eq()
        Jp = J @ P.T
        Jv = Jp[:,-dof:]
        Ju = Jp[:,:-dof]
        H  = solve(-Ju, Jv)

        u0 = (P@q0)[:-dof, :]
        ud0 = (P@qd0)[:-dof, :]
        udd0 = (P@qdd0)[:-dof, :]
        u_guess = u0 + (h * ud0) + ((h**2/4) * udd0)
        constraint_res = super()._eval_pos_eq()
        u_delta = solve(Ju, -constraint_res)
        itr = 0
        while np.linalg.norm(u_delta) >1e-3:
            u_guess = u_guess + u_delta
            q = solve(P, np.concatenate([u_guess, v]))
            self._set_gen_coordinates(q)
            constraint_res = super()._eval_pos_eq()
            u_delta = solve(Ju, -constraint_res)
            if itr > 50:
                print("RHS Iterations exceded \n")
                raise ValueError("Integration Iterations exceded \n")
                break
            itr += 1
        
        q = solve(P, np.concatenate([u_guess, v]))
        self._set_gen_coordinates(q)

        # 2) getting dependent velocities ud from vd
        #J  = self._jac[:-dof]
        ud = H@vd
        qd = solve(P, np.concatenate([ud, vd]))
        self._set_gen_velocities(qd)

        # 3) evaluating the mass matrix and force vector
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        Qd = self._eval_acc_eq()

        Mp = P @ M @ P.T
        Qp = P@Qt

        Mvv = Mp[-dof:, -dof:]
        Mvu = Mp[-dof:, :-dof]
        Muu = Mp[:-dof, :-dof]
        Muv = Mp[:-dof, -dof:]
        
        Qv = Qp[-dof:]
        Qu = Qp[:-dof]

        # 4) getting the dependent accelerations (udd)
        udd = H@vdd + solve(Ju, -Qd)
        qdd = solve(P, np.concatenate([udd, vdd]))

        # 5) getting the lagrange multipliers
        lamda = solve(Ju.T, (Qu - Muv@vdd - Muu@udd))

        # 6) evaluating the error
        residual = Mvv@vdd + Mvu@udd + Jv.T@lamda - Qv

        M_hat = Mvv + (Mvu @ H) + H.T@(Muv + Muu@H)
        return residual, q, qd, qdd, lamda, M_hat

    def perform_coordinates_partition(self, M, J, Q):
        P   = self.permutaion_mat
        dof = self.dof
        
        Mp = P @ M @ P.T
        Qp = P@Q
        Jp = J@P.T
        
        self.Jv = Jp[:,-dof:]
        self.Ju = Jp[:,:-dof]
        
        self.H = -solve(Ju, Jv)

        self.Mvv = Mp[-dof:, -dof:]
        self.Mvu = Mp[-dof:, :-dof]
        self.Muu = Mp[:-dof, :-dof]
        self.Muv = Mp[:-dof, -dof:]
        
        self.Qv = Qp[-dof:]
        self.Qu = Qp[:-dof]

    def _extract_independent_coordinates(self):
        P = self.permutaion_mat
        self.independent_cols = P.T[:, self.model.nc:]
        self.dof = self.independent_cols.shape[1]
        self.independent_cord = [self._coordinates_indicies[np.argmax(self.independent_cols[:,i])] for i in range(self.dof) ]
    
    def _eval_augmented_matricies(self, q , qd):
        self._set_gen_coordinates(q)
        self._set_gen_velocities(qd)
        J  = super()._eval_jac_eq()
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        Qd = self._eval_acc_eq()
        return M, J, Qt, Qd
    
        
    def _solve_augmented_system(self, M, J, Qt, Qd):
        A = self.eval_coefficient_matrix(M, J)
        
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
        dof = self.dof
        P = self.permutaion_mat
        qp = P@q
        qv = qp[-dof:, :]
        return qv    
    
    def _eval_lagrange_multipliers(self, i):
        self._set_gen_coordinates(self._pos_history[i])
        return self._lgr_history[i]

    def eval_coefficient_matrix(self, M, J):
        z = np.zeros((self.model.nc, self.model.nc))
        u = np.concatenate([M, J.T], axis=1)
        l = np.concatenate([J, z], axis=1)
        A = np.concatenate([u, l], axis=0)
        return A


################################################################

class dds_solver5(abstract_solver):
    
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
        
        pos_t0 = self._pos_history[0]
        vel_t0 = self._vel_history[0]
    
        self.factorize_constraints_jacobian()
        self._extract_independent_coordinates()

        self._newton_raphson(pos_t0)

        M, J, Qt, Qd = self._eval_augmented_matricies(pos_t0, vel_t0)
        acc_t0, lamda_t0 = self._solve_augmented_system(M, J, Qt, Qd)        
        self._acc_history[0] = acc_t0
        self._lgr_history[0] = lamda_t0
        
        print('\nRunning System Dynamic Analysis:')
        i = 0
        while i != bar_length:
            t = time_array[i+1]
            progress_bar(bar_length, i, t0, t)
            self._set_time(t)
            self.factorize_constraints_jacobian(self._jac[:-self.dof])
            self._extract_independent_coordinates()
            self._solve_time_step(t, dt, i)
            i += 1            
        print('\n')
        self._creat_results_dataframes()

    def _solve_time_step(self, t, h, i):
        print('SOLVING TIME STEP (%s)'%i)
        print('======================')
        qdd0 = self._acc_history[i]
        lgr0 = self._lgr_history[i]

        fi0 = np.concatenate([qdd0, lgr0])

        q, qd, qdd, lamda = self.step(fi0, t, h, i)
        
        self._pos_history[i+1] = q
        self._vel_history[i+1] = qd
        self._acc_history[i+1] = qdd
        self._lgr_history[i+1] = lamda

        print('FINISHED TIME STEP (%s)'%i)
        print('=======================\n')


    def factorize_constraints_jacobian(self, jacobian=None):
        A = super()._eval_jac_eq() if jacobian is None else jacobian
        P, L, U = sc.linalg.lu(A.T)
        self.permutaion_mat  = P.T
    
    def step(self, fi, t, h, i):
        print('Entring Step Calculation')
        soln = self.integrator_rhs(fi, t, h, i)
        residual, q, qd, qdd, lamda, M_hat = soln
        delta = solve(M_hat, -residual)
        norm = np.linalg.norm(delta)
        itr = 0
        while norm >1e-3:
            print('Itr : %s'%itr)
            print('ResNorm = %s' %np.linalg.norm(residual))
            print('DelNorm = %s\n' %np.linalg.norm(delta))
            fi = fi + delta
            soln = self.integrator_rhs(fi, t, h, i)
            residual, q, qd, qdd, lamda, M_hat = soln
            delta = solve(M_hat, -residual)
            norm = np.linalg.norm(delta)
            if itr > 150:
                print("Integration Iterations exceded \n")
                raise ValueError("Integration Iterations exceded \n")
                break
            itr += 1
        return q, qd, qdd, lamda
        

    def integrator_rhs(self, fi, t, h, i):
        print('Entring Integrator RHS')

        self._set_time(t+h)

        qdd   = fi[:self.model.n]
        lamda = fi[self.model.n:]

        # Getting last time step states
        q0   = self._pos_history[i]
        qd0  = self._vel_history[i]
        qdd0 = self._acc_history[i]

        # getting the new (n+1) independent position (v)
        q_bar = q0 + (h * qd0) + ((h**2/4) * qdd0)
        q = q_bar + (h**2/4) * qdd

        # getting the new (n+1) independent velocity (vd)
        qd_bar = qd0 + (h * qdd0)
        qd = qd_bar + (h/2) * qdd

        v  = self.get_indpenednt_q(q)
        vd = self.get_indpenednt_q(qd)

        guess = q + (qd * h) + (0.5 * qdd * (h**2))
        for c in range(self.dof): 
            guess[np.argmax(self.independent_cols[:, c]), 0] = v[c,0]

        self._newton_raphson(guess)
        A  = self._jac
        q_mod = self._pos
        
        vel_rhs = self._eval_vel_eq(vd)
        qd_mod = solve(A, -vel_rhs)
        
        self._set_gen_coordinates(q_mod)
        self._set_gen_velocities(qd_mod)

        J  = A[:-self.dof]
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        Qd = self._eval_acc_eq()

        M_hat = self.eval_coefficient_matrix(M, J)
        residual = (M_hat @ fi) - np.concatenate([Qt, -Qd])

        return residual, q_mod, qd_mod, qdd, lamda, M_hat

    def _extract_independent_coordinates(self):
        P = self.permutaion_mat
        self.independent_cols = P.T[:, self.model.nc:]
        self.dof = self.independent_cols.shape[1]
        self.independent_cord = [self._coordinates_indicies[np.argmax(self.independent_cols[:,i])] for i in range(self.dof) ]
    
    def _eval_augmented_matricies(self, q , qd):
        self._set_gen_coordinates(q)
        self._set_gen_velocities(qd)
        J  = super()._eval_jac_eq()
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        Qd = self._eval_acc_eq()
        return M, J, Qt, Qd
    
        
    def _solve_augmented_system(self, M, J, Qt, Qd):
        A = self.eval_coefficient_matrix(M, J)
        
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
        dof = self.dof
        P = self.permutaion_mat
        qp = P@q
        qv = qp[-dof:, :]
        return qv    
    
    def _eval_lagrange_multipliers(self, i):
        self._set_gen_coordinates(self._pos_history[i])
        return self._lgr_history[i]

    def eval_coefficient_matrix(self, M, J):
        z = np.zeros((self.model.nc, self.model.nc))
        u = np.concatenate([M, J.T], axis=1)
        l = np.concatenate([J, z], axis=1)
        A = np.concatenate([u, l], axis=0)
        return A
