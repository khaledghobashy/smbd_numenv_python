# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:10:35 2020

@author: khaled.ghobashy
"""

import numpy as np

class integrator(object):

    def __init__(self, SSODE, y0, t0, t_end, h, **kwargs):

        self.SSODE = SSODE
        self.h = h
        self.t = t0

    def step(self):
        raise NotImplementedError

class RKMethods(integrator):

    n_stages = NotImplemented

    A = NotImplemented
    B = NotImplemented
    C = NotImplemented

    def __init__(self, SSODE, y0, t0, t_end, h, **kwargs):
        self.K = np.empty((self.n_stages + 1, y0.shape[0]))
        super().__init__(SSODE, y0, t0, t_end, h, **kwargs)

    def step(self, state_vector, i, fi):
        
        h = self.h
        t = self.t
        func = self.SSODE

        A = self.A
        B = self.B
        C = self.C
        K = self.K

        K[0] = fi.flat

        for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
            dy   = h * (K[:s].T @ a[:s][:, None])
            K[s] = func(state_vector + dy, t + c * h, i).flat
        
        yn = state_vector +  (h * (K[:-1].T @ B[:, None]))

        #print(yn, yn.shape)
        #f_new = func(yn, t + h, i)
        #K[-1] = f_new.flat

        self.t = t + h
        self.y = yn
    

class Explicit_RK4(RKMethods):

    n_stages = 4

    A = np.array([[0, 0, 0,   0],
                  [1/2, 0, 0, 0],
                  [0, 1/2, 0, 0],
                  [0,  0, 1,  0]])
    
    B = np.array([1/6, 1/3, 1/3, 1/6])
    C = np.array([0, 1/2, 1/2, 0])


class Explicit_RK2(RKMethods):

    n_stages = 2

    A = np.array([[0, 0],
                  [2/3, 0]])
    
    B = np.array([1/4, 3/4])
    C = np.array([0, 2/3])


class Explicit_RK23(RKMethods):

    n_stages = 3

    A = np.array([[0, 0, 0],
                  [1/2, 0, 0],
                  [0, 3/4, 0]])
    B = np.array([2/9, 1/3, 4/9])
    C = np.array([0, 1/2, 3/4])


class Explicit_RK45(RKMethods):

    n_stages = 6

    A = np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])

    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])


from scipy.optimize import fsolve

class Implicit_Trapezoidal(integrator):


    pass
