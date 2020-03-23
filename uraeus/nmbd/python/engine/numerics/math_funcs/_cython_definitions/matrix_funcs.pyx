#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: infertypes=True
#cython: initializedcheck=False
#cython: cdivision=True

cimport cython
cimport numpy as cnp
from cpython cimport array

import numpy as np
from numpy.linalg import multi_dot, norm
from cython.parallel import prange, parallel
from libc.math cimport fabs, exp


@cython.wraparound (False)
@cython.boundscheck(False)
cpdef E(double[:,:] p):
    
    cdef double e0 = p[0,0]
    cdef double e1 = p[1,0]
    cdef double e2 = p[2,0]
    cdef double e3 = p[3,0]
        
    m = np.empty((3,4),dtype=np.float64)
    cdef double[:,:] result = m
    
    result[0,0] = -e1
    result[0,1] =  e0
    result[0,2] = -e3
    result[0,3] =  e2
    
    result[1,0] = -e2
    result[1,1] =  e3
    result[1,2] =  e0
    result[1,3] = -e1
    
    result[2,0] = -e3
    result[2,1] = -e2
    result[2,2] =  e1
    result[2,3] =  e0

    return m

@cython.wraparound (False)
@cython.boundscheck(False)
cpdef G(double[:,:] p):
    
    cdef double e0 = p[0,0]
    cdef double e1 = p[1,0]
    cdef double e2 = p[2,0]
    cdef double e3 = p[3,0]
        
    m = np.empty((3,4),dtype=np.float64)
    cdef double[:,:] result = m
    
    result[0,0] = -e1
    result[0,1] =  e0
    result[0,2] =  e3
    result[0,3] = -e2
    
    result[1,0] = -e2
    result[1,1] = -e3
    result[1,2] =  e0
    result[1,3] =  e1
    
    result[2,0] = -e3
    result[2,1] =  e2
    result[2,2] = -e1
    result[2,3] =  e0

    return m

@cython.wraparound (False)
@cython.boundscheck(False)
cpdef A(double[:,:] p):
    m = E(p).dot(G(p).T)
    return m


@cython.wraparound (False)
@cython.boundscheck(False)
cpdef B(double[:,:] p, double[:,:] u):
    
    cdef double e0 = p[0,0]
    cdef double e1 = p[1,0]
    cdef double e2 = p[2,0]
    cdef double e3 = p[3,0]
    
    cdef double ux = u[0,0]
    cdef double uy = u[1,0]
    cdef double uz = u[2,0]
    
    m = np.empty((3,4),dtype=np.float64)
    cdef double[:,:] result = m
    
    result[0,0] = 2*e0*ux + 2*e2*uz - 2*e3*uy
    result[0,1] = 2*e1*ux + 2*e2*uy + 2*e3*uz
    result[0,2] = 2*e0*uz + 2*e1*uy - 2*e2*ux
    result[0,3] = -2*e0*uy + 2*e1*uz - 2*e3*ux
    
    result[1,0] = 2*e0*uy - 2*e1*uz + 2*e3*ux
    result[1,1] = -2*e0*uz - 2*e1*uy + 2*e2*ux
    result[1,2] = 2*e1*ux + 2*e2*uy + 2*e3*uz
    result[1,3] = 2*e0*ux + 2*e2*uz - 2*e3*uy
    
    result[2,0] = 2*e0*uz + 2*e1*uy - 2*e2*ux
    result[2,1] = 2*e0*uy - 2*e1*uz + 2*e3*ux
    result[2,2] = -2*e0*ux - 2*e2*uz + 2*e3*uy
    result[2,3] = 2*e1*ux + 2*e2*uy + 2*e3*uz

    return m


@cython.wraparound (False)
@cython.boundscheck(False)
cpdef skew_matrix(double[:,:] v):
    cdef double x = v[0,0]
    cdef double y = v[1,0]
    cdef double z = v[2,0]
    
    m = np.zeros((3,3),dtype=np.float64)
    cdef double[:,:] result = m
    
    result[0,1] = -z
    result[0,2] = y
    result[1,0] = z
    result[1,2] = -x
    result[2,0] = -y
    result[2,1] = x

    return m

@cython.wraparound (False)
@cython.boundscheck(False)
cpdef orthogonal_vector(double[:,:] v):
    
    dummy = np.ones((3,1),dtype=np.float64)
    cdef int i = np.argmax(np.abs(v))
    dummy[i] = 0
    m = multi_dot([skew_matrix(v),dummy])
    return m

@cython.wraparound (False)
@cython.boundscheck(False)
cpdef triad(double[:,:] v1, double[:,:] v2=None):
    cdef cnp.ndarray i
    cdef cnp.ndarray j
    cdef cnp.ndarray k
    
    k = v1/norm(v1)
    
    if v2 is not None:
        i = v2/norm(v2)
    else:
        i = orthogonal_vector(k)
        i = i/norm(i)

    j = multi_dot([skew_matrix(k),i])
    j = j/norm(j)
    
    m = np.empty((3,3),dtype=np.float64)
    cdef double[:,:] result = m

    cdef double[:,:] k_v = k
    cdef double[:,:] i_v = i
    cdef double[:,:] j_v = j

    
    result[0,0] = i[0,0]
    result[0,1] = j[0,0]              
    result[0,2] = k[0,0]
    
    result[1,0] = i[1,0]
    result[1,1] = j[1,0]
    result[1,2] = k[1,0]
    
    result[2,0] = i[2,0]
    result[2,1] = j[2,0]
    result[2,2] = k[2,0]
    
    return m

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef matrix_assembler(tuple data, int[:] rows, int[:] cols, tuple shape):
    
    cdef:
        int n_rows = shape[0]
        int i, k
        double[:] e_data = np.zeros((14*n_rows, ), dtype=np.float64)
        int[:] e_rows = np.zeros((14*n_rows, ), dtype=np.int32)
        int[:] e_cols = np.zeros((14*n_rows, ), dtype=np.int32)

        mat = np.zeros(shape, dtype=np.float64)
        double[:, ::1] view = mat
    
    k = sparse_assembler(data, rows, cols, e_data, e_rows, e_cols)
    construct_matrix(view, e_data, e_rows, e_cols, k)    
    return mat


cdef void construct_matrix(double[:,:] view, double[:] e_data, int[:] e_rows, int[:] e_cols, int k) nogil:
    cdef int i
    for i in prange(k, nogil=True):
        view[e_rows[i], e_cols[i]] = e_data[i]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int sparse_assembler(tuple blocks, int[:] b_rows, int[:] b_cols,
                       double[:] e_data, int[:] e_rows, int[:] e_cols):
    
    cdef:
        int row_counter = 0
        int nnz_counter = 0
        int prev_rows_size = 0
        int prev_cols_size = 0
        int nnz = len(b_rows)
        int v, i, j, vi, vj, m, n
        
        double value
        double[:,:] arr

    for v in range(nnz):
        vi = b_rows[v]
        vj = b_cols[v]
        
        if vi != row_counter:
            row_counter +=1
            prev_rows_size += m
            prev_cols_size  = 0
        
        arr = blocks[v]
        m = arr.shape[0]
        n = arr.shape[1]
                    
        if n == 3:
            prev_cols_size = 7*(vj//2)
        elif n == 4:
            prev_cols_size = 7*(vj//2) + 3
        
        nnz_counter = update_arrays(e_data, e_rows, e_cols, m, n, 
                      prev_rows_size, prev_cols_size, arr, 
                      nnz_counter)    
    return nnz_counter


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int update_arrays(double[:] e_data, int[:] e_rows, int[:] e_cols, 
                       int m, int n, int r, int c, double[:,:] arr, int nnz_counter) nogil:
        
    cdef:
        int i, j
        double value
    
    for i in range(m):
        for j in range(n):
            value = arr[i, j]
            if fabs(value)> 1e-5:
                e_rows[nnz_counter] = (r + i)
                e_cols[nnz_counter] = (c + j)
                e_data[nnz_counter] = (value)
                
                nnz_counter += 1
    return nnz_counter

