import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
import time
import pandas as pd

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef void phi_acc_at_pos_double(double[:,:] particles, double[:] masses, double[:,:] pos, double G, double eps, double[:,:] out):
    cdef Py_ssize_t n_particles = particles.shape[0]
    cdef Py_ssize_t n_pos = pos.shape[0]
    cdef int pos_idx
    cdef int part_idx
    cdef double acc_mul
    cdef double dist
    for pos_idx in prange(n_pos,nogil=True):

        out[pos_idx,0] = 0
        out[pos_idx,1] = 0
        out[pos_idx,2] = 0
        out[pos_idx,3] = 0

        for part_idx in range(n_particles):
            dist = sqrt((pos[pos_idx,0] - particles[part_idx,0])**2 + (pos[pos_idx,1] - particles[part_idx,1])**2 + (pos[pos_idx,2] - particles[part_idx,2])**2 + eps**2)
            if dist != 0:
                acc_mul = G * masses[part_idx]/(dist**3)
                out[pos_idx,0] += (particles[part_idx,0] - pos[pos_idx,0]) * acc_mul
                out[pos_idx,1] += (particles[part_idx,1] - pos[pos_idx,1]) * acc_mul
                out[pos_idx,2] += (particles[part_idx,2] - pos[pos_idx,2]) * acc_mul
                out[pos_idx,3] += (-1) * G * (masses[part_idx])/(dist)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef void phi_acc_at_pos_single(float[:,:] particles, float[:] masses, float[:,:] pos, float G, float eps, float[:,:] out):
    cdef Py_ssize_t n_particles = particles.shape[0]
    cdef Py_ssize_t n_pos = pos.shape[0]
    cdef int pos_idx
    cdef int part_idx
    cdef float acc_mul
    cdef float dist
    for pos_idx in prange(n_pos,nogil=True):

        out[pos_idx,0] = 0
        out[pos_idx,1] = 0
        out[pos_idx,2] = 0
        out[pos_idx,3] = 0

        for part_idx in range(n_particles):
            dist = sqrt((pos[pos_idx,0] - particles[part_idx,0])**2 + (pos[pos_idx,1] - particles[part_idx,1])**2 + (pos[pos_idx,2] - particles[part_idx,2])**2 + eps**2)
            if dist != 0:
                acc_mul = G * masses[part_idx]/(dist**3)
                out[pos_idx,0] += (particles[part_idx,0] - pos[pos_idx,0]) * acc_mul
                out[pos_idx,1] += (particles[part_idx,1] - pos[pos_idx,1]) * acc_mul
                out[pos_idx,2] += (particles[part_idx,2] - pos[pos_idx,2]) * acc_mul
                out[pos_idx,3] += (-1) * G * (masses[part_idx])/(dist)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef double[:,:,:] measure_phi_double(double[:,:,:] particles, double[:] masses, int[:] steps, double[:,:] pos, double G = 1, double eps = 0):
    cdef Py_ssize_t n_steps = steps.shape[0]
    cdef Py_ssize_t n_pos = pos.shape[0]
    cdef double[:,:,:] out = np.zeros((n_steps,n_pos,4),dtype=np.float64)

    cdef int i
    cdef int step
    for i in range(n_steps):
        step = steps[i]
        phi_acc_at_pos_double(particles[step],masses,pos,G,eps,out[i])
    
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef float[:,:,:] measure_phi_single(float[:,:,:] particles, float[:] masses, int[:] steps, float[:,:] pos, float G = 1, float eps = 0):
    cdef Py_ssize_t n_steps = steps.shape[0]
    cdef Py_ssize_t n_pos = pos.shape[0]
    cdef float[:,:,:] out = np.zeros((n_steps,n_pos,4),dtype=np.float32)

    cdef int i
    cdef int step
    for i in range(n_steps):
        step = steps[i]
        phi_acc_at_pos_single(particles[step],masses,pos,G,eps,out[i])
    
    return out

