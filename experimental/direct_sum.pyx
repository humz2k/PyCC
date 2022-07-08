import cython
from cython.parallel import prange
import numpy as np
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] phi_acc(double[:,:] particles, double[:] masses, double[:,:] pos, double G, double eps):
    cdef Py_ssize_t n_pos = pos.shape[0]
    cdef Py_ssize_t n_particles = particles.shape[0]
    cdef double[:,:] out = np.zeros((n_pos,4),dtype=float)
    cdef int pos_idx
    cdef int part_idx
    cdef double acc_mul
    cdef double dist
    for pos_idx in prange(n_pos,nogil=True):
        for part_idx in range(n_particles):
            dist = sqrt((pos[pos_idx,0] - particles[part_idx,0])**2 + (pos[pos_idx,1] - particles[part_idx,1])**2 + (pos[pos_idx,2] - particles[part_idx,2])**2)
            if dist != 0:
                if eps != 0:
                    dist = sqrt(dist**2 + eps**2)
                acc_mul = G * masses[part_idx]/(dist**3)
                out[pos_idx,0] += (particles[part_idx,0] - pos[pos_idx,0]) * acc_mul
                out[pos_idx,1] += (particles[part_idx,1] - pos[pos_idx,1]) * acc_mul
                out[pos_idx,2] += (particles[part_idx,2] - pos[pos_idx,2]) * acc_mul
                out[pos_idx,3] += (-1) * G * masses[part_idx]/dist
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] phi(double[:,:] particles, double[:] masses, double[:,:] pos, double G, double eps):
    cdef Py_ssize_t n_pos = pos.shape[0]
    cdef Py_ssize_t n_particles = particles.shape[0]
    cdef double[:] out = np.zeros((n_pos),dtype=float)
    cdef int pos_idx
    cdef int part_idx
    cdef double acc_mul
    cdef double dist
    for pos_idx in prange(n_pos,nogil=True):
        for part_idx in range(n_particles):
            dist = sqrt((pos[pos_idx,0] - particles[part_idx,0])**2 + (pos[pos_idx,1] - particles[part_idx,1])**2 + (pos[pos_idx,2] - particles[part_idx,2])**2)
            if dist != 0:
                if eps != 0:
                    dist = sqrt(dist**2 + eps**2)
                acc_mul = G * masses[part_idx]/(dist**3)
                out[pos_idx] += (-1) * G * masses[part_idx]/dist
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy(double[:,:] source, double[:,:] dest):
    cdef Py_ssize_t x_source = source.shape[0]
    cdef Py_ssize_t y_source = source.shape[1]
    cdef int x
    cdef int y
    for x in prange(x_source,nogil=True):
        for y in range(y_source):
            dest[x,y] = source[x,y]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple evaluate(double[:,:] particles, double[:,:] velocities, double[:] masses, double[:,:] pos = None, int steps = 0, double eps = 0, double G = 6.6743e-11):
    cdef Py_ssize_t n_particles = particles.shape[0]
    cdef double[:,:,:] out_particles = np.zeros((steps + 1,n_particles,10),dtype=float)
    cdef double[:,:] current_part_pos = np.zeros((n_particles,3),dtype=float)
    cdef double[:,:] current_part_vel = np.zeros((n_particles,3),dtype=float)

    copy(particles,current_part_pos)
    copy(velocities,current_part_vel)

    cdef double[:,:,:] out_pos
    cdef Py_ssize_t n_pos
    if pos == None:
        out_pos = None
    else:
        n_pos = pos.shape[0]
        out_pos = np.zeros((steps + 1,n_pos,7),dtype=float)
    


    return out_particles,out_pos