import cython
from cython.parallel import prange
import numpy as np
from libc.math cimport sqrt
import time
import pandas as pd

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void phi_acc(double[:,:] particles, double[:] masses, double[:,:] pos, double G, double eps, double[:,:] acc, double[:] phi):
    cdef Py_ssize_t n_pos = pos.shape[0]
    cdef Py_ssize_t n_particles = particles.shape[0]
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
                acc[pos_idx,0] += (particles[part_idx,0] - pos[pos_idx,0]) * acc_mul
                acc[pos_idx,1] += (particles[part_idx,1] - pos[pos_idx,1]) * acc_mul
                acc[pos_idx,2] += (particles[part_idx,2] - pos[pos_idx,2]) * acc_mul
                phi[pos_idx] += (-1) * G * masses[part_idx]/dist

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy2d(double[:,:] source, double[:,:] dest):
    cdef Py_ssize_t x_source = source.shape[0]
    cdef Py_ssize_t y_source = source.shape[1]
    cdef int x
    cdef int y
    for x in prange(x_source,nogil=True):
        for y in range(y_source):
            dest[x,y] = source[x,y]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy2out2d(double[:,:] source, int index, double[:,:,:] dest):
    cdef Py_ssize_t x_source = source.shape[0]
    cdef Py_ssize_t y_source = source.shape[1]
    cdef int x
    cdef int y
    for x in prange(x_source,nogil=True):
        for y in range(y_source):
            dest[index,x,y] = source[x,y]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy2out1d(double[:] source, int index, double[:,:] dest):
    cdef Py_ssize_t x_source = source.shape[0]
    cdef int x
    cdef int y
    for x in prange(x_source,nogil=True):
        dest[index,x] = source[x]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add2d(double[:,:] source, double[:,:] dest, double mul = 1):
    cdef Py_ssize_t x_source = source.shape[0]
    cdef Py_ssize_t y_source = source.shape[1]
    cdef int x
    cdef int y
    for x in prange(x_source,nogil=True):
        for y in range(y_source):
            dest[x,y] += source[x,y] * mul

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple c_evaluate(double[:,:] particles, double[:,:] velocities, double[:] masses, double[:,:] eval_pos = None, int steps = 0, double eps = 0, double G = 6.6743e-11):

    cdef Py_ssize_t n_particles = particles.shape[0]
    cdef double[:,:,:] part_pos = np.zeros((steps + 1,n_particles,3),dtype=float)
    cdef double[:,:,:] part_vel = np.zeros((steps + 1,n_particles,3),dtype=float)
    cdef double[:,:,:] part_acc = np.zeros((steps + 1,n_particles,3),dtype=float)
    cdef double[:,:] part_phi = np.zeros((steps + 1,n_particles),dtype=float)

    cdef double[:,:] current_part_pos = np.zeros((n_particles,3),dtype=float)
    cdef double[:,:] current_part_vel = np.zeros((n_particles,3),dtype=float)
    cdef double[:,:] current_part_acc = np.zeros((n_particles,3),dtype=float)
    cdef double[:] current_part_phi = np.zeros((n_particles),dtype=float)

    cdef double[:,:,:] eval_acc
    cdef double[:,:] eval_phi
    cdef double[:,:] current_eval_acc
    cdef double[:] current_eval_phi

    cdef Py_ssize_t n_eval_pos
    if eval_pos == None:
        eval_acc = None
        eval_phi = None
    else:
        n_eval_pos = eval_pos.shape[0]
        eval_acc = np.zeros((steps + 1,n_eval_pos,3),dtype=float)
        eval_phi = np.zeros((steps + 1,n_eval_pos),dtype=float)
        current_eval_acc = np.zeros((n_eval_pos,3),dtype=float)
        current_eval_phi = np.zeros((n_eval_pos),dtype=float)

    copy2d(particles,current_part_pos)
    copy2d(velocities,current_part_vel)

    phi_acc(current_part_pos,masses,current_part_pos,G,eps,current_part_acc,current_part_phi)

    copy2out2d(current_part_pos,0,part_pos)
    copy2out2d(current_part_vel,0,part_vel)
    copy2out2d(current_part_acc,0,part_acc)
    copy2out1d(current_part_phi,0,part_phi)

    if eval_pos != None:
        phi_acc(current_part_pos,masses,eval_pos,G,eps,current_eval_acc,current_eval_phi)
        copy2out2d(current_eval_acc,0,eval_acc)
        copy2out1d(current_eval_phi,0,eval_phi)

    cdef int step

    for step in range(steps):
        add2d(current_part_vel,current_part_pos,0.5)
        add2d(current_part_acc,current_part_vel)
        add2d(current_part_vel,current_part_pos,0.5)

        copy2out2d(current_part_pos,step+1,part_pos)
        copy2out2d(current_part_vel,step+1,part_vel)

        phi_acc(current_part_pos,masses,current_part_pos,G,eps,current_part_acc,current_part_phi)

        copy2out2d(current_part_acc,step+1,part_acc)
        copy2out1d(current_part_phi,step+1,part_phi)

        if eval_pos != None:
            phi_acc(current_part_pos,masses,eval_pos,G,eps,current_eval_acc,current_eval_phi)
            copy2out2d(current_eval_acc,step+1,eval_acc)
            copy2out1d(current_eval_phi,step+1,eval_phi)
    
    return (np.asarray(part_pos),np.asarray(part_vel),np.asarray(part_acc),np.asarray(part_phi)),(np.asarray(eval_acc),np.asarray(eval_phi))

def evaluate(particles,velocities,masses,eval_pos = None, steps = 0, eps = 0, G = 6.6743e-11):
    (part_pos,part_vel,part_acc,part_phi),(eval_acc,eval_phi) = c_evaluate(particles,velocities,masses,eval_pos=eval_pos,steps=steps,eps=eps,G=G)
    print(part_pos)
    return None,None