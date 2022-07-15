import cython
from cython.parallel import prange
import numpy as np
from libc.math cimport sqrt
import time
import pandas as pd
from scipy import constants

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

        acc[pos_idx,0] = 0
        acc[pos_idx,1] = 0
        acc[pos_idx,2] = 0
        phi[pos_idx] = 0

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
    for x in range(x_source):
        for y in range(y_source):
            dest[x,y] = source[x,y]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy2out2d(double[:,:] source, int index, int offset, double[:,:,:] dest):
    cdef Py_ssize_t x_source = source.shape[0]
    cdef Py_ssize_t y_source = source.shape[1]
    cdef int x
    cdef int y
    for x in range(x_source):
        for y in range(y_source):
            dest[index,x,y+offset] = source[x,y]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy2out1d(double[:] source, int index, int offset, double[:,:,:] dest):
    cdef Py_ssize_t x_source = source.shape[0]
    cdef int x
    cdef int y
    for x in range(x_source):
        dest[index,x,offset] = source[x]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add2d(double[:,:] source, double[:,:] dest, double mul = 1):
    cdef Py_ssize_t x_source = source.shape[0]
    cdef Py_ssize_t y_source = source.shape[1]
    cdef int x
    cdef int y
    for x in range(x_source):
        for y in range(y_source):
            dest[x,y] += source[x,y] * mul

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple c_evaluate(double[:,:] particles, double[:,:] velocities, double[:] masses, double[:,:] eval_pos = None, int steps = 0, double eps = 0, double G = 6.6743e-11, double dt = 1000):
    cdef Py_ssize_t n_particles = particles.shape[0]
    
    cdef double[:,:,:] part_out = np.zeros((steps + 1,n_particles,12),dtype=float)

    cdef double[:,:] current_part_pos = np.zeros((n_particles,3),dtype=float)
    cdef double[:,:] current_part_vel = np.zeros((n_particles,3),dtype=float)
    cdef double[:,:] current_part_acc = np.zeros((n_particles,3),dtype=float)
    cdef double[:] current_part_phi = np.zeros((n_particles),dtype=float)

    cdef double[:,:,:] eval_out
    cdef double[:,:] current_eval_acc
    cdef double[:] current_eval_phi

    cdef Py_ssize_t n_eval_pos
    if eval_pos == None:
        eval_out = None
    else:
        n_eval_pos = eval_pos.shape[0]
        eval_out = np.zeros((steps + 1,n_eval_pos,6),dtype=float)
        current_eval_acc = np.zeros((n_eval_pos,3),dtype=float)
        current_eval_phi = np.zeros((n_eval_pos),dtype=float)

    copy2d(particles,current_part_pos)
    copy2d(velocities,current_part_vel)

    phi_acc(current_part_pos,masses,current_part_pos,G,eps,current_part_acc,current_part_phi)

    cdef int i

    for i in range(n_particles):
        part_out[0,i,1] = i
        part_out[0,i,0] = 0

    copy2out2d(current_part_pos,0,2,part_out)
    copy2out2d(current_part_vel,0,5,part_out)
    copy2out2d(current_part_acc,0,8,part_out)
    copy2out1d(current_part_phi,0,11,part_out)

    if eval_pos != None:
        phi_acc(current_part_pos,masses,eval_pos,G,eps,current_eval_acc,current_eval_phi)

        for i in range(n_eval_pos):
            eval_out[0,i,1] = i
            eval_out[0,i,0,] = 0

        copy2out2d(current_eval_acc,0,2,eval_out)
        copy2out1d(current_eval_phi,0,5,eval_out)

    cdef int step

    for step in range(steps):

        for i in range(n_particles):
            part_out[step+1,i,0] = step+1
            part_out[step+1,i,1] = i

        add2d(current_part_vel,current_part_pos,0.5 * dt)
        add2d(current_part_acc,current_part_vel,dt)
        add2d(current_part_vel,current_part_pos,0.5 * dt)

        copy2out2d(current_part_pos,step+1,2,part_out)
        copy2out2d(current_part_vel,step+1,5,part_out)

        phi_acc(current_part_pos,masses,current_part_pos,G,eps,current_part_acc,current_part_phi)

        copy2out2d(current_part_acc,step+1,8,part_out)
        copy2out1d(current_part_phi,step+1,11,part_out)

        if eval_pos != None:
            for i in range(n_eval_pos):
                eval_out[step+1,i,0] = step+1
                eval_out[step+1,i,1] = i

            phi_acc(current_part_pos,masses,eval_pos,G,eps,current_eval_acc,current_eval_phi)
            copy2out2d(current_eval_acc,step+1,2,eval_out)
            copy2out1d(current_eval_phi,step+1,5,eval_out)
    
    return part_out,eval_out

def evaluate(particles,velocities,masses,eval_pos = None, steps = 0, eps = 0, G = constants.G,dt = 1000):
    first = time.perf_counter()

    part_out,eval_out = c_evaluate(particles,velocities,masses,eval_pos=eval_pos,steps=steps,eps=eps,G=G, dt = dt)

    second = time.perf_counter()

    stats = {"eval_time": second-first}

    part_out_array = np.asarray(part_out)

    part_out_array = np.reshape(part_out_array,(part_out_array.shape[0] * part_out_array.shape[1],part_out_array.shape[2]))
    part_out_df = pd.DataFrame(part_out_array,columns=["step","id","x","y","z","vx","vy","vz","ax","ay","az","phi"])
    part_out_df["step"] = part_out_df["step"].astype(int)
    part_out_df["id"] = part_out_df["id"].astype(int)

    if type(eval_pos) != type(None):
        eval_out_array = np.asarray(eval_out)
        eval_out_array = np.reshape(eval_out_array,(eval_out_array.shape[0] * eval_out_array.shape[1],eval_out_array.shape[2]))
        eval_out_df = pd.DataFrame(eval_out_array,columns=["step","id","ax","ay","az","phi"])
        eval_out_df["step"] = eval_out_df["step"].astype(int)
        eval_out_df["id"] = eval_out_df["id"].astype(int)

        return part_out_df,eval_out_df,stats

    return part_out_df,stats