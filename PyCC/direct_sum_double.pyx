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
cdef void phi_acc(double[:,:] particles, double[:] masses, double G, double eps, double[:,:] acc, double[:] phi):
    cdef Py_ssize_t n_particles = particles.shape[0]
    cdef int pos_idx
    cdef int part_idx
    cdef double acc_mul
    cdef double dist
    for pos_idx in prange(n_particles,nogil=True):

        acc[pos_idx,0] = 0
        acc[pos_idx,1] = 0
        acc[pos_idx,2] = 0
        phi[pos_idx] = 0

        for part_idx in range(n_particles):
            dist = sqrt((particles[pos_idx,0] - particles[part_idx,0])**2 + (particles[pos_idx,1] - particles[part_idx,1])**2 + (particles[pos_idx,2] - particles[part_idx,2])**2 + eps**2)
            if dist != 0:
                acc_mul = G * masses[part_idx]/(dist**3)
                acc[pos_idx,0] += (particles[part_idx,0] - particles[pos_idx,0]) * acc_mul
                acc[pos_idx,1] += (particles[part_idx,1] - particles[pos_idx,1]) * acc_mul
                acc[pos_idx,2] += (particles[part_idx,2] - particles[pos_idx,2]) * acc_mul
                phi[pos_idx] += (-1) * G * (masses[part_idx] * masses[pos_idx])/(dist)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
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
@cython.cdivision(True)
@cython.nonecheck(False)
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
@cython.cdivision(True)
@cython.nonecheck(False)
cdef void copy2out1d(double[:] source, int index, int offset, double[:,:,:] dest):
    cdef Py_ssize_t x_source = source.shape[0]
    cdef int x
    cdef int y
    for x in range(x_source):
        dest[index,x,offset] = source[x]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
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
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double[:,:,:] c_evaluate(double[:,:] particles, double[:,:] velocities, double[:] masses, int steps = 0, double eps = 0, double G = 1, double dt = 1):
    cdef Py_ssize_t n_particles = particles.shape[0]
    
    cdef double[:,:,:] part_out = np.zeros((steps + 1,n_particles,12),dtype=np.float64)

    cdef double[:,:] current_part_pos = np.zeros((n_particles,3),dtype=np.float64)
    cdef double[:,:] current_part_vel = np.zeros((n_particles,3),dtype=np.float64)
    cdef double[:,:] current_part_acc = np.zeros((n_particles,3),dtype=np.float64)
    cdef double[:] current_part_phi = np.zeros((n_particles),dtype=np.float64)

    copy2d(particles,current_part_pos)
    copy2d(velocities,current_part_vel)

    phi_acc(current_part_pos,masses,G,eps,current_part_acc,current_part_phi)

    cdef int i

    for i in range(n_particles):
        part_out[0,i,1] = i
        part_out[0,i,0] = 0

    copy2out2d(current_part_pos,0,2,part_out)
    copy2out2d(current_part_vel,0,5,part_out)
    copy2out2d(current_part_acc,0,8,part_out)
    copy2out1d(current_part_phi,0,11,part_out)

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

        phi_acc(current_part_pos,masses,G,eps,current_part_acc,current_part_phi)

        copy2out2d(current_part_acc,step+1,8,part_out)
        copy2out1d(current_part_phi,step+1,11,part_out)
    
    return part_out

def evaluate(particles, velocities, masses, steps = 0, eps = 0, G = 1,dt = 1):
    first = time.perf_counter()

    part_out = c_evaluate(particles, velocities, masses, steps=steps, eps=eps, G=G, dt = dt)

    second = time.perf_counter()

    stats = {"eval_time": second-first}

    part_out_array = np.asarray(part_out)

    part_out_array = np.reshape(part_out_array,(part_out_array.shape[0] * part_out_array.shape[1],part_out_array.shape[2]))
    part_out_df = pd.DataFrame(part_out_array,columns=["step","id","x","y","z","vx","vy","vz","ax","ay","az","phi"])
    part_out_df["step"] = part_out_df["step"].astype(int)
    part_out_df["id"] = part_out_df["id"].astype(int)

    return part_out_df,stats