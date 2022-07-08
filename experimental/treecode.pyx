import cython
from cython.parallel import prange
import numpy as np
from libc.math cimport sqrt
import time
import pandas as pd
from scipy import constants

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:,:] divide_box(double[:,:] box):

    cdef double[:,:,:] new_boxes = np.zeros((8,3,2),dtype=float)

    cdef int i

    cdef double temp

    temp = (box[0,0] + box[0,1])/2

    for i in range(0,4):
        new_boxes[i,0,0] = box[0,0]
        new_boxes[i,0,1] = temp

    for i in range(4,8):
        new_boxes[i,0,0] = temp
        new_boxes[i,0,1] = box[0,1]

    temp = (box[1,0] + box[1,1])/2

    for i in range(0,4):
        new_boxes[i*2,1,0] = temp
        new_boxes[i*2,1,1] = box[1,1]

    for i in range(0,4):
        new_boxes[i*2+1,1,0] = box[1,0]
        new_boxes[i*2+1,1,1] = temp
    
    temp = (box[2,0] + box[2,1])/2
    
    new_boxes[0,2,0] = box[2,0]
    new_boxes[0,2,1] = temp
    new_boxes[1,2,0] = box[2][0]
    new_boxes[1,2,1] = temp

    new_boxes[2][2][0] = temp
    new_boxes[2][2][1] = box[2][1]
    new_boxes[3][2][0] = temp
    new_boxes[3][2][1] = box[2][1]

    new_boxes[4][2][0] = box[2][0]
    new_boxes[4][2][1] = temp
    new_boxes[5][2][0] = box[2][0]
    new_boxes[5][2][1] = temp

    new_boxes[6][2][0] = temp
    new_boxes[6][2][1] = box[2][1]
    new_boxes[7][2][0] = temp
    new_boxes[7][2][1] = box[2][1]

    return new_boxes

cpdef tuple separate_particles(double[:,:,:] boxes, double[:,:] particles, double[:] masses):

    cdef list sorted_particles = []
    cdef list sorted_masses = []

    cdef Py_ssize_t n_particles = particles.shape[0]
    cdef int[:] indexes = np.zeros((n_particles),dtype=np.int32)
    cdef int[:] n_parts_in_box = np.zeros((8),dtype=np.int32)

    cdef int i
    cdef int box
    for i in range(n_particles):
        for box in range(8):
            if (particles[i,0] >= boxes[box,0,0] and particles[i,0] <= boxes[box,0,1]) and (particles[i,1] >= boxes[box,1,0] and particles[i,1] <= boxes[box,1,1]) and (particles[i,2] >= boxes[box,2,0] and particles[i,2] <= boxes[box,2,1]):
                indexes[i] = box
                n_parts_in_box[box] += 1
                break

    for i in range(8):
        sorted_particles.append(np.zeros((n_parts_in_box[i],3),dtype=float))
        sorted_masses.append(np.zeros((n_parts_in_box[i]),dtype=float))
    
    cdef int temp
    for i in range(n_particles):
        n_parts_in_box[indexes[i]] -= 1
        temp = n_parts_in_box[indexes[i]]

        sorted_particles[indexes[i]][temp,0] = particles[i,0]
        sorted_particles[indexes[i]][temp,1] = particles[i,1]
        sorted_particles[indexes[i]][temp,2] = particles[i,2]
        sorted_masses[indexes[i]][temp] = masses[i]

    return sorted_particles,sorted_masses

cdef class Node:
    cdef double[:] center
    cdef double mass
    cdef list children

cpdef void build_tree():
    cdef Node a = Node()
    a.center = np.zeros((3),dtype=float)
    a.mass = 10.
    a.children = []