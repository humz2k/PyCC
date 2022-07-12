import cython
from cython.parallel import prange
import numpy as np
from libc.math cimport sqrt,fabs
from libc.stdlib cimport malloc, free
import time
import pandas as pd
from scipy import constants

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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

cdef struct Node:
    double* center
    double size
    int id
    int start_index
    int n_particles
    double mass
    Node* children

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[:,:] sort_particles(double[:,:,:] boxes, int start_index, int id, Py_ssize_t n_particles, double[:,:] particle_array, int[:] indexes):
    cdef int i
    cdef int box
    cdef int total_parts = 0
    cdef int[:] n_parts_in_box = np.zeros((8),dtype=np.int32)
    cdef int[:] sort_array = np.zeros((8),dtype=np.int32)
    cdef int[:,:] info = np.zeros((8,2),dtype=np.int32)

    for i in range(start_index,n_particles):
        if indexes[i] != id:
            break
        for box in range(8):
            if (particle_array[i,0] >= boxes[box,0,0] and particle_array[i,0] < boxes[box,0,1]) and (particle_array[i,1] >= boxes[box,1,0] and particle_array[i,1] < boxes[box,1,1]) and (particle_array[i,2] >= boxes[box,2,0] and particle_array[i,2] < boxes[box,2,1]):
                indexes[i] += box + 1 + 8
                n_parts_in_box[box] += 1
                total_parts += 1
                break
    
    for i in range(7):
        sort_array[i+1] = n_parts_in_box[i] + sort_array[i]
    
    for i in range(8):
        info[i,0] = sort_array[i] + start_index
        info[i,1] = id + 1 + i + 8
    
    cdef double[:,:] swap_array = np.zeros((total_parts,4),dtype=float)
    cdef int[:] swap_index = np.zeros(total_parts,dtype=np.int32)

    for i in range(start_index,start_index+total_parts):
        box = sort_array[indexes[i] - 1 - 8 - id]
        swap_array[box,0] = particle_array[i,0]
        swap_array[box,1] = particle_array[i,1]
        swap_array[box,2] = particle_array[i,2]
        swap_array[box,3] = particle_array[i,3]
        swap_index[box] = indexes[i]
        sort_array[indexes[i] - 1 - 8 - id] += 1

    for i in range(start_index,start_index+total_parts):
        box = i - start_index
        particle_array[i,0] = swap_array[box,0]
        particle_array[i,1] = swap_array[box,1]
        particle_array[i,2] = swap_array[box,2]
        particle_array[i,3] = swap_array[box,3]
        indexes[i] = swap_index[box]

    return info

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Node c_build_tree(double size, double[:,:] particle_array, int[:] indexes, Py_ssize_t n_particles):
    cdef int[:,:] info

    cdef double[:,:] box = np.array([[-size,size],[-size,size],[-size,size]],dtype=float)

    cdef Node base_node = ret_node()

    base_node = make_node(base_node,0,0,n_particles,box,particle_array,indexes)

    return base_node

cdef Node ret_node():
    cdef Node out
    out.size = 0
    return out

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Node make_node(Node node, int id, int start_index, Py_ssize_t n_particles, double[:,:] box, double[:,:] particle_array, int[:] indexes):
    node.size = box[0,1] - box[0,0]
    node.center = <double *> malloc(3 * sizeof(double))
    node.center[0] = box[0,0] + node.size/2
    node.center[1] = box[1,0] + node.size/2
    node.center[2] = box[2,0] + node.size/2
    node.id = id
    node.mass = 0
    node.n_particles = 0
    node.start_index = start_index

    cdef int i
    for i in range(start_index,n_particles):
        if indexes[i] != id:
            break
        node.mass += particle_array[i,3]
        node.n_particles += 1
    
    if node.n_particles <= 1:
        return node

    cdef double[:,:,:] boxes = divide_box(box)

    cdef int[:,:] info = sort_particles(boxes,start_index,id,n_particles,particle_array,indexes)

    node.children = <Node *>malloc(8 * sizeof(Node))

    for i in range(8):
        node.children[i] = make_node(ret_node(),info[i,1],info[i,0],n_particles,boxes[i],particle_array,indexes)
    
    return node

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int phi_acc(Node base_node, double[:,:] particle_array, double[:,:] pos, double G, double eps, double theta, double[:,:] acc, double[:] phi):
    cdef Py_ssize_t n_pos = pos.shape[0]
    cdef int pos_idx
    cdef int part_idx
    cdef double acc_mul
    cdef double dist
    cdef int truncations = 0

    for pos_idx in prange(n_pos,nogil=True):

        acc[pos_idx,0] = 0
        acc[pos_idx,1] = 0
        acc[pos_idx,2] = 0
        phi[pos_idx] = 0

        traverse(base_node,particle_array,pos,pos_idx,G,eps,theta,acc,phi,pos_idx,&truncations)
    return truncations

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef void traverse(Node node, double[:,:] particle_array, double[:,:] pos, int pos_idx, double G, double eps, double theta, double[:,:] acc, double[:] phi, int idx, int* truncations) nogil:

    cdef double acc_mul
    
    if node.n_particles == 1:
        dist = sqrt((pos[pos_idx,0] - particle_array[node.start_index,0])**2 + (pos[pos_idx,1] - particle_array[node.start_index,1])**2 + (pos[pos_idx,2] - particle_array[node.start_index,2])**2)

        if dist != 0:
            if eps != 0:
                dist = sqrt(dist**2 + eps**2)
            
            acc_mul = G * particle_array[node.start_index,3]/(dist**3)
            acc[idx,0] += (particle_array[node.start_index,0] - pos[pos_idx,0]) * acc_mul
            acc[idx,1] += (particle_array[node.start_index,1] - pos[pos_idx,1]) * acc_mul
            acc[idx,2] += (particle_array[node.start_index,2] - pos[pos_idx,2]) * acc_mul
            phi[idx] += (-1) * G * particle_array[node.start_index,3]/dist

        return

    elif node.n_particles == 0:
        return

    else:
        dist = sqrt((pos[pos_idx,0] - node.center[0])**2 + (pos[pos_idx,1] - node.center[1])**2 + (pos[pos_idx,2] - node.center[2])**2)
    
        if dist != 0:
            if node.size**3/dist <= theta:
                truncations[0] += 1
                if eps != 0:
                    dist = sqrt(dist**2 + eps**2)
                acc_mul = G * node.mass/(dist**3)
                acc[idx,0] += (node.center[0] - pos[pos_idx,0]) * acc_mul
                acc[idx,1] += (node.center[1] - pos[pos_idx,1]) * acc_mul
                acc[idx,2] += (node.center[2] - pos[pos_idx,2]) * acc_mul
                phi[idx] += (-1) * G * node.mass/dist
                return
    
        for i in range(8):
            traverse(node.children[i],particle_array,pos,pos_idx,G,eps,theta,acc,phi,idx,truncations)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple evaluate(double[:,:] particles, double[:,:] velocities, double[:] masses, double[:,:] eval_pos = None, int steps = 0, double eps = 0, double G = 6.6743e-11, double dt = 1000, double theta = 0):
    cdef Py_ssize_t n_particles = particles.shape[0]

    cdef double radius = 0

    cdef double[:,:] particle_array = np.zeros((n_particles,4),dtype=float)
    cdef int[:] indexes = np.zeros(n_particles,dtype=np.int32)

    cdef int i
    for i in range(n_particles):
        particle_array[i,0] = particles[i,0]
        particle_array[i,1] = particles[i,1]
        particle_array[i,2] = particles[i,2]

        if fabs(particles[i,0]) > radius:
            radius = fabs(particles[i,0])
        if fabs(particles[i,1]) > radius:
            radius = fabs(particles[i,1])
        if fabs(particles[i,2]) > radius:
            radius = fabs(particles[i,2])
        
        particle_array[i,3] = masses[i]

    radius = radius * 1.01

    cdef Node base_node = c_build_tree(radius,particle_array,indexes, n_particles)

    cdef double[:,:] current_part_acc = np.zeros((n_particles,3),dtype=float)
    cdef double[:] current_part_phi = np.zeros((n_particles),dtype=float)

    cdef int truncs = phi_acc(base_node,particle_array,particles,G,eps,theta,current_part_acc,current_part_phi)

    return np.asarray(current_part_acc),np.asarray(current_part_phi),truncs