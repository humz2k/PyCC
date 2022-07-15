import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
import time
from cython.view cimport array as cvarray

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double[:,:] test_cython(int xdim, int ydim):
    cdef double[:,:] array = cvarray(shape=(xdim,ydim), itemsize=sizeof(double), format="d")
    return array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double[:,:] test_np(int xdim, int ydim):
    cdef double[:,:] array = np.zeros((xdim,ydim),dtype=np.float64)
    return array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double[:,:] test_np_empty(int xdim, int ydim):
    cdef double[:,:] array = np.empty((xdim,ydim),dtype=np.float64)
    return array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef void check_indexing(double[:,:] array, int xdim, int ydim):
    cdef int i
    cdef int j
    for i in range(xdim):
        for j in range(ydim):
            array[i,j] = 1.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef void do_test_np(int xdim, int ydim):
    cdef double[:,:] arr = test_np(xdim,ydim)
    check_indexing(arr,xdim,ydim)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef void do_test_cy(int xdim, int ydim):
    cdef double[:,:] arr = test_cython(xdim,ydim)
    check_indexing(arr,xdim,ydim)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef void do_test_npe(int xdim, int ydim):
    cdef double[:,:] arr = test_np_empty(xdim,ydim)
    check_indexing(arr,xdim,ydim)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef void test(int xdim, int ydim):
    cdef double first
    cdef double second
    first = time.perf_counter()
    do_test_np(xdim,ydim)
    second = time.perf_counter()
    print(second-first)

    first = time.perf_counter()
    do_test_np(xdim,ydim)
    second = time.perf_counter()
    print(second-first)

    first = time.perf_counter()
    do_test_cy(xdim,ydim)
    second = time.perf_counter()
    print(second-first)

    first = time.perf_counter()
    do_test_npe(xdim,ydim)
    second = time.perf_counter()
    print(second-first)

    first = time.perf_counter()
    do_test_np(xdim,ydim)
    second = time.perf_counter()
    print(second-first)