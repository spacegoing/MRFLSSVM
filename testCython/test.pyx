import numpy as np

cdef api void pythonAdd(int[] a):
    a[1] = 5
    a[0] = 4
    # pycaddhaha(a)

cdef api int* numpyArray(int rows, int cols):
    cdef int[:,:] arr = np.arange(rows*cols,dtype=np.int32).reshape([rows, cols])
    return &arr[0,0]
