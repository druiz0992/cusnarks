import numpy as np
cimport numpy as np
cimport types as ct

assert sizeof(ct.uint32_t) == sizeof(np.uint32)

cdef extern from "../cuda/bigint.hh":
    cdef cppclass C_BigInt "BigInt":
        C_BigInt(np.int32_t *, ct.uint32)
        void BigInt_ModAdd256()
        void retreive()

cdef class BigInt:
    cdef C_BigInt* g
    cdef int dim1

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.int32_t] arr):
        self.initbigint(arr)
    
    def initbigint(self, np.ndarray[ndim=1, dtype=np.int32_t] arr):
        self.dim1 = len(arr)
        self.g = new C_BigInt(&arr[0], self.dim1)

    def mod_add(self):
        self.g.BigInt_ModAdd256()

    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] a = np.zeros(self.dim1, dtype=np.int32)

        self.g.retreive()

        return a

