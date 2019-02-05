"""
    Copyright 2018 0kims association.

    This file is part of cusnarks.

    cusnarks is a free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    cusnarks is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
    more details.

    You should have received a copy of the GNU General Public License along with
    cusnarks. If not, see <https://www.gnu.org/licenses/>.

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : bigint_pyx
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Big Integer wrapper function
// ------------------------------------------------------------------

"""

import numpy as np
cimport numpy as np
cimport types as ct

assert sizeof(ct.uint32_t) == sizeof(np.uint32)

cdef extern from "../cuda/bigint.h":
    cdef cppclass C_BigInt "BigInt":
        C_BigInt(ct.uint256_t *, ct.uint256_t *, ct.uint32_t)
        void addm()
        void retreive()
        cdef VWIDTH
        cdef XOFFSET
        cdef YOFFSET
        cdef ZOFFSET

_VWIDTH = VWIDTH

cdef class BigInt:
    cdef C_BigInt* g
    cdef int dim1

    def __cinit__(self, np.ndarray[ndim=2, dtype=np.uint32_t] arr, np.ndarray[ndim=1, dtype=np.uint32_t] p):
        self.initbigint(arr, p)
    
    def initbigint(self, np.ndarray[ndim=2, dtype=np.uint32_t] x,
                         np.ndarray[ndim=2, dtype=np.uint32_t] y, 
                         np.ndarray[ndim=1, dtype=np.uint32_t] p):
        if x.shape != y.shape pr x.shape[1] != ct.NWORDS_256BIT
            return 0

        self.dim1 = y.shape[0]
        cdef np.ndarray [ndim=2, dtype=np.uint32_t]z = np.zeros((3*self.dim1, ct.NWORDS_256BIT), dtyp=np.uint32_t)
        z[XOFFSET::VWIDTH,:] = x
        z[YOFFSET::VWIDTH,:] = y

        self.g = new C_BigInt(&z[0], &p[0], self.dim1)

    def addm(self):
        self.g.addm()

    def retreive(self):
        cdef np.ndarray[ndim=2, dtype=np.uint32_t] a = np.zeros((self.dim1, ct.NWORDS_256BIT), dtype=np.uint32_t)
        cdef np.ndarray[ndim=2, dtype=np.uint32_t] b = np.zeros((self.dim1, ct.NWORDS_256BIT), dtype=np.uint32_t)

        self.g.retreive(&a[0],self.dim1)

        b = a[ZOFFWSET::VWIDTH,:]

        return b

