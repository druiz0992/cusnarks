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

cimport _types as ct

from _bigint cimport C_BigInt

assert sizeof(ct.uint32_t) == sizeof(np.uint32)

#_VWIDTH = VWIDTH

cdef class BigInt:
    cdef C_BigInt* g
    cdef int dim

    def __cinit__(self, np.ndarray[ndim=2, dtype=np.uint32_t] v,
                         np.ndarray[ndim=1, dtype=np.uint32_t] p):

        if  v.shape[1] != ct.NWORDS_256BIT:
            return

        self.dim = v.shape[0]
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] v2 = np.zeros(self.dim * v.shape[1], dtype=np.uint32)
        v2 = np.concatenate(v)

        self.g = new C_BigInt(&v2[0], &p[0], self.dim)

    def addm(self):
        self.g.addm()

    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] vout = np.zeros(self.dim * ct.NWORDS_256BIT, dtype=np.uint32)

        self.g.retrieve(&vout[0])

        return vout

