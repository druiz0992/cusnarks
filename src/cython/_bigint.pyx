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


cdef class BigInt:
    cdef C_BigInt* g
    cdef int dim

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.uint32_t] p, ct.uint32_t vec_len):
        self.dim = vec_len

        self.g = new C_BigInt(&p[0], self.dim)

    def addm(self, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec):

        if  in_vec.shape[1] != ct.NWORDS_256BIT or in_vec.shape[0] > self.dim:
            return

        cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat = np.zeros(in_vec.shape[0] * ct.NWORDS_256BIT, dtype=np.uint32)
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat = np.zeros(in_vec.shape[0]/2 * ct.NWORDS_256BIT, dtype=np.uint32)
        in_vec_flat = np.concatenate(in_vec)
        self.g.addm(&in_vec_flat[0], &out_vec_flat[0], in_vec.shape[0])
       
        return np.reshape(out_vec_flat,(-1,ct.NWORDS_256BIT))

