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
// File name  : _u256.pyx
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   U256 Integer wrapper function
// ------------------------------------------------------------------

"""

import numpy as np
cimport numpy as np

cimport _types as ct

from _u256 cimport C_U256
from cython cimport view


cdef class U256:
    cdef C_U256* g
    cdef int dim

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.uint32_t] p, ct.uint32_t vec_len, ct.uint32_t seed=0):
        self.dim = vec_len

        self.g = new C_U256(&p[0], self.dim, seed)

    def __dealloc__(self):
        del self.g

    def addm(self, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec, ct.uint32_t premod=1):

        if  in_vec.shape[1] != ct.NWORDS_256BIT or in_vec.shape[0] > self.dim:
            return

        cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat = np.zeros(in_vec.shape[0] * ct.NWORDS_256BIT, dtype=np.uint32)
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat = np.zeros(in_vec.shape[0]/2 * ct.NWORDS_256BIT, dtype=np.uint32)
        in_vec_flat = np.concatenate(in_vec)
        self.g.addm(&out_vec_flat[0],&in_vec_flat[0], in_vec.shape[0], premod)
       
        return np.reshape(out_vec_flat,(-1,ct.NWORDS_256BIT))

    def subm(self, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec, ct.uint32_t premod=1):

        if  in_vec.shape[1] != ct.NWORDS_256BIT or in_vec.shape[0] > self.dim:
            return

        cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat = np.zeros(in_vec.shape[0] * ct.NWORDS_256BIT, dtype=np.uint32)
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat = np.zeros(in_vec.shape[0]/2 * ct.NWORDS_256BIT, dtype=np.uint32)
        in_vec_flat = np.concatenate(in_vec)
        self.g.subm(&out_vec_flat[0],&in_vec_flat[0], in_vec.shape[0], premod)
       
        return np.reshape(out_vec_flat,(-1,ct.NWORDS_256BIT))

    def mod(self, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec):

        if  in_vec.shape[1] != ct.NWORDS_256BIT or in_vec.shape[0] > self.dim:
            return

        cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat = np.zeros(in_vec.shape[0] * ct.NWORDS_256BIT, dtype=np.uint32)
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat = np.zeros(in_vec.shape[0] * ct.NWORDS_256BIT, dtype=np.uint32)
        in_vec_flat = np.concatenate(in_vec)
        self.g.mod(&out_vec_flat[0],&in_vec_flat[0], in_vec.shape[0])
       
        return np.reshape(out_vec_flat,(-1,ct.NWORDS_256BIT))

    def mulm(self, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec, ct.uint32_t nprime, ct.uint32_t premod=1):

        if  in_vec.shape[1] != ct.NWORDS_256BIT or in_vec.shape[0] > self.dim:
            return

        cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat = np.zeros(in_vec.shape[0] * ct.NWORDS_256BIT, dtype=np.uint32)
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat = np.zeros(in_vec.shape[0]/2 * ct.NWORDS_256BIT, dtype=np.uint32)
        in_vec_flat = np.concatenate(in_vec)
        self.g.mulmont(&out_vec_flat[0], &in_vec_flat[0], in_vec.shape[0], nprime, premod)
       
        return np.reshape(out_vec_flat,(-1,ct.NWORDS_256BIT))


