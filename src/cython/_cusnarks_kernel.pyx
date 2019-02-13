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
from libc.stdlib cimport malloc, free

cimport _types as ct

from _cusnarks_kernel cimport C_CUSnarks
from cython cimport view

def get_mod(np.ndarray[ndim=1, dtype=np.uint32_t] p, np.ndarray[ndim=1, dtype=np.uint32_t] p_,
            np.ndarray[ndim=1, dtype=np.uint32_t] r, np.ndarray[ndim=1, dtype=np.uint32_t]r_):

    cdef ct.mod_info_t mod_info = ct.mod_info_t(p, p_, r, r_)

    return mod_info

cdef class CUSnarks:
    cdef C_CUSnarks* g
    cdef int dim

    def __cinit__(self, ct.mod_info_t *p, ct.uint32_t vec_len, ct.uint32_t in_size, ct.uint32_t out_size, ct.uint32_t seed=0):
        self.dim = vec_len
    
        cdef ct.mod_info_t mod_info
        mod_info.p = &p->p[0]
        mod_info.p_ = &p->p[0]
        mod_info.r = &p->r[0]
        mod_info.r_ = &p->r_[0]

        self.g = new C_CUSnarks(&mod_info, self.dim, in_size, out_size, seed)

    def __dealloc__(self):
        del self.g

    def kernelLaunch( void *func_p, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec, dict vdim_out,  ct.kernel_config_t *config, ...):
        """
          vdim_out['len']
          vdim_out['width']
        """
        if  in_vec.shape[0] > self.dim or out_vec_len > self.dim:
            return

        cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat = np.zeros(in_vec.shape[0] * in_vec.shape[1], dtype=np.uint32)
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat = np.zeros(vdim_out['len'] * vdim_out['width'], dtype=np.uint32)
        in_vec_flat = np.concatenate(in_vec)
        self.g.kernelLaunch(func_p, &out_vec_flat[0],&in_vec_flat[0], in_size, out_size, config, ...) 
        
        return np.reshape(out_vec_flat,(-1,vdim_out['width']))

    """
    def rand(self, np.ndarray[ndim=2, dtype=np.uint32_t ]samples):
        if samples.shape[1] != ct.NWORDS_256BIT:
           return

        cdef np.uint32_t [:] samples1d = samples.flatten()
        self.g.rand(&samples1d[0],samples.shape[0])

        return np.asarray(samples1d, dtype=np.uint32).reshape((-1,ct.NWORDS_256BIT))

    """
    def rand(self, ct.uint32_t n_samples, ct.uint32_t size):
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] samples = np.zeros(n_samples * ct.NWORDS_256BIT, dtype=np.uint32)
        self.g.rand(&samples[0],n_samples, size)

        return samples.reshape((-1,ct.NWORDS_256BIT))
     

    def getDeviceInfo():
       self.g.getDeviceInfo()
