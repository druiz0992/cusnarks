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
// File name  : _cusnarks_kernel.pyx
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Cusnarks cython wrapper
// ------------------------------------------------------------------

"""

import numpy as np
cimport numpy as np

cimport _types as ct


from _cusnarks_kernel cimport C_CUSnarks, C_U256
from cython cimport view

cdef class CUSnarks:
    cdef C_CUSnarks* _cusnarks_ptr
    cdef ct.uint32_t in_dim, in_size, out_dim, out_size

    def __cinit__(self, ct.uint32_t in_len, ct.uint32_t out_len=0,  ct.uint32_t in_size=0, ct.uint32_t out_size=0, ct.uint32_t seed=0):
        self.in_dim = in_len
        if out_len == 0:
           self.out_dim = self.in_dim
        if in_size == 0:
           self.in_size = in_len * sizeof(ct.uint32_t) * ct.NWORDS_256BIT
        if out_size == 0:
           self.out_size = self.out_dim * sizeof(ct.uint32_t) *ct.NWORDS_256BIT

        #self._cusnarks_ptr = new C_CUSnarks(self.in_dim, self.in_size, self.out_dim, self.out_size, seed, 0)

    #def __dealloc__(self):
    #    del self._cusnarks_ptr

    def kernelLaunch(self, ct.uint32_t kernel_idx, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec, dict config, dict params):
        cdef ct.vector_t out_v
        cdef ct.vector_t in_v
        
        out_v.length = params['length']
        in_v.length  = in_vec.shape[0]

        if  in_v.length > self.in_dim  or out_v.length > self.out_dim:
            return

        print "XXXXXXXXXXX"
        print out_v.length, out_v.size, in_v.length, in_v.size, in_vec.shape[0], in_vec.shape[1]

        cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat = np.zeros(in_v.length * in_vec.shape[1], dtype=np.uint32)
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat = np.zeros(out_v.length * in_vec.shape[1], dtype=np.uint32)
        in_vec_flat = np.concatenate(in_vec)
        out_v.data = <ct.uint32_t *>&out_vec_flat[0]
        in_v.data  = <ct.uint32_t *>&in_vec_flat[0]

        cdef ct.kernel_config_t kconfig
        kconfig.blockD = config['blockD']
        if 'gridD' in config:
            kconfig.gridD  = config.config['gridD']
        if 'smemS' in config:
            kconfig.smemS  = config.config['smemS']

        cdef ct.kernel_params_t kparams
        kparams.midx = params['midx']
        kparams.length = params['length']
        kparams.stride = params['stride']
        if 'premod' in params:
            kparams.premod = params['premod']

        self._cusnarks_ptr.kernelLaunch (kernel_idx, &out_v, &in_v, &kconfig, &kparams) 
        
        return np.reshape(out_vec_flat,(-1,in_vec.shape[1]))
    
    """
    def rand(self, np.ndarray[ndim=2, dtype=np.uint32_t ]samples):
        if samples.shape[1] != ct.NWORDS_256BIT:
           return

        cdef np.uint32_t [:] samples1d = samples.flatten()
        self._cusnarks_ptr.rand(&samples1d[0],samples.shape[0])

        return np.asarray(samples1d, dtype=np.uint32).reshape((-1,ct.NWORDS_256BIT))

    """
    def rand(self, ct.uint32_t n_samples):
        print n_samples, ct.NWORDS_256BIT
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] samples = np.zeros(n_samples * ct.NWORDS_256BIT, dtype=np.uint32)
        self._cusnarks_ptr.rand(&samples[0],n_samples)

        return samples.reshape((-1,ct.NWORDS_256BIT))
     

    def getDeviceInfo(self):
       self._cusnarks_ptr.getDeviceInfo()

cdef class U256 (CUSnarks):
    cdef C_U256* _u256_ptr
    #cdef C_U256* _ptr

    def __cinit__(self, ct.uint32_t in_len, ct.uint32_t out_len=0,  ct.uint32_t in_size=0, ct.uint32_t out_size=0, ct.uint32_t seed=0):
        if out_len == 0:
            out_len = in_len
        self._u256_ptr = new C_U256(in_len,seed)
        self._cusnarks_ptr = <C_CUSnarks *>self._u256_ptr

    def __dealloc__(self):
        del self._u256_ptr
    
    #def addm(self, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec, ct.mod_t mod_idx= ct.MOD_GROUP, ct.uint32_t premod=1):
#
        #if  in_vec.shape[1] != ct.NWORDS_256BIT or in_vec.shape[0] > self.dim:
            #return
#
        #cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat = np.zeros(in_vec.shape[0] * ct.NWORDS_256BIT, dtype=np.uint32)
        #cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat = np.zeros(in_vec.shape[0]/2 * ct.NWORDS_256BIT, dtype=np.uint32)
        #in_vec_flat = np.concatenate(in_vec)
        #self._u256_ptr.addm(&out_vec_flat[0],&in_vec_flat[0], in_vec.shape[0], mod_idx, premod)
       #
        #return np.reshape(out_vec_flat,(-1,ct.NWORDS_256BIT))
#
    #def subm(self, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec, ct.mod_t mod_idx = ct.MOD_GROUP, ct.uint32_t premod=1):
#
        #if  in_vec.shape[1] != ct.NWORDS_256BIT or in_vec.shape[0] > self.dim:
            #return
#
        #cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat = np.zeros(in_vec.shape[0] * ct.NWORDS_256BIT, dtype=np.uint32)
        #cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat = np.zeros(in_vec.shape[0]/2 * ct.NWORDS_256BIT, dtype=np.uint32)
        #in_vec_flat = np.concatenate(in_vec)
        #self._u256_ptr.subm(&out_vec_flat[0],&in_vec_flat[0], in_vec.shape[0], mod_idx, premod)
       #
        #return np.reshape(out_vec_flat,(-1,ct.NWORDS_256BIT))
#
    #def mod(self, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec, ct.mod_t mod_idx=ct.MOD_GROUP):
#
        #if  in_vec.shape[1] != ct.NWORDS_256BIT or in_vec.shape[0] > self.dim:
            #return
#
        #cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat = np.zeros(in_vec.shape[0] * ct.NWORDS_256BIT, dtype=np.uint32)
        #cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat = np.zeros(in_vec.shape[0] * ct.NWORDS_256BIT, dtype=np.uint32)
        #in_vec_flat = np.concatenate(in_vec)
        #self._u256_ptr.mod(&out_vec_flat[0],&in_vec_flat[0], in_vec.shape[0], mod_idx)
       #
        #return np.reshape(out_vec_flat,(-1,ct.NWORDS_256BIT))
#
    #def mulm(self, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec, ct.mod_t mod_idx = ct.MOD_GROUP, ct.uint32_t premod=1):
#
        #if  in_vec.shape[1] != ct.NWORDS_256BIT or in_vec.shape[0] > self.dim:
            #return
#
        #cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat = np.zeros(in_vec.shape[0] * ct.NWORDS_256BIT, dtype=np.uint32)
        #cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat = np.zeros(in_vec.shape[0]/2 * ct.NWORDS_256BIT, dtype=np.uint32)
        #in_vec_flat = np.concatenate(in_vec)
        #self._u256_ptr.mulm(&out_vec_flat[0], &in_vec_flat[0], in_vec.shape[0], mod_idx, premod)
       #
        #return np.reshape(out_vec_flat,(-1,ct.NWORDS_256BIT))



