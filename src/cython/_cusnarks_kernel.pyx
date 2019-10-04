#cython: language_level=3
#cython.wraparound(False)
#cython.boundscheck(False)
#cython.nonecheck(False)

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
//   Cusnarks cython wrapper implementation
// ------------------------------------------------------------------

"""

import numpy as np
cimport numpy as np
import os

cimport _types as ct
cimport _utils_host as uh

from cython cimport view
from constants import *
from libc.stdlib cimport malloc, free
from libc.math cimport log2, ceil

IF CUDA_DEF:
  from _cusnarks_kernel cimport C_CUSnarks, C_U256, C_ECBN128, C_EC2BN128, C_ZCUPoly, C_AsyncBuf


  # CUSnarks class cython wrapper
  cdef class CUSnarks:
      cdef C_CUSnarks* _cusnarks_ptr
      cdef ct.uint32_t in_dim, in_size, out_dim, out_size, in_ndim, out_ndim
  
      def __cinit__(self, ct.uint32_t in_len, ct.uint32_t out_len=0,  ct.uint32_t in_size=0, ct.uint32_t out_size=0, ct.uint32_t seed=0):
          self.in_dim = in_len
          if out_len == 0:
             self.out_dim = self.in_dim
          if in_size == 0:
             self.in_size = in_len * sizeof(ct.uint32_t) * ct.NWORDS_256BIT
          if out_size == 0:
             self.out_size = self.out_dim * sizeof(ct.uint32_t) *ct.NWORDS_256BIT
 
      def kernelLaunch(self, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec, dict config, dict params, ct.uint32_t gpu_id=0,
                       ct.uint32_t stream_id=0, ct.uint32_t n_kernels=1):
          cdef ct.uint32_t i=0
          cdef ct.vector_t out_v
          cdef ct.vector_t in_v
         
          out_v.length = params['out_length']
          in_v.length  = in_vec.shape[0]
  
          #print (in_v.length, self.in_dim, out_v.length, self.out_dim)
          if  in_v.length > self.in_dim  or out_v.length > self.out_dim:
              print(in_v.length, self.in_dim, out_v.length, self.out_dim)
              assert False, "Incorrect arguments"
              return 0.0
   
          # create kernel config data
          cdef ct.kernel_config_t *kconfig = <ct.kernel_config_t *> malloc(n_kernels * sizeof(ct.kernel_config_t))
          
          for i in range(n_kernels):
             kconfig[i].blockD = config['blockD'][i]
             kconfig[i].kernel_idx = config['kernel_idx'][i]
             # gridD and smemS and return_val and return offset do not need to exist
             if 'return_val' in config:
                 kconfig[i].return_val  = config['return_val'][i]
             else :
                 kconfig[i].return_val  = 1
             if 'return_offset' in config:
                 kconfig[i].return_offset  = config['return_offset'][i]
             else :
                 kconfig[i].return_offset  = 0
             if 'gridD' in config:
                 kconfig[i].gridD  = config['gridD'][i]
             else :
                 kconfig[i].gridD  = 0
             if 'smemS' in config:
                 kconfig[i].smemS  = config['smemS'][i]
             else:
                 kconfig[i].smemS  = 0
             if 'in_offset' in config:
                 kconfig[i].in_offset  = config['in_offset'][i]
             else:
                 kconfig[i].in_offset = 0
  
          #Template case. Not working, but leave for now
          """
          print("allocating async buffer in_vec\n");
          cdef C_AsyncBuf [ct.uint32_t] *in_vec_flat = new C_AsyncBuf[ct.uint32_t](in_v.length  * in_vec.shape[1])
          print("Assigning value\n")
          in_vec_flat.setBuf(&in_vec[0,0], in_v.length*in_vec.shape[1])
          print("storing\n")
          in_v.data  = <ct.uint32_t *>in_vec_flat.getBuf()
  
          print("allocating async buffer out_vec\n");
          cdef C_AsyncBuf [ct.uint32_t] *out_vec_flat = new C_AsyncBuf[ct.uint32_t](out_v.length  * in_vec.shape[1])
          out_v.data = <ct.uint32_t *>out_vec_flat.getBuf()
  
          # TODO :I am trying to represent input data as ndarray. I don't
          # know how other way to do this but to overwrite ndarray with input data
  
          # create kernel params data
          #cdef ct.kernel_params_t *kparams = <ct.kernel_params_t *> malloc(n_kernels * sizeof(ct.kernel_params_t))
          print("allocating async buffer params\n");
          cdef C_AsyncBuf [ct.kernel_params_t] *kparams_buffer = new C_AsyncBuf[ct.kernel_params_t](n_kernels)
          cdef ct.kernel_params_t *kparams = <ct.kernel_params_t *>kparams_buffer.getBuf()
          """
          # create kernel params data
          cdef ct.kernel_params_t *kparams

          cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat_sync
          cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat_sync

          cdef C_AsyncBuf *in_vec_flat_async 
          cdef C_AsyncBuf *out_vec_flat_async 
          cdef C_AsyncBuf *kparams_buffer_async

          if stream_id == -1:

             in_vec_flat_sync = np.zeros(in_v.length * in_vec.shape[1], dtype=np.uint32)
             in_vec_flat_sync = np.concatenate(in_vec)
             in_v.data  = <ct.uint32_t *>&in_vec_flat_sync[0]
  
             out_vec_flat_sync = np.zeros(out_v.length * in_vec.shape[1], dtype=np.uint32)
             out_v.data = <ct.uint32_t *>&out_vec_flat_sync[0]
  
             kparams = <ct.kernel_params_t *> malloc(n_kernels * sizeof(ct.kernel_params_t))

          else :
             in_vec_flat_async = new C_AsyncBuf(in_v.length  * in_vec.shape[1], sizeof(ct.uint32_t))
             in_vec_flat_async.setBuf(&in_vec[0,0], in_v.length*in_vec.shape[1])
             in_v.data  = <ct.uint32_t *>in_vec_flat_async.getBuf()
  
             out_vec_flat_async = new C_AsyncBuf(out_v.length  * in_vec.shape[1], sizeof(ct.uint32_t))
             out_v.data = <ct.uint32_t *>out_vec_flat_async.getBuf()
  
             # create kernel params data
             kparams_buffer_async = new C_AsyncBuf(n_kernels, sizeof(ct.kernel_params_t))
             kparams = <ct.kernel_params_t *>kparams_buffer_async.getBuf()

          for i in range(n_kernels):
            kparams[i].midx = params['midx'][i]
            kparams[i].in_length = params['in_length'][i]
            kparams[i].out_length = params['out_length']
            kparams[i].stride = params['stride'][i]
            if 'premod' in params:
              kparams[i].premod = params['premod'][i]
            if 'premul' in params:
              kparams[i].premul = params['premul'][i]
            if 'fft_Nx' in params:
              kparams[i].fft_Nx = params['fft_Nx'][i]
            if 'fft_Ny' in params:
              kparams[i].fft_Ny = params['fft_Ny'][i]
            if 'N_fftx' in params:
              kparams[i].N_fftx = params['N_fftx'][i]
            if 'N_ffty' in params:
              kparams[i].N_ffty = params['N_ffty'][i]
            if 'forward' in params:
              kparams[i].forward = params['forward'][i]
            else:
              kparams[i].forward = 1
            if 'padding_idx' in params:
              kparams[i].padding_idx = params['padding_idx'][i]
            else:
              kparams[i].padding_idx = 0
            if 'as_mont' in params:
              kparams[i].as_mont = params['as_mont'][i]
            else:
              kparams[i].as_mont = 1

          cdef double exec_time = self._cusnarks_ptr.kernelLaunch(&out_v, &in_v, kconfig, kparams,gpu_id, stream_id, n_kernels) 
          cdef ct.uint32_t [:] kdata = <ct.uint32_t [:out_v.length * in_vec.shape[1]]>out_v.data

          if stream_id == -1:  
            free(kparams)

          free(kconfig)
  
          return np.copy(np.asarray(kdata).reshape(-1,in_vec.shape[1])), exec_time

      def streamDel(self, ct.uint32_t gpu_id, ct.uint32_t stream_id):
          self._cusnarks_ptr.streamDel(gpu_id, stream_id)

      def streamSync(self, ct.uint32_t gpu_id, ct.uint32_t stream_id):

          cdef double t=self._cusnarks_ptr.streamSync(gpu_id, stream_id)
          cdef ct.uint32_t vlen = self._cusnarks_ptr.streamGetOutputDataLen(gpu_id, stream_id)
          cdef ct.uint32_t *data = self._cusnarks_ptr.streamGetOutputData(gpu_id, stream_id)
          #cdef ct.uint32_t nwords_256bit = <int>(self.out_size/(self.out_dim*32))

          #cdef ct.uint32_t [:] kdata = <ct.uint32_t [:vlen * nwords_256bit * NWORDS_256BIT]>data
          cdef ct.uint32_t [:] kdata = <ct.uint32_t [:vlen * NWORDS_256BIT]>data
          cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_v = np.zeros(vlen * NWORDS_256BIT, dtype=np.uint32)

          # copy  C data to python data and delete C pointer
          out_v = np.copy(np.asarray(kdata))
          self._cusnarks_ptr.streamDel(gpu_id, stream_id)

          return np.asarray(out_v).reshape(-1, NWORDS_256BIT), t 
   
      def rand(self, ct.uint32_t n_samples):
          cdef np.ndarray[ndim=1, dtype=np.uint32_t] samples = np.zeros(n_samples * ct.NWORDS_256BIT, dtype=np.uint32)
          self._cusnarks_ptr.rand(&samples[0],n_samples)
  
          return samples.reshape((-1,ct.NWORDS_256BIT))
  
      def randu256(self, ct.uint32_t n_samples, np.ndarray[ndim=1, dtype=np.uint32_t] mod):
          cdef np.ndarray[ndim=1, dtype=np.uint32_t] samples = np.zeros(n_samples * ct.NWORDS_256BIT, dtype=np.uint32)
          self._cusnarks_ptr.randu256(&samples[0],n_samples, &mod[0])
  
          return samples.reshape((-1,ct.NWORDS_256BIT))
  
      def randu256(self, ct.uint32_t n_samples):
          cdef np.ndarray[ndim=1, dtype=np.uint32_t] samples = np.zeros(n_samples * ct.NWORDS_256BIT, dtype=np.uint32)
          self._cusnarks_ptr.randu256(&samples[0],n_samples, <ct.uint32_t *>0)
  
          return samples.reshape((-1,ct.NWORDS_256BIT))
       
      def saveFile(self, np.ndarray[ndim=1, dtype=np.uint32_t] samples, bytes fname):
          cdef char *fname_c = fname
          self._cusnarks_ptr.saveFile(&samples[0],<int>(len(samples)/8), fname_c)
          
      def getDeviceInfo(self):
         self._cusnarks_ptr.getDeviceInfo()
  
      #def __dealloc__(self):
          #del self._cusnarks_ptr
  
  # CU256 class cython wrapper
  cdef class U256 (CUSnarks):
      cdef C_U256* _u256_ptr
  
      def __cinit__(self, ct.uint32_t in_len, ct.uint32_t out_len=0,  ct.uint32_t in_size=0, ct.uint32_t out_size=0, ct.uint32_t seed=0):
          if out_len == 0:
              out_len = in_len
          self._u256_ptr = new C_U256(in_len,seed)
          self._cusnarks_ptr = <C_CUSnarks *>self._u256_ptr
  
      def __dealloc__(self):
          del self._u256_ptr
      
  # CUECBN128 class cython wrapper
  cdef class ECBN128 (CUSnarks):
      cdef C_ECBN128* _ecbn128_ptr
  
      def __cinit__(self, ct.uint32_t in_len, ct.uint32_t out_len=0,  ct.uint32_t in_size=0, ct.uint32_t out_size=0, ct.uint32_t seed=0):
          if out_len == 0:
              out_len = in_len
          self._ecbn128_ptr = new C_ECBN128( in_len,seed)
          self._cusnarks_ptr = <C_CUSnarks *>self._ecbn128_ptr
          self.in_dim = self.in_dim * 3
          self.out_dim = self.out_dim  * 3
          self.out_size = self.out_dim * sizeof(ct.uint32_t) *ct.NWORDS_256BIT
          self.in_size = self.in_dim * sizeof(ct.uint32_t) *ct.NWORDS_256BIT
     
      def __dealloc__(self):
          del self._ecbn128_ptr
  
  
  # CUEC2BN128 class cython wrapper
  cdef class EC2BN128 (CUSnarks):
      cdef C_EC2BN128* _ec2bn128_ptr
  
      def __cinit__(self, ct.uint32_t in_len, ct.uint32_t out_len=0,  ct.uint32_t in_size=0, ct.uint32_t out_size=0, ct.uint32_t seed=0):
          if out_len == 0:
              out_len = in_len
          self._ec2bn128_ptr = new C_EC2BN128( in_len,seed)
          self._cusnarks_ptr = <C_CUSnarks *>self._ec2bn128_ptr
          # TODO : add correct dimension
          self.in_dim = self.in_dim * 6
          self.out_dim = self.out_dim  * 6
          self.out_size = self.out_dim * sizeof(ct.uint32_t) *ct.NWORDS_256BIT
          self.in_size = self.in_dim * sizeof(ct.uint32_t) *ct.NWORDS_256BIT
     
      def __dealloc__(self):
          del self._ec2bn128_ptr
  
  
  
  # CUPoly class cython wrapper
  cdef class ZCUPoly (CUSnarks):
      cdef C_ZCUPoly* _zpoly_ptr
  
      def __cinit__(self, ct.uint32_t in_len, ct.uint32_t out_len=0,  ct.uint32_t in_size=0, ct.uint32_t out_size=0, ct.uint32_t seed=0):
          if out_len == 0:
              out_len = in_len
          self._zpoly_ptr = new C_ZCUPoly(in_len,seed)
          self._cusnarks_ptr = <C_CUSnarks *>self._zpoly_ptr
  
      def __dealloc__(self):
          del self._zpoly_ptr
  
  # C_AsyncBuf class cython wrapper
  """
  cdef class AsyncBuf:
      cdef void* buffer
      cdef ct.uint32_t max_nelems
      cdef ct.uint32_t el_size

      cdef C_AsyncBuf* _async_buffer_ptr
  
      def __cinit__(self, ct.uint32_t nelems, ct.uint32_t el_size):
          self._async_buffer_ptr = new C_AsyncBuf(nelems, el_size)
  
      def __dealloc__(self):
          if self._async_buffer_ptr != NULL:
            del self._async_buffer_ptr

      def  getBufUint32(self):
        cdef ct.uint32_t n_elems = self._async_buffer_ptr.getNelems()
        cdef ct.uint32_t [:] r= <ct.uint32_t [:n_elems]>self._async_buffer_ptr.getBuf()

        return np.asarray(r)
        
      def getBufKernelParams(self):       
        cdef ct.kernel_params_t *kparams = self._async_buffer_ptr.getBuf()
        dict kparams_dict = {}

        return kparams_dict

      def  getNelems(self):
        return self._async_buffer_ptr.getNelems()

  
      def  setBuf(self, np.ndarray[ndim=1, dtype=np.uint32_t] in_data, ct.uint32_t nelems):
        return self._async_buffer_ptr.setBuf(&in_data[0], nelems)
      """

def montsquare_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_veca, ct.uint32_t pidx):
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec = np.zeros(len(in_veca), dtype=np.uint32)

        uh.cmontsquare_h(&out_vec[0], &in_veca[0], pidx)
  
        return np.reshape(out_vec, (-1, NWORDS_256BIT))

def montmultN_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_veca, np.ndarray[ndim=1, dtype=np.uint32_t] in_vecb, ct.uint32_t pidx):
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec = np.zeros(len(in_veca), dtype=np.uint32)
        cdef ct.uint32_t n = <int>(len(in_veca)/NWORDS_256BIT)
        cdef ct.uint32_t i,offset=0

        for i in xrange(n):
           uh.cmontmult_h(&out_vec[offset], &in_veca[offset], &in_vecb[offset], pidx)
           offset += NWORDS_256BIT
  
        return np.reshape(out_vec, (-1, NWORDS_256BIT))

def montmult_neg_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_veca, np.ndarray[ndim=1, dtype=np.uint32_t] in_vecb, ct.uint32_t pidx):
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec = np.zeros(len(in_veca), dtype=np.uint32)
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] zero = np.zeros(len(in_veca), dtype=np.uint32)

        uh.cmontmult_h(&out_vec[0], &in_veca[0], &in_vecb[0], pidx)
        uh.csubm_h(&out_vec[0], &zero[0] , &out_vec[0], pidx)

        return out_vec

def addm_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_veca, np.ndarray[ndim=1, dtype=np.uint32_t] in_vecb, ct.uint32_t pidx):
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec = np.zeros(len(in_veca), dtype=np.uint32)

        uh.caddm_h(&out_vec[0], &in_veca[0], &in_vecb[0], pidx)
  
        return out_vec

def addu256_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_veca, np.ndarray[ndim=1, dtype=np.uint32_t] in_vecb):
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec = np.zeros(len(in_veca), dtype=np.uint32)

        uh.caddu256_h(&out_vec[0], &in_veca[0], &in_vecb[0])
  
        return out_vec

def subm_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_veca, np.ndarray[ndim=1, dtype=np.uint32_t] in_vecb, ct.uint32_t pidx):
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec = np.zeros(len(in_veca), dtype=np.uint32)

        uh.csubm_h(&out_vec[0], &in_veca[0], &in_vecb[0], pidx)
  
        return out_vec

def ntt_h(np.ndarray[ndim=2, dtype=np.uint32_t] in_A, 
          np.ndarray[ndim=2, dtype=np.uint32_t] in_roots, ct.uint32_t pidx):

      cdef ct.uint32_t n = in_A.shape[1]
      cdef ct.uint32_t L = int(np.log2(len(in_roots)))

      cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_roots_flat = np.zeros(in_roots.shape[0] * in_roots.shape[1], dtype=np.uint32)
      cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_A_flat = np.zeros(in_A.shape[0] * in_A.shape[1], dtype=np.uint32)

      in_roots_flat = np.concatenate(in_roots)
      in_A_flat = np.concatenate(in_A)

      uh.cntt_h(&in_A_flat[0], &in_roots_flat[0], L, pidx)

      return np.reshape(in_A_flat,(-1,n))

def intt_h(np.ndarray[ndim=2, dtype=np.uint32_t] in_A, 
          np.ndarray[ndim=2, dtype=np.uint32_t] in_roots, ct.uint32_t fmat, ct.uint32_t pidx):

      cdef ct.uint32_t n = in_A.shape[1]
      cdef ct.uint32_t L = int(np.log2(len(in_roots)))

      cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_roots_flat = np.zeros(in_roots.shape[0] * in_roots.shape[1], dtype=np.uint32)
      cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_A_flat = np.zeros(in_A.shape[0] * in_A.shape[1], dtype=np.uint32)

      in_roots_flat = np.concatenate(in_roots)
      in_A_flat = np.concatenate(in_A)

      uh.cintt_h(&in_A_flat[0], &in_roots_flat[0], fmat, L, pidx)

      return np.reshape(in_A_flat,(-1,n))

def find_roots_h (np.ndarray[ndim=1, dtype=np.uint32_t] in_proot, ct.uint32_t nroots, ct.uint32_t pidx):
 
      cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_roots_flat = np.zeros(nroots * len(in_proot), dtype=np.uint32)
      uh.cfind_roots_h(&out_roots_flat[0], &in_proot[0], nroots, pidx)

      return np.reshape(out_roots_flat,(-1, len(in_proot)), 0)

def ntt_parallel_h(np.ndarray[ndim=2, dtype=np.uint32_t] in_A, 
          np.ndarray[ndim=2, dtype=np.uint32_t] in_roots, ct.uint32_t Nrows, ct.uint32_t Ncols, ct.uint32_t pidx,
          ct.uint32_t mode=0):

      cdef ct.uint32_t n = in_A.shape[1]
      cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_roots_flat = np.zeros(in_roots.shape[0] * in_roots.shape[1], dtype=np.uint32)
      cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_A_flat = np.zeros(in_A.shape[0] * in_A.shape[1], dtype=np.uint32)

      in_roots_flat = np.concatenate(in_roots)
      in_A_flat = np.concatenate(in_A)

      uh.cntt_parallel_h(&in_A_flat[0], &in_roots_flat[0], Nrows, Ncols, pidx, mode)

      return np.reshape(in_A_flat,(-1,n))

def ntt_parallel2D_h(np.ndarray[ndim=2, dtype=np.uint32_t] in_A, 
          np.ndarray[ndim=2, dtype=np.uint32_t] in_roots, ct.uint32_t Nrows, ct.uint32_t fft_Ny, ct.uint32_t Ncols, ct.uint32_t fft_Nx, ct.uint32_t pidx, ct.uint32_t mode=0):

      cdef ct.uint32_t n = in_A.shape[1]
      cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_roots_flat = np.zeros(in_roots.shape[0] * in_roots.shape[1], dtype=np.uint32)
      cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_A_flat = np.zeros(in_A.shape[0] * in_A.shape[1], dtype=np.uint32)

      in_roots_flat = np.concatenate(in_roots)
      in_A_flat = np.concatenate(in_A)

      uh.cntt_parallel2D_h(&in_A_flat[0], &in_roots_flat[0], Nrows, fft_Ny, Ncols, fft_Nx, pidx, mode)

      return np.reshape(in_A_flat,(-1,n))

def ntt_build_h(ct.uint32_t nsamples):
     cdef ct.fft_params_t *fft_params = <ct.fft_params_t *> malloc(sizeof(ct.fft_params_t))
     cdef ct.uint32_t i

     uh.cntt_build_h(fft_params, nsamples)

     py_fft_params={}
     py_fft_params['fft_type'] = fft_params.fft_type
     py_fft_params['padding'] = fft_params.padding
     py_fft_params['levels'] = fft_params.levels
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] fft_sizes = np.zeros(1<<fft_params.fft_type, dtype=np.uint32)
     py_fft_params['fft_N'] = fft_sizes

     for i in xrange(1<<fft_params.fft_type):
        py_fft_params['fft_N'][i] = fft_params.fft_N[i]

     free(fft_params)

     return py_fft_params
    
    
def rangeu256_h(ct.uint32_t nsamples, np.ndarray[ndim=1, dtype=np.uint32_t] start, ct.uint32_t inc,
                                      np.ndarray[ndim=1, dtype=np.uint32_t] mod):

     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_samples = np.zeros(nsamples * ct.NWORDS_256BIT, dtype=np.uint32)
     
     uh.crangeu256_h(&out_samples[0], nsamples, &start[0], inc, &mod[0])
  
     return out_samples.reshape((-1,ct.NWORDS_256BIT))

def mpoly_eval_h(np.ndarray[ndim=2, dtype=np.uint32_t] scldata, np.ndarray[ndim=1, dtype=np.uint32_t] pdata,
              ct.uint32_t reduce_coeff, ct.uint32_t ncoeff, ct.uint32_t start_idx, ct.uint32_t last_idx, ct.uint32_t max_threads, ct.uint32_t pidx):

     cdef np.ndarray[ndim=1, dtype=np.uint32_t] outd = np.zeros(ncoeff*8,dtype=np.uint32)
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] sclflat = np.zeros(scldata.shape[0] * scldata.shape[1],dtype=np.uint32)
     sclflat = np.reshape(scldata,-1)
     cdef ct.mpoly_eval_t *args_c = <ct.mpoly_eval_t *> malloc(sizeof(ct.mpoly_eval_t))
         
     args_c.pout = &outd[0]
     args_c.scalar = &sclflat[0]
     args_c.pin = &pdata[0]
     args_c.reduce_coeff = reduce_coeff
     args_c.start_idx = start_idx
     args_c.last_idx = last_idx
     args_c.max_threads = max_threads
     args_c.pidx = pidx

     if args_c.max_threads == 0:
       uh.cmpoly_eval_h(args_c)
     else:
       with nogil:
         uh.cmpoly_eval_server_h(args_c)

     free(args_c)

     return np.reshape(outd,(-1,8))

def zpoly_norm_h(np.ndarray[ndim=2, dtype=np.uint32_t] pin_data):
    cdef ct.uint32_t ncoeff = pin_data.shape[0]
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] pin_data_flat = np.zeros(ncoeff * pin_data.shape[1],dtype=np.uint32)
    pin_data_flat = np.reshape(pin_data,-1)
    cdef ct.uint32_t idx = uh.czpoly_norm_h(&pin_data_flat[0],ncoeff)

    return idx

def sortu256_idx_h(np.ndarray[ndim=2, dtype=np.uint32_t] vin):
    if len(vin)==0:
      return np.asarray([],dtype=np.uint32)

    cdef np.ndarray[ndim=1, dtype=np.uint32_t] vin_flat = np.zeros(vin.shape[0] * vin.shape[1],dtype=np.uint32)
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] idx_flat = np.zeros(vin.shape[0],dtype=np.uint32)


    vin_flat = np.reshape(vin,-1)

    with nogil:
      uh.csortu256_idx_h(&idx_flat[0],&vin_flat[0],vin.shape[0])

    return idx_flat

def writeU256DataFile_h(np.ndarray[ndim=1, dtype=np.uint32_t] vin, bytes fname):
    cdef unsigned long long nWords = np.uint64(vin.shape[0])
    uh.cwriteU256DataFile_h(&vin[0], <char *>fname, nWords)

def writeWitnessFile_h(np.ndarray[ndim=1, dtype=np.uint32_t] vin, bytes fname):
    cdef unsigned long long vlen = vin.shape[0]/NWORDS_256BIT
    uh.cwriteWitnessFile_h(&vin[0], <char *>fname, vlen)

def readU256DataFile_h(bytes fname, ct.uint32_t insize, ct.uint32_t outsize):
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] vout = np.zeros(outsize * NWORDS_256BIT,dtype=np.uint32)

    uh.creadU256DataFile_h(&vout[0], <char *>fname, insize, outsize)

    return vout.reshape((-1,NWORDS_256BIT))

def readWitnessFile_h(bytes fname, ct.uint32_t fmt, unsigned long long insize):
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] vout = np.zeros(insize * NWORDS_256BIT,dtype=np.uint32)

    uh.creadWitnessFile_h(&vout[0], <char *>fname, fmt, insize)

    return vout.reshape((-1,NWORDS_256BIT))

def readU256CircuitFileHeader_h(bytes fname):
    cdef ct.cirbin_hfile_t header
    uh.creadU256CircuitFileHeader_h(&header, <char *>fname)
   
    header_d = {'nWords' : header.nWords,
                'nPubInputs' : header.nPubInputs,
                'nOutputs' : header.nOutputs,
                'nVars' : header.nVars,
                'nConstraints' : header.nConstraints,
                'cirformat' : header.cirformat,
                'R1CSA_nWords' : header.R1CSA_nWords,
                'R1CSB_nWords' : header.R1CSB_nWords,
                'R1CSC_nWords' : header.R1CSC_nWords }
    
    return header_d

def readU256CircuitFile_h(bytes fname):
    header_d = readU256CircuitFileHeader_h(<char *>fname)
    cdef unsigned long long nWords = np.uint64(os.path.getsize(fname)/4)
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] cir_data = np.zeros(nWords,dtype=np.uint32)
   
    uh.creadU256CircuitFile_h(&cir_data[0], <char *>fname, nWords)

    return cir_data

def readU256PKFile_h(bytes fname):
    header_d = readU256PKFileHeader_h(<char *>fname)
    cdef unsigned long long nWords = np.uint64(os.path.getsize(fname)/4)
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] pk_data = np.zeros(nWords,dtype=np.uint32)
   
    uh.creadU256PKFile_h(&pk_data[0], <char *>fname, nWords)

    return pk_data

def readU256PKFileHeader_h(bytes fname):
    cdef ct.pkbin_hfile_t header
    uh.creadU256PKFileHeader_h(&header, <char *>fname)
   
    header_d = {'nWords' : header.nWords,
                'ftype' : header.ftype,
                'protocol' : header.protocol,
                'Rbitlen' : header.Rbitlen,
                'k_binformat' : header.k_binformat,
                'k_ecformat' : header.k_ecformat,
                'nVars' : header.nVars,
                'nPublic' : header.nPublic,
                'domainSize' : header.domainSize }
    
    return header_d

def readR1CSFile_h(bytes filename):
    cdef ct.r1csv1_t *r1cs_header = <ct.r1csv1_t *> malloc(sizeof(ct.r1csv1_t))

    uh.creadR1CSFileHeader_h(r1cs_header, filename)
    
    cdef ct.uint32_t ncoeff_A, ncoeff_B, ncoeff_C
    ncoeff_A = r1cs_header.R1CSA_nCoeff
    ncoeff_B = r1cs_header.R1CSB_nCoeff
    ncoeff_C = r1cs_header.R1CSC_nCoeff

    cdef np.ndarray[ndim=1, dtype=np.uint32_t] r1csA_samples = np.zeros(1 + r1cs_header.nConstraints + ncoeff_A * (NWORDS_256BIT + 1),dtype=np.uint32)
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] r1csB_samples = np.zeros(1 + r1cs_header.nConstraints + ncoeff_B * (NWORDS_256BIT + 1),dtype=np.uint32)
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] r1csC_samples = np.zeros(1 + r1cs_header.nConstraints + ncoeff_C * (NWORDS_256BIT + 1),dtype=np.uint32)

    uh.creadR1CSFile_h(&r1csA_samples[0], filename, r1cs_header, ct.R1CSA_IDX)
    uh.creadR1CSFile_h(&r1csB_samples[0], filename, r1cs_header, ct.R1CSB_IDX)
    uh.creadR1CSFile_h(&r1csC_samples[0], filename, r1cs_header, ct.R1CSC_IDX)

    
    header = {}
    header['magic_number'] = r1cs_header.magic_number
    header['version'] = r1cs_header.version
    header['word_width_bytes'] = r1cs_header.word_width_bytes
    header['nVars'] = r1cs_header.nVars
    header['nPubOutputs'] = r1cs_header.nPubOutputs
    header['nPubInputs'] = r1cs_header.nPubInputs
    header['nPrivInputs'] = r1cs_header.nPrivInputs
    header['nConstraints'] = r1cs_header.nConstraints

    header['R1CSA_nCoeff'] = r1cs_header.R1CSA_nCoeff
    header['R1CSB_nCoeff'] = r1cs_header.R1CSB_nCoeff
    header['R1CSC_nCoeff'] = r1cs_header.R1CSC_nCoeff
    
    free(r1cs_header)

    return header, r1csA_samples, r1csB_samples, r1csC_samples
  
def r1cs_to_mpoly_len_h(np.ndarray[ndim=1, dtype=np.uint32_t] r1cs_len, dict header, ct.uint32_t extend):
    cdef ct.cirbin_hfile_t *header_c = <ct.cirbin_hfile_t *> malloc(sizeof(ct.cirbin_hfile_t))

    header_c.nWords = header['nWords']
    header_c.nPubInputs = header['nPubInputs']
    header_c.nOutputs = header['nOutputs']
    header_c.nVars = header['nVars']
    header_c.nConstraints = header['nConstraints']
    header_c.cirformat = header['cirformat']
    header_c.R1CSC_nWords = header['R1CSA_nWords']
    header_c.R1CSB_nWords = header['R1CSB_nWords']
    header_c.R1CSC_nWords = header['R1CSC_nWords']

    cdef np.ndarray[ndim=1, dtype=np.uint32_t] plen_out = np.zeros(header_c.nVars,dtype=np.uint32)
    uh.cr1cs_to_mpoly_len_h(&plen_out[0], &r1cs_len[0], header_c, extend)
    free(header_c)
     
    return plen_out

def r1cs_to_mpoly_h(np.ndarray[ndim=1, dtype=np.uint32_t] plen, 
                    np.ndarray[ndim=1, dtype=np.uint32_t] r1cs, dict header, ct.uint32_t to_mont, ct.uint32_t pidx, ct.uint32_t extend):

    cdef ct.cirbin_hfile_t *header_c = <ct.cirbin_hfile_t *> malloc(sizeof(ct.cirbin_hfile_t))
    #cdef np.ndarray[ndim=1, dtype=np.uint32_t] pout = np.zeros(pwords*(NWORDS_256BIT+1)+1,dtype=np.uint32)
    #cdef np.ndarray[ndim=1, dtype=np.uint32_t] pout = np.zeros(MAX_R1CSPOLY_NWORDS*NWORDS_256BIT,dtype=np.uint32)
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] pout = np.zeros(int(1 + np.sum(plen)*(NWORDS_256BIT+1)+ plen.shape[0]),dtype=np.uint32)

    header_c.nWords = header['nWords']
    header_c.nPubInputs = header['nPubInputs']
    header_c.nOutputs = header['nOutputs']
    header_c.nVars = header['nVars']
    header_c.nConstraints = header['nConstraints']
    header_c.cirformat = header['cirformat']
    header_c.R1CSC_nWords = header['R1CSA_nWords']
    header_c.R1CSB_nWords = header['R1CSB_nWords']
    header_c.R1CSC_nWords = header['R1CSC_nWords']

    pout[0] = header_c.nVars
    pout[1:header_c.nVars+1] = plen

    uh.cr1cs_to_mpoly_h(&pout[0], &r1cs[0], header_c, to_mont, pidx, extend)
    #ret_val = uh.cr1cs_to_mpoly_h(&pout[0], &r1cs[0], header_c, extend)
    free(header_c)
     
    #ncoeff = int(np.sum(pout[1:pout[0]+1])*(NWORDS_256BIT+2)+1)
    return pout

def mpoly_madd_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_veca,
                  np.ndarray[ndim=1, dtype=np.uint32_t] in_vecb, ct.uint32_t nVars, ct.uint32_t pidx):
        cdef np.ndarray[ndim=2, dtype=np.uint32_t] out_vec = np.zeros((nVars,NWORDS_256BIT), dtype=np.uint32)
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] tmp_vec = np.zeros(NWORDS_256BIT, dtype=np.uint32)
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] one = np.zeros(NWORDS_256BIT, dtype=np.uint32)
        cdef ct.uint32_t i, j,idx, v_offset=in_veca[0]+1, c_offset=1+in_veca[0], offset

        one[0] = 1;

        for j in xrange(nVars):
           if in_veca[j+1] == 0: 
              continue
           v_offset += in_veca[j+1]
           offset = in_veca[c_offset]*NWORDS_256BIT

           uh.cmontmult_h(&out_vec[j,0], &in_veca[v_offset], &in_vecb[offset], pidx)
           for i in xrange(in_veca[j+1]-1):
             v_offset += NWORDS_256BIT
             c_offset +=1
             offset = in_veca[c_offset]*NWORDS_256BIT
             uh.cmontmult_h(&tmp_vec[0], &in_veca[v_offset], &in_vecb[offset], pidx)
             uh.caddm_h(&out_vec[j,0], &out_vec[j,0], &tmp_vec[0], pidx)
  
           uh.cmontmult_h(&out_vec[j,0], &out_vec[j,0], &one[0], pidx)
           v_offset += NWORDS_256BIT
           c_offset = v_offset
        #TODO I could add n constraints here and not affect previous calculation
        """
        for j in xrange(nVars-1):
            in_veca[j+2]+= in_veca[j+1]
        """

        return out_vec

def evalLagrangePoly_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_t,
                       np.ndarray[ndim=1, dtype=np.uint32_t] in_l,
                       np.ndarray[ndim=2, dtype=np.uint32_t] in_roots, ct.uint32_t pidx):
     cdef np.ndarray[ndim=2, dtype=np.uint32_t] out_vec = np.zeros((len(in_roots),NWORDS_256BIT), dtype=np.uint32)
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] x = np.zeros(NWORDS_256BIT, dtype=np.uint32)
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] x_inv = np.zeros(NWORDS_256BIT, dtype=np.uint32)
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] l = in_l
     cdef int i,m = len(in_roots)

     for i in xrange(m):
        uh.csubm_h(&x[0],&in_t[0], &in_roots[i,0],pidx)
        uh.cmontinv_h(&x_inv[0],&x[0], pidx)
        uh.cmontmult_h(&out_vec[i,0],&l[0],&x_inv[0],pidx)
        uh.cmontmult_h(&l[0],&l[0],&in_roots[1,0],pidx)

     return out_vec.reshape((-1,NWORDS_256BIT))

def GrothSetupComputePS_h( np.ndarray[ndim=1, dtype=np.uint32_t]in_kA,
                        np.ndarray[ndim=1, dtype=np.uint32_t]in_kB,
                        np.ndarray[ndim=1, dtype=np.uint32_t]in_invD,
                        np.ndarray[ndim=2, dtype=np.uint32_t]in_veca,
                        np.ndarray[ndim=2, dtype=np.uint32_t]in_vecb,
                        np.ndarray[ndim=2, dtype=np.uint32_t]in_vecc, ct.uint32_t start, ct.uint32_t end, ct.uint32_t pidx):
     cdef ct.uint32_t s,n=0
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] t1 = np.zeros(NWORDS_256BIT, dtype=np.uint32)
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] t2 = np.zeros(NWORDS_256BIT, dtype=np.uint32)
     cdef np.ndarray[ndim=2, dtype=np.uint32_t] out_vec = np.zeros((end-start,NWORDS_256BIT), dtype=np.uint32)

     for s in xrange(start,end):
        uh.cmontmult_h(&t1[0],&in_veca[s,0],&in_kB[0], pidx)
        uh.cmontmult_h(&t2[0],&in_vecb[s,0],&in_kA[0], pidx)
        uh.caddm_h(&t1[0],&t1[0], &t2[0], pidx)
        uh.caddm_h(&t1[0],&t1[0], &in_vecc[s,0], pidx)
        uh.cmontmult_h(&out_vec[n,0],&t1[0],&in_invD[0], pidx)
        n+=1

     return out_vec


def GrothSetupComputeeT_h( np.ndarray[ndim=1, dtype=np.uint32_t]in_t,
                        np.ndarray[ndim=1, dtype=np.uint32_t]in_z,
                        ct.uint32_t maxH, ct.uint32_t pidx):
     cdef ct.uint32_t s
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] t1 = np.zeros(NWORDS_256BIT, dtype=np.uint32)
     cdef np.ndarray[ndim=2, dtype=np.uint32_t] out_vec = np.zeros((maxH,NWORDS_256BIT), dtype=np.uint32)

     t1 = np.copy(in_t)
     out_vec[0] = np.copy(in_z)

     for s in xrange(1,maxH):
        uh.cmontmult_h(&out_vec[s,0],&t1[0],&in_z[0], pidx)
        uh.cmontmult_h(&t1[0],&t1[0],&in_t[0], pidx)

     return out_vec

def ec_jac2aff_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_v, ct.uint32_t pidx, ct.uint32_t strip_last=0 ):
     cdef ct.uint32_t lenv = <int>(len(in_v)/(NWORDS_256BIT*ECP_JAC_OUTDIMS))
     cdef ct.uint32_t out_len
     if strip_last == 0:
         out_len = <int>len(in_v)
     else:
         out_len = <int> (lenv * 2 * NWORDS_256BIT)

     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec = np.zeros(out_len, dtype=np.uint32)
    
     with nogil:
       uh.cec_jac2aff_h(&out_vec[0],&in_v[0],lenv, pidx, strip_last)

     return out_vec.reshape((-1,NWORDS_256BIT))

def ec2_jac2aff_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_v, ct.uint32_t pidx, ct.uint32_t strip_last=0 ):
     cdef ct.uint32_t lenv = <int>(len(in_v)/(NWORDS_256BIT*ECP2_JAC_OUTDIMS))
     cdef ct.uint32_t out_len
     if strip_last == 0:
         out_len = <int>len(in_v)
     else:
         out_len = <int> (lenv * 4 * NWORDS_256BIT)
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec = np.zeros(out_len, dtype=np.uint32)
   
     with nogil: 
       uh.cec2_jac2aff_h(&out_vec[0],&in_v[0],lenv, pidx, strip_last)

     return out_vec.reshape((-1,NWORDS_256BIT))

def ec_jacdouble_h(np.ndarray[ndim=1, dtype = np.uint32_t] in_eca, ct.uint32_t pidx):
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_ecz = np.zeros(in_eca.shape[0], dtype=np.uint32)

     uh.cec_jacdouble_h(&out_ecz[0], &in_eca[0], pidx)

     return np.reshape(out_ecz,(-1, NWORDS_256BIT))

def ec_jacadd_h(np.ndarray[ndim=1, dtype = np.uint32_t] in_eca, 
                np.ndarray[ndim=1, dtype=np.uint32_t] in_ecb,  ct.uint32_t pidx):
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_ecz = np.zeros(in_eca.shape[0], dtype=np.uint32)

     uh.cec_jacadd_h(&out_ecz[0], &in_eca[0], &in_ecb[0], pidx)

     return np.reshape(out_ecz,(-1, NWORDS_256BIT))

def ec_jacscmul_h(np.ndarray[ndim=1, dtype = np.uint32_t] in_scl, 
                np.ndarray[ndim=1, dtype=np.uint32_t] in_eca,  ct.uint32_t pidx, ct.uint32_t add_last=0):
     cdef ct.uint32_t n
     if add_last:
        n= <int> (in_eca.shape[0]/(NWORDS_256BIT*ECP_JAC_INDIMS) )
     else:
        n = <int> (in_eca.shape[0]/(NWORDS_256BIT*ECP_JAC_OUTDIMS) )
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_ecz = np.zeros(n * NWORDS_256BIT * ECP_JAC_OUTDIMS, dtype=np.uint32)

     uh.cec_jacscmul_h(&out_ecz[0], &in_scl[0], &in_eca[0], n, pidx, add_last)

     return np.reshape(out_ecz,(-1, NWORDS_256BIT))

def ec_isoncurve_h(np.ndarray[ndim=1, dtype = np.uint32_t] in_p, ct.uint32_t is_affine, ct.uint32_t ec2, ct.uint32_t pidx):
     if ec2:
       return uh.cec2_isoncurve_h(&in_p[0], is_affine, pidx)
     else:
       return uh.cec_isoncurve_h(&in_p[0], is_affine, pidx)


def ec_jacaddreduce_h(np.ndarray[ndim=1, dtype = np.uint32_t] inv,
                             ct.uint32_t pidx, ct.uint32_t to_aff, ct.uint32_t add_in,
                             ct.uint32_t strip_last):

  cdef ct.uint32_t outdims = ECP_JAC_OUTDIMS
  cdef ct.uint32_t indims = ECP_JAC_OUTDIMS

  if strip_last:
      outdims = ECP_JAC_INDIMS

  if add_in:
      indims = ECP_JAC_INDIMS

  cdef np.ndarray[ndim=1, dtype=np.uint32_t] outv = np.zeros(outdims*NWORDS_256BIT, dtype=np.uint32)

  cdef ct.uint32_t n = <int> (inv.shape[0] / (indims*NWORDS_256BIT)  )

  uh.cec_jacaddreduce_h(&outv[0], &inv[0], n, pidx, to_aff, add_in, strip_last)

  return np.reshape(outv,(-1, NWORDS_256BIT))

def ec2_jacaddreduce_h(np.ndarray[ndim=1, dtype = np.uint32_t] inv,
                             ct.uint32_t pidx, ct.uint32_t to_aff, ct.uint32_t add_in,
                             ct.uint32_t strip_last):

  cdef ct.uint32_t outdims = ECP2_JAC_OUTDIMS
  cdef ct.uint32_t indims = ECP2_JAC_OUTDIMS

  if strip_last:
      outdims = ECP2_JAC_INDIMS

  if add_in:
      indims = ECP2_JAC_INDIMS

  cdef np.ndarray[ndim=1, dtype=np.uint32_t] outv = np.zeros(outdims*NWORDS_256BIT, dtype=np.uint32)

  cdef ct.uint32_t n = <int> (inv.shape[0] / (indims*NWORDS_256BIT) )

  uh.cec2_jacaddreduce_h(&outv[0], &inv[0], n, pidx, to_aff, add_in, strip_last)

  return np.reshape(outv,(-1, NWORDS_256BIT))



def mpoly_to_sparseu256_h(np.ndarray[ndim=1, dtype=np.uint32_t]in_mpoly):
    cdef list sp_poly_list=[]
    cdef dict sp_poly={}
    cdef ct.uint32_t n_keys=0
    cdef ct.uint32_t k,sumc=0
    cdef ct.uint32_t npoly = in_mpoly[0]
    cdef ct.uint32_t i,j, c_offset = 1 + npoly, v_offset = 1 + npoly , ncoeff 

    for i in xrange(npoly):
       ncoeff = in_mpoly[i+1]
       if ncoeff==0:
          #c_offset+=1
          sp_poly_list.append({})
          continue
       v_offset += ncoeff
       sp_poly={}
       n_keys=0
       for j in xrange(ncoeff):
           sumc=0
           for k in xrange(v_offset,v_offset+ct.NWORDS_256BIT):
              sumc += in_mpoly[k]
           if sumc != 0 or (ncoeff==1 and sumc == 0):
              sp_poly[str(in_mpoly[c_offset])] = in_mpoly[v_offset:v_offset+ct.NWORDS_256BIT]
              n_keys+=1
           c_offset +=1
           v_offset +=ct.NWORDS_256BIT
       if n_keys > 0:
          sp_poly_list.append(sp_poly)
       c_offset = v_offset
  
    return sp_poly_list

def to_montgomeryN_h(np.ndarray[ndim=1, dtype=np.uint32_t]in_v, ct.uint32_t pidx ):
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_v = np.zeros(in_v.shape[0], dtype=np.uint32)
     cdef ct.uint32_t n = <int>(in_v.shape[0]/ct.NWORDS_256BIT)

     with nogil:
       uh.cto_montgomeryN_h(&out_v[0], &in_v[0], n, pidx)

     return out_v.reshape((-1,ct.NWORDS_256BIT))
    
def from_montgomeryN_h(np.ndarray[ndim=1, dtype=np.uint32_t]in_v, ct.uint32_t pidx, ct.uint32_t strip_last ):
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_v = np.zeros(in_v.shape[0], dtype=np.uint32)
     cdef ct.uint32_t n = <int>(in_v.shape[0]/ct.NWORDS_256BIT)

     with nogil:
       uh.cfrom_montgomeryN_h(&out_v[0], &in_v[0], n, pidx, strip_last)

     return out_v.reshape((-1,ct.NWORDS_256BIT))

def ec_stripc_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_v):
     cdef ct.uint32_t n = <int>(in_v.shape[0]*2/3)
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_v = np.zeros(n, dtype=np.uint32)

     uh.cec_stripc_h(&out_v[0], &in_v[0], <int>(n/(2*NWORDS_256BIT)))

     return out_v.reshape((-1,ct.NWORDS_256BIT))

def ec2_stripc_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_v):
     cdef ct.uint32_t n = <int>(in_v.shape[0]*4/6)
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_v = np.zeros(n, dtype=np.uint32)

     uh.cec2_stripc_h(&out_v[0], &in_v[0], <int>(n/(4*NWORDS_256BIT)))

     return out_v.reshape((-1,ct.NWORDS_256BIT))

def field_roots_compute_h(ct.uint32_t nbits):
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_v = np.zeros((1<<nbits) * NWORDS_256BIT, dtype=np.uint32)
     
     uh.cfield_roots_compute_h(&out_v[0], nbits)
     return out_v.reshape((-1,ct.NWORDS_256BIT))
     

def mpoly_to_montgomery_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_vec, ct.uint32_t pidx):
     uh.cmpoly_to_montgomery_h(&in_vec[0], pidx)
     return

def mpoly_from_montgomery_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_vec, ct.uint32_t pidx):
     uh.cmpoly_from_montgomery_h(&in_vec[0], pidx)
     return

def init_h():
     uh.cinit_h()

def release_h():
    uh.crelease_h()

