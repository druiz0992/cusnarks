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

cimport _types as ct
cimport _utils_host as uh

from cython cimport view
from constants import *
from libc.stdlib cimport malloc, free

IF CUDA_DEF:
  from _cusnarks_kernel cimport C_CUSnarks, C_U256, C_ECBN128, C_EC2BN128, C_ZCUPoly


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
  
      def kernelLaunch(self, np.ndarray[ndim=2, dtype=np.uint32_t] in_vec, dict config, dict params, ct.uint32_t n_kernel=1):
          cdef ct.vector_t out_v
          cdef ct.vector_t in_v
         
          out_v.length = params['out_length']
          in_v.length  = in_vec.shape[0]
  
          #print in_v.length, self.in_dim, out_v.length, self.out_dim
          if  in_v.length > self.in_dim  or out_v.length > self.out_dim:
              assert False, "Incorrect arguments"
              return 0.0
  
          cdef np.ndarray[ndim=1, dtype=np.uint32_t] in_vec_flat
          cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec_flat
  
          # create kernel config data
          cdef ct.kernel_config_t *kconfig = <ct.kernel_config_t *> malloc(n_kernel * sizeof(ct.kernel_config_t))
  
          for i in range(n_kernel):
             kconfig[i].blockD = config['blockD'][i]
             kconfig[i].kernel_idx = config['kernel_idx'][i]
             # gridD and smemS and return_val do not need to exist
             if 'return_val' in config:
                 kconfig[i].return_val  = config['return_val'][i]
             else :
                 kconfig[i].return_val  = 1
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
  
          in_vec_flat = np.zeros(in_v.length * in_vec.shape[1], dtype=np.uint32)
          in_vec_flat = np.concatenate(in_vec)
          in_v.data  = <ct.uint32_t *>&in_vec_flat[0]
  
          out_vec_flat = np.zeros(out_v.length * in_vec.shape[1], dtype=np.uint32)
          out_v.data = <ct.uint32_t *>&out_vec_flat[0]
  
          # TODO :I am trying to represent input data as ndarray. I don't
          # know how other way to do this but to overwrite ndarray with input data
  
          # create kernel params data
          cdef ct.kernel_params_t *kparams = <ct.kernel_params_t *> malloc(n_kernel * sizeof(ct.kernel_params_t))
          for i in range(n_kernel):
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
  
          exec_time = self._cusnarks_ptr.kernelLaunch(&out_v, &in_v, kconfig, kparams, n_kernel) 
         
          kdata =  np.reshape(out_vec_flat,(-1,in_vec.shape[1]))
  
          free(kconfig)
          free(kparams)
  
          return kdata, exec_time
   
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
          self._cusnarks_ptr.saveFile(&samples[0],len(samples)/8, fname_c)
          
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
  
  
def montmult_h(np.ndarray[ndim=1, dtype=np.uint32_t] in_veca, np.ndarray[ndim=1, dtype=np.uint32_t] in_vecb, ct.uint32_t pidx):
        cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_vec = np.zeros(len(in_veca), dtype=np.uint32)

        uh.cmontmult_h(&out_vec[0], &in_veca[0], &in_vecb[0], pidx)
  
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
   
     uh.cntt_build_h(fft_params, nsamples)

     py_fft_params={}
     py_fft_params['fft_type'] = fft_params.fft_type
     py_fft_params['padding'] = fft_params.padding
     py_fft_params['levels'] = fft_params.levels
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] fft_sizes = np.zeros(1<<(ct.FFT_T_N-1), dtype=np.uint32)
     py_fft_params['fft_N'] = fft_sizes

     py_fft_params['fft_N'][0] = fft_params.fft_N[0]
     py_fft_params['fft_N'][1] = fft_params.fft_N[1]
     py_fft_params['fft_N'][2] = fft_params.fft_N[2]
     py_fft_params['fft_N'][3] = fft_params.fft_N[3]
     py_fft_params['fft_N'][4] = fft_params.fft_N[4]
     py_fft_params['fft_N'][5] = fft_params.fft_N[5]
     py_fft_params['fft_N'][6] = fft_params.fft_N[6]
     py_fft_params['fft_N'][7] = fft_params.fft_N[7]

     free(fft_params)

     return py_fft_params
    
    
def rangeu256_h(ct.uint32_t nsamples, np.ndarray[ndim=1, dtype=np.uint32_t] start, ct.uint32_t inc,
                                      np.ndarray[ndim=1, dtype=np.uint32_t] mod):

     cdef np.ndarray[ndim=1, dtype=np.uint32_t] out_samples = np.zeros(nsamples * ct.NWORDS_256BIT, dtype=np.uint32)
     
     uh.crangeu256_h(&out_samples[0], nsamples, &start[0], inc, &mod[0])
  
     return out_samples.reshape((-1,ct.NWORDS_256BIT))

def zpoly_maddm_h(np.ndarray[ndim=2, dtype=np.uint32_t] scldata, np.ndarray[ndim=1, dtype=np.uint32_t] pdata,
              ct.uint32_t ncoeff, ct.uint32_t last_idx, ct.uint32_t pidx):

     cdef np.ndarray[ndim=1, dtype=np.uint32_t] outd = np.zeros(ncoeff*8,dtype=np.uint32)
     cdef np.ndarray[ndim=1, dtype=np.uint32_t] sclflat = np.zeros(scldata.shape[0] * scldata.shape[1],dtype=np.uint32)
     sclflat = np.reshape(scldata,-1)
         
     uh.czpoly_maddm_h(&outd[0],&sclflat[0], &pdata[0], ncoeff, last_idx, pidx)
     #uh.cmaddm_h(&outd[0],&scldata[0], &pdata[0], last_idx, pout_d, pidx)

     return np.reshape(outd,(-1,8))

def zpoly_norm_h(np.ndarray[ndim=2, dtype=np.uint32_t] pin_data, ct.uint32_t pidx):
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] pin_data_flat = np.zeros(pin_data.shape[0] * pin_data.shape[1],dtype=np.uint32)
    pin_data_flat = np.reshape(pin_data,-1)
    cdef ct.uint32_t idx = uh.czpoly_norm_h(&pin_data_flat[0],pidx)

    return idx

def sortu256_idx_h(np.ndarray[ndim=2, dtype=np.uint32_t] vin):
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] vin_flat = np.zeros(vin.shape[0] * vin.shape[1],dtype=np.uint32)
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] idx_flat = np.zeros(vin.shape[0],dtype=np.uint32)

    vin_flat = np.reshape(vin,-1)

    uh.csortu256_idx_h(&idx_flat[0],&vin_flat[0],vin.shape[0])

    return idx_flat

def writeU256CircuitFile_h(np.ndarray[ndim=1, dtype=np.uint32_t] vin, bytes fname):
    uh.cwriteU256CircuitFile_h(&vin[0], <char *>fname, vin.shape[0])

def readU256CircuitFileHeader_h(bytes fname):
    cdef ct.cirbin_hfile_t header
    uh.creadU256CircuitFileHeader_h(&header, <char *>fname)
   
    header_d = {'nWords' : header.nWords,
                'nPubInputs' : header.nPubInputs,
                'nOutputs' : header.nOutputs,
                'nVars' : header.nVars,
                'nConstraints' : header.nConstraints,
                'constA_nWords' : header.constA_nWords,
                'constB_nWords' : header.constB_nWords,
                'constC_nWords' : header.constC_nWords }
    
    return header_d

def readU256CircuitFile_h(bytes fname):
    header_d = readU256CircuitFileHeader_h(<char *>fname)
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] cir_data = np.zeros(header_d['nWords'],dtype=np.uint32)
   
    uh.creadU256CircuitFile_h(&cir_data[0], <char *>fname, header_d['nWords'])

    return cir_data

