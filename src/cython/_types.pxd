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
// File name  : types.pxd
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Definition of basic data types for wrapper functions
// ------------------------------------------------------------------

"""


import numpy as np
cimport numpy as np

cdef extern from "types.h":
  
  # Types
  ctypedef unsigned int uint32_t
  ctypedef int int32_t

  #Constants 
  cdef uint32_t NWORDS_256BIT
  cdef uint32_t MAX_R1CSPOLY_NWORDS
  cdef uint32_t MAX_R1CSPOLYTMP_NWORDS

  ctypedef struct kernel_config_t:
        int blockD
        int gridD
        int smemS
        int kernel_idx
        int return_val
        int in_offset

  ctypedef struct vector_t:
      uint32_t *data
      uint32_t length
      uint32_t size

  ctypedef enum mod_t:
      MOD_FIELD, MOD_GROUP, MOD_N 

  ctypedef enum fft_size_t:
     FFT_SIZE_2 = 1, FFT_SIZE_4, FFT_SIZE_8, FFT_SIZE_16, FFT_SIZE_32, FFT_SIZE_1024, FFT_SIZE_1M, FFT_SIZE_N

  ctypedef enum fft_t:
    FFT_T_1D = 0, FFT_T_2D, FFT_T_3D,  FFT_T_4D,  FFT_T_N 

  ctypedef struct fft_params_t:
     fft_t fft_type
     uint32_t fft_N[8]
     uint32_t padding
     uint32_t levels
  
  ctypedef struct kernel_params_t:
      uint32_t premod
      uint32_t premul
      uint32_t in_length
      uint32_t out_length
      uint32_t stride
      fft_size_t fft_Nx
      uint32_t N_fftx
      fft_size_t fft_Ny
      uint32_t N_ffty
      uint32_t forward
      uint32_t padding_idx
      uint32_t as_mont
      mod_t midx


  ctypedef void (*kernel_cb)(uint32_t *out_vector_data,
                             uint32_t *in_vector_data,
                             kernel_params_t *params)


  ctypedef enum u256_callback_t:
     CB_U256_ADDM , CB_U256_SUBM, CB_U256_MOD, CB_U256_MULM, CB_U256_ADDM_REDUCE,  CB_U256_ADDM_REDUCE_SHFL, CB_U256_SHR1, CB_U256_SHL1, CB_U256_N

  ctypedef enum ec_callback_t:
   #CB_EC_LDR_ADD, CB_EC_LDR_DOUBLE, CB_EC_LDR_MUL, CB_EC_LDR_MAD, CB_EC_JAC_ADD, CB_EC_JAC_DOUBLE, CB_EC_JAC_MUL, CB_EC_JAC_MAD, CB_EC_N
   CB_EC_JACAFF_ADD, CB_EC_JAC_ADD, CB_EC_JACAFF_DOUBLE, CB_EC_JAC_DOUBLE, CB_EC_JAC_MUL, CB_EC_JAC_MUL1, CB_EC_JAC_MAD, CB_EC_JAC_MAD_SHFL, CB_EC_N
  
  ctypedef enum ec2_callback_t:
   CB_EC2_JACAFF_ADD, CB_EC2_JAC_ADD, CB_EC2_JACAFF_DOUBLE, CB_EC2_JAC_DOUBLE, CB_EC2_JAC_MUL, CB_EC2_JAC_MUL1, CB_EC2_JAC_MAD, CB_EC2_JAC_MAD_SHFL, CB_EC2_N
      
  ctypedef enum zpoly_callback_t:
   CB_ZPOLY_FFT32, CB_ZPOLY_IFFT32, CB_ZPOLY_MUL32, CB_ZPOLY_FFTN, CB_ZPOLY_IFFTN, CB_ZPOLY_MULN, 
   CB_ZPOLY_FFT2DX, CB_ZPOLY_FFT2DY, CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXXPREV, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY,
   CB_ZPOLY_ADD, CB_ZPOLY_SUB, CB_ZPOLY_SUBPREV, CB_ZPOLY_MULC, CB_ZPOLY_MULCPREV, CB_ZPOLY_MULK, CB_ZPOLY_MADPREV, CB_ZPOLY_ADDPREV,
   CB_ZPOLY_DIVSNARKS, CB_ZPOLY_N

  ctypedef struct cirbin_hfile_t:
     uint32_t nWords
     uint32_t nPubInputs
     uint32_t nOutputs
     uint32_t nVars
     uint32_t nConstraints
     uint32_t tformat
     uint32_t R1CSA_nWords
     uint32_t R1CSB_nWords
     uint32_t R1CSC_nWords
  

  ctypedef enum cirbin_hfile_offset_t:
      CIRBIN_H_NWORDS_OFFSET = 0, CIRBIN_H_NPUBINPUTS_OFFSET, CIRBIN_H_NOUTPUTS_OFFSET, 
      CIRBIN_H_NVARS_OFFSET, CIRBIN_H_NCONSTRAINTS_OFFSET, CIRBIN_H_FORMAT_OFFSET, CIRBIN_H_CONSTA_NWORDS_OFFSET,
      CIRBIN_H_CONSTB_NWORDS_OFFSET, CIRBIN_H_CONSTC_NWORDS_OFFSET, CIRBIN_H_N_OFFSET



_NWORDS_256BIT = NWORDS_256BIT
_MAX_R1CSPOLY_NWORDS = MAX_R1CSPOLY_NWORDS
_MAX_R1CSPOLYTMP_NWORDS = MAX_R1CSPOLYTMP_NWORDS
