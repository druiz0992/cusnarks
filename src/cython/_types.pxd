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
  ctypedef unsigned long long int t_uint64
  ctypedef long long int t_int64

  #Constants 
  cdef uint32_t NWORDS_256BIT
  cdef uint32_t MAX_R1CSPOLY_NWORDS
  cdef uint32_t MAX_R1CSPOLYTMP_NWORDS
  cdef uint32_t N_STREAMS_PER_GPU
  cdef uint32_t GROTH_PROOF_N_ECPOINTS
  cdef uint32_t MAX_NCORES_OMP
  cdef uint32_t U256_BSELM

  ctypedef struct kernel_config_t:
        int blockD
        int gridD
        int smemS
        int kernel_idx
        int return_val
        int in_offset
        int return_offset
        int n_kernels

  ctypedef struct vector_t:
      uint32_t *data
      uint32_t length
      uint32_t size

  ctypedef enum mod_t:
      MOD_FIELD, MOD_GROUP, MOD_N 

  ctypedef enum fft_mode_t:
      FFT_T_DIT, FFT_T_DIF, FFT_T_MODE_N

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
     CB_U256_ADDM , CB_U256_SUBM, CB_U256_MOD, CB_U256_MULM, CB_U256_ADDM_REDUCE,  CB_U256_ADDM_REDUCE_SHFL, CB_U256_SHR1, CB_U256_SHL1, CB_U256_SHL, CB_U256_ALMINV, CB_U256_MULM2, CB_U256_N

  ctypedef enum ec_callback_t:
   #CB_EC_LDR_ADD, CB_EC_LDR_DOUBLE, CB_EC_LDR_MUL, CB_EC_LDR_MAD, CB_EC_JAC_ADD, CB_EC_JAC_DOUBLE, CB_EC_JAC_MUL, CB_EC_JAC_MAD, CB_EC_N
   CB_EC_JACAFF_ADD, CB_EC_JAC_ADD, CB_EC_JACAFF_DOUBLE, CB_EC_JAC_DOUBLE, CB_EC_JAC_MUL, CB_EC_JAC_MUL1, CB_EC_JAC_MAD, CB_EC_JAC_MAD_SHFL, CB_EC_JAC_MUL_OPT, CB_EC_JAC_RED, CB_EC_JAC_MUL_PRECOMP, CB_EC_N
  
  ctypedef enum ec2_callback_t:
   CB_EC2_JACAFF_ADD, CB_EC2_JAC_ADD, CB_EC2_JACAFF_DOUBLE, CB_EC2_JAC_DOUBLE, CB_EC2_JAC_MUL, CB_EC2_JAC_MUL1, CB_EC2_JAC_MAD, CB_EC2_JAC_MAD_SHFL, CB_EC2_JAC_MUL_OPT, CB_EC2_JAC_RED, CB_EC2_N
      
  ctypedef enum zpoly_callback_t:
   CB_ZPOLY_FFT32, CB_ZPOLY_IFFT32, CB_ZPOLY_MUL32, CB_ZPOLY_FFTN, CB_ZPOLY_IFFTN, CB_ZPOLY_MULN, 
   CB_ZPOLY_FFT2DX, CB_ZPOLY_FFT2DY, 
   CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXXPREV, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY,
   CB_ZPOLY_INTERP3DXX, CB_ZPOLY_INTERP3DXY, CB_ZPOLY_INTERP3DYX, CB_ZPOLY_INTERP3DYY, CB_ZPOLY_INTERP3DFINISH,
   CB_ZPOLY_FFT4DXX, CB_ZPOLY_FFT4DXY, CB_ZPOLY_FFT4DYX, CB_ZPOLY_FFT4DYY,
   CB_ZPOLY_INTERP4DXX, CB_ZPOLY_INTERP4DXY, CB_ZPOLY_INTERP4DYX, CB_ZPOLY_INTERP4DYY, CB_ZPOLY_INTERP4DFINISH,
   CB_ZPOLY_ADD, CB_ZPOLY_SUB, CB_ZPOLY_SUBPREV, CB_ZPOLY_MULC, CB_ZPOLY_MULCPREV, CB_ZPOLY_MULK, CB_ZPOLY_MADPREV, CB_ZPOLY_ADDPREV,
   CB_ZPOLY_DIVSNARKS, CB_ZPOLY_N

  ctypedef struct cirbin_hfile_t:
     unsigned long long nWords
     unsigned long long nPubInputs
     unsigned long long nOutputs
     unsigned long long nVars
     unsigned long long nConstraints
     unsigned long long cirformat
     unsigned long long R1CSA_nWords
     unsigned long long R1CSB_nWords
     unsigned long long R1CSC_nWords
  

  ctypedef enum cirbin_hfile_offset_t:
      CIRBIN_H_NWORDS_OFFSET = 0, CIRBIN_H_NPUBINPUTS_OFFSET=2, CIRBIN_H_NOUTPUTS_OFFSET=4, 
      CIRBIN_H_NVARS_OFFSET=6, CIRBIN_H_NCONSTRAINTS_OFFSET=8, CIRBIN_H_FORMAT_OFFSET=10,
      CIRBIN_H_CONSTA_NWORDS_OFFSET=12, CIRBIN_H_CONSTB_NWORDS_OFFSET=14,
      CIRBIN_H_CONSTC_NWORDS_OFFSET=16, CIRBIN_H_N_OFFSET=18

  ctypedef struct pkbin_hfile_t:
     uint32_t nWords
     uint32_t ftype
     uint32_t protocol
     uint32_t Rbitlen
     uint32_t k_binformat
     uint32_t k_ecformat
     uint32_t nVars
     uint32_t nPublic
     uint32_t domainSize

  ctypedef struct mpoly_eval_t:
    uint32_t *pout
    const uint32_t *scalar
    uint32_t *pin
    uint32_t reduce_coeff
    uint32_t start_idx
    uint32_t last_idx
    uint32_t max_threads
    uint32_t thread_id
    uint32_t pidx

  ctypedef struct ntt_interpolandmul_t:
    uint32_t *A
    uint32_t *B
    uint32_t *roots
    uint32_t Nrows
    uint32_t Ncols
    uint32_t mNrows
    uint32_t mNcols
    uint32_t nroots
    uint32_t rstride
    uint32_t pidx
    uint32_t max_threads
    uint32_t start_idx
    uint32_t last_idx
    uint32_t thread_id

  ctypedef struct jacadd_reduced_t:
    uint32_t *out_ep
    uint32_t *scl
    uint32_t *x
    uint32_t n
    uint32_t ec2
    uint32_t *ec_table
    uint32_t pidx
    uint32_t max_threads
    uint32_t start_idx
    uint32_t last_idx
    uint32_t thread_id
    t_uint64 offset
    t_uint64 total_words 
    uint32_t order
    char *filename

  ctypedef struct r1csv1_t:
    uint32_t magic_number
    uint32_t version
    uint32_t word_width_bytes
    uint32_t nVars
    uint32_t nPubOutputs
    uint32_t nPubInputs
    uint32_t nPrivInputs
    uint32_t nLabels
    uint32_t nConstraints

    uint32_t R1CSA_nCoeff
    uint32_t R1CSB_nCoeff
    uint32_t R1CSC_nCoeff
    uint32_t constraintOffset
    t_int64  constraintLen

  ctypedef struct ec_table_offset_t:
    uint32_t table_order
    t_uint64 woffset_A
    t_uint64 woffset_B2
    t_uint64 woffset_B1
    t_uint64 woffset_C
    t_uint64 woffset_hExps
    t_uint64 nwords_tdata

  ctypedef enum r1cs_idx_t:
     R1CSA_IDX=0, R1CSB_IDX, R1CSC_IDX, R1CS_N_IDX
  
  ctypedef enum misc_const_len_t:
    MISC_K_1 = 0, MISC_K_INF = 2, MISC_K_INF2 = 5, MISC_K_N = 11 
  
  ctypedef enum shmem_t:
      SHMEM_T_WITNESS_32M = 0, SHMEM_T_WITNESS_64M, SHMEM_T_WITNESS_128M, SHMEM_T_N

  ctypedef enum snarks_file_t:
      SNARKSFILE_T_CIRCUIT = 0, SNARKSFILE_T_PK, SNARKSFILE_T_VK, SNARKSFILE_T_WITNESS,
      SNARKSFILE_T_PROOF, SNARKSFILE_T_PDATA, SNARKSFILE_T_N


  ctypedef enum pkbin_hfile_offset_t:
      PKBIN_H_NWORDS_OFFSET=0, PKBIN_H_FTYPE_OFFSET, PKBIN_H_PROTOCOL_OFFSET, PKBIN_H_RBITLEN_OFFSET,
      PKBIN_H_BINFORMAT_OFFSET, PKBIN_H_ECFORMAT_OFFSET, PKBIN_H_NVARS_OFFSET, PKBIN_H_NPUBLIC_OFFSET,
      PKBIN_H_DOMAINBITS_OFFSET, PKBIN_H_DOMAINSIZE_OFFSET, PKBIN_H_N_OFFSET
  
  ctypedef enum gpu_id_t:
    GPU_ID0 = 0, GPU_ID1, GPU_ID2, GPU_ID3

  ctypedef enum kernel_t:
     KERNEL_T_ZPOLY = 0, KERNEL_T_ECBN128_T, KERNEL_T_EC2BN128_T, KERNEL_T_N

_NWORDS_256BIT = NWORDS_256BIT
_MAX_NCORES_OMP = MAX_NCORES_OMP
_MAX_R1CSPOLY_NWORDS = MAX_R1CSPOLY_NWORDS
_MAX_R1CSPOLYTMP_NWORDS = MAX_R1CSPOLYTMP_NWORDS
_NSTREAMS_PER_GPU = N_STREAMS_PER_GPU
_GROTH_PROOF_N_ECPOINTS = GROTH_PROOF_N_ECPOINTS
_U256_BSELM = U256_BSELM
