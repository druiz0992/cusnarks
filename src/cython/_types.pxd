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

  ctypedef struct kernel_config_t:
        int blockD
        int gridD
        int smemS

  ctypedef struct vector_t:
      uint32_t *data
      uint32_t length
      uint32_t size

  ctypedef enum mod_t:
      MOD_FIELD, MOD_GROUP, MOD_N 

  ctypedef struct kernel_params_t:
      uint32_t premod
      uint32_t premul
      uint32_t in_length
      uint32_t out_length
      uint32_t stride
      mod_t midx


  ctypedef void (*kernel_cb)(uint32_t *out_vector_data,
                             uint32_t *in_vector_data,
                             kernel_params_t *params)

  ctypedef enum u256_callback_t:
     CB_U256_ADDM , CB_U256_SUBM, CB_U256_MOD, CB_U256_MULM, CB_U256_ADDM_REDUCE,  CB_U256_SHL1, CB_U256_N

  ctypedef enum ec_callback_t:
   CB_EC_LDR_ADD, CB_EC_LDR_DOUBLE, CB_EC_LDR_MUL, CB_EC_LDR_MAD, CB_EC_JAC_ADD, CB_EC_JAC_DOUBLE, CB_EC_JAC_MUL, CB_EC_JAC_MAD, CB_EC_N
  
  ctypedef enum ec2_callback_t:
   CB_EC2_JAC_ADD , CB_EC2_JAC_DOUBLE, CB_EC2_JAC_MUL, CB_EC2_JAC_MAD, CB_EC2_N
      
  ctypedef enum zpoly_callback_t:
   CB_ZPOLY_FFT, CB_ZPOLY_IFFT, CB_ZPOLY_N

_NWORDS_256BIT = NWORDS_256BIT
