/*
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
// File name  : types.h
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Definition of basic data types
// ------------------------------------------------------------------

*/

#ifndef _TYPES_H_
#define _TYPES_H_

#define NWORDS_256BIT      (8)
#define NWORDS_256BIT_FIOS (NWORDS_256BIT + 3)
#define U256_XOFFSET            (0 * NWORDS_256BIT)
#define U256_YOFFSET            (1 * NWORDS_256BIT)
#define U256_NDIMS              (1)
#define U256K_OFFSET            (U256_NDIMS * NWORDS_256BIT)
#define ECP_XOFFSET             (1 * NWORDS_256BIT)
#define ECP_ZOFFSET             (2 * NWORDS_256BIT)
#define ECP_SCLOFFSET           (0 * NWORDS_256BIT)
#define ECPOINT_NDIMS  (2)
#define ECK_NDIMS      (ECPOINT_NDIMS + U256_NDIMS)
#define ECK_OFFSET     (ECK_NDIMS * NWORDS_256BIT)
#define CUSNARKS_BLOCK_DIM  (256)
#define CUSNARKS_MAX_NCB = (32)
#define U256_BLOCK_DIM  (256)

typedef unsigned int uint32_t;
typedef int int32_t;

typedef struct {
   uint32_t p[NWORDS_256BIT];
   uint32_t p_[NWORDS_256BIT];
   uint32_t r_[NWORDS_256BIT];
   // r =  1 << 256
   // p * p_ - r * r_ = 1 

}mod_info_t;

/**
 * Holds the parameters necessary to "launch" a CUDA kernel (i.e. schedule it for
 * execution on some stream of some device).
 */
typedef struct {
        int blockD;
        int gridD;
        int smemS;  // in bytes
} kernel_config_t;

typedef enum{
   MOD_GROUP = 0,
   MOD_FIELD,
   MOD_N

}mod_t;

typedef struct{
  uint32_t *data;
  uint32_t length;
  uint32_t size;

}vector_t;

typedef struct{
   uint32_t premod; 
   uint32_t length;
   uint32_t stride;
   mod_t    midx;

}kernel_params_t;

typedef void (*kernel_cb)(uint32_t *out_vector_data,
                          uint32_t *in_vector_data,
                          kernel_params_t* params);

typedef enum{
   CB_U256_ADDM = 0,
   CB_U256_SUBM,
   CB_U256_MOD,
   CB_U256_MULM,
   CB_U256_N

}u256_callback_t;

typedef enum{
   CB_EC_ADD = 0,
   CB_EC_DOUBLE,
   CB_EC_MUL,
   CB_EC_ADDRED,
   CB_EC_MULRED,
   CB_EC_N

}ec_callback_t;

#endif
