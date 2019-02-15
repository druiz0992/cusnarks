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
//   Definition of basic data types and general constants
// ------------------------------------------------------------------

*/

#ifndef _TYPES_H_
#define _TYPES_H_

#define NWORDS_256BIT           (8)
#define NWORDS_256BIT_FIOS (NWORDS_256BIT + 3)
#define U256_XOFFSET            (0 * NWORDS_256BIT)
#define U256_YOFFSET            (1 * NWORDS_256BIT)
#define U256_NDIMS              (1)
#define U256K_OFFSET            (U256_NDIMS * NWORDS_256BIT)
#define ECP_XOFFSET             (1 * NWORDS_256BIT)
#define ECP_ZOFFSET             (2 * NWORDS_256BIT)
#define ECP_SCLOFFSET           (0 * NWORDS_256BIT)
#define ECPOINT_NDIMS           (2)
#define ECK_NDIMS               (ECPOINT_NDIMS + U256_NDIMS)
#define ECK_OFFSET              (ECK_NDIMS * NWORDS_256BIT)
#define CUSNARKS_BLOCK_DIM      (256)
#define CUSNARKS_MAX_NCB        (32)
#define U256_BLOCK_DIM          (256)

typedef unsigned int uint32_t;
typedef int int32_t;

// prime number info for finite fields
typedef struct {
   uint32_t p[NWORDS_256BIT];
   uint32_t p_[NWORDS_256BIT];
   uint32_t r_[NWORDS_256BIT];
   // r =  1 << 256
   // p * p_ - r * r_ = 1 

}mod_info_t;

// BN128 curve defition : Y^2 = X^3 + b
// Generator point G=(gx, gy) is on the curve
// gx = 1 -> I defined it as an array because i need to conver it to Montgomery??
// gy = 2
typedef struct {
  uint32_t b[NWORDS_256BIT];
  uint32_t g1x[NWORDS_256BIT];
  uint32_t g1y[NWORDS_256BIT];
  uint32_t g2x[2*NWORDS_256BIT];
  uint32_t g2y[2*NWORDS_256BIT];

}ecbn128_t;

// additional constants required
typedef struct {
  uint32_t twelve[NWORDS_256BIT];

}misc_const_t;
/**
 * Holds the parameters necessary to "launch" a CUDA kernel (i.e. schedule it for
 * execution on some stream of some device).
 */
typedef struct {
        int blockD;
        int gridD;
        int smemS;  // in bytes
} kernel_config_t;


// index to different primes used
typedef enum{
   MOD_GROUP = 0,
   MOD_FIELD,
   MOD_N

}mod_t;

// data vector
typedef struct{
  uint32_t *data;
  uint32_t length;
  uint32_t size;

}vector_t;


// kernel input parameters
typedef struct{
   uint32_t premod; // data requires to be mod-ded as preprocessing stage
   uint32_t length; // input data length (number of elements)
   uint32_t stride; // data elemements processed by thread
   mod_t    midx;   // index to prime number to be used by kernel

}kernel_params_t;


// kernel callback defition
typedef void (*kernel_cb)(uint32_t *out_vector_data,
                          uint32_t *in_vector_data,
                          kernel_params_t* params);

// index to u256 class kernels
typedef enum{
   CB_U256_ADDM = 0,
   CB_U256_SUBM,
   CB_U256_MOD,
   CB_U256_MULM,
   CB_U256_N

}u256_callback_t;

// index to ec128bn class kernels
typedef enum{
   CB_EC_ADD = 0,
   CB_EC_DOUBLE,
   CB_EC_MUL,
   CB_EC_ADDRED,
   CB_EC_MULRED,
   CB_EC_N

}ec_callback_t;

#endif
