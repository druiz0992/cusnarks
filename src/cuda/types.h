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
#define NWORDS_256BIT_SOS  ((NWORDS_256BIT) * 2 + 2)
#define PRIME_BASE           (30)
#define NBITS_WORD           (32)
#define MAX_R1CSPOLY_NWORDS  (10000000)
#define MAX_R1CSPOLYTMP_NWORDS  (100000)

#define U256_XOFFSET            (0 * NWORDS_256BIT)
#define U256_YOFFSET            (1 * NWORDS_256BIT)
#define U256_NDIMS              (1)
#define U256K_OFFSET            (U256_NDIMS * NWORDS_256BIT)

#define ECP_SCLOFFSET           (0 * NWORDS_256BIT)
#define ECK_INDIMS               (3)

// Jacobian
#define ECP_JAC_N256W                 (1)
#define ECP_JAC_INDIMS                (2) // X, Y
#define ECP_JAC_OUTDIMS               (3) // X, Y, Z
#define ECP_JAC_XOFFSET_BASE          (0)
#define ECP_JAC_YOFFSET_BASE          (1)
#define ECP_JAC_ZOFFSET_BASE          (2)
#define ECP_JAC_INXOFFSET             (0 * NWORDS_256BIT)
#define ECP_JAC_INYOFFSET             (1 * NWORDS_256BIT)
#define ECP_JAC_OUTXOFFSET            (0 * NWORDS_256BIT)
#define ECP_JAC_OUTYOFFSET            (1 * NWORDS_256BIT)
#define ECP_JAC_OUTZOFFSET            (2 * NWORDS_256BIT)
#define ECP_JAC_INOFFSET              (ECP_JAC_INDIMS * NWORDS_256BIT)
#define ECP_JAC_OUTOFFSET             (ECP_JAC_OUTDIMS * NWORDS_256BIT)

#define ECP2_JAC_N256W                 (2)
#define ECP2_JAC_INDIMS                (4) // X, Y
#define ECP2_JAC_OUTDIMS               (6) // X, Y, Z
#define ECP2_JAC_XOFFSET_BASE          (0)
#define ECP2_JAC_YOFFSET_BASE          (1)
#define ECP2_JAC_ZOFFSET_BASE          (2)
#define ECP2_JAC_INXOFFSET             (0 * 2 * NWORDS_256BIT)
#define ECP2_JAC_INYOFFSET             (1 * 2 * NWORDS_256BIT)
#define ECP2_JAC_OUTXOFFSET            (0 * 2 * NWORDS_256BIT)
#define ECP2_JAC_OUTYOFFSET            (1 * 2 * NWORDS_256BIT)
#define ECP2_JAC_OUTZOFFSET            (2 * 2 * NWORDS_256BIT)
#define ECP2_JAC_INOFFSET              (ECP2_JAC_INDIMS *  NWORDS_256BIT)
#define ECP2_JAC_OUTOFFSET             (ECP2_JAC_OUTDIMS * NWORDS_256BIT)

#define CUSNARKS_BLOCK_DIM      (256)
#define CUSNARKS_MAX_NCB        (32)
#define U256_BLOCK_DIM          (256)
#define ECBN128_BLOCK_DIM          (256)

#define MAX_NCORES_OMP        (32)

typedef unsigned int uint32_t;
typedef int int32_t;

typedef unsigned int uint256_t[NWORDS_256BIT];
typedef unsigned int uint512_t[2*NWORDS_256BIT];

// prime number info for finite fields
typedef struct {
   uint32_t p[NWORDS_256BIT];
   uint32_t p_[NWORDS_256BIT];
   uint32_t r_[NWORDS_256BIT];
   uint32_t nonres[NWORDS_256BIT];
   uint32_t r2modp[NWORDS_256BIT];
   uint32_t r2[NWORDS_256BIT];
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
  uint32_t _1[2*NWORDS_256BIT];
  uint32_t _inf[3*NWORDS_256BIT];
  uint32_t _inf2[6*NWORDS_256BIT];

}misc_const_t;

typedef enum{
  MISC_K_1 = 0,
  MISC_K_INF = 2, 
  MISC_K_INF2 = 5,
  MISC_K_N = 11
}misc_const_len_t;
/**
 * Holds the parameters necessary to "launch" a CUDA kernel (i.e. schedule it for
 * execution on some stream of some device).
 */
typedef struct {
        int blockD;
        int gridD;
        int smemS;  // in bytes
        int kernel_idx;
        int return_val;
        int in_offset;
} kernel_config_t;


// index to different primes used
typedef enum{
   MOD_GROUP = 0,
   MOD_FIELD,
   MOD_N

}mod_t;

typedef enum{
  MOD_INFO_P,
  MOD_INFO_P_,
  MOD_INFO_R_,
  MOD_INFO_NONRES,
  MOD_INFO_R2MODP,
  MOD_INFO_R2,
  MOD_INFO_N

}mod_info_name_t;

typedef enum {
  FMT_EXT = 0,
  FMT_MONT,
  FMT_N

}fmt_t;



// data vector
typedef struct{
  uint32_t *data;
  uint32_t length;
  uint32_t size;

}vector_t;

typedef enum{
  FFT_SIZE_2 = 1,
  FFT_SIZE_4,
  FFT_SIZE_8,
  FFT_SIZE_16,
  FFT_SIZE_32,
  FFT_SIZE_1024,
  FFT_SIZE_1M,
  FFT_SIZE_N

}fft_size_t;
  

typedef enum{
  FFT_T_1D = 0, // N < 32
  FFT_T_2D,     // 32 < N < 1024
  FFT_T_3D,     // 1024 < N < 2^20
  FFT_T_4D,     // 2^20 < N < 2^40
  FFT_T_N 

}fft_t;

typedef struct{
  fft_t fft_type;
  uint32_t fft_N[1<<(FFT_T_N-1)];
  uint32_t padding;
  uint32_t levels;
  
}fft_params_t;

// kernel input parameters
typedef struct{
   uint32_t premod; // data requires to be mod-ded as preprocessing stage  
   uint32_t premul;
   uint32_t in_length; // input data length (number of elements)
   uint32_t out_length; // output data length (number of elements)
   uint32_t stride; // data elemements processed by thread
   fft_size_t fft_Nx;
   uint32_t N_fftx;
   fft_size_t fft_Ny;
   uint32_t N_ffty;
   uint32_t forward;
   uint32_t padding_idx;
   uint32_t as_mont;
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
   CB_U256_ADDM_REDUCE,
   CB_U256_ADDM_REDUCE_SHFL,
   CB_U256_SHR1,
   CB_U256_SHL1,
   CB_U256_N

}u256_callback_t;

// index to ec128bn class kernels
typedef enum{
   //CB_EC_LDR_ADD = 0,
   //CB_EC_LDR_DOUBLE,
   //CB_EC_LDR_MUL,
   //CB_EC_LDR_MAD,
   CB_EC_JACAFF_ADD = 0,
   CB_EC_JAC_ADD,
   CB_EC_JACAFF_DOUBLE,
   CB_EC_JAC_DOUBLE,
   CB_EC_JAC_MUL,
   CB_EC_JAC_MUL1,
   CB_EC_JAC_MAD,
   CB_EC_JAC_MAD_SHFL,
   CB_EC_N

}ec_callback_t;

// index to ec128bn class kernels
typedef enum{
   CB_EC2_JACAFF_ADD = 0,
   CB_EC2_JAC_ADD,
   CB_EC2_JACAFF_DOUBLE,
   CB_EC2_JAC_DOUBLE,
   CB_EC2_JAC_MUL,
   CB_EC2_JAC_MUL1,
   CB_EC2_JAC_MAD,
   CB_EC2_JAC_MAD_SHFL,
   CB_EC2_N

}ec2_callback_t;

typedef enum{
   CB_ZPOLY_FFT32 = 0,
   CB_ZPOLY_IFFT32,
   CB_ZPOLY_MUL32,
   CB_ZPOLY_FFTN, 
   CB_ZPOLY_IFFTN,
   CB_ZPOLY_MULN,
   CB_ZPOLY_FFT2DX,
   CB_ZPOLY_FFT2DY,
   CB_ZPOLY_FFT3DXX,
   CB_ZPOLY_FFT3DXX_PREV,
   CB_ZPOLY_FFT3DXY,
   CB_ZPOLY_FFT3DYX,
   CB_ZPOLY_FFT3DYY,
   CB_ZPOLY_ADD,
   CB_ZPOLY_SUB,
   CB_ZPOLY_SUBPREV,
   CB_ZPOLY_MULC,
   CB_ZPOLY_MULCPREV,
   CB_ZPOLY_MULK,
   CB_ZPOLY_MADPREV,
   CB_ZPOLY_ADDPREV,
   CB_ZPOLY_DIVSNARKS,
   CB_ZPOLY_N

}zpoly_callback_t;

typedef struct{
  uint32_t nWords;
  uint32_t nPubInputs;
  uint32_t nOutputs;
  uint32_t nVars;
  uint32_t nConstraints;
  uint32_t cirformat;
  uint32_t R1CSA_nWords;
  uint32_t R1CSB_nWords;
  uint32_t R1CSC_nWords;
  
}cirbin_hfile_t;

typedef enum{
  CIRBIN_H_NWORDS_OFFSET = 0,
  CIRBIN_H_NPUBINPUTS_OFFSET,
  CIRBIN_H_NOUTPUTS_OFFSET,
  CIRBIN_H_NVARS_OFFSET,
  CIRBIN_H_NCONSTRAINTS_OFFSET,
  CIRBIN_H_FORMAT_OFFSET,
  CIRBIN_H_CONSTA_NWORDS_OFFSET,
  CIRBIN_H_CONSTB_NWORDS_OFFSET,
  CIRBIN_H_CONSTC_NWORDS_OFFSET,
  CIRBIN_H_N_OFFSET

}cirbin_hfile_offset_t;

typedef struct{
  uint32_t nWords;
  uint32_t ftype;
  uint32_t protocol;
  uint32_t Rbitlen;
  uint32_t k_binformat;
  uint32_t k_ecformat;
  uint32_t nVars;
  uint32_t nPublic;
  uint32_t domainSize;
  
}pkbin_hfile_t;

typedef enum{
  SNARKSFILE_T_CIRCUIT = 0,
  SNARKSFILE_T_PK, 
  SNARKSFILE_T_VK,
  SNARKSFILE_T_WITNESS,
  SNARKSFILE_T_PROOF,
  SNARKSFILE_T_PDATA,

  SNARKSFILE_T_N

}snarks_file_t;

typedef enum {
  PKBIN_H_NWORDS_OFFSET=0,
  PKBIN_H_FTYPE_OFFSET,
  PKBIN_H_PROTOCOL_OFFSET,
  PKBIN_H_RBITLEN_OFFSET,
  PKBIN_H_BINFORMAT_OFFSET,
  PKBIN_H_ECFORMAT_OFFSET,
  PKBIN_H_NVARS_OFFSET,
  PKBIN_H_NPUBLIC_OFFSET,
  PKBIN_H_DOMAINBITS_OFFSET,
  PKBIN_H_DOMAINSIZE_OFFSET,

  PKBIN_H_N_OFFSET
  
}pkbin_hfile_offset_t;

typedef enum{
  PROTOCOL_T_NORMAL=0,
  PROTOCOL_T_GROTH,
  PROTOCOL_T_N,

}protocol_t;

typedef enum{
  EC_T_PROJECTIVE = 0,
  EC_T_JACOBIAN,
  EC_T_AFFINE,
  EC_T_N

}ec_format_t;

typedef enum{
  GPU_ID0 = 0,
  GPU_ID1,
  GPU_ID2,
  GPU_ID3

}gpu_id_t;

typedef struct{
  uint32_t *pout;
  const uint32_t *scalar;
  uint32_t *pin;
  uint32_t reduce_coeff;
  uint32_t start_idx;
  uint32_t last_idx;
  uint32_t max_threads;
  uint32_t thread_id;
  uint32_t pidx;
  
}mpoly_eval_t;

#endif
