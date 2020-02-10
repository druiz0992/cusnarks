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
#define NBITS_254BIT           (254)
#define NWORDS_256BIT_SHIFT     (3)
#define NWORDS_256BIT_FIOS (NWORDS_256BIT + 3)
#define NWORDS_256BIT_SOS  ((NWORDS_256BIT) * 2 + 2)
#define PRIME_BASE           (30)
#define NBITS_WORD           (32)
#define NBITS_WORD_LOG2      (5)
#define NBITS_WORD_MOD       (0x1F)
#define MAX_R1CSPOLY_NWORDS  (10000000)
#define MAX_R1CSPOLYTMP_NWORDS  (100000)

#define NBITS_BYTE (8)
#define EC_JACREDUCE_BATCH_SIZE (1<<5)
#define EC_JACREDUCE_FLAGS_INIT   (1)
#define EC_JACREDUCE_FLAGS_FINISH (1<<1)
#define EC_JACREDUCE_FLAGS_REDUCTION (1<<2)

#define R1CS_HDR_MAGIC_NUMBER  (0x73633172)
#define R1CS_HDR_V01           (1)
#define R1CS_HDR_START_OFFSET_NWORDS (8)

#define WITNESS_HEADER_N_OFFSET_NWORDS (0)
#define WITNESS_HEADER_SIZE_OFFSET_NWORDS (2)
#define WITNESS_HEADER_OTHER_OFFSET_NWORDS (3)
#define WITNESS_HEADER_W_OFFSET_NWORDS (4)

#define WITNESS_HEADER_N_LEN_NWORDS (2)
#define WITNESS_HEADER_SIZE_LEN_NWORDS (1)
#define WITNESS_HEADER_OTHER_LEN_NWORDS (1)

#define WITNESS_HEADER_LEN_NWORDS (WITNESS_HEADER_W_OFFSET_NWORDS)

#define U256_BSELM  (8)
#define WARP_SIZE  (32)
#define WARP_HALF_SIZE (16)
#define WARP_DOUBLE_SIZE_NBITS (6)

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

#define N_STREAMS_PER_GPU (1+4)

typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned long long int t_uint64;

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
  uint32_t b2[2*NWORDS_256BIT];
  uint32_t g1x[NWORDS_256BIT];
  uint32_t g1y[NWORDS_256BIT];
  uint32_t g2x[2*NWORDS_256BIT];
  uint32_t g2y[2*NWORDS_256BIT];

}ecbn128_t;

typedef enum{
  ECBN128_PARAM_B = 0,
  ECBN128_PARAM_B2X = 8,
  ECBN128_PARAM_B2Y = 16,
  ECBN128_PARAM_G1X = 24,
  ECBN128_PARAM_G1Y = 32,
  ECBN128_PARAM_G2X = 40,
  ECBN128_PARAM_G2Y = 56,

  ECBN128_PARAM_N = 72,
}ecbn128_params_t;

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
        int return_offset;
} kernel_config_t;


// index to different primes used
typedef enum{
   MOD_GROUP = 0,
   MOD_FIELD,
   MOD_N

}mod_t;

typedef enum{
   FFT_T_DIT,
   FFT_T_DIF,
   FFT_T_MODE_N

}fft_mode_t;

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

typedef struct{
  uint32_t *x0;
  uint32_t *x1;
  uint32_t *x2;
  uint32_t *x3;
  uint32_t *x4;
}inv_t;

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
   CB_U256_SHL,
   CB_U256_ALMINV,
   CB_U256_MULM2,
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
   CB_EC_JAC_MUL_OPT,
   CB_EC_JAC_RED,
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
   CB_EC2_JAC_MUL_OPT,
   CB_EC2_JAC_RED,
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
   CB_ZPOLY_FFT3DXXPREV,
   CB_ZPOLY_FFT3DXY,
   CB_ZPOLY_FFT3DYX,
   CB_ZPOLY_FFT3DYY,
   CB_ZPOLY_INTERP3DXX,
   CB_ZPOLY_INTERP3DXY,
   CB_ZPOLY_INTERP3DYX,
   CB_ZPOLY_INTERP3DYY,
   CB_ZPOLY_INTERP3DFINISH,
   CB_ZPOLY_FFT4DXX,
   CB_ZPOLY_FFT4DXY,
   CB_ZPOLY_FFT4DYX,
   CB_ZPOLY_FFT4DYY,
   CB_ZPOLY_INTERP4DXX,
   CB_ZPOLY_INTERP4DXY,
   CB_ZPOLY_INTERP4DYX,
   CB_ZPOLY_INTERP4DYY,
   CB_ZPOLY_INTERP4DFINISH,
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
  unsigned long long nWords;
  unsigned long long nPubInputs;
  unsigned long long nOutputs;
  unsigned long long nVars;
  unsigned long long nConstraints;
  unsigned long long cirformat;
  unsigned long long R1CSA_nWords;
  unsigned long long R1CSB_nWords;
  unsigned long long R1CSC_nWords;
  
}cirbin_hfile_t;

typedef enum{
  CIRBIN_H_NWORDS_OFFSET = 0,
  CIRBIN_H_NPUBINPUTS_OFFSET=2,
  CIRBIN_H_NOUTPUTS_OFFSET=4,
  CIRBIN_H_NVARS_OFFSET=6,
  CIRBIN_H_NCONSTRAINTS_OFFSET=8,
  CIRBIN_H_FORMAT_OFFSET=10,
  CIRBIN_H_CONSTA_NWORDS_OFFSET=12,
  CIRBIN_H_CONSTB_NWORDS_OFFSET=14,
  CIRBIN_H_CONSTC_NWORDS_OFFSET=16,
  CIRBIN_H_N_OFFSET=18

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

typedef struct{
  uint32_t *A;
  uint32_t *B;
  uint32_t *roots;
  uint32_t Nrows;
  uint32_t Ncols;
  uint32_t mNrows;
  uint32_t mNcols;
  uint32_t nroots;
  uint32_t rstride;
  uint32_t pidx;
  uint32_t max_threads;
  uint32_t start_idx;
  uint32_t last_idx;
  uint32_t thread_id;

}ntt_interpolandmul_t;

typedef struct{
  uint32_t *out_ep;
  uint32_t *scl;
  uint32_t *x;
  uint32_t n;
  uint32_t pidx;
  uint32_t max_threads;
  uint32_t start_idx;
  uint32_t last_idx;
  uint32_t thread_id;

}jacadd_reduced_t;

typedef enum{
  KERNEL_T_ZPOLY = 0,
  KERNEL_T_ECBN128_T,
  KERNEL_T_EC2BN128_T,

  KERNEL_T_N
}kernel_t;

typedef struct {
  uint32_t magic_number;
  uint32_t version;
  uint32_t word_width_bytes;
  uint32_t nVars;
  uint32_t nPubOutputs;
  uint32_t nPubInputs;
  uint32_t nPrivInputs;
  uint32_t nConstraints;

  uint32_t R1CSA_nCoeff;
  uint32_t R1CSB_nCoeff;
  uint32_t R1CSC_nCoeff;

}r1csv1_t;

typedef enum{
  R1CSA_IDX = 0,
  R1CSB_IDX = 1,
  R1CSC_IDX = 2,
  R1CS_N_IDX = 3

}r1cs_idx_t;
#endif
