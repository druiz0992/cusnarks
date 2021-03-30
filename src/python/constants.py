"""
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
*/

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : constnts.py
//
// Date       : 15/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Constant defition for python. I am using same values as types.h
//
//  TODO . I need to find a way to either generate this file automatically
//   of have python use types.h directly. Otherwise, this is a souce of pain
//
"""

from _ff import *

NWORDS_256BIT  =    8
NBITS_WORD = 32
N_STREAMS_PER_GPU =  1 + 7
GROTH_PROOF_N_ECPOINTS = 4
MAX_R1CSPOLY_NWORDS = 10000000 
MAX_R1CSPOLYTMP_NWORDS = 100000
#MAX_R1CSPOLY_NWORDS = 100000 
#MAX_R1CSPOLYTMP_NWORDS = 10000
U256_XOFFSET  = 0 * NWORDS_256BIT
U256_YOFFSET  = 1 * NWORDS_256BIT
U256_NDIMS    =1
U256K_OFFSET  = U256_NDIMS * NWORDS_256BIT
MAX_U256_BSELM  = 12
DEFAULT_U256_BSELM  = 6
DEFAULT_U256_BSELM_CUDA = 8
MAX_PIPPENGERS_CONF =20
DEFAULT_PIPPENGERS_CONF=0
WIT_SHMEMKEY = 123456

ECP_SCLOFFSET  = 0 * NWORDS_256BIT
ECK_INDIMS      =         3

ECP_JAC_INDIMS=                2 
ECP_JAC_OUTDIMS=               3 
ECP_JAC_XOFFSET_BASE=          0
ECP_JAC_YOFFSET_BASE=          1
ECP_JAC_ZOFFSET_BASE=          2
ECP_JAC_INXOFFSET=             0 * NWORDS_FP
ECP_JAC_INYOFFSET=             1 * NWORDS_FP
ECP_JAC_OUTXOFFSET=            0 * NWORDS_FP
ECP_JAC_OUTYOFFSET =           1 * NWORDS_FP
ECP_JAC_OUTZOFFSET=            2 * NWORDS_FP
ECK_JAC_INOFFSET=             ECP_JAC_INDIMS * NWORDS_FP
ECK_JAC_OUTOFFSET=            ECP_JAC_OUTDIMS * NWORDS_FP

ECP2_JAC_INDIMS=                4
ECP2_JAC_OUTDIMS=               6
ECP2_JAC_XOFFSET_BASE=          0
ECP2_JAC_YOFFSET_BASE=          2
ECP2_JAC_ZOFFSET_BASE=          4
ECP2_JAC_INXOFFSET=             0 * NWORDS_FP
ECP2_JAC_INYOFFSET=             2 * NWORDS_FP
ECP2_JAC_OUTXOFFSET=            0 * NWORDS_FP
ECP2_JAC_OUTYOFFSET=            2 * NWORDS_FP
ECP2_JAC_OUTZOFFSET=            4 * NWORDS_FP
ECP2_JAC_INOFFSET=              ECP2_JAC_INDIMS * 2 * NWORDS_FP
ECP2_JAC_OUTOFFSET=             ECP2_JAC_OUTDIMS * 2 * NWORDS_FP

CUSNARKS_BLOCK_DIM  =256
CUSNARKS_MAX_NCB = 32
U256_BLOCK_DIM  =256
ECBN128_BLOCK_DIM  =128

MOD_FP = 0
MOD_FR = 1
MOD_N = 2


FFT_SIZE_2 = 1
FFT_SIZE_4 = 2
FFT_SIZE_8 = 3
FFT_SIZE_16 = 4
FFT_SIZE_32 = 5
FFT_SIZE_1024 = 6
FFT_SIZE_1M = 7
FFT_SIZE_N = 8


FFT_T_1D = 0
FFT_T_2D = 1 
FFT_T_3D = 2
FFT_T_4D = 3
FFT_T_N  = 4

CB_U256_ADDM = 0
CB_U256_SUBM = 1
CB_U256_MOD = 2
CB_U256_MULM = 3
CB_U256_ADDM_REDUCE = 4
CB_U256_ADDM_REDUCE_SHFL = 5
CB_U256_SHR1 = 6
CB_U256_SHL1 = 7
CB_U256_SHL = 8
CB_U256_ALMINV = 9
CB_U256_MULM2 = 10
CB_U256_N = 11

#CB_EC_LDR_ADD = 0
#CB_EC_LDR_DOUBLE = 1
#CB_EC_LDR_MUL = 2
#CB_EC_LDR_MAC = 3
CB_EC_JACAFF_ADD = 0
CB_EC_JAC_ADD = 1
CB_EC_JACAFF_DOUBLE = 2
CB_EC_JAC_DOUBLE = 3
CB_EC_JAC_MUL = 4
CB_EC_JAC_MUL1 = 5
CB_EC_JAC_MAD = 6
CB_EC_JAC_MAD_SHFL = 7
CB_EC_JAC_MUL_OPT = 8
CB_EC_JAC_RED = 9
CB_EC_JAC_MUL_PRECOMP = 10
CB_EC_JAC_MUL_PIPPEN = 11
CB_EC_JAC_RED1_PIPPEN = 12
CB_EC_JAC_RED2_PIPPEN = 13
CB_EC_JAC_RED3_PIPPEN = 14
CB_EC_N = 15

CB_EC2_JACAFF_ADD = 15
CB_EC2_JAC_ADD = 16
CB_EC2_JACAFF_DOUBLE=17
CB_EC2_JAC_DOUBLE=18
CB_EC2_JAC_MUL=19
CB_EC2_JAC_MUL1=20
CB_EC2_JAC_MAD=21
CB_EC2_JAC_MAD_SHFL=22
CB_EC2_JAC_MUL_OPT = 23
CB_EC2_JAC_RED = 24
CB_EC2_JAC_MUL_PIPPEN = 25
CB_EC2_JAC_RED1_PIPPEN = 26
CB_EC2_JAC_RED2_PIPPEN = 27
CB_EC2_JAC_RED3_PIPPEN = 28
CB_EC2_N=29

CB_ZPOLY_FFT32 = 0
CB_ZPOLY_IFFT32=1
CB_ZPOLY_MUL32=2
CB_ZPOLY_FFTN = 3
CB_ZPOLY_IFFTN=4
CB_ZPOLY_MULN=5
CB_ZPOLY_FFT2DX=6
CB_ZPOLY_FFT2DY=7
CB_ZPOLY_FFT3DXX=8
CB_ZPOLY_FFT3DXXPREV=9
CB_ZPOLY_FFT3DXY=10
CB_ZPOLY_FFT3DYX=11
CB_ZPOLY_FFT3DYY=12
CB_ZPOLY_INTERP3DXX=13
CB_ZPOLY_INTERP3DXY=14
CB_ZPOLY_INTERP3DYX=15
CB_ZPOLY_INTERP3DYY=16
CB_ZPOLY_INTERP3DFINISH=17
CB_ZPOLY_FFT4DXX=18
CB_ZPOLY_FFT4DXY=19
CB_ZPOLY_FFT4DYX=20
CB_ZPOLY_FFT4DYY=21
CB_ZPOLY_INTERP4DXX=22
CB_ZPOLY_INTERP4DXY=23
CB_ZPOLY_INTERP4DYX=24
CB_ZPOLY_INTERP4DYY=25
CB_ZPOLY_INTERP4DFINISH=26
CB_ZPOLY_ADD=27
CB_ZPOLY_SUB=28
CB_ZPOLY_SUBPREV=29
CB_ZPOLY_MULC=30
CB_ZPOLY_MULCPREV=31
CB_ZPOLY_MULK=32
CB_ZPOLY_MADPREV=33
CB_ZPOLY_ADDPREV=34
CB_ZPOLY_DIVSNARKS=35
CB_ZPOLY_N=36

CIRBIN_H_NWORDS_OFFSET = 0
CIRBIN_H_NPUBINPUTS_OFFSET = 2
CIRBIN_H_NOUTPUTS_OFFSET = 4
CIRBIN_H_NVARS_OFFSET = 6
CIRBIN_H_NCONSTRAINTS_OFFSET = 8
CIRBIN_H_FORMAT_OFFSET = 10
CIRBIN_H_CONSTA_NWORDS_OFFSET = 12
CIRBIN_H_CONSTB_NWORDS_OFFSET = 14
CIRBIN_H_CONSTC_NWORDS_OFFSET = 16
CIRBIN_H_N_OFFSET = 18

ECTABLE_DATA_OFFSET_WORDS = 13
EC_JACREDUCE_BATCH_SIZE=5


FMT_EXT  = 0
FMT_MONT = 1
FMT_N = 2


GroupIDX = 0
FieldIDX = 1
PROTOCOL_T_NORMAL =0
PROTOCOL_T_GROTH = 1
PROTOCOL_T_N = 2

EC_T_PROJECTIVE = 0
EC_T_JACOBIAN = 1
EC_T_AFFINE = 2
EC_T_N = 3

SNARKSFILE_T_CIRCUIT = 0
SNARKSFILE_T_PK = 1
SNARKSFILE_T_VK = 2
SNARKSFILE_T_WITNESS = 3
SNARKSFILE_T_PROOF = 4
SNARKSFILE_T_PDATA = 5
SNARKSFILE_T_N = 6

PKBIN_H_NWORDS_OFFSET=0
PKBIN_H_FTYPE_OFFSET = 1 
PKBIN_H_PROTOCOL_OFFSET = 2
PKBIN_H_RBITLEN_OFFSET = 3
PKBIN_H_BINFORMAT_OFFSET = 4 
PKBIN_H_ECFORMAT_OFFSET = 5
PKBIN_H_NVARS_OFFSET = 6
PKBIN_H_NPUBLIC_OFFSET = 7
PKBIN_H_DOMAINBITS_OFFSET = 8
PKBIN_H_DOMAINSIZE_OFFSET = 9
PKBIN_H_N_OFFSET = 10


GPU_ID0 = 0
GPU_ID1 = 1
GPU_ID2 = 2
GPU_ID3 = 3

KERNEL_T_ZPOLY = 0
KERNEL_T_ECBN128_T = 1
KERNEL_T_EC2BN128_T = 2
KERNEL_T_N = 3


ZKEY_COEFF_NWORDS  = (NWORDS_256BIT + 3)

#ECP vector alignment
EC_ALIGNMENT_FACTOR=128
