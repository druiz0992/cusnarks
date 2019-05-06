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

NWORDS_256BIT  =    8
NWORDS_256BIT_FIOS = NWORDS_256BIT + 3
U256_XOFFSET  = 0 * NWORDS_256BIT
U256_YOFFSET  = 1 * NWORDS_256BIT
U256_NDIMS    =1
U256K_OFFSET  = U256_NDIMS * NWORDS_256BIT

ECP_SCLOFFSET  = 0 * NWORDS_256BIT
ECK_INDIMS      =         3

ECP_LDR_INDIMS  =2
ECP_LDR_OUTDIMS = 2
ECP_LDR_INXOFFSET =1 * NWORDS_256BIT
ECP_LDR_INZOFFSET =2 * NWORDS_256BIT
ECP_LDR_OUTXOFFSET = 0 * NWORDS_256BIT
ECP_LDR_OUTZOFFSET = 1 * NWORDS_256BIT

ECK_LDR_INDIMS =ECP_LDR_INDIMS  + U256_NDIMS
ECK_LDR_OUTDIMS =ECP_LDR_OUTDIMS
ECK_LDR_INOFFSET =ECK_LDR_INDIMS * NWORDS_256BIT
ECK_LDR_OUTOFFSET = ECK_LDR_OUTDIMS * NWORDS_256BIT


ECP_JAC_INDIMS=                2 
ECP_JAC_OUTDIMS=               3 
ECP_JAC_XOFFSET_BASE=          0
ECP_JAC_YOFFSET_BASE=          1
ECP_JAC_ZOFFSET_BASE=          2
ECP_JAC_INXOFFSET=             0 * NWORDS_256BIT
ECP_JAC_INYOFFSET=             1 * NWORDS_256BIT
ECP_JAC_OUTXOFFSET=            0 * NWORDS_256BIT
ECP_JAC_OUTYOFFSET =           1 * NWORDS_256BIT
ECP_JAC_OUTZOFFSET=            2 * NWORDS_256BIT
ECK_JAC_INOFFSET=             ECP_JAC_INDIMS * NWORDS_256BIT
ECK_JAC_OUTOFFSET=            ECP_JAC_OUTDIMS * NWORDS_256BIT

ECP2_JAC_INDIMS=                4
ECP2_JAC_OUTDIMS=               6
ECP2_JAC_XOFFSET_BASE=          0
ECP2_JAC_YOFFSET_BASE=          2
ECP2_JAC_ZOFFSET_BASE=          4
ECP2_JAC_INXOFFSET=             0 * NWORDS_256BIT
ECP2_JAC_INYOFFSET=             2 * NWORDS_256BIT
ECP2_JAC_OUTXOFFSET=            0 * NWORDS_256BIT
ECP2_JAC_OUTYOFFSET=            2 * NWORDS_256BIT
ECP2_JAC_OUTZOFFSET=            4 * NWORDS_256BIT
ECP2_JAC_INOFFSET=              ECP2_JAC_INDIMS * 2 * NWORDS_256BIT
ECP2_JAC_OUTOFFSET=             ECP2_JAC_OUTDIMS * 2 * NWORDS_256BIT

CUSNARKS_BLOCK_DIM  =256
CUSNARKS_MAX_NCB = 32
U256_BLOCK_DIM  =256
ECBN128_BLOCK_DIM  =256

MOD_GROUP = 0
MOD_FIELD = 1
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
CB_U256_SHL1 = 6
CB_U256_N = 7

#CB_EC_LDR_ADD = 0
#CB_EC_LDR_DOUBLE = 1
#CB_EC_LDR_MUL = 2
#CB_EC_LDR_MAC = 3
CB_EC_JACAFF_ADD = 0
CB_EC_JAC_ADD = 1
CB_EC_JACAFF_DOUBLE = 2
CB_EC_JAC_DOUBLE = 3
CB_EC_JAC_MUL = 4
CB_EC_JAC_MAD = 5
CB_EC_JAC_MAD_SHFL = 6
CB_EC_N = 7

CB_EC2_JACAFF_ADD = 0
CB_EC2_JAC_ADD = 1
CB_EC2_JACAFF_DOUBLE=2
CB_EC2_JAC_DOUBLE=3
CB_EC2_JAC_MUL=4
CB_EC2_JAC_MAD=5
CB_EC2_JAC_MAD_SHFL=6
CB_EC2_N=7

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
CB_ZPOLY_ADD=13
CB_ZPOLY_SUB=14
CB_ZPOLY_MULC=15
CB_ZPOLY_MULCPREV=16
CB_ZPOLY_MULK=17
CB_ZPOLY_MADPREV=18
CB_ZPOLY_ADDPREV=19
CB_ZPOLY_DIVSNARKS=20
CB_ZPOLY_N=21

