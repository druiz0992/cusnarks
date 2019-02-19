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
ECP_JAC_INXOFFSET=             1 * NWORDS_256BIT
ECP_JAC_INYOFFSET=             2 * NWORDS_256BIT
ECP_JAC_OUTXOFFSET=            0 * NWORDS_256BIT
ECP_JAC_OUTYOFFSET =           1 * NWORDS_256BIT
ECP_JAC_OUTZOFFSET=            2 * NWORDS_256BIT

ECK_JAC_INDIMS=               ECP_JAC_INDIMS  + U256_NDIMS
ECK_JAC_OUTDIMS=              ECP_JAC_OUTDIMS
ECK_JAC_INOFFSET=             ECK_JAC_INDIMS * NWORDS_256BIT
ECK_JAC_OUTOFFSET=            ECK_JAC_OUTDIMS * NWORDS_256BIT

CUSNARKS_BLOCK_DIM  =256
CUSNARKS_MAX_NCB = 32
U256_BLOCK_DIM  =256
ECBN128_BLOCK_DIM  =256

MOD_GROUP = 0
MOD_FIELD = 1
MOD_N = 2


CB_U256_ADDM = 0
CB_U256_SUBM = 1
CB_U256_MOD = 2
CB_U256_MULM = 3
CB_U256_ADDM_REDUCE = 4
CB_U256_SHL1 = 5
CB_U256_N = 6

CB_EC_LDR_ADD = 0,
CB_EC_LDR_DOUBLE = 1
CB_EC_LDR_MUL = 2
CB_EC_LDR_MAC = 3
CB_EC_JAC_ADD = 4
CB_EC_JAC_DOUBLE = 5
CB_EC_JAC_MUL = 6
CB_EC_JAC_MAD = 7
CB_EC_N = 8
