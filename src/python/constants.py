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
ECP_XOFFSET   =1 * NWORDS_256BIT
ECP_ZOFFSET   =2 * NWORDS_256BIT
ECP_SCLOFFSET =          0 * NWORDS_256BIT
ECPOINT_NDIMS = 2
ECK_NDIMS     = ECPOINT_NDIMS + U256_NDIMS
ECK_OFFSET    = ECK_NDIMS * NWORDS_256BIT
CUSNARKS_BLOCK_DIM  =256
CUSNARKS_MAX_NCB = 32
U256_BLOCK_DIM  =256

MOD_GROUP = 0
MOD_FIELD = 1
MOD_N = 2


CB_U256_ADDM = 0
CB_U256_SUBM = 1
CB_U256_MOD = 2
CB_U256_MULM = 3
CB_U256_ADDM_REDUCE = 4
CB_U256_N = 5

CB_EC_ADD = 0,
CB_EC_DOUBLE = 1
CB_EC_MUL = 2
CB_EC_ADDRED = 3
CB_EC_MULRED = 4
CB_EC_N = 5
