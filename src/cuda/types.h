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
#define U256_BLOCK_DIM  (256)

typedef unsigned int uint32_t;
typedef int int32_t;

#endif
