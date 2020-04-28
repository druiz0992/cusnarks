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
// File name  : transpose.h
//
// Date       : 06/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of Matrix Transpose
// ------------------------------------------------------------------

*/
#ifndef _TRANSPOSE_H_
#define _TRANSPOSE_H_
void transpose_h(uint32_t *mout, const uint32_t *min, uint32_t in_nrows, uint32_t in_ncols);
void transpose_h(uint32_t *min, uint32_t in_nrows, uint32_t in_ncols);
void transpose_h(uint32_t *mout, const uint32_t *min,  uint32_t start_row, uint32_t last_row, uint32_t in_nrows, uint32_t in_ncols);
void transposeBlock_h(uint32_t *mout, uint32_t *min, uint32_t start_row, uint32_t last_row, uint32_t block_size);
void transposeBlock_h(uint32_t *mout, uint32_t *min, uint32_t start_row, uint32_t last_row, uint32_t in_nrows, uint32_t in_ncols, uint32_t block_size);
void transpose_square_h(uint32_t *min, uint32_t in_nrows);
const uint32_t *inplaceTransposeTidxGet();
void printU256M(const char *, uint32_t nrows, uint32_t ncols);

#endif


