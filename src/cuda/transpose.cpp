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
// File name  : transpose.cpp
//
// Date       : 5/09/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Functionality to transpose a matrix
//
// ------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "types.h"
#include "bigint.h"
#include "constants.h"
#include "transpose.h"

#ifdef PARALLEL_EN
static  uint32_t parallelism_enabled =  1;
#else
static  uint32_t parallelism_enabled =  0;
#endif
/*
  Transpose matrix of 256 bit coefficients
 
  uint32_t *mout : Output transposed matrix
  uint32_t *min : Input matrix
  uint32_t in_nrows : Number of rows in input matrix
  uint32_t in_ncols : Nuimber of columns in input matrix 
*/
void transpose_h(uint32_t *mout, const uint32_t *min, uint32_t in_nrows, uint32_t in_ncols)
{
  uint32_t i,j,k;

  for (i=0; i<in_nrows; i++){
    for(j=0; j<in_ncols; j++){
      for (k=0; k<NWORDS_FR; k++){
        //printf("OUT: %d, IN : %d\n",(j*in_nrows+i)*NWORDS_FR+k, (i*in_ncols+j)*NWORDS_FR+k);
        mout[(j*in_nrows+i)*NWORDS_FR+k] = min[(i*in_ncols+j)*NWORDS_FR+k];
      }
    }
  }
}
void transpose_h(uint32_t *mout, const uint32_t *min,  uint32_t start_row, uint32_t last_row, uint32_t in_nrows, uint32_t in_ncols)
{
  uint32_t i,j,k;

  for (i=start_row; i<last_row; i++){
    for(j=0; j<in_ncols; j++){
      for (k=0; k<NWORDS_FR; k++){
        //printf("OUT: %d, IN : %d\n",(j*in_nrows+i)*NWORDS_FR+k, (i*in_ncols+j)*NWORDS_FR+k);
        mout[(j*in_nrows+i)*NWORDS_FR+k] = min[(i*in_ncols+j)*NWORDS_FR+k];
      }
    }
  }
}
// input matrix mxn
void transpose_h(uint32_t *min, uint32_t in_nrows, uint32_t in_ncols)
{
   uint32_t m = sizeof(uint32_t) * NBITS_BYTE - __builtin_clz(in_nrows) - 1;
   uint32_t n = sizeof(uint32_t) * NBITS_BYTE - __builtin_clz(in_ncols) - 1;
   uint32_t nelems = in_nrows*in_ncols-2;

   if (in_nrows == in_ncols){
     transpose_square_h(min, in_nrows);
     return;
   }
   const uint32_t *tt = inplaceTransposeTidxGet();

   uint32_t idx = tt[m];
   const uint32_t N_1 = (1 << n) - 1;
   const uint32_t NM_1 = nelems+1;

   uint32_t val[NWORDS_FR];
   uint32_t cur_pos = tt[idx++], trans_pos, step=tt[idx++];
   uint32_t max_count = tt[idx++], max_el = tt[idx++];

   while (nelems > 0){
     memcpy(val, &min[cur_pos * NWORDS_FR], sizeof(uint32_t)*NWORDS_FR);
     for (uint32_t ccount=0; ccount < max_count; ccount++){
       for (uint32_t elcount=0; elcount < max_el; elcount++){
          trans_pos = (cur_pos >> n) + ((cur_pos & N_1) << m);
          swapuBI_h(&min[trans_pos * NWORDS_FR], val, NWORDS_FR);
          nelems--;
          //cur_pos = (trans_pos >> n ) + ((trans_pos & N_1) << m);
          cur_pos = trans_pos;
       }
       cur_pos = (trans_pos + step) & NM_1;
       memcpy(val, &min[cur_pos * NWORDS_FR], sizeof(uint32_t)*NWORDS_FR);
     }
     cur_pos = tt[idx++];
     step = tt[idx++];
     max_count = tt[idx++];
     max_el = tt[idx++];
   }
}

void transpose_square_h(uint32_t *min, uint32_t in_nrows)
{
  uint32_t m = sizeof(uint32_t) * NBITS_BYTE - __builtin_clz(in_nrows) - 1;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for(uint32_t i=0; i<in_nrows; i++){
    for(uint32_t j=i+1; j <in_nrows; j++){
        uint32_t cur_pos = (i << m) + j;
        uint32_t trans_pos = (j << m) + i;
        swapuBI_h(&min[trans_pos  * NWORDS_FR], &min[cur_pos * NWORDS_FR], NWORDS_FR);
    } 
  }
}

void transposeBlock_h(uint32_t *mout, uint32_t *min, uint32_t in_nrows, uint32_t in_ncols, uint32_t block_size)
{
    for (uint32_t i = 0; i < in_nrows; i += block_size) {
        for(uint32_t j = 0; j < in_ncols; ++j) {
            for(uint32_t b = 0; b < block_size && i + b < in_nrows; ++b) {
               for (uint32_t k=0; k< NWORDS_FR; k++){
                  mout[(j*in_nrows + i + b)*NWORDS_FR+k] = min[((i + b)*in_ncols + j)*NWORDS_FR + k];
               }
            }
        }
    }
}
void transposeBlock_h(uint32_t *mout, uint32_t *min, uint32_t start_row, uint32_t last_row, uint32_t in_nrows, uint32_t in_ncols, uint32_t block_size)
{

    for (uint32_t i = start_row; i < last_row; i += block_size) {
        for(uint32_t j = 0; j < in_ncols; ++j) {
            for(uint32_t b = 0; b < block_size && i + b < last_row; ++b) {
               for (uint32_t k=0; k< NWORDS_FR; k++){
                  mout[(j*in_nrows + i + b)*NWORDS_FR+k] = min[((i + b)*in_ncols + j)*NWORDS_FR + k];
               }
            }
        }
    }
}

void printUM(const uint32_t *x, uint32_t nrows, uint32_t ncols)
{
  uint32_t i,j;

  for(i=0; i< nrows; i++){
    for (j=0; j< ncols; j++) {
       printUBINumber(&x[i * ncols * NWORDS_FR + j*NWORDS_FR], NWORDS_FR);
    }
  }
}


