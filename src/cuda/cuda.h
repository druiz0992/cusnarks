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
// File name  : cuda.h
//
// Date       : 11/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of CUDA specific macros. Also, all global variables that
//  need to be visible by GPU are declared as extern here
// ------------------------------------------------------------------

*/

#ifndef _CUDA_H_
#define _CUDA_H_

// Prime information for Finite fields is declared in constant memory
extern __constant__ mod_info_t mod_info_ct[MOD_N];
extern __constant__ ecbn128_t ecbn128_params_ct[MOD_N];
extern __constant__ misc_const_t misc_const_ct[MOD_N];
extern __constant__ uint32_t W32_ct[NWORDS_256BIT * 16];
extern __constant__ uint32_t IW32_ct[NWORDS_256BIT * 16];
extern __constant__ uint32_t IW32_nroots_ct[NWORDS_256BIT * (FFT_SIZE_N - 1)];

#ifndef assert
#define assert(X)         \
  do {                    \
    if ( !(X) ) {         \
        printf("Assert. tid %d: %s, %d\n",threadIdx.x, __FILE__, __LINE__);\
        return;           \
    }                     \
  } while(0)
#endif

#ifndef CCHECK
#define CCHECK(call)        \
  do{                    \
     const cudaError_t error = call;  \
     if (error != cudaSuccess)        \
     {                                \
        printf("Error: %s:%d,  ",__FILE__, __LINE__);  \
        printf("code:%d, reason: %s\n",error, cudaGetErrorString(error)); \
        exit(1);                      \
     }                                \
   } while(0)
#endif

#endif
