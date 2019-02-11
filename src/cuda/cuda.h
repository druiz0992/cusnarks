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
//  Definition of CUDA specifica constants
// ------------------------------------------------------------------

*/

#ifndef _CUDA_H_
#define _CUDA_H_

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
