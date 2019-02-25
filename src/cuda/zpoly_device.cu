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
// File name  : zpoly_kernel.cu
//
// Date       : 25/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of zpoly arithmetic
// ------------------------------------------------------------------

*/

#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "log.h"
#include "zpoly_device.h"

/*
    Modular addition kernel

*/
__global__ void fft_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t __restrict__ *x;
    uint32_t __restrict__ *z;
   
    if(tid >= params->in_length/params->stride) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * params->stride * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * params->stride * U256K_OFFSET];
    
    if (params->premod){
      #pragma unroll
      for (i=0; i< params->stride/2-1; i++){
        modu256(&x[i*NWORDS_256BIT],&x[i*NWORDS_256BIT], params->midx);
        modu256(&y[i*NWORDS_256BIT],&y[i*NWORDS_256BIT], params->midx);
      }
    }

   #pragma unroll
   for (i=0; i< params->stride/2-1; i++){
      addmu256(&z[i*NWORDS_256BIT],(const uint32_t *)&x[i*NWORDS_256BIT], (const uint32_t *)&y[i*NWORDS_256BIT], params->midx);
   }   
}

__global__ void ifft_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t __restrict__ *x;
    uint32_t __restrict__ *y;
    uint32_t __restrict__ *z;
   
    if(tid >= params->in_length/params->stride) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * params->stride * U256K_OFFSET + U256_XOFFSET * params->stride/2];
    y = (uint32_t *) &in_vector[tid * params->stride * U256K_OFFSET + U256_YOFFSET * params->stride/2];
    z = (uint32_t *) &out_vector[tid * params->stride/2 * U256K_OFFSET];
    
    if (params->premod){
      #pragma unroll
      for (i=0; i< params->stride/2-1; i++){
        modu256(&x[i*NWORDS_256BIT],&x[i*NWORDS_256BIT], params->midx);
        modu256(&y[i*NWORDS_256BIT],&y[i*NWORDS_256BIT], params->midx);
      }
    }

   #pragma unroll
   for (i=0; i< params->stride/2-1; i++){
      addmu256(&z[i*NWORDS_256BIT],(const uint32_t *)&x[i*NWORDS_256BIT], (const uint32_t *)&y[i*NWORDS_256BIT], params->midx);
   }   
}


