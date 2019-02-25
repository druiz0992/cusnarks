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
      for (i=0; i< params->stride; i++){
        modu256(&x[i*NWORDS_256BIT],&x[i*NWORDS_256BIT], params->midx);
      }
    }
    fft(x);
}

__global__ void ifft_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
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
      for (i=0; i< params->stride; i++){
        modu256(&x[i*NWORDS_256BIT],&x[i*NWORDS_256BIT], params->midx);
      }
    }
    ifft(x);
}

__device__ void fft(uint32_t *x)
{
  uint32_t i, j, k;
  uint32_t s, t;
  uint32_t val[]={0,0,0,0,0,0,0,0};

  memcpy(val,x,sizeof(uint32_t)*NWORDS_256BIT);

  val += __shfl_xor_sync(0xffffffff, val[0], 1);
  val += __shfl_xor_sync(0xffffffff, val[1], 1);
  val += __shfl_xor_sync(0xffffffff, val[2], 1);
  val += __shfl_xor_sync(0xffffffff, val[3], 1);
  val += __shfl_xor_sync(0xffffffff, val[4], 1);
  val += __shfl_xor_sync(0xffffffff, val[5], 1);
  val += __shfl_xor_sync(0xffffffff, val[6], 1);
  val += __shfl_xor_sync(0xffffffff, val[7], 1);

  val += __shfl_xor_sync(0xffffffff, val, 2);
  val += __shfl_xor_sync(0xffffffff, val, 4);
  val += __shfl_xor_sync(0xffffffff, val, 8);
  val += __shfl_xor_sync(0xffffffff, val, 16);

        }
     }
  }
}
