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
// File name  : u256_kernel.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of uint256 arithmetic
// ------------------------------------------------------------------

*/

#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "log.h"
#include "u256_device.h"

/*
    Modular addition kernel

*/
__global__ void addmu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
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
      for (i=0; i< params->stride/2; i++){
        modu256(&x[i*NWORDS_256BIT],&x[i*NWORDS_256BIT], params->midx);
        modu256(&y[i*NWORDS_256BIT],&y[i*NWORDS_256BIT], params->midx);
      }
    }

   #pragma unroll
   for (i=0; i< params->stride/2; i++){
      addmu256(&z[i*NWORDS_256BIT],(const uint32_t *)&x[i*NWORDS_256BIT], (const uint32_t *)&y[i*NWORDS_256BIT], params->midx);
   }   
}

/*
    Modular addition + reduction kernel 
      In : x[N]   
      Out :z[N/(blockDim * stride)] 
      Ex:
         N        = 512
         stride   = 4
         BlockDim = 128
         Grid     = 1
         
         in sample    : x[0] x[1] x[2] x[3] ......................................................................................................... x[511]     
         thread       |    0           |   1            |...|      8         |...|     16         |...|     32         |...|     64         |...|       127       |
                      --------------------------------------------------------------------------------------------------------------------------------------------
         1)           |    Z[0]        |   Z[1]         |   |     Z[8]       |   |    Z[16]       |   |    Z[32]       |   |    Z[64]       |   |      Z[127]     |
                      -----------------------------------------------------------------------------------------------------------------------------------------
                      | x[0]+x[1]      | x[4]+x[5]+     |...| x[32]+x[33]+   |...| x[64]+x[65]+   |...| x[128]+x[129]+ |...| x[256]+x[257]+ |...|  x[508]+x[509]+ |
                      | x[2]+x[3]      | x[6]+x[7]      |...| x[34]+x[35]    |...| x[66]+x[67]    |...| x[130]+x[131]  |...| x[258]+x[259]  |...|  x[510]+x[511]  |  
                      |                |                |...|                |...|                |...|                |...|                |...|                 |
                      ---------------------------------------------------------------------------------------------------------------------------------------------
         2)           | Z[64k]         |  Z[64k+1]      |...| Z[64k+8]       |...| Z[64k+16]      |...| Z[64k+32]      |...|  ->Z[0]        |   |   -> Z[63]      |
                      ---------------------------------------------------------------------------------------------------------------------------------------------
                      | x[0]+x[1]+     | x[4]+x[5]+     |...| x[32]+x[33]+   |...| x[64]+x[65]+   |...| x[128]+x[129]+ |...|                |...|                 |
                      | x[2]+x[3]+     | x[6]+x[7]+     |...| x[34]+x[35]+   |...| x[66]+x[67]+   |...| x[130]+x[131]+ |...|  ..........    |...| ..........      |
                      | x[256]+x[257]+ | x[260]+x[261]+ |...| x[288]+x[289]+ |...| x[320]+x[321]+ |...| x[384]+x[385]+ |...|                |...|                 |
                      | x[258]+x[259]  | x[262]+x[263]  |...| x[290]+x[291]  |...| x[322]+x[323]  |...| x[386]+x[387]+ |...|                |...|                 |
                      ---------------------------------------------------------------------------------------------------------------------------------------------
         3)           | Z[32k]         |  Z[32k+1]      |...| Z[32k+8]       |...| Z[32k+16]      |...|  ->Z[0]        |...|                |   |                 |
                      ---------------------------------------------------------------------------------------------------------------------------------------------
                      ---------------------------------------------------------------------------------------------------------------------------------------------
         4)           | Z[16k]         |  Z[16k+1]      |...| Z[16k+8]       |...|    -> Z[0]     |...|                |...|                |   |                 |
                      ---------------------------------------------------------------------------------------------------------------------------------------------
                      ---------------------------------------------------------------------------------------------------------------------------------------------
         5)           | Z[8k]          |  Z[8k+1]       |...|  ->Z[0]        |...| .........      |...|                |...|                |   |                 |
                      ---------------------------------------------------------------------------------------------------------------------------------------------
                      ---------------------------------------------------------------------------------------------------------------------------------------------
         6)           | Z[4k]          |  Z[4k+1]       |...| ..........     |...| .........      |...|                |...|                |   |                 |
                      ---------------------------------------------------------------------------------------------------------------------------------------------
                      ---------------------------------------------------------------------------------------------------------------------------------------------
         7)           | Z[2k]          |  Z[2k+1]       |...| ..........     |...| .........      |...|                |...|                |   |                 |
                      ---------------------------------------------------------------------------------------------------------------------------------------------
                      ---------------------------------------------------------------------------------------------------------------------------------------------
         8)           | Z[k]           |  -> Z[0]       |...|                |...|                |...|                |...|                |...|                 |


*/
__global__ void addmu256_reduce_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t i;

    uint32_t debug_idx = 0;

    extern __shared__ uint32_t smem[];
    uint32_t *smem_ptr = &smem[tid*NWORDS_256BIT];  // 0 .. blockDim
    //uint32_t smem_ptr[128*NWORDS_256BIT];

    uint32_t __restrict__ *x;
    uint32_t __restrict__ *z;
   
    if(idx >= params->in_length/params->stride) {
      return;
    }

    x = (uint32_t *) &in_vector[idx  * params->stride * U256K_OFFSET + U256_XOFFSET]; // 0 .. N-1
    z = (uint32_t *) &out_vector[blockIdx.x * U256K_OFFSET];  // 
    //memset(smem, 0, blockDim.x * NWORDS_256BIT * sizeof(uint32_t));
    
    /*
    if (idx == debug_idx){
       logDebugBigNumber("smem[0]\n",smem_ptr);
      for (i =0; i < params->stride; i++){
       logDebugBigNumber("X[0]\n",&x[i * U256K_OFFSET]);
      }
    }
    */
    if (params->premod){
      #pragma unroll
      for (i =0; i < params->stride; i++){
        modu256(&x[i*U256K_OFFSET],&x[i*U256K_OFFSET], params->midx);
      }
    }

    addmu256(smem_ptr, (const uint32_t *)x, (const uint32_t *)&x[U256K_OFFSET], params->midx);
    /*
    if (idx == debug_idx){
       logInfoBigNumber("smem\n",smem_ptr);
    }
    */

    #pragma unroll
    for (i =0; i < params->stride-2; i++){
      addmu256(smem_ptr, (const uint32_t *)smem_ptr, (const uint32_t *)&x[(i+2)*U256K_OFFSET], params->midx);
      /*
      if (idx == debug_idx){
        logInfo("idx:%d, %d\n",idx,i);
        logInfoBigNumber("smem[i]\n",smem_ptr);
      }
      */
    }
    __syncthreads();

    if (idx == debug_idx){
       logInfoBigNumber("smem[i]\n",smem_ptr);
    }
    // reduction global mem
    if (blockDim.x >= 1024 && tid < 512){
      /*
      if (idx == debug_idx){
        logDebugBigNumber("+smem[0]\n",smem_ptr);
        logDebugBigNumber("+smem[512]\n",&smem[(tid+512)*NWORDS_256BIT]);
      }
      */
      addmu256(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+512)*NWORDS_256BIT], params->midx);
      /*
      if (idx == debug_idx){
        logDebugBigNumber("smem[0]\n",smem_ptr);
      }
      */
    }
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256){
      /*
      if (idx == debug_idx){
        logDebugBigNumber("+smem[0]\n",smem_ptr);
        logDebugBigNumber("+smem[256]\n",&smem[(tid+256)*NWORDS_256BIT]);
      }
       */
      addmu256(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+256)*NWORDS_256BIT], params->midx);
       /*
      if (idx == debug_idx){
        logDebugBigNumber("smem[=256]\n",smem_ptr);
      }
      */
    }
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128){
       /*
       if (idx == debug_idx){
         logDebugBigNumber("+smem[0]\n",smem_ptr);
        logDebugBigNumber("+smem[128]\n",&smem[(tid+128)*NWORDS_256BIT]);
       }
       */
      addmu256(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+128)*NWORDS_256BIT], params->midx);
       /*
       if (idx == debug_idx){
         logDebugBigNumber("smem[=128+0]\n",smem_ptr);
       }
       */
    }
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64){
      /*
      if (idx == debug_idx){
        logDebugBigNumber("+smem[0]\n",smem_ptr);
        logDebugBigNumber("+smem[64]\n",&smem[(tid+64)*NWORDS_256BIT]);
      }
      */
      addmu256(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+64)*NWORDS_256BIT], params->midx);
      if (idx == debug_idx){
        logInfoBigNumber("smem[=64+0]\n",smem_ptr);
      }
    }
    __syncthreads();
      
    // unrolling warp
    if (tid < 32)
    {
        volatile uint32_t *vsmem = smem;
          /*
        if (idx == debug_idx){
          logInfoBigNumber("+smem[0]\n",(uint32_t *)vsmem);
          logInfoBigNumber("+smem[32]\n",&smem[(tid+32)*NWORDS_256BIT]);
        }
         */
        addmu256((uint32_t *)&vsmem[tid * NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid * NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+32)*NWORDS_256BIT], params->midx);
       /*
        if (idx == debug_idx){
        logInfoBigNumber("smem[=32+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("+smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("+smem[16]\n",&smem[(tid+16)*NWORDS_256BIT]);
        }
        */
        addmu256((uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+16)*NWORDS_256BIT], params->midx);
        /*
        if (idx == debug_idx){
        logDebugBigNumber("smem[=16+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("+smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("+smem[8]\n",&smem[(tid+8)*NWORDS_256BIT]);
        }
        */
        addmu256((uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+8)*NWORDS_256BIT], params->midx);
        /*
        if (idx == debug_idx){
          logDebugBigNumber("smem[=8+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("smem[4]\n",&smem[(tid+4)*NWORDS_256BIT]);
        }
        */
        addmu256((uint32_t *) &vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+4)*NWORDS_256BIT], params->midx);
        /*
        if (idx == debug_idx){
          logDebugBigNumber("smem[=4+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("smem[2]\n",&smem[(tid+2)*NWORDS_256BIT]);
        }
        */
        addmu256((uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+2)*NWORDS_256BIT], params->midx);
        /*
        if (idx == debug_idx){
          logDebugBigNumber("smem[=2+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("smem[1]\n",&smem[(tid+1)*NWORDS_256BIT]);
        }
        */
        addmu256((uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+1)*NWORDS_256BIT], params->midx);
        if (idx == debug_idx){
          logInfoBigNumber("smem[=0+1]\n",(uint32_t *)vsmem);
        }

        if (tid==0) {
           memcpy(z, smem_ptr, sizeof(uint32_t) * NWORDS_256BIT);
        }
    }

      
}


__global__ void addmu256_reduce_kernel2(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t i;

    uint32_t debug_idx = 0;

    extern __shared__ uint32_t smem[];
    uint32_t *smem_ptr = &smem[tid*NWORDS_256BIT];  // 0 .. blockDim
    //uint32_t smem_ptr[128*NWORDS_256BIT];

    uint32_t __restrict__ *x;
    uint32_t __restrict__ *z;
   
    if(idx >= params->in_length/params->stride) {
      return;
    }

    x = (uint32_t *) &in_vector[idx  * params->stride * U256K_OFFSET + U256_XOFFSET]; // 0 .. N-1
    z = (uint32_t *) &out_vector[blockIdx.x * U256K_OFFSET];  // 
    //memset(smem, 0, blockDim.x * NWORDS_256BIT * sizeof(uint32_t));
    
    /*
    if (idx == debug_idx){
       logDebugBigNumber("smem[0]\n",smem_ptr);
      for (i =0; i < params->stride; i++){
       logDebugBigNumber("X[0]\n",&x[i * U256K_OFFSET]);
      }
    }
    */
    if (params->premod){
      #pragma unroll
      for (i =0; i < params->stride; i++){
        modu256(&x[i*U256K_OFFSET],&x[i*U256K_OFFSET], params->midx);
      }
    }

    addmu256_2(smem_ptr, (const uint32_t *)x, (const uint32_t *)&x[U256K_OFFSET], params->midx);
  /*
    if (idx == debug_idx){
       logInfoBigNumber("smem2\n",smem_ptr);
    }
  */

    #pragma unroll
    for (i =0; i < params->stride-2; i++){
      addmu256_2(smem_ptr, (const uint32_t *)smem_ptr, (const uint32_t *)&x[(i+2)*U256K_OFFSET], params->midx);
/*
      if (idx == debug_idx){
        logDebug("idx:%d, %d\n",idx,i);
        logInfoBigNumber("smem2[i]\n",smem_ptr);
      }
*/
    }
    __syncthreads();

    
    if (idx == debug_idx){
       logInfoBigNumber("smem2[i]\n",smem_ptr);
    }
    
    // reduction global mem
    if (blockDim.x >= 1024 && tid < 512){
      /*
      if (idx == debug_idx){
        logDebugBigNumber("+smem[0]\n",smem_ptr);
        logDebugBigNumber("+smem[512]\n",&smem[(tid+512)*NWORDS_256BIT]);
      }
      */
      addmu256_2(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+512)*NWORDS_256BIT], params->midx);
      /*
      if (idx == debug_idx){
        logDebugBigNumber("smem[0]\n",smem_ptr);
      }
      */
    }
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256){
      /*
      if (idx == debug_idx){
        logDebugBigNumber("+smem[0]\n",smem_ptr);
        logDebugBigNumber("+smem[256]\n",&smem[(tid+256)*NWORDS_256BIT]);
      }
       */
      addmu256_2(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+256)*NWORDS_256BIT], params->midx);
       /*
      if (idx == debug_idx){
        logDebugBigNumber("smem[=256]\n",smem_ptr);
      }
      */
    }
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128){
       /*
       if (idx == debug_idx){
         logDebugBigNumber("+smem[0]\n",smem_ptr);
        logDebugBigNumber("+smem[128]\n",&smem[(tid+128)*NWORDS_256BIT]);
       }
       */
      addmu256_2(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+128)*NWORDS_256BIT], params->midx);
       /*
       if (idx == debug_idx){
         logDebugBigNumber("smem[=128+0]\n",smem_ptr);
       }
       */
    }
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64){
      /*
      if (idx == debug_idx){
        logDebugBigNumber("+smem[0]\n",smem_ptr);
        logDebugBigNumber("+smem[64]\n",&smem[(tid+64)*NWORDS_256BIT]);
      }
      */
      addmu256_2(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+64)*NWORDS_256BIT], params->midx);
      if (idx == debug_idx){
        logInfoBigNumber("smem2[=64+0]\n",smem_ptr);
      }
    }
    __syncthreads();
      
    // unrolling warp
    if (tid < 32)
    {
        volatile uint32_t *vsmem = smem;
        /*
        if (idx == debug_idx){
          logInfoBigNumber("+smem2[0]\n",(uint32_t *)vsmem);
        logInfoBigNumber("+smem2[32]\n",&smem[(tid+32)*NWORDS_256BIT]);
        }
        */
        addmu256_2((uint32_t *)&vsmem[tid * NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid * NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+32)*NWORDS_256BIT], params->midx);
        
         /*
        if (idx == debug_idx){
        logInfoBigNumber("smem2[=32+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logInfoBigNumber("+smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("+smem[16]\n",&smem[(tid+16)*NWORDS_256BIT]);
        }
        */
        addmu256_2((uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+16)*NWORDS_256BIT], params->midx);
        /*
        if (idx == debug_idx){
        logDebugBigNumber("smem[=16+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("+smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("+smem[8]\n",&smem[(tid+8)*NWORDS_256BIT]);
        }
        */
        addmu256_2((uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+8)*NWORDS_256BIT], params->midx);
        /*
        if (idx == debug_idx){
          logDebugBigNumber("smem[=8+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("smem[4]\n",&smem[(tid+4)*NWORDS_256BIT]);
        }
        */
        addmu256_2((uint32_t *) &vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+4)*NWORDS_256BIT], params->midx);
        /*
        if (idx == debug_idx){
          logDebugBigNumber("smem[=4+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("smem[2]\n",&smem[(tid+2)*NWORDS_256BIT]);
        }
        */
        addmu256_2((uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+2)*NWORDS_256BIT], params->midx);
        /*
        if (idx == debug_idx){
          logDebugBigNumber("smem[=2+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("smem[1]\n",&smem[(tid+1)*NWORDS_256BIT]);
        }
        */
        addmu256_2((uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[tid*NWORDS_256BIT],
                 (const uint32_t *)&vsmem[(tid+1)*NWORDS_256BIT], params->midx);
        if (idx == debug_idx){
          logInfoBigNumber("smem2[=0+1]\n",(uint32_t *)vsmem);
        }

        if (tid==0) {
           memcpy(z, smem_ptr, sizeof(uint32_t) * NWORDS_256BIT);
        }
    }

      
}


/*
    Modular Sub kernel

*/
__global__ void submu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
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
      for (i=0; i< params->stride/2; i++){
        modu256(&x[i*NWORDS_256BIT],&x[i*NWORDS_256BIT], params->midx);
        modu256(&y[i*NWORDS_256BIT],&y[i*NWORDS_256BIT], params->midx);
      }
    }

   #pragma unroll
   for (i=0; i< params->stride/2; i++){
      submu256(&z[i*NWORDS_256BIT],(const uint32_t *)&x[i*NWORDS_256BIT], (const uint32_t *)&y[i*NWORDS_256BIT], params->midx);
   }   
} 
/*
    Modulo

*/
__global__ void modu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    const uint32_t __restrict__ *x;
    uint32_t __restrict__ *z;
 
    if(tid >= params->in_length) {
      return;
    }

    x = (const uint32_t *) &in_vector[tid * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    
    modu256(z, x, params->midx);
}

/*
   Montgomery multiplication
*/
__global__ void mulmontu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid >= params->in_length/2) {
      return;
    }

    uint32_t __restrict__ *A, *B, *U;
    uint32_t i,j; 
 
    A = (uint32_t *) &in_vector[tid * 2 * U256K_OFFSET + U256_XOFFSET];
    B = (uint32_t *) &in_vector[tid * 2 * U256K_OFFSET + U256_YOFFSET];
    U = (uint32_t *) &out_vector[tid * U256K_OFFSET];
   
    // ensure A, B < p 
    // TODO : If numbers are already in Montgomery format
    // premod operations doesn't make any sense. I leave it 
    // for the moment as during testing I multiply random numbers without
    // checking if they are greated than the prime number
    if (params->premod){
      modu256(A,A, params->midx);
      modu256(B,B, params->midx);
    }

    mulmontu256(U, (const uint32_t *)A, (const uint32_t *) B, params->midx);

   return;
}

__global__ void mulmontu256_2_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid >= params->in_length/4) {
      return;
    }

    uint32_t __restrict__ *A, *B, *U;
    uint32_t i,j; 
 
    A = (uint32_t *) &in_vector[tid * 4 * U256K_OFFSET + U256_XOFFSET];
    B = (uint32_t *) &in_vector[tid * 4 * U256K_OFFSET + U256_YOFFSET];
    U = (uint32_t *) &out_vector[tid * U256K_OFFSET];
   
    // ensure A, B < p 
    // TODO : If numbers are already in Montgomery format
    // premod operations doesn't make any sense. I leave it 
    // for the moment as during testing I multiply random numbers without
    // checking if they are greated than the prime number
    if (params->premod){
      modu256(A,A, params->midx);
      modu256(B,B, params->midx);
      modu256(&A[2*NWORDS_256BIT],&A[2*NWORDS_256BIT], params->midx);
      modu256(&B[2*NWORDS_256BIT],&B[2*NWORDS_256BIT], params->midx);
    }

    mulmontu256_2(U, (const uint32_t *)A, (const uint32_t *) B, params->midx);

   return;
}


__global__ void shr1u256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;
    uint32_t b_shifted;

    uint32_t __restrict__ *x;
    uint32_t __restrict__ *y;
    uint32_t __restrict__ *z;
   
    if(tid >= params->in_length) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * U256K_OFFSET + U256_XOFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    memset(z, 0, NWORDS_256BIT*sizeof(uint32_t));
 
    #pragma unroll
    for (i=0; i< NWORDS_256BIT*32; i++){   
      b_shifted = shr1u256((const uint32_t *)x);
      z[i/32] |= (b_shifted << (i % 32));
    }
}
/*
   x + y for 256 bit numbers
*/
__forceinline__ __device__ void addu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y)
{
  // z[i] = x[i] + y[i] for 8x32 bit words
  asm("{                             \n\t"
      ".reg .u32         %x_;        \n\t"
      ".reg .u32         %y_;        \n\t"
      "mov.u32           %x_,%8;     \n\t"
      "mov.u32           %y_,%9;     \n\t"
      "add.cc.u32        %0, %x_, %y_;\n\t"             // sum with carry out
      "mov.u32           %x_,%10;     \n\t"
      "mov.u32           %y_,%11;     \n\t"
      "addc.cc.u32       %1, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%12;     \n\t"
      "mov.u32           %y_,%13;     \n\t"
      "addc.cc.u32       %2, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%14;     \n\t"
      "mov.u32           %y_,%15;     \n\t"
      "addc.cc.u32       %3, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%16;     \n\t"
      "mov.u32           %y_,%17;     \n\t"
      "addc.cc.u32       %4, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%18;     \n\t"
      "mov.u32           %y_,%19;     \n\t"
      "addc.cc.u32       %5, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%20;     \n\t"
      "mov.u32           %y_,%21;     \n\t"
      "addc.cc.u32       %6, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%22;     \n\t"
      "mov.u32           %y_,%23;     \n\t"
      "addc.u32          %7, %x_, %y_;\n\t"             // sum with carry in 
      "}                             \n\n"
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(x[0]), "r"(y[0]), "r"(x[1]), "r"(y[1]),
        "r"(x[2]), "r"(y[2]), "r"(x[3]), "r"(y[3]),
        "r"(x[4]), "r"(y[4]), "r"(x[5]), "r"(y[5]),
        "r"(x[6]), "r"(y[6]), "r"(x[7]), "r"(y[7]));
}

/*
   z(288 bits) = x(288 bits) + y(256 bits)
   TODO : pending
*/
__forceinline__ __device__ void addu288(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y)
{
  // z[i] = x[i] + y[i] for 8x32 bit words
  asm("{                             \n\t"
      //".reg .u32         %x_;        \n\t"
      //".reg .u32         %y_;        \n\t"
      //"mov.u32           %x_,%8;     \n\t"
      //"mov.u32           %y_,%9;     \n\t"
      "add.cc.u32        %0, %8, %9;\n\t"             // sum with carry out
      //"mov.u32           %x_,%10;     \n\t"
      //"mov.u32           %y_,%11;     \n\t"
      "addc.cc.u32       %1, %10, %11;\n\t"             // sum with carry in and carry out
      //"mov.u32           %x_,%12;     \n\t"
      //"mov.u32           %y_,%13;     \n\t"
      "addc.cc.u32       %2, %12, %13;\n\t"             // sum with carry in and carry out
      //"mov.u32           %x_,%14;     \n\t"
      //"mov.u32           %y_,%15;     \n\t"
      "addc.cc.u32       %3, %14, %15;\n\t"             // sum with carry in and carry out
      //"mov.u32           %x_,%16;     \n\t"
      //"mov.u32           %y_,%17;     \n\t"
      "addc.cc.u32       %4, %16, 17;\n\t"             // sum with carry in and carry out
      //"mov.u32           %x_,%18;     \n\t"
      //"mov.u32           %y_,%19;     \n\t"
      "addc.cc.u32       %5, %18, %19;\n\t"             // sum with carry in and carry out
      //"mov.u32           %x_,%20;     \n\t"
      //"mov.u32           %y_,%21;     \n\t"
      "addc.cc.u32       %6, %20, %21;\n\t"             // sum with carry in and carry out
      //"mov.u32           %x_,%22;     \n\t"
      //"mov.u32           %y_,%23;     \n\t"
      "addc.u32          %7, %22, %23;\n\t"             // sum with carry in 
      "}                             \n\n"
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7]), "=r"(z[8])
      : "r"(x[0]), "r"(y[0]), "r"(x[1]), "r"(y[1]),
        "r"(x[2]), "r"(y[2]), "r"(x[3]), "r"(y[3]),
        "r"(x[4]), "r"(y[4]), "r"(x[5]), "r"(y[5]),
        "r"(x[6]), "r"(y[6]), "r"(x[7]), "r"(y[7]),"r"(x[8]));
}


/*
   x - y for 256 bit numbers
*/
__device__ void subu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y)
{
  // z[i] = x[i] - y[i] for 8x32 bit words
  asm("{                             \n\t"
      ".reg .u32          %x_;        \n\t"
      ".reg .u32          %y_;        \n\t"
      "mov.u32           %x_,%8;     \n\t"
      "mov.u32           %y_,%9;     \n\t"
      "sub.cc.u32        %0, %x_, %y_;\n\t"             // sum with borrow out
      "mov.u32           %x_,%10;     \n\t"
      "mov.u32           %y_,%11;     \n\t"
      "subc.cc.u32       %1, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%12;     \n\t"
      "mov.u32           %y_,%13;     \n\t"
      "subc.cc.u32       %2, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%14;     \n\t"
      "mov.u32           %y_,%15;     \n\t"
      "subc.cc.u32       %3, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%16;     \n\t"
      "mov.u32           %y_,%17;     \n\t"
      "subc.cc.u32       %4, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%18;     \n\t"
      "mov.u32           %y_,%19;     \n\t"
      "subc.cc.u32       %5, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%20;     \n\t"
      "mov.u32           %y_,%21;     \n\t"
      "subc.cc.u32       %6, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%22;     \n\t"
      "mov.u32           %y_,%23;     \n\t"
      "subc.u32          %7, %x_, %y_;\n\t"             // sum with borrow in 
      "}                             \n\t"
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(x[0]), "r"(y[0]), "r"(x[1]), "r"(y[1]),
        "r"(x[2]), "r"(y[2]), "r"(x[3]), "r"(y[3]),
        "r"(x[4]), "r"(y[4]), "r"(x[5]), "r"(y[5]),
        "r"(x[6]), "r"(y[6]), "r"(x[7]), "r"(y[7]));

}


/*
   x == 0 for 256 bit numbers
*/
__device__ uint32_t eq0u256(const uint32_t __restrict__ *x)
{
  if (x[0] == 0 && x[1] ==  0 && x[2] == 0 && x[3] == 0 && x[4] == 0 && x[5] == 0 && x[6] == 0 && x[7] == 0){
    return 1;
  } else { 
    return 0;
  }
}

/*
   x == y for 256 bit numbers
*/
__device__ uint32_t equ256(const uint32_t __restrict__ *x, const uint32_t __restrict__ *y)
{
  if (x[0] == y[0] && x[1] ==  y[1] && x[2] == y[2] && x[3] == y[3] &&
		  x[4] == y[4] && x[5] == y[5] && x[6] == y[6] && x[7] == y[7]){
    return 1;
  } else { 
    return 0;
  }
}


/*
   x < y for 256 bit numbers
*/
__device__ uint32_t ltu256(const uint32_t __restrict__ *x, const uint32_t __restrict__ *y)
{
   if (x[7] > y[7]) return 0;
   else if (x[7] < y[7]) return 1;
   else if (x[6] > y[6]) return 0;
   else if (x[6] < y[6]) return 1;
   else if (x[5] > y[5]) return 0;
   else if (x[5] < y[5]) return 1;
   else if (x[4] > y[4]) return 0;
   else if (x[4] < y[4]) return 1;
   else if (x[3] > y[3]) return 0;
   else if (x[3] < y[3]) return 1;
   else if (x[2] > y[2]) return 0;
   else if (x[2] < y[2]) return 1;
   else if (x[1] > y[1]) return 0;
   else if (x[1] < y[1]) return 1;
   else if (x[0] >= y[0]) return 0;
   else return 1;
}
/*
   x + y (mod N) for 256 bit numbers

   NOTE. Function requires that x, y < N
*/
__device__ void addmu256(uint32_t __restrict__*z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y, mod_t midx)
{
  uint32_t const __restrict__ *p = mod_info_ct[midx].p;
  uint32_t tmp[NWORDS_256BIT];

   addu256(tmp,x,y);
   //if (tmp[NWORDS_256BIT-1] > p[NWORDS_256BIT-1]){
    if (ltu256(p,tmp)){
      subu256(tmp,tmp,p);
   } 
   asm("mov.u32     %0,  %8;\n\t"
       "mov.u32     %1,  %9;\n\t"
       "mov.u32     %2,  %10;\n\t"
       "mov.u32     %3,  %11;\n\t"
       "mov.u32     %4,  %12;\n\t"
       "mov.u32     %5,  %13;\n\t"
       "mov.u32     %6,  %14;\n\t"
       "mov.u32     %7,  %15;\n\t"
    : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
      "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
    : "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]), "r"(tmp[3]),
      "r"(tmp[4]), "r"(tmp[5]), "r"(tmp[6]), "r"(tmp[7]));
#if 0
  // if (ltu256(p,z)) {
   if (eq0u256(y)) {
   asm("mov.u32     %0,  %8;\n\t"
       "mov.u32     %1,  %9;\n\t"
       "mov.u32     %2,  %10;\n\t"
       "mov.u32     %3,  %11;\n\t"
       "mov.u32     %4,  %12;\n\t"
       "mov.u32     %5,  %13;\n\t"
       "mov.u32     %6,  %14;\n\t"
       "mov.u32     %7,  %15;\n\t"
    : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
      "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
    : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
      "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));
      //memcpy(z,x,NWORDS_256BIT * sizeof(uint32_t));

   } else {
      //subu256(tmp,p,y);
      subu256(tmp,mod_info_ct[midx].p,y);
      submu256(z,x,tmp, midx);
   }

#endif
  
}
__device__ void addmu256_2(uint32_t __restrict__*z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y, mod_t midx)
{
  uint32_t const __restrict__ *p = mod_info_ct[midx].p;
  uint32_t tmp[NWORDS_256BIT];
  // if (ltu256(p,z)) {
   if (eq0u256(y)) {
   asm("mov.u32     %0,  %8;\n\t"
       "mov.u32     %1,  %9;\n\t"
       "mov.u32     %2,  %10;\n\t"
       "mov.u32     %3,  %11;\n\t"
       "mov.u32     %4,  %12;\n\t"
       "mov.u32     %5,  %13;\n\t"
       "mov.u32     %6,  %14;\n\t"
       "mov.u32     %7,  %15;\n\t"
    : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
      "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
    : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
      "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));
      //memcpy(z,x,NWORDS_256BIT * sizeof(uint32_t));

   } else {
      //subu256(tmp,p,y);
      subu256(tmp,mod_info_ct[midx].p,y);
      submu256(z,x,tmp, midx);

   }

}

/*
   x - y (mod N) for 256 bit numbers

   NOTE. Function requires that x, y < N
*/

__device__ void submu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ *y, mod_t midx)
{

  uint32_t tmp[NWORDS_256BIT];
  if (ltu256(x,y)){
    subu256(tmp,mod_info_ct[midx].p,y);
    addu256(z,x,tmp);

  } else {
    subu256(z,x,y);
  }
  
}

// multiply by small constant (<= 32) 
__device__ void mulku256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ k, mod_t midx)
{
  uint32_t i;
  uint32_t tmp[NWORDS_256BIT+1];  //288 bits. I can multiply by up to 32
  memcpy(tmp, x, NWORDS_256BIT * sizeof(uint32_t));

  assert(k < U256_MAX_SMALLK);

  addu288(z,x,x); 

  #pragma unroll
  for (i=0; i<k-1; i++){
    addu288(z,z,x); 
  }

  // TODO
  // modulo p and return
}
	

/*
   aA = X[0] * Y[0]
   bB = X[1] * Y[1]
   Z[0] = aA + bB * residue
   Z[1] = (X[0] + X[1]) * (Y[0] + Y[1]) - aA - bB
*/
__device__ void mulmontu256_2(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, const uint32_t __restrict__ *B, mod_t midx)
{
    uint32_t const __restrict__ *nonres = mod_info_ct[midx].nonres;
    uint32_t tmp1[NWORDS_256BIT], tmp2[NWORDS_256BIT];

    addmu256(tmp1, A, &A[NWORDS_256BIT], midx);                // tmp1 = X[0] + X[1]
    addmu256(tmp2, B, &B[NWORDS_256BIT], midx);                // tmp2 = Y[0] + Y[1]
     
    mulmontu256(tmp1,tmp1,tmp2,midx);                          // tmp1 = tmp1 * tmp2  
    mulmontu256(tmp2,A,B,midx);                                // tmp2 = aA = X[0] * Y[0] 
    mulmontu256(U,&A[NWORDS_256BIT],&B[NWORDS_256BIT],midx);   // Z[0] = bB = X[1] * Y[1]

    submu256(&U[NWORDS_256BIT], tmp1, tmp2, midx);             //Z[1] = tmp1 - aA
    submu256(&U[NWORDS_256BIT], &U[NWORDS_256BIT], U, midx);   //Z[1] = tmp1 - aA - bB
    
    mulmontu256(U,U,nonres,midx);                                    // Z[0] = bB * non_residue
    addmu256(U,U,tmp2, midx);                                  // Z[0] = Z[0] + aA
}

/*
   ab =  X[0] * X[1]
   t1 =  X[0] + nonres * X[1] 
   a2  = X[0] + X[1]
   Z[0] = t1 * a2 - ab + nonres * ab
   Z[1] = ab + ab
*/
__device__ void sqmontu256_2(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, mod_t midx)
{
    uint32_t const __restrict__ *nonres = mod_info_ct[midx].nonres;
    uint32_t tmp1[NWORDS_256BIT], tmp2[NWORDS_256BIT];

    mulmontu256(tmp1, A, &A[NWORDS_256BIT], midx);             // tmp1 = ab = X[0] * X[1]
    mulmontu256(tmp2, nonres, &A[NWORDS_256BIT], midx);       //  tmp2 = nonres * X[1]
    addmu256(tmp2, tmp2,A, midx);                             //  tmp2 = t1 

    mulmontu256(tmp2, tmp2,A, midx);                          //  tmp2 = t1 * X[0]
    mulmontu256(tmp2, tmp2,&A[NWORDS_256BIT], midx);          //  tmp2 = t1 * a2
    submu256(tmp2, tmp2, tmp1, midx);                         //  tmp2 = t1 * a2 - ab
    addmu256(&U[NWORDS_256BIT],tmp1, tmp1,midx);               // Z[1] = ab + ab
    submu256(tmp1, tmp1, nonres, midx);                       //  tmp1 = nonres * ab


    addmu256(&U[NWORDS_256BIT],tmp1, tmp2,midx);              // Z[2] = t1 * a2 - ab + nonres * ab
}


/*
   Montgomery Multiplication(xr^(-1),y^r(-1)) = xr^(-1) * yr^(-1) * r (mod N)  for 256 bit numbers
     FIOS implementatin

   NOTE. Function requires that x, y < N
   NOTE. If x or y are not in Montgomery format, output is 
    in standard format multiplication of x * y
     ex: MontMul(xr^(-1),  y) = xr^(-1) * y * r = x * y

   NOTE : According to Tolg Acar's thesis*:
      SOS   2s^2+s MUL, 4s^2+4s+2 ADD 
      FIOS  2s^2+s MUL, 5s^2+3s+2 ADD

   * www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf

*/
__device__ void mulmontu256(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, const uint32_t __restrict__ *B, mod_t midx)
{ 
    uint32_t i,j;
    uint32_t S, C=0, C1, C2;
    uint32_t __restrict__ M[2], X[2];
    uint32_t __restrict__ T[]={0,0,0,0,0,0,0,0,0,0,0};
    uint32_t const __restrict__ *PN_u256 = mod_info_ct[midx].p_;
    uint32_t const __restrict__ *P_u256 = mod_info_ct[midx].p;

    logDebugBigNumber("P\n",(uint32_t *)P_u256);
    logDebugBigNumber("PN\n",(uint32_t *)PN_u256);
    logDebugBigNumber("A\n", (uint32_t*) A);
    logDebugBigNumber("B\n", (uint32_t *)B);

    #pragma unroll
    for(i=0; i<NWORDS_256BIT; i++)
    {
      // (C,S) = t[0] + a[0]*b[i], worst case 2 words
      madcu32(&C,&S,A[0],B[i],T[0]);
      logDebug("0 - C : %u, S: %u\n",C,S);
      logDebug("0 - A[0] : %u, B[i]: %u T[0] : %u\n",A[0],B[i], T[0]);

      // ADD(t[1],C)
      propcu32(T, C, 1);
      logDebugBigNumber("T\n",T);

     // m = S*n'[0] mod W, where W=2^32
     // Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
     mulu32(M, S, PN_u256[0]);
     logDebug("M[0] : %u, M[1] : %u\n",M[0], M[1]);

     // (C,S) = S + m*n[0], worst case 2 words
     madcu32(&C,&S,M[0],P_u256[0],S);
     logDebug("1 - C : %u, S: %u\n",C,S);

     #pragma unroll
     for(j=1; j<NWORDS_256BIT; j++)
     {
       // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
       mulu32(X, A[j], B[i]);
       addcu32(&C1, &S, T[j], C);
       logDebug("2 - C1 : %u, S: %u\n",C1,S);
       addcu32(&C2, &S, S, X[0]);
       logDebug("3 - C2 : %u, S: %u\n",C2,S);
       logDebug("X[0] : %u, X[1]: %u\n",X[0], X[1]);
       addcu32(&X[0], &C, C1, X[1]);
       logDebug("4 - C : %u\n",C);
       addcu32(&X[0], &C, C, C2);
       logDebug("5 - C : %u\n",C);

       // ADD(t[j+1],C)
       propcu32(T,C,j+1);
       logDebugBigNumber("T\n",T);

       // (C,S) = S + m*n[j]
       madcu32(&C,&S,M[0], P_u256[j],S);
       logDebug("6 - C : %u, S: %u\n",C,S);

       // t[j-1] = S
       T[j-1] = S;
       logDebugBigNumber("T\n",T);
     }

     // (C,S) = t[s] + C
     addcu32(&C,&S, T[NWORDS_256BIT], C);
     logDebug("6 - C : %u, S: %u\n",C,S);
     // t[s-1] = S
     T[NWORDS_256BIT-1] = S;
     // t[s] = t[s+1] + C
     addcu32(&X[0],&T[NWORDS_256BIT], T[NWORDS_256BIT+1], C);
     // t[s+1] = 0
     T[NWORDS_256BIT+1] = 0;
   }

   logDebugBigNumber("T before mod\n",T);
  
   /* Step 3: if(u>=n) return u-n else return u */
   //if (ltu256(P_u256,T)){
   if (P_u256[NWORDS_256BIT-1] < T[NWORDS_256BIT-1]){
      subu256(U,T,P_u256);
   } else {
      //memcpy(U, T, sizeof(uint32_t) * NWORDS_256BIT);
   asm("mov.u32     %0,  %8;\n\t"
       "mov.u32     %1,  %9;\n\t"
       "mov.u32     %2,  %10;\n\t"
       "mov.u32     %3,  %11;\n\t"
       "mov.u32     %4,  %12;\n\t"
       "mov.u32     %5,  %13;\n\t"
       "mov.u32     %6,  %14;\n\t"
       "mov.u32     %7,  %15;\n\t"
    : "=r"(U[0]), "=r"(U[1]), "=r"(U[2]), "=r"(U[3]),
      "=r"(U[4]), "=r"(U[5]), "=r"(U[6]), "=r"(U[7])
    : "r"(T[0]), "r"(T[1]), "r"(T[2]), "r"(T[3]),
      "r"(T[4]), "r"(T[5]), "r"(T[6]), "r"(T[7]));
   }
   logDebugBigNumber("U after mod\n",U);

   return;
}
__device__ void sqmontu256(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, mod_t midx)
{
   //TODO : implement proper squaring
   mulmontu256(U,A,A,midx);
}
/*
   x mod N

   NOTE : It requires that prime is at least 253 bit number. In reality is 254 bits the prime
    i am using
   */
__device__ void modu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, mod_t midx)
{
   const uint32_t __restrict__ *p = mod_info_ct[midx].p;

   asm("mov.u32     %0,  %8;\n\t"
       "mov.u32     %1,  %9;\n\t"
       "mov.u32     %2,  %10;\n\t"
       "mov.u32     %3,  %11;\n\t"
       "mov.u32     %4,  %12;\n\t"
       "mov.u32     %5,  %13;\n\t"
       "mov.u32     %6,  %14;\n\t"
       "mov.u32     %7,  %15;\n\t"
    : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
      "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
    : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
      "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));

  // x(255 bit number worst case) - p (253 bit number) = z (255 bit number) : ex 31(5 b) - 4(3 b) =27 (5 b) 
  if (!ltu256(z,p)){
  //if (z[NWORDS_256BIT-1] > p[NWORDS_256BIT-1]){
     subu256(z,z,p);
  } else { return; }
  // x(255 bit ) - p (253 bit number) = z (255 bit number) : ex 27(5 b) - 4(3 b) =23 (5 b) 
  //if (z[NWORDS_256BIT-1] > p[NWORDS_256BIT-1]){
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }
  // x(255 bit ) - p (253 bit number) = z (255 bit number) : ex 23(5 b) - 4(3 b) = 19 (5 b) 
  //if (z[NWORDS_256BIT-1] > p[NWORDS_256BIT-1]){
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }
  // x(255 bit ) - p (253 bit number) = z (254 bit number) : ex 19(5 b) - 4(3 b) = 15 (4 b) 
  //if (z[NWORDS_256BIT-1] > p[NWORDS_256BIT-1]){
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }
  // x(254 bit ) - p (253 bit number) = z (254 bit number) : ex 15(5 b) - 4(3 b) = 11 (4 b) 
  //if (z[NWORDS_256BIT-1] > p[NWORDS_256BIT-1]){
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }
  // x(254 bit ) - p (253 bit number) = z (254 bit number) : ex 11(5 b) - 4(3 b) = 7 (3 b) 
  //if (z[NWORDS_256BIT-1] > p[NWORDS_256BIT-1]){
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }
  // x(254 bit ) - p (253 bit number) = z (254 bit number) : ex 7(5 b) - 4(3 b) = 3 (3 b) 
  //if (z[NWORDS_256BIT-1] > p[NWORDS_256BIT-1]){
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }

  assert(0);

}

/*
   x >> 1 for 256 bit number
*/
__device__ uint32_t shr1u256(const uint32_t __restrict__ *x)
{
   uint32_t c; 
   asm("{                                    \n\t"
       ".reg .u32          %tmp;             \n\t"
       "and.b32            %8, %9,  1;     \n\t"  // x[0] |= (x[1] & 1) << 31
       "shr.u32            %0,   %9,  1;     \n\t"  // x[0] = x[0] >> 1
       "and.b32            %tmp, %10,  1;     \n\t"  // x[0] |= (x[1] & 1) << 31
       "shl.b32            %tmp, %tmp, 31;   \n\t"
       "or.b32             %0,   %0,  %tmp;  \n\t"
       "shr.u32            %1,   %10,   1;     \n\t"  // x[1] = x[1] >> 1
       "and.b32            %tmp, %11,  1;     \n\t"  // x[1] |= (x[2] & 1) << 31
       "shl.b32            %tmp, %tmp, 31;   \n\t"
       "or.b32             %1,   %1,  %tmp;   \n\t"
       "shr.u32            %2,   %11,  1;     \n\t"  // x[2] = x[2] >> 1
       "and.b32            %tmp, %12,  1;     \n\t"  // x[2] |= (x[3] & 1) << 31
       "shl.b32            %tmp, %tmp, 31;   \n\t"
       "or.b32             %2,   %2,  %tmp;   \n\t"
       "shr.u32            %3,   %12,  1;     \n\t"  // x[3] = x[3] >> 1
       "and.b32            %tmp, %13,  1;     \n\t"  // x[3] |= (x[4] & 1) << 31
       "shl.b32            %tmp, %tmp, 31;   \n\t"
       "or.b32             %3,   %3,  %tmp;   \n\t"
       "shr.u32            %4,   %13,  1;     \n\t"  // x[4] = x[4] >> 1
       "and.b32            %tmp, %14,  1;     \n\t"  // x[4] |= (x[5] & 1) << 31
       "shl.b32            %tmp, %tmp, 31;   \n\t"
       "or.b32             %4,   %4,  %tmp;   \n\t"
       "shr.u32            %5,   %14,  1;     \n\t"  // x[5] = x[5] >> 1
       "and.b32            %tmp, %15,  1;     \n\t"  // x[5] |= (x[6] & 1) << 31
       "shl.b32            %tmp, %tmp, 31;   \n\t"
       "or.b32             %5,   %5,  %tmp;   \n\t"
       "shr.u32            %6,   %15,  1;     \n\t"  // x[6] = x[6] >> 1
       "and.b32            %tmp, %16,  1;     \n\t"  // x[6] |= (x[7] & 1) << 31
       "shl.b32            %tmp, %tmp, 31;   \n\t"
       "or.b32             %6,   %6,  %tmp;   \n\t"
       "shr.u32            %7,   %16,  1;     \n\t"  // x[7] = x[7] >> 1
       "}                               \n\t"
       : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]), 
         "=r"(x[4]), "=r"(x[5]), "=r"(x[6]), "=r"(x[7]), "=r"(c)
       : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), 
         "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));

      return c;
}
/*
   x * y for 32 bit number. Returns 64 bit number
 */
__device__ void mulu32(uint32_t __restrict__ *z, uint32_t x, uint32_t y)
{
  // z[i] = x * y for 32 bit words
  asm("{                                    \n\t"
      ".reg .u64 %prod;                     \n\t"
      "mul.wide.u32        %prod, %2,    %3;\n\t"           
      "cvt.u32.u64         %0,    %prod;    \n\t"
      "shr.u64             %prod, %prod, 32;\n\t"
      "cvt.u32.u64         %1,    %prod;    \n\t"
      "}                                    \n\t"
      : "=r"(z[0]), "=r"(z[1])
      : "r"(x), "r"(y));
}


/*
   (c,s) = x + y * a for 32 bit number. Returns 32 bit number and 32 bit carry
*/
__forceinline__ __device__ void madcu32(uint32_t *c, uint32_t *s, uint32_t x, uint32_t y, uint32_t a)
{
   // (C,S) =  x * y + a -> No carry in
   asm("{                                   \n\t"
       "mad.lo.cc.u32  %0, %2, %3, %4;      \n\t"
       "madc.hi.u32    %1, %2, %3, 0;       \n\t"
       "}                                   \n\t"
       : "=r"(s[0]), "=r"(c[0])
       : "r"(x), "r"(y), "r"(a));
}
/*
   (c.s) = x + y for 32 bit numbers. Returns 32 bit number and 1 bit carry
*/
__forceinline__ __device__ void addcu32(uint32_t *c, uint32_t *s, uint32_t x, uint32_t y)
{
   asm("{                               \n\t"
       "add.cc.u32         %0, %2, %3;    \n\t"
       "set.lt.u32.u32     %1, %0, %2; \n\t"
       "and.b32            %1, %1,1;     \n\t" 
       "}                               \n\t"
       : "=r"(s[0]), "=r"(c[0]) 
       : "r"(x), "r"(y));
}
/*
   Propagate carry bit across a 256 bit number starting in 32 bit word indexed by digit
*/
__forceinline__ __device__ void propcu32(uint32_t *x, uint32_t c, uint32_t digit)
{
   #pragma unroll
   for (; digit < NWORDS_256BIT_FIOS-1 ; digit++)
   {
     asm("{                                   \n\t"
         "add.cc.u32      %0,   %3, %2;   \n\t"
         "set.lt.u32.u32  %1,   %0, %2;   \n\t"
         "and.b32         %1,   %1,  1;      \n\t"
         "}                                   \n\t"
         : "=r"(x[digit]), "=r"(c) 
         : "r"(x[digit]), "r"(c));
   }
}
