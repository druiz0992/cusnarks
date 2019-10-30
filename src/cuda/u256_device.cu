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
#include "utils_device.h"
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
        modu255(&x[i*NWORDS_256BIT],&x[i*NWORDS_256BIT], params->midx);
        modu255(&y[i*NWORDS_256BIT],&y[i*NWORDS_256BIT], params->midx);
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

    extern __shared__ uint32_t smem[];
    uint32_t *smem_ptr = &smem[tid*NWORDS_256BIT];  // 0 .. blockDim

    uint32_t __restrict__ *x;
    uint32_t __restrict__ *z;
  
    if(idx >= params->in_length/params->stride) {
      return;
    }

    x = (uint32_t *) &in_vector[idx  * params->stride * U256K_OFFSET]; // 0 .. N-1

    if (gridDim.x == 1){
       z = (uint32_t *) out_vector;
    } else {
       z = (uint32_t *) &in_vector[blockIdx.x * U256K_OFFSET];  // 
    }

    if (params->premod){
      #pragma unroll
      for (i =0; i < params->stride; i++){
        modu255(&x[i*U256K_OFFSET],&x[i*U256K_OFFSET], params->midx);
      }
    }

    logDebugBigNumberTid(1,"smem[0]\n",smem_ptr);
    logDebugBigNumberTid(params->stride,"X[0]\n",&x[i * U256K_OFFSET]);

    addmu256(smem_ptr, (const uint32_t *)x, (const uint32_t *)&x[U256K_OFFSET], params->midx);

    logDebugBigNumberTid(1,"smem[i]\n",smem_ptr);

    #pragma unroll
    for (i =0; i < params->stride-2; i++){
      addmu256(smem_ptr, (const uint32_t *)smem_ptr, (const uint32_t *)&x[(i+2)*U256K_OFFSET], params->midx);

      logDebugTid("idx:%d\n",i);
      logDebugBigNumberTid(1,"smem[i]\n",smem_ptr);
    }
    __syncthreads();

    logDebugBigNumberTid(1,"smem[0]\n",smem_ptr);

    // reduction global mem
    if (blockDim.x >= 1024 && tid < 512){
      logDebugBigNumberTid(1,"+smem[0]\n",smem_ptr);
      logDebugBigNumberTid(1,"+smem[512]\n",&smem[(tid+512)*NWORDS_256BIT]);

      addmu256(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+512)*NWORDS_256BIT], params->midx);

      logDebugBigNumberTid(1,"smem[0]\n",smem_ptr);
    }
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256){
      logDebugBigNumberTid(1,"+smem[0]\n",smem_ptr);
      logDebugBigNumberTid(1,"+smem[256]\n",&smem[(tid+256)*NWORDS_256BIT]);

      addmu256(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+256)*NWORDS_256BIT], params->midx);

      logDebugBigNumberTid(1,"smem[=256]\n",smem_ptr);
    }
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128){
      logDebugBigNumberTid(1,"+smem[0]\n",smem_ptr);
      logDebugBigNumberTid(1,"+smem[128]\n",&smem[(tid+128)*NWORDS_256BIT]);

      addmu256(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+128)*NWORDS_256BIT], params->midx);

      logDebugBigNumberTid(1,"smem[=128+0]\n",smem_ptr);
    }
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64){
      logDebugBigNumberTid(1,"+smem[0]\n",smem_ptr);
      logDebugBigNumberTid(1,"+smem[64]\n",&smem[(tid+64)*NWORDS_256BIT]);

      addmu256(smem_ptr,
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+64)*NWORDS_256BIT], params->midx);

      logDebugBigNumberTid(1,"smem[=64+0]\n",smem_ptr);
    }
    __syncthreads();
    
    logDebugBigNumberTid(1,"smem[0]\n",smem_ptr);

    // unrolling warp
    if (tid < 32)
    {
        volatile uint32_t *vsmem = smem;
        logDebugBigNumberTid(1,"+smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumberTid(1,"+smem[32]\n",&smem[(tid+32)*NWORDS_256BIT]);

        addmu256(&vsmem[tid * NWORDS_256BIT],
                 &vsmem[tid * NWORDS_256BIT],
                 &vsmem[(tid+32)*NWORDS_256BIT], params->midx);

        logDebugBigNumberTid(1,"smem[=32+0]\n",(uint32_t *)vsmem);
        logDebugBigNumberTid(1,"+smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumberTid(1,"+smem[16]\n",&smem[(tid+16)*NWORDS_256BIT]);

        addmu256(&vsmem[tid*NWORDS_256BIT],
                 &vsmem[tid*NWORDS_256BIT],
                 &vsmem[(tid+16)*NWORDS_256BIT], params->midx);

        logDebugBigNumberTid(1,"smem[=16+0]\n",(uint32_t *)vsmem);
        logDebugBigNumberTid(1,"+smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumberTid(1,"+smem[8]\n",&smem[(tid+8)*NWORDS_256BIT]);

        addmu256(&vsmem[tid*NWORDS_256BIT],
                 &vsmem[tid*NWORDS_256BIT],
                 &vsmem[(tid+8)*NWORDS_256BIT], params->midx);

        logDebugBigNumberTid(1,"smem[=8+0]\n",(uint32_t *)vsmem);
        logDebugBigNumberTid(1,"smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumberTid(1,"smem[4]\n",&smem[(tid+4)*NWORDS_256BIT]);

        addmu256(&vsmem[tid*NWORDS_256BIT],
                 &vsmem[tid*NWORDS_256BIT],
                 &vsmem[(tid+4)*NWORDS_256BIT], params->midx);

        logDebugBigNumberTid(1,"smem[=4+0]\n",(uint32_t *)vsmem);
        logDebugBigNumberTid(1,"smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumberTid(1,"smem[2]\n",&smem[(tid+2)*NWORDS_256BIT]);

        addmu256(&vsmem[tid*NWORDS_256BIT],
                 &vsmem[tid*NWORDS_256BIT],
                 &vsmem[(tid+2)*NWORDS_256BIT], params->midx);

        logDebugBigNumberTid(1,"smem[=2+0]\n",(uint32_t *)vsmem);
        logDebugBigNumberTid(1,"smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumberTid(1,"smem[1]\n",&smem[(tid+1)*NWORDS_256BIT]);

        addmu256(&vsmem[tid*NWORDS_256BIT],
                 &vsmem[tid*NWORDS_256BIT],
                 &vsmem[(tid+1)*NWORDS_256BIT], params->midx);

        logDebugBigNumberTid(1,"smem[=0+1]\n",(uint32_t *)vsmem);

        if (tid==0) {
	   //TODO change be movu256
           memcpy(z, smem_ptr, sizeof(uint32_t) * NWORDS_256BIT);
           //movu256(z, smem_ptr);
           logDebugBigNumberTid(1,"Z : \n",smem_ptr);
        }
    }

      
}

__global__ void addmu256_reduce_shfl_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t sumX[] =  {0,0,0,0,0,0,0,0};
    uint32_t sumY[] =  {0,0,0,0,0,0,0,0};
    uint32_t i, size1,size2;

    extern __shared__ uint32_t smem[];

    uint32_t __restrict__ *z;
    uint32_t laneIdx = tid % warpSize;
    uint32_t warpIdx = tid / warpSize;
  
    if(idx >= params->in_length) {
      return;
    }

    movu256(sumX, &in_vector[idx * U256K_OFFSET]);
    if (params->premod){
      modu255(sumX, sumX, params->midx);
    }

    if (params->premul){
       size1 = blockDim.x >> 6;
       size2 = blockDim.x >= 32 ? 16 : blockDim.x/2;
    } else {
       size1 = 16;
       //asm("clz.b32    %0,%1;\n\t"
         //:"=r"(size2) : "r"(blockDim.x >> 6));
       size2 = blockDim.x >> 6;
    }

    logInfoTid("Size1 :%d\n",size1);
    logInfoTid("Size2 :%d\n",size2);
    // last step
    if (gridDim.x == 1){
       z = (uint32_t *) out_vector;
    } else {
       z = (uint32_t *) &in_vector[idx/blockDim.x * U256K_OFFSET];  // 
    }

    logInfoBigNumberTid(1,"X[0]\n",sumX);

    // block wide warp reduce
    #pragma unroll
    for (i = size1; i > 0; i >>= 1){
      shflxoru256(sumY, sumX, i);
      logInfoTid("idx:%d\n",i);
      logInfoBigNumberTid(1,"sumX\n",sumX);
      logInfoBigNumberTid(1,"sumY\n",sumY);

      addmu256(sumX, sumX, sumY, params->midx);

      logInfoBigNumberTid(1,"sumX+\n",sumX);
    }

    if (laneIdx == 0) {
       movu256(&smem[warpIdx*NWORDS_256BIT], sumX);
       logInfoTid("save idx:%d\n",warpIdx);
       logInfoBigNumberTid(1,"val\n",sumX);
    }

    __syncthreads();

    if (tid < size2*2) {
      logInfoTid("blockDim :%d\n",blockDim.x);
      logInfoTid("LaneIdx :%d\n",laneIdx);
      logInfoTid("Size :%d\n",size2);
      movu256(sumX,&smem[laneIdx*NWORDS_256BIT]);
      logInfoBigNumberTid(size2*2-idx,"Save\n",&smem[laneIdx*NWORDS_256BIT]);
    } else {
      set0u256(sumX);
    }
    logInfoBigNumberTid(1,"Second\n",sumX);
    #pragma unroll
    // last warp reduce
    for (i=size2; i > 0; i >>=1){
      shflxoru256(sumY, sumX, i);
      logInfoTid("idx:%d\n",i);
      logInfoBigNumberTid(1,"sumY\n",sumY);
      logInfoBigNumberTid(1,"sumX\n",sumX);
      addmu256(sumX, sumX, sumY, params->midx);
      logInfoBigNumberTid(1,"sumX+\n",sumX);
    }

    if (tid==0) {
     //TODO change be movu256
     //memcpy(z, sumX, sizeof(uint32_t) * NWORDS_256BIT);
     movu256(z, sumX);
     logInfoBigNumberTid(1,"Z : \n",sumX);
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
        modu255(&x[i*NWORDS_256BIT],&x[i*NWORDS_256BIT], params->midx);
        modu255(&y[i*NWORDS_256BIT],&y[i*NWORDS_256BIT], params->midx);
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
    
    modu255(z, x, params->midx);
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
      modu255(A,A, params->midx);
      modu255(B,B, params->midx);
    }

    mulmontu256(U, (const uint32_t *)A, (const uint32_t *) B, params->midx);

   return;
}

/*
  Montgomery multiplication extended field number
*/
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
      modu255(A,A, params->midx);
      modu255(B,B, params->midx);
      modu255(&A[2*NWORDS_256BIT],&A[2*NWORDS_256BIT], params->midx);
      modu255(&B[2*NWORDS_256BIT],&B[2*NWORDS_256BIT], params->midx);
    }

    mulmontu256_2(U, (const uint32_t *)A, (const uint32_t *) B, params->midx);

   return;
}

/*
  Right logical shift kernel
*/
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

    //memset causes blocking operations on current device
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
  Left logical shift kernel
*/
__global__ void shl1u256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i;
    uint32_t b_shifted;

    uint32_t __restrict__ *x;
    uint32_t __restrict__ *y;
    uint32_t __restrict__ *z;
   
    if(tid >= params->in_length) {
      return;
    }

    //memset causes blocking operations on current device
    x = (uint32_t *) &in_vector[tid * U256K_OFFSET + U256_XOFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    memset(z, 0, NWORDS_256BIT*sizeof(uint32_t));

    logInfoBigNumberTid(1,"X: \n",x);
    #pragma unroll
    for (i= NWORDS_256BIT*32-1; i>=0; i--){   
      b_shifted = shl1u256((const uint32_t *)x);
      z[i/32] |= (b_shifted << (i % 32));
      logInfoTid("C : %d\n",b_shifted);
      logInfoBigNumberTid(1,"X: \n",x);
      logInfoBigNumberTid(1,"Z: \n",z);
    }
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


// multiply by small constant (<= 32) 
__device__ void mulku256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, const uint32_t __restrict__ k, mod_t midx)
{
  uint32_t i;
  uint32_t tmp[NWORDS_256BIT+1];  //288 bits. I can multiply by up to 32

	   //TODO change be movu256
  //memcpy(tmp, x, NWORDS_256BIT * sizeof(uint32_t));
  movu256(tmp, (uint32_t *)x);

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
    //int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x * NWORDS_256BIT;
    uint32_t const __restrict__ *nonres = mod_info_ct[midx].nonres;
    uint32_t tmp1[NWORDS_256BIT],tmp2[NWORDS_256BIT],tmp3[NWORDS_256BIT],tmp4[NWORDS_256BIT];

    #if 0
    mulmontu256(tmp1, A,B,midx);                  
    mulmontu256(tmp2, &A[NWORDS_256BIT],&B[NWORDS_256BIT],midx); 

    mulmontu256(tmp3, A,&B[NWORDS_256BIT],midx);                  
    mulmontu256(tmp4, &A[NWORDS_256BIT],B,midx);                  

    submu256(U, tmp1, tmp2, midx);   
    addmu256(&U[NWORDS_256BIT], tmp3, tmp4, midx);                
    #else
    mulmontu256(tmp1, A,B,midx);                  
    mulmontu256(tmp2, &A[NWORDS_256BIT],&B[NWORDS_256BIT],midx); 

    addmu256(tmp3,A,&A[NWORDS_256BIT], midx);                
    addmu256(tmp4,B,&B[NWORDS_256BIT], midx);                
    mulmontu256(tmp3, tmp3,tmp4,midx); 
    submu256(U, tmp1, tmp2, midx);   
    addmu256(&U[NWORDS_256BIT], tmp1, tmp2, midx);                
    submu256(&U[NWORDS_256BIT], tmp3, &U[NWORDS_256BIT], midx);                

    #endif
 
    /* 
    addmu256(tmp1, A, &A[NWORDS_256BIT], midx);                // tmp1 = X[0] + X[1]
    addmu256(tmp2, B, &B[NWORDS_256BIT], midx);                // tmp2 = Y[0] + Y[1]
     
    mulmontu256(tmp1,tmp1,tmp2,midx);                          // tmp1 = tmp1 * tmp2  
    mulmontu256(tmp2,A,B,midx);                                // tmp2 = aA = X[0] * Y[0] 
    mulmontu256(U,&A[NWORDS_256BIT],&B[NWORDS_256BIT],midx);   // Z[0] = bB = X[1] * Y[1]

    submu256(&U[NWORDS_256BIT], tmp1, tmp2, midx);             //Z[1] = tmp1 - aA
    submu256(&U[NWORDS_256BIT], &U[NWORDS_256BIT], U, midx);   //Z[1] = tmp1 - aA - bB
    
    mulmontu256(U,U,nonres,midx);                                    // Z[0] = bB * non_residue
    addmu256(U,U,tmp2, midx);                                  // Z[0] = Z[0] + aA 
    */
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
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t const __restrict__ *nonres = mod_info_ct[midx].nonres;
    uint32_t tmp1[NWORDS_256BIT],tmp2[NWORDS_256BIT],tmp3[NWORDS_256BIT];

    /*
    logInfoBigNumberTid(1,"X[0]:\n",(uint32_t *)A);
    logInfoBigNumberTid(1,"X[1]:\n",(uint32_t *)&A[NWORDS_256BIT]);
    */

    mulmontu256(tmp1, A,&A[NWORDS_256BIT], midx);             // Z[1] = 2 * X[0] * X[1]

    sqmontu256(tmp2, A, midx);
    sqmontu256(tmp3, &A[NWORDS_256BIT], midx);

    addmu256(&U[NWORDS_256BIT],tmp1,tmp1,midx);
    submu256(U,tmp2, tmp3,midx);                                 // Z[0] = X[0] * X[0] - (X[1] * X[1])
    
    /*
    logInfoBigNumberTid(1,"U[0]:\n",(uint32_t *)U);
    logInfoBigNumberTid(1,"U[1]:\n",(uint32_t *)&U[NWORDS_256BIT]);
    */
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
    uint32_t S, C=0, C1, C2,C3;
    uint32_t __restrict__ M, X[2];
    uint32_t __restrict__ __align__(16) T[]={0,0,0,0,0,0,0,0,0,0};
    uint32_t const __restrict__ *PN_u256 = mod_info_ct[midx].p_;
    uint32_t const __restrict__ *P_u256 = mod_info_ct[midx].p;

    //movu256(Ar,(uint32_t *)A);
    //movu256(Br,(uint32_t *)B);

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    //logInfoBigNumberTid(1,"A\n",(uint32_t *)A);
    //logInfoBigNumberTid(1,"B\n",(uint32_t *)B);
    #pragma unroll
    for(i=0; i<NWORDS_256BIT; i++)
    {
      // (C,S) = t[0] + a[0]*b[i], worst case 2 words
      madcu32(&C,&S,A[0],B[i],T[0]);
      //logInfoTid("0 - C:%u\n",C);
      //logInfoTid("0 - S:%u\n",S);

      // ADD(t[1],C)
      //propcu32(T, C, 1);
      addcu32(&C3, &T[1], T[1], C);
      //logInfoTid("C3: %u\n",C3);
      //logInfoBigNumberTid(1,"T\n",T);

      // m = S*n'[0] mod W, where W=2^32
      // Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
      mulu32lo(&M, S, PN_u256[0]);
      //logInfoTid("M[0]: %u\n", M);

      // (C,S) = S + m*n[0], worst case 2 words
      madcu32(&C,&S,M,P_u256[0],S);
      //logInfoTid("1 - C: %u\n",C );
      //logInfoTid("1 - S: %u\n", S);
  
      // FIRST IT
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      mulu32(X, A[1], B[i]);
      addcu32(&C1, &S, T[1], C);
      addcu32(&C2, &S, S, X[0]);
      addcu32(&X[0], &C, C1, X[1]);
      addcu32(&C1, &C, C, C2);

      // ADD(t[2],C)
      //C +=C3;
      addcu32(&C2, &C, C, C3);
      C1 +=C2;
      addcu32(&C2, &T[2], T[2], C);
      C3 =C2 + C1;

      // (C,S) = S + m*n[1]
      madcu32(&C,&T[0],M, P_u256[1],S);

      //j=2
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      mulu32(X, A[2], B[i]);
      addcu32(&C1, &S, T[2], C);
      addcu32(&C2, &S, S, X[0]);
      addcu32(&X[0], &C, C1, X[1]);
      addcu32(&C1, &C, C, C2);
 
      // ADD(t[j+1],C)
      //C +=C3;
      addcu32(&C2, &C, C, C3);
      C1 +=C2;
      addcu32(&C2, &T[3], T[3], C);
      C3 =C2 + C1;
 
      // (C,S) = S + m*n[j]
      madcu32(&C,&T[1],M, P_u256[2],S);
 
      // j = 3
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      mulu32(X, A[3], B[i]);
      addcu32(&C1, &S, T[3], C);
      addcu32(&C2, &S, S, X[0]);
      addcu32(&X[0], &C, C1, X[1]);
      addcu32(&C1, &C, C, C2);

      // ADD(t[j+1],C)
      //C +=C3;
      addcu32(&C2, &C, C, C3);
      C1 +=C2;
      addcu32(&C2, &T[4], T[4], C);
      C3 =C2 + C1;

      // (C,S) = S + m*n[j]
      madcu32(&C,&T[2],M, P_u256[3],S);

      // j = 4
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      mulu32(X, A[4], B[i]);
      addcu32(&C1, &S, T[4], C);
      addcu32(&C2, &S, S, X[0]);
      addcu32(&X[0], &C, C1, X[1]);
      addcu32(&C1, &C, C, C2);

      // ADD(t[j+1],C)
      //C +=C3;
      addcu32(&C2, &C, C, C3);
      C1 +=C2;
      addcu32(&C2, &T[5], T[5], C);
      C3 =C2 + C1;

      // (C,S) = S + m*n[j]
      madcu32(&C,&T[3],M, P_u256[4],S);

      // j = 5
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      mulu32(X, A[5], B[i]);
      addcu32(&C1, &S, T[5], C);
      addcu32(&C2, &S, S, X[0]);
      addcu32(&X[0], &C, C1, X[1]);
      addcu32(&C1, &C, C, C2);

      // ADD(t[j+1],C)
      //C +=C3;
      addcu32(&C2, &C, C, C3);
      C1 +=C2;
      addcu32(&C2, &T[6], T[6], C);
      C3 =C2 + C1;

      // (C,S) = S + m*n[j]
      madcu32(&C,&T[4],M, P_u256[5],S);

      // j = 6
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      mulu32(X, A[6], B[i]);
      addcu32(&C1, &S, T[6], C);
      addcu32(&C2, &S, S, X[0]);
      addcu32(&X[0], &C, C1, X[1]);
      addcu32(&C1, &C, C, C2);

      // ADD(t[j+1],C)
      //C +=C3;
      addcu32(&C2, &C, C, C3);
      C1 +=C2;
      addcu32(&C2, &T[7], T[7], C);
      C3 =C2 + C1;

      // (C,S) = S + m*n[j]
      madcu32(&C,&T[5],M, P_u256[6],S);

      // j = 7
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      mulu32(X, A[7], B[i]);
      addcu32(&C1, &S, T[7], C);
      addcu32(&C2, &S, S, X[0]);
      addcu32(&X[0], &C, C1, X[1]);
      addcu32(&C1, &C, C, C2);

      // ADD(t[j+1],C)
      //C +=C3;
      addcu32(&C2, &C, C, C3);
      C1 +=C2;
      addcu32(&C2, &T[8], T[8], C);
      C3 =C2 + C1;

      // (C,S) = S + m*n[j]
      madcu32(&C,&T[6],M, P_u256[7],S);

      //propcu32_extend(T,C3);
      // (C,S) = t[s] + C
      addcu32(&C,&T[7], T[NWORDS_256BIT], C);
      //logInfoTid("6 - C: %u\n",C );
      //logInfoTid("6 - S: %u\n",S );
      // t[s-1] = S
      // t[s] = t[s+1] + C
      addcu32(&X[0],&T[NWORDS_256BIT], T[NWORDS_256BIT+1], C);
      // t[s+1] = 0
      T[NWORDS_256BIT+1] = 0;
   }

   /* Step 3: if(u>=n) return u-n else return u */
   if (ltu256(P_u256,T)){
   //if (P_u256[NWORDS_256BIT-1] < T[NWORDS_256BIT-1]){
      subu256(U,T,P_u256);
   } else {
       movu256(U,T);
   }
   return;
}

#if 0
// SOS Version is commented out because FIOS performs better (even squaring). I leave it here in case
// i decide to optimized

// SOS
__device__ void mulmontu256(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, const uint32_t __restrict__ *B, mod_t midx)
{ 
    uint32_t i,j;
    uint32_t S, C=0, C1, C2,C3=0,C4=0;
    uint32_t __restrict__ M[2], X[2];
    uint32_t __restrict__ T[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    uint32_t const __restrict__ *PN_u256 = mod_info_ct[midx].p_;
    uint32_t const __restrict__ *P_u256 = mod_info_ct[midx].p;

    logDebugBigNumber("P\n",(uint32_t *)P_u256);
    logDebugBigNumber("PN\n",(uint32_t *)PN_u256);
    logDebugBigNumber("A\n", (uint32_t*) A);
    logDebugBigNumber("B\n", (uint32_t *)B);

    #if 0
    #pragma unroll
    for(i=0; i<NWORDS_256BIT; i++) {
       C=0;
       #pragma unroll
       for (j=0; j < NWORDS_256BIT; j++){
          //(C,S) := t[i+j] + a[j]*b[i] + C
          madcu32(&C, &S,A[j],B[i],C);
          addcu32(&C1, &T[i+j], T[i+j], S);
	  C += C1;
       }
       T[i+NWORDS_256BIT] = C;
    }
    #endif

    #pragma unroll
    for(i=0; i<NWORDS_256BIT; i++) {
       //(C,S) := t[i+i] + a[i]*a[i]
       madcu32(&C, &T[i,i],A[i],A[i],T[i+i]);
       #pragma unroll
       for (j=i+1; j < NWORDS_256BIT; j++){
          //(C,S) := t[i+j] + 2*a[j]*a[i] + C
          madcu32(&C, &S,A[j],A[i],C);
          addcu32(&C1, &T[i+j], T[i+j], S);
	  C += C1;
       }
       T[i+NWORDS_256BIT] += C;
    }

    #pragma unroll
    for (i=0; i < NWORDS_256BIT; i++) {
       C = 0;
       // m = S*n'[0] mod W, where W=2^32
       // Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
       mulu32(M, T[i], PN_u256[0]);
       #pragma unroll
       for (j=0; j < NWORDS_256BIT; j++){
         //(C,S) := t[i+j] + m*n[j] + C
         madcu32(&C,&S,M[0], P_u256[j],C);
         addcu32(&C1, &T[i+j], T[i+j], S);
         C += C1;
	 //C += ((C3 >> (i+j+1)) & 0x1);
	 //C3 &= (0xFFFFFFFF ^ (1 << (1+j+i)));
       }
       addcu32(&C4, &T[i+NWORDS_256BIT], T[i+NWORDS_256BIT], C);
       //C3 |= (C4 << (i+NWORDS_256BIT +1));
    }

    movu256(U,&T[NWORDS_256BIT]);
  
    /* Step 3: if(u>=n) return u-n else return u */
   if (ltu256(P_u256,U)){
      subu256(U,U,P_u256);
   }
   logDebugBigNumber("U after mod\n",U);

   return;
}
#endif

__device__ void sqmontu256(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, mod_t midx)
{
   //TODO : implement proper squaring
   #if 1
   mulmontu256(U,A,A,midx);
   #else
    uint32_t i,j;
    uint32_t S, C=0, C1, C2,C3,C4;
    uint32_t __restrict__ M[2], X[2];
    uint32_t __restrict__ __align__(16) T[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    uint32_t const __restrict__ *PN_u256 = mod_info_ct[midx].p_;
    uint32_t const __restrict__ *P_u256 = mod_info_ct[midx].p;

    #pragma unroll
    for(i=0; i<NWORDS_256BIT; i++)
    {
      // (C,S) = t[i+i] + a[i]*a[i], worst case 2 words
      madcu32(&C,&S,A[i],A[i],T[i+i]);
      T[i+i] = S;

      #pragma unroll
      for (j=i; j < NWORDS_256BIT; j++){
        //(C,S) := t[i+j] + 2*a[j]*a[i] + C
        madcu32(&C,&S,A[j],A[i],T[i+j]);
        C1 = (X[0] >> 31)+C2;
        X[0] <<= 1;
        C2 = X[1] >> 31;
        X[1] = (X[1] << 1) + C1;
        addcu32(&C1, &X[0], X[0], C);
        addcu32(&C, &S, T[i+j],X[0]);
        C += C1;
        addcu32(&X[0], &C, C,X[1]);
        T[i+j] = S;
      }
      T[i+NWORDS_256BIT] += C;
    }

    #pragma unroll
    for (i=0; i< NWORDS_256BIT; i++){
       C = 0;
       //m := t[i]*n'[0] mod W
       mulu32(M, T[i], PN_u256[0]);

       #pragma unroll
       for (j=0; j < NWORDS_256BIT; j++){
           //(C,S) := t[i+j] + m*n[j] + C
           mulu32(X, M[0], P_u256[j]);
           addcu32(&C1, &X[0], X[0],C);
           addcu32(&C, &S, T[i+j], X[0]);
           C += C1;
           C += ((C3 >> (i+j+1)) & 1);
	   C3 &= (0xFFFFFFFF ^ (1 << (1+j+i))); 
           addcu32(&X[0], &C, C, X[1]);
           T[i+j] = S;
       }
         addcu32(&C4,&T[i+NWORDS_256BIT], T[i+NWORDS_256BIT], C);
         C3 |= (C4 << (i+NWORDS_256BIT+1));  
    }
    movu256(U,&T[NWORDS_256BIT]);

    /* Step 3: if(u>=n) return u-n else return u */
    if (ltu256(P_u256,U)){
       subu256(U,U,P_u256);
    }
   return;

   #endif
}

/*
   x mod N

   NOTE : It requires that prime is at least 253 bit number. In reality is 254 bits the prime
    i am using
   */
__device__ void modu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, mod_t midx)
{
   const uint32_t __restrict__ *p = mod_info_ct[midx].p;

   movu256(z,(uint32_t *)x);

   // x(255 bit number worst case) - p (253 bit number) = z (255 bit number) : ex 31(5 b) - 4(3 b) =27 (5 b) 
   if (!ltu256(z,p)){
      subu256(z,z,p);
   } else { return; }
   // x(255 bit ) - p (253 bit number) = z (255 bit number) : ex 27(5 b) - 4(3 b) =23 (5 b) 
   if (!ltu256(z,p)){
      subu256(z,z,p);
   } else { return; }
   // x(255 bit ) - p (253 bit number) = z (255 bit number) : ex 23(5 b) - 4(3 b) = 19 (5 b) 
   if (!ltu256(z,p)){
      subu256(z,z,p);
   } else { return; }
   // x(255 bit ) - p (253 bit number) = z (254 bit number) : ex 19(5 b) - 4(3 b) = 15 (4 b) 
   if (!ltu256(z,p)){
      subu256(z,z,p);
   } else { return; }
   // x(254 bit ) - p (253 bit number) = z (254 bit number) : ex 15(5 b) - 4(3 b) = 11 (4 b) 
   if (!ltu256(z,p)){
      subu256(z,z,p);
   } else { return; }
   // x(254 bit ) - p (253 bit number) = z (254 bit number) : ex 11(5 b) - 4(3 b) = 7 (3 b) 
   if (!ltu256(z,p)){
      subu256(z,z,p);
   } else { return; }
   // x(254 bit ) - p (253 bit number) = z (254 bit number) : ex 7(5 b) - 4(3 b) = 3 (3 b) 
   if (!ltu256(z,p)){
      subu256(z,z,p);
   } else { return; }
 
   assert(0);
}

/*
   x mod N

   NOTE : It requires that prime is at least 253 bit number and less than 256 bit (msb must be 0). In reality is 254 bits the prime
    i am using. modu255 is more efficient that modu256
   */
__device__ void modu255(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, mod_t midx)
{
   const uint32_t __restrict__ *p = mod_info_ct[midx].p;

   #if 1
   movu256(z,(uint32_t *)x);
   #else
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
  #endif

  if (!subgtu256(z,p)) return;
  else if (!subgtu256(z,p)) return;
  else if (!subgtu256(z,p)) return;
  else if (!subgtu256(z,p)) return;
  else if (!subgtu256(z,p)) return;
  else if (!subgtu256(z,p)) return;
  else if (!subgtu256(z,p)) return;

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
   x << 1 for 256 bit number
*/
__device__ uint32_t shl1u256(const uint32_t __restrict__ *x)
{
   uint32_t c; 
   asm("{                                    \n\t"
       ".reg .u32          %tmp;             \n\t"
       "bfe.u32            %8,   %16,  31,1;  \n\t"       // c = x[7] & (1<<31)
       "shl.b32            %7,   %16,  1;         \n\t"   // x[7] = x[7] << 1
       "bfe.u32            %tmp, %15, 31,1;  \n\t"        // tmp = x[6] & (1<<31)
       "or.b32             %7,   %7,  %tmp;  \n\t"        // x[7] |= tmp
       "shl.b32            %6,   %15,  1;         \n\t"   // x[6] = x[6] << 1
       "bfe.u32            %tmp, %14, 31,1;  \n\t"        // tmp = x[5] & (1<<31)
       "or.b32             %6,   %6,  %tmp;  \n\t"        // x[6] |= tmp
       "shl.b32            %5,   %14,  1;         \n\t"   // x[5] = x[5] << 1
       "bfe.u32            %tmp, %13, 31,1;  \n\t"        // tmp = x[4] & (1<<31)
       "or.b32             %5,   %5,  %tmp;  \n\t"        // x[5] |= tmp
       "shl.b32            %4,   %13,  1;         \n\t"   // x[4] = x[4] << 1
       "bfe.u32            %tmp, %12, 31,1;  \n\t"        // tmp = x[3] & (1<<31)
       "or.b32             %4,   %4,  %tmp;  \n\t"        // x[4] |= tmp
       "shl.b32            %3,   %12,  1;         \n\t"   // x[3] = x[3] << 1
       "bfe.u32            %tmp, %11, 31,1;  \n\t"        // tmp = x[2] & (1<<31)
       "or.b32             %3,   %3,  %tmp;  \n\t"        // x[3] |= tmp
       "shl.b32            %2,   %11,  1;         \n\t"   // x[2] = x[2] << 1
       "bfe.u32            %tmp, %10, 31,1;  \n\t"        // tmp = x[1] & (1<<31)
       "or.b32             %2,   %2,  %tmp;  \n\t"        // x[2] |= tmp
       "shl.b32            %1,   %10,  1;         \n\t"   // x[1] = x[1] << 1
       "bfe.u32            %tmp, %9, 31,1;  \n\t"        // tmp = x[0] & (1<<31)
       "or.b32             %1,   %1,  %tmp;  \n\t"        // x[1] |= tmp
       "shl.b32            %0,   %9,  1;         \n\t"   // x[0] = x[0] << 1
       "}                               \n\t"
       : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]), 
         "=r"(x[4]), "=r"(x[5]), "=r"(x[6]), "=r"(x[7]), "=r"(c)
       : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), 
         "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));

      return c;
}


/*
   (x & (1<< bsel)) >> bsel  for 256 bit number
*/
__device__ uint32_t bselu256(const uint32_t __restrict__ *x, uint32_t bsel)
{
   uint32_t c;
   uint32_t word = bsel >> NBITS_WORD_LOG2; // bsel/32 gives the word number
   uint32_t bit = bsel & NBITS_WORD_MOD; // bsel % 32 gives bit number

   asm("{                                       \n\t"
         "bfe.u32            %0,   %1,  %2, 1;  \n\t"      
       "}                                       \n\t"
       : "=r"(c)
       : "r"(x[word]), "r"(bit));
    
   return c;
}

__device__ uint32_t bselMu256(const uint32_t __restrict__ *x, uint32_t bsel)
{
   uint32_t c,i, rc=0; 
   uint32_t word = bsel >> NBITS_WORD_LOG2; // bsel/32 gives the word number
   uint32_t bit = bsel & NBITS_WORD_MOD; // bsel % 32 gives bit number

   #pragma unroll
   for (i=0; i< U256_BSELM; i++){
     asm("{                                       \n\t"
           "bfe.u32            %0,   %1,  %2, 1;  \n\t"      
         "}                                       \n\t"
         : "=r"(c)
         : "r"(x[NWORDS_256BIT*i+word]), "r"(bit));
    
     rc += (c << i);  
   }

   return rc;
}



/*
  returns number of leading zeros in a 256 bit number
*/
__device__ uint32_t clzMu256(const uint32_t __restrict__ *x)
{
   uint32_t i,j, c, rc, mrc=255; 
  
   #pragma unroll 
   for (i=0; i< U256_BSELM; i++){
     c = 32;    
     rc = 0;
     for (j=NWORDS_256BIT; j >= 1 && c == 32; j--){
        asm("{                                    \n\t"
            "   clz.b32           %0,%2;          \n\t"
            "   add.u32           %1, %3, %0;     \n\t"      
            "}                   \n\t"
            :"=r"(c), "=r"(rc) : "r"(x[NWORDS_256BIT*i+j-1]), "r"(rc));
     }
     if (rc <= mrc) { mrc = rc; }
   }

   return mrc;
}

__device__ uint32_t clzu256(const uint32_t __restrict__ *x)
{
   uint32_t j, c, rc, mrc=255; 
   
   c = 32;    
   rc = 0;
   for (j=NWORDS_256BIT; j >= 1 && c == 32; j--){
      asm("{                                    \n\t"
          "   clz.b32           %0,%2;          \n\t"
          "   add.u32           %1, %3, %0;     \n\t"      
          "}                   \n\t"
          :"=r"(c), "=r"(rc) : "r"(x[j-1]), "r"(rc));
   }

   return rc;
}

// returns 1 if x - y >= y and x = x-y
// returns 0 if x - y <= y
__forceinline__ __device__ uint32_t subgtu256(uint32_t __restrict__ *x, const uint32_t __restrict__ *y)
{
   uint32_t z[NWORDS_256BIT];
   uint32_t r, flag;

   // 
   subu256(z,x,y);

   asm("clz.b32    %0,%1;\n\t"
       :"=r"(r) : "r"(z[NWORDS_256BIT-1]));
   flag = r > 0;
   if ((r == 32) && eq0u256(z)){
     flag = 0;
   } 
   if (flag){
     movu256(x,z);
   }
   return flag;
}


__forceinline__ __device__ void shflxoru256(uint32_t *d_out, uint32_t *d_in, uint32_t srcLane )
{
    ulonglong4 in, *out;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    in = *(ulonglong4 *)d_in;
    out = (ulonglong4 *)d_out;

    out->x = __shfl_xor_sync(0xffffffff, in.x, srcLane);
    out->y = __shfl_xor_sync(0xffffffff, in.y, srcLane);
    out->z = __shfl_xor_sync(0xffffffff, in.z, srcLane);
    out->w = __shfl_xor_sync(0xffffffff, in.w, srcLane);
}
