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
#include "utils_device.h"
#include "u256_device.h"
#include "zpoly_device.h"

/*
    Modular addition kernel

*/
__global__ void fft32_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t __restrict__ *x;
    uint32_t __restrict__ *z;
   
    if(tid >= (params->in_length)) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    
    if (params->premod){
       modu256(x,x, params->midx);
    }

    if (tid == 0){
        logInfo("len : %d\n",params->in_length);
        for (uint32_t i=0; i< 32; i++){
           logInfoBigNumber("x[i]: \n",&x[i*NWORDS_256BIT]);
        }
    }
    fft32(x, params->midx);
    memcpy(z, x , sizeof(uint32_t) * NWORDS_256BIT);
}

__global__ void ifft32_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t __restrict__ *x;
    uint32_t __restrict__ *z;
   
    if(tid >= (params->in_length >> 5)) {
      return;
    }

    x = (uint32_t *) &in_vector[ (tid * U256K_OFFSET) << 5];
    z = (uint32_t *) &out_vector[(tid * U256K_OFFSET) << 5]; 

    
    if (params->premod){
      #pragma unroll
      for (i=0; i< 32; i++){
        modu256(&x[i*NWORDS_256BIT],&x[i*NWORDS_256BIT], params->midx);
      }
    }
    ifft32(x, params->midx);
    memcpy(z, x , sizeof(uint32_t) * 32 * NWORDS_256BIT);
}

/*
   32Point DIF FFT inplace. Input samples are ordered. Output samples are unordered
*/
__device__ void fft32(uint32_t *x, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t otherX[] = {0,0,0,0,0,0,0,0};
  uint32_t const *W32 = W32_ct;
  uint32_t lane = threadIdx.x % warpSize;
  uint32_t i, size;

  #pragma unroll
  for (i=ZPOLY_FFT_N-1; i>=1 ; i--){
    size = 1 << i;
    if (tid == 16){
      logInfoBigNumber("PreB OtherX :\n",otherX);
      logInfoBigNumber("PreB X :\n",x);
      logInfo("size : %d\n",size);
    }

    fft_butterfly(otherX, x, size);

    if (tid == 16){
      logInfoBigNumber("AfterB OtherX :\n",otherX);
    }

    if (lane%(size<<1) < size){  //  It 0: 0-15      W0,W1,...,W15
                                 //  It 1: 0-7, 16-23    W0,W2,W4,..W14
                                 //  It 2: 0-3, 8-11, 16-19, 24-27  W0, W4, W8, W12
                                 //  It 3: 0-1, 4-5, 8-9, 12-13,...  W0, W8

      addmu256(x, x, otherX, midx);
      if (tid == 16){ 
        logInfo("Op : x + otherX : lane%(size>>1) : %d, size : %d\n", lane%(size<<1),size);
      }
    } else {
      submu256(x,otherX, x,  midx);
      mulmontu256(x, x, &W32[ (1<< (ZPOLY_FFT_N-i+1)) * (lane%size)], midx);
      if (tid == 16){ 
        logInfo("Op : otherX - x : lane%(size>>1) : %d, size : %d\n", lane%(size<<1),size);
      }
    }
    if (tid == 16){
      logInfoBigNumber("AfterB x:\n",x);
      logInfoBigNumber("Root : \n",(uint32_t *)&W32[ (1 << (ZPOLY_FFT_N-i+1)) * (lane%size)]);
      logInfo("idx : %d\n",(1 << (ZPOLY_FFT_N-i+1)) * (lane % size));
    }
  }
  // I can skip mulypliying by 1 in the last iteration
  //  It 4: 0,2,4,6,...    W0
    if (tid == 16){
      logInfoBigNumber("PreB OtherX :\n",otherX);
      logInfoBigNumber("PreB X :\n",x);
      logInfo("size : %d\n",size);
    }
  fft_butterfly(otherX, x, 1);
    if (tid == 16){
      logInfoBigNumber("AfterB OtherX :\n",otherX);
    }
  if (lane % 2 == 0){  
      addmu256(x, x, otherX, midx);
      if (tid == 16){ 
        logInfo("Op : x + otherX : lane%(size>>1) : %d\n", lane%2);
      }
  } else {
      submu256(x, otherX, x, midx);
      if (tid == 16){ 
        logInfo("Op : otherX - x : lane%(size>>1) : %d\n", lane%2);
      }
  }
    if (tid == 16){
      logInfoBigNumber("AfterB x:\n",x);
      logInfoBigNumber("Root : \n",(uint32_t *)&W32[ (1 << (ZPOLY_FFT_N-i+1)) * (lane%size)]);
      logInfo("idx : %d\n",(1 << (ZPOLY_FFT_N-i+1)) * (lane % size));
    }
}

/*
   32Point DIT FFT inplace. Input samples are unordered. Output samples are ordered
*/
__device__ void ifft32(uint32_t *x, mod_t midx)
{
  uint32_t otherX[] = {0,0,0,0,0,0,0,0};
  uint32_t const *IW32 = IW32_ct;
  uint32_t const *inv_scaler = IW32_nroots_ct;
  uint32_t lane = threadIdx.x % warpSize;
  uint32_t i, size;

  //  It 0: 0,2,4,6,...    W0
  fft_butterfly(otherX, x, 1);
  if (lane % 2){
    addmu256(x, x, otherX, midx);
  } else {
    submu256(x, x, otherX, midx);
  }

  #pragma unroll
  for (i=1; i < ZPOLY_FFT_N ; i++){
    size = 1 << i;    // size = 2,4,8,16

    if (lane % (size >> 1) >= size){  
    mulmontu256(x, x, &IW32[ (1 >> (ZPOLY_FFT_N - 1 - i)) * lane%size], midx);
    }
    fft_butterfly(otherX, x, size);
    if (lane % (size >> 1) < size){
      addmu256(x, x, otherX, midx);
    } else {
      submu256(x, x, otherX, midx);
    }
 }

  // TODO . For now, I will use scaler in montgomery assuming that everything is in mongomery. Althugh this is going to change
  // depending on the format of input data, scaler must be in Montgomery or normal format.
  // If X is normal, W is montgomery => result is normal and scaler must be normal. 
  //     Scaler = 32. Inv Scaler 21204235282094297871551205565717985242031228012903033270457635305745314480129L
  // If X is Montgomery, W is Mongtgomery => Result is montgomery and scaler can be normal or Montgomery
  //    Scaler =.  Inv Scaler = 3618502788666131106986593281521497120414687020801267626233049500247285301248L => 1<< 251
 mulmontu256(x,x,inv_scaler, midx);
}


__forceinline__ __device__ void fft_butterfly(uint32_t *d_out, uint32_t *d_in, uint32_t srcLane )
{
    ulonglong4 in, *out;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    in = *(ulonglong4 *)d_in;
    out = (ulonglong4 *)d_out;

    # if 0
    if (tid == 0){
      logInfoBigNumber("BUT IN: \n",d_in);
      logInfoBigNumber("BUT OUT: \n",d_out);
      logInfo("IN.X[0]: %x\n",in.x&0xFFFFFFFF);
      logInfo("IN.X[1]: %x\n",(in.x >> 32) &0xFFFFFFFF);
      logInfo("IN.Y[0]: %x\n",in.y&0xFFFFFFFF);
      logInfo("IN.Y[1]: %x\n",(in.y >> 32) &0xFFFFFFFF);
      logInfo("IN.Z[0]: %x\n",in.z&0xFFFFFFFF);
      logInfo("IN.Z[1]: %x\n",(in.z >> 32) &0xFFFFFFFF);
      logInfo("IN.W[0]: %x\n",in.w&0xFFFFFFFF);
      logInfo("IN.W[1]: %x\n",(in.w >> 32) &0xFFFFFFFF);
    }
    #endif

    out->x = __shfl_xor_sync(0xffffffff, in.x, srcLane);
    out->y = __shfl_xor_sync(0xffffffff, in.y, srcLane);
    out->z = __shfl_xor_sync(0xffffffff, in.z, srcLane);
    out->w = __shfl_xor_sync(0xffffffff, in.w, srcLane);
    # if 0
    if (tid == 0){
      logInfo("OUT.X[0]: %x\n",out->x&0xFFFFFFFF);
      logInfo("OUT.X[1]: %x\n",(out->x >> 32) &0xFFFFFFFF);
      logInfo("OUT.Y[0]: %x\n",out->y&0xFFFFFFFF);
      logInfo("OUT.Y[1]: %x\n",(out->y >> 32) &0xFFFFFFFF);
      logInfo("OUT.Z[0]: %x\n",out->z&0xFFFFFFFF);
      logInfo("OUT.Z[1]: %x\n",(out->z >> 32) &0xFFFFFFFF);
      logInfo("OUT.W[0]: %x\n",out->w&0xFFFFFFFF);
      logInfo("OUT.W[1]: %x\n",(out->w >> 32) &0xFFFFFFFF);
    }
    #endif
}

