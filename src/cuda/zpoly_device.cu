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
__global__ void zpoly_fft32_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t *x, *z;
   
    if(tid >= (params->in_length)) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    
    if (params->premod){
       modu256(x,x, params->midx);
    }

    /*
    if (tid == 0){
        logInfo("len : %d\n",params->in_length);
        for (uint32_t i=0; i< 32; i++){
           logInfoBigNumber("x[i]: \n",&x[i*NWORDS_256BIT]);
        }
    }
    */
    fft32_dif(z,x, params->midx);
    //memcpy(z, x , sizeof(uint32_t) * NWORDS_256BIT);
}

__global__ void zpoly_ifft32_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t *x, *z;
   
    if(tid >= params->in_length) {
      return;
    }

    x = (uint32_t *) &in_vector[ tid * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET]; 

    if (params->premod){
      modu256(x,x, params->midx);
    }

    ifft32_dit(z, x, params->midx);
    //memcpy(z, x , sizeof(uint32_t) * NWORDS_256BIT);
}

__global__ void zpoly_mul32_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t *x, *y, *z;
   
    if(tid >= params->in_length/2) {
      return;
    }

    x = (uint32_t *) &in_vector[ tid * U256K_OFFSET];
    y = (uint32_t *) &in_vector[ params->in_length/2 * U256K_OFFSET + tid * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET]; 

    if (params->premod){
      modu256(x,x, params->midx);
    }

    if (tid==0){
      for (i=0; i<32 ; i++){
         logInfoBigNumber("X in \n",&x[i*U256K_OFFSET]);
      }
    }
    fft32_dif(x, x, params->midx);
    if (tid==0){
      for (i=0; i<32 ; i++){
         logInfoBigNumber("X out \n",&x[i*U256K_OFFSET]);
      }
    }
    if (tid==0){
      for (i=0; i<32 ; i++){
         logInfoBigNumber("Y in \n",&y[i*U256K_OFFSET]);
      }
    }
    fft32_dif(y, y, params->midx);
    if (tid==0){
      for (i=0; i<32 ; i++){
         logInfoBigNumber("Y out \n",&y[i*U256K_OFFSET]);
      }
    }
    mul_poly(z,x,y, 31, params->midx);
    if (tid==0){
      for (i=0; i<32 ; i++){
         logInfoBigNumber("X*Y out \n",&z[i*U256K_OFFSET]);
      }
    }
    ifft32_dit(z,z, params->midx);
    if (tid==0){
      for (i=0; i<32 ; i++){
         logInfoBigNumber("result \n",&z[i*U256K_OFFSET]);
      }
    }
    //memcpy(z, x , sizeof(uint32_t) * NWORDS_256BIT);
}

/*
    Modular addition kernel

*/
__global__ void zpoly_fftN_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t *x, *z;
    uint32_t N = params->fft_Nx;
   
    if(tid >= (params->in_length)) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    
    if (params->premod){
       modu256(x,x, params->midx);
    }

    /*
    if (tid == 0){
        logInfo("len : %d\n",params->in_length);
        for (uint32_t i=0; i< 32; i++){
           logInfoBigNumber("x[i]: \n",&x[i*NWORDS_256BIT]);
        }
    }
    */
    fftN_dif(z,x, N, params->midx);
    //memcpy(z, x , sizeof(uint32_t) * NWORDS_256BIT);
}

__global__ void zpoly_ifftN_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t *x, *z;
    uint32_t N = params->fft_Nx;
   
    if(tid >= params->in_length) {
      return;
    }

    x = (uint32_t *) &in_vector[ tid * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET]; 

    if (params->premod){
      modu256(x,x, params->midx);
    }

    ifftN_dit(z, x, N, params->midx);
    //memcpy(z, x , sizeof(uint32_t) * NWORDS_256BIT);
}

__global__ void zpoly_mulN_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t *x, *y, *z;
    uint32_t N = params->fft_Nx;
   
    if(tid >= params->in_length/2) {
      return;
    }

    x = (uint32_t *) &in_vector[ tid * U256K_OFFSET];
    y = (uint32_t *) &in_vector[ params->in_length/2 * U256K_OFFSET + tid * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET]; 

    if (params->premod){
      modu256(x,x, params->midx);
    }

    fftN_dif(x, x,N, params->midx);
    fftN_dif(y, y,N, params->midx);
    mul_poly(z,x,y, 31, params->midx);
    ifftN_dit(z,z,N, params->midx);
    //memcpy(z, x , sizeof(uint32_t) * NWORDS_256BIT);
}


__global__ void zpoly_fft2DX_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    uint32_t Nx = params->fft_Nx;
    uint32_t Ny = params->fft_Ny;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t *x;
    //uint32_t *x, *z;
    if (tid == 0){
      logInfo("Nx: %d\n",Nx);
      logInfo("Ny: %d\n",Ny);
    }
   
    if(tid > params->in_length/2){
      return;
    }

    x = (uint32_t *) &in_vector[tid * U256K_OFFSET];
    //z = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    
    if (params->premod){
       modu256(x,x, params->midx);
    }

    /*
    if (tid == 0){
        logInfo("len : %d\n",params->in_length);
        for (uint32_t i=0; i< 32; i++){
           logInfoBigNumber("x[i]: \n",&x[i*NWORDS_256BIT]);
        }
    }
    */
    fft2Dx_dif(out_vector,in_vector, Nx, Ny, params->midx);
    //memcpy(z, x , sizeof(uint32_t) * NWORDS_256BIT);
}

__global__ void zpoly_fft2DY_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    uint32_t Nx = params->fft_Nx;
    uint32_t Ny = params->fft_Ny;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //uint32_t *x, *z;
    if (tid == 0){
      logInfo("Nx: %d\n",Nx);
      logInfo("Ny: %d\n",Ny);
    }
   
    if(tid > params->in_length){
      return;
    }

    //z = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    
    /*
    if (tid == 0){
        logInfo("len : %d\n",params->in_length);
        for (uint32_t i=0; i< 32; i++){
           logInfoBigNumber("x[i]: \n",&x[i*NWORDS_256BIT]);
        }
    }
    */
    fft2Dy_dif(out_vector,out_vector, Nx, Ny, params->midx);
    //memcpy(z, x , sizeof(uint32_t) * NWORDS_256BIT);
}



__global__ void zpoly_fft3D_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
  return;
}

/*
  Input : X[0], X[1],....,X[Nx*Ny-1], for Nx,Ny = 1,2,4,8,16,32. Min Nx*Ny = 32

  Output : Z[0], Z[1],...Z[(Nx*Ny-1], where Z[i] are the FFT coefficients. FFt is done in place and results unordered

  Steps
    1) Store results in 2D matrix, filling it by column  X[0]     X[Ny]    X[2Ny]   ....  X[(Nx-1) * Ny]
                                                         X[1]     X[Ny+1]  X[2Ny+1] ..... X[(Nx-1) * Ny + 1]
                                                         ..........................................
                                                         X[Ny-1]  X[2Ny-1] X[3Ny-1] ..... X[Nx     * Ny - 1]

    2) Perform Ny x Nx point FFt (FFT of rows), and store results same matrix
    3) Mulitiply matrix elements Aik by root[j*k]
    4) Perform Nx x Ny point FFT (FFT of columns)
    5) Convert to 1D verctor, filling it by row
*/
__device__ void fft2Dx_dif(uint32_t *z, uint32_t *x, uint32_t Nx, uint32_t Ny, mod_t midx)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t new_ridx = (tid << Ny) & ((1 << (Nx + Ny))-1);
  uint32_t new_cidx = tid >> Nx;
  uint32_t new_cidx2, new_ridx2;
  uint32_t *roots = &x[(1<<(Nx+Ny)) * U256K_OFFSET];

  // FFT rows (Every Ny points)
  //fftN_dif(&z[tid*U256K_OFFSET], &x[(new_cidx + new_ridx)*U256K_OFFSET],Nx,midx);
  //fftN_dif(&z[(new_cidx + new_ridx)*U256K_OFFSET], &x[(new_cidx + new_ridx)*U256K_OFFSET],Nx,midx);
  fftN_dif(&z[tid*U256K_OFFSET], &x[(new_cidx + new_ridx)*U256K_OFFSET],Nx,midx);
  mul_poly(&z[tid*U256K_OFFSET],&z[tid*U256K_OFFSET],&roots[(tid/(1<<Nx)) * (tid % (1<<Ny))*U256K_OFFSET], 0, midx);
  //mul_poly(&z[(new_cidx + new_ridx)*U256K_OFFSET],&x[(new_cidx + new_ridx)*U256K_OFFSET],roots, (1<<N)-1, params->midx);

    if (tid == 0){
        for (uint32_t i=0; i< 32; i++){
           new_ridx2 = (i << Ny) & ((1 << (Nx + Ny))-1);
           new_cidx2 = i >> Nx;
           logInfo("idx: %d\n",new_cidx2 + new_ridx2);
           logInfoBigNumber("x[i]:\n",&z[(new_cidx2 + new_ridx2)*U256K_OFFSET]);
           //logInfoBigNumber("x[i]: %d \n",&z[i*U256K_OFFSET]);
        }
    }
  // FFT columns (Every point)
  //fftN_dif(&z[tid*U256K_OFFSET], &z[(new_cidx + new_ridx) * U256K_OFFSET],Ny,midx);
  //fftN_dif(&z[tid*U256K_OFFSET], &z[tid* U256K_OFFSET],Ny,midx);
}
__device__ void fft2Dy_dif(uint32_t *z, uint32_t *x, uint32_t Nx, uint32_t Ny, mod_t midx)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t new_ridx = (tid << Ny) & ((1 << (Nx + Ny))-1);
  uint32_t new_cidx = tid >> Nx;
  uint32_t new_cidx2, new_ridx2;
  uint32_t *roots = &x[(1<<(Nx+Ny)) * U256K_OFFSET];

  // FFT rows (Every Ny points)
  //fftN_dif(&z[tid*U256K_OFFSET], &x[(new_cidx + new_ridx)*U256K_OFFSET],Nx,midx);
  //fftN_dif(&z[(new_cidx + new_ridx)*U256K_OFFSET], &x[(new_cidx + new_ridx)*U256K_OFFSET],Nx,midx);
  //fftN_dif(&z[tid*U256K_OFFSET], &x[(new_cidx + new_ridx)*U256K_OFFSET],Nx,midx);
  //mul_poly(&z[tid*U256K_OFFSET],&z[tid*U256K_OFFSET],&roots[(tid/(1<<Nx)) * (tid % (1<<Ny))*U256K_OFFSET], 0, midx);
  //mul_poly(&z[(new_cidx + new_ridx)*U256K_OFFSET],&x[(new_cidx + new_ridx)*U256K_OFFSET],roots, (1<<N)-1, params->midx);

    if (tid == 0){
        logInfo("Ny: %d\n",Ny);
        for (uint32_t i=0; i< 32; i++){
           new_ridx2 = (i << Ny) & ((1 << (Nx + Ny))-1);
           new_cidx2 = i >> Nx;
           logInfo("idx: %d\n",new_cidx2 + new_ridx2);
           logInfoBigNumber("x[i]:\n",&z[(new_cidx2 + new_ridx2)*U256K_OFFSET]);
           //logInfoBigNumber("x[i]: %d \n",&z[i*U256K_OFFSET]);
        }
    }
  // FFT columns (Every point)
  //fftN_dif(&z[tid*U256K_OFFSET], &z[(new_cidx + new_ridx) * U256K_OFFSET],Ny,midx);
  fftN_dif(&z[tid*U256K_OFFSET], &z[tid* U256K_OFFSET],Ny,midx);
}
/*
  Multiply 2 poly of degre d
*/
__device__ void mul_poly(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t d, mod_t midx)
{
  uint32_t i, coeff_idx;

  #pragma unroll
  for (i=0; i<= d; i++) {
    coeff_idx = i * NWORDS_256BIT;
    mulmontu256(&z[coeff_idx],&x[coeff_idx],&y[coeff_idx],midx);
  }
}
/*
   32Point DIF FFT inplace. Input samples are ordered. Output samples are unordered
*/
__device__ void fft32_dif(uint32_t *z, uint32_t *x, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t otherX[] = {0,0,0,0,0,0,0,0};
  uint32_t thisX[] =  {0,0,0,0,0,0,0,0};
  uint32_t const *W32 = W32_ct;
  uint32_t lane = threadIdx.x % warpSize;
  uint32_t i, size, root_idx;

  movu256(thisX,x);

  #pragma unroll
  for (i=ZPOLY_FFT_32-1; i>=1 ; i--){
    size = 1 << i;

    /*
    if (tid == 2){
      logInfoBigNumber("PreB OtherX :\n",otherX);
      logInfoBigNumber("PreB X :\n",thisX);
      logInfo("size : %d\n",size);
    }
    */

    fft_butterfly(otherX, thisX, size);

    /*
    if (tid == 2){
      logInfoBigNumber("AfterB OtherX :\n",otherX);
    }
    */

    root_idx = (1<< (ZPOLY_FFT_32-i-1)) * (lane%size) * NWORDS_256BIT;
    if (lane%(size<<1) < size){  //  It 0: 0-15      W0,W1,...,W15
                                 //  It 1: 0-7, 16-23    W0,W2,W4,..W14
                                 //  It 2: 0-3, 8-11, 16-19, 24-27  W0, W4, W8, W12
                                 //  It 3: 0-1, 4-5, 8-9, 12-13,...  W0, W8

      addmu256(thisX, thisX, otherX, midx);
      /*
        if (tid == 2){ 
          logInfo("Op : x + otherX : lane%(size>>1) : %d, size : %d\n", lane%(size<<1),size);
        }
      */
    } else {
      submu256(thisX, otherX, thisX,  midx);
      mulmontu256(thisX, thisX, &W32[ root_idx], midx);
      /*
      if (tid == 2){ 
        logInfo("Op : otherX - x : lane%(size>>1) : %d, size : %d\n", lane%(size<<1),size);
      }
      */
    }
    /*
    if (tid == 2){
      logInfoBigNumber("AfterB x:\n",thisX);
      logInfoBigNumber("Root : \n",(uint32_t *)&W32[ root_idx]);
      logInfo("idx : %d\n", root_idx / NWORDS_256BIT);
    }
    */
  }
    /*
    if (tid == 2){
      logInfoBigNumber("PreB OtherX :\n",otherX);
      logInfoBigNumber("PreB X :\n",thisX);
      logInfo("size : %d\n",size);
    }
    */
  // I can skip mulypliying by 1 in the last iteration
  //  It 4: 0,2,4,6,...    W0
  fft_butterfly(otherX, thisX, 1);
    /*
    if (tid == 2){
      logInfoBigNumber("AfterB OtherX :\n",otherX);
    }
    */
  if (lane % 2 == 0){  
      addmu256(z, thisX, otherX, midx);
      /*
      if (tid == 2){ 
        logInfo("Op : x + otherX : lane%(size>>1) : %d\n", lane%2);
      }
      */
  } else {
      submu256(z, otherX, thisX, midx);
       /*
      if (tid == 2){ 
        logInfo("Op : otherX - x : lane%(size>>1) : %d\n", lane%2);
      }
      */
  }
    /*
    if (tid == 2){
      logInfoBigNumber("AfterB x:\n",z);
      logInfoBigNumber("Root : \n",(uint32_t *)&W32[ (1 << (ZPOLY_FFT_32-i-1)) * (lane%size) * NWORDS_256BIT]);
      logInfo("idx : %d\n",(1 << (ZPOLY_FFT_32-i-1)) * (lane % size));
    }
    */
}

/*
   I32Point DIT FFT inplace. Input samples are unordered. Output samples are ordered
*/
__device__ void ifft32_dit(uint32_t *z, uint32_t *x, mod_t midx)
{
  uint32_t otherX[] = {0,0,0,0,0,0,0,0};
  uint32_t thisX[] = {0,0,0,0,0,0,0,0};
  uint32_t const *IW32 = IW32_ct;
  uint32_t const *inv_scaler = &IW32_nroots_ct[(ZPOLY_FFT_32-1)*NWORDS_256BIT];
  uint32_t lane = threadIdx.x % warpSize;
  uint32_t i, size, root_idx;

  movu256(thisX,x);

  //  It 0: 0,2,4,6,...    W0
  fft_butterfly(otherX, thisX, 1);
  if (lane % 2 == 0){
    addmu256(thisX, thisX, otherX, midx);
  } else {
    submu256(thisX, otherX, thisX, midx);
  }

  #pragma unroll
  for (i=1; i < ZPOLY_FFT_32 ; i++){
    size = 1 << i;    // size = 2,4,8,16
    root_idx = (1 << (ZPOLY_FFT_32 - 1 - i)) * (lane%size) * NWORDS_256BIT;

    if (lane % (size << 1) >= size){  
       mulmontu256(thisX, thisX, &IW32[root_idx], midx);
    }
    fft_butterfly(otherX, thisX, size);
    if (lane % (size << 1) < size){
      addmu256(thisX, thisX, otherX, midx);
    } else {
      submu256(thisX, otherX, thisX, midx);
    }
 }

  // TODO . For now, I will use scaler in montgomery assuming that everything is in mongomery. Althugh this is going to change
  // depending on the format of input data, scaler must be in Montgomery or normal format.
  // If X is normal, W is montgomery => result is normal and scaler must be normal. 
  //     Scaler = 32. Inv Scaler 21204235282094297871551205565717985242031228012903033270457635305745314480129L
  // If X is Montgomery, W is Mongtgomery => Result is montgomery and scaler can be normal or Montgomery
  //    Scaler =.  Inv Scaler = 3618502788666131106986593281521497120414687020801267626233049500247285301248L => 1<< 251
 mulmontu256(z,thisX,inv_scaler, midx);
}

/*
   N-Point DIF FFT inplace. Input samples are ordered. Output samples are unordered
*/
__device__ void fftN_dif(uint32_t *z, uint32_t *x, uint32_t N, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t otherX[] = {0,0,0,0,0,0,0,0};
  uint32_t thisX[] =  {0,0,0,0,0,0,0,0};
  uint32_t const *W32 = W32_ct;
  uint32_t lane = threadIdx.x % warpSize;
  //uint32_t lane = threadIdx.x % (1 << N);
  uint32_t i, size, root_idx, debug_tidx =0;

  movu256(thisX,x);

  if (tid == debug_tidx){
    logInfo("N : %d\n",N);
    for (i=0;i<32; i++){
       logInfoBigNumber("X in\n",&x[i*U256K_OFFSET]);
    }
   logInfoBigNumber("ThisX in\n",thisX);
  }

  #pragma unroll
  for (i=N-1; i>=1 ; i--){
    size = 1 << i;

    if (tid == debug_tidx){
      logInfoBigNumber("PreB OtherX :\n",otherX);
      logInfoBigNumber("PreB X :\n",thisX);
      logInfo("size : %d\n",size);
    }

    fft_butterfly(otherX, thisX, size);

    if (tid == debug_tidx){
      logInfoBigNumber("AfterB OtherX :\n",otherX);
    }

    //root_idx = ((1<< (N-i-1)) * (lane%size) * NWORDS_256BIT) >> (ZPOLY_FFT_32 - N);
    root_idx = ((1<< (N-i-1)) * (lane%size) * NWORDS_256BIT) << (ZPOLY_FFT_32 - N);
    if (lane%(size<<1) < size){  //  It 0: 0-15      W0,W1,...,W15
                                 //  It 1: 0-7, 16-23    W0,W2,W4,..W14
                                 //  It 2: 0-3, 8-11, 16-19, 24-27  W0, W4, W8, W12
                                 //  It 3: 0-1, 4-5, 8-9, 12-13,...  W0, W8

      addmu256(thisX, thisX, otherX, midx);
        if (tid == debug_tidx){ 
          logInfo("Op : x + otherX : lane%(size>>1) : %d, size : %d\n", lane%(size<<1),size);
        }
    } else {
      submu256(thisX, otherX, thisX,  midx);
      mulmontu256(thisX, thisX, &W32[ root_idx], midx);
      if (tid == debug_tidx){ 
        logInfo("Op : otherX - x : lane%(size>>1) : %d, size : %d\n", lane%(size<<1),size);
      }
    }
    if (tid == debug_tidx){
      logInfoBigNumber("AfterB x:\n",thisX);
      logInfoBigNumber("Root : \n",(uint32_t *)&W32[ root_idx]);
      logInfo("idx : %d\n", root_idx / (NWORDS_256BIT * (1 << (ZPOLY_FFT_32 - N))) );
    }
  }
    if (tid == debug_tidx){
      logInfoBigNumber("PreB OtherX :\n",otherX);
      logInfoBigNumber("PreB X :\n",thisX);
      logInfo("size : %d\n",size);
    }
  // I can skip mulypliying by 1 in the last iteration
  //  It 4: 0,2,4,6,...    W0
  fft_butterfly(otherX, thisX, 1);
    if (tid == debug_tidx){
      logInfoBigNumber("AfterB OtherX :\n",otherX);
    }
  if (lane % 2 == 0){  
      addmu256(z, thisX, otherX, midx);
      if (tid == debug_tidx){ 
        logInfo("Op : x + otherX : lane%(size>>1) : %d\n", lane%2);
      }
  } else {
      submu256(z, otherX, thisX, midx);
      if (tid == debug_tidx){ 
        logInfo("Op : otherX - x : lane%(size>>1) : %d\n", lane%2);
      }
  }
    if (tid == debug_tidx){
      logInfoBigNumber("AfterB x:\n",z);
      logInfo("idx : %d\n",(1 << (N-i-1)) * (lane % size));
      logInfoBigNumber("Root : \n",(uint32_t *)&W32[ (1 << (N-i-1)) * (lane%size) * NWORDS_256BIT]);
      for (i=0;i<32; i++){
         logInfoBigNumber("X out\n",&z[i*U256K_OFFSET]);
      }
    }
}

/*
   I32Point DIT FFT inplace. Input samples are unordered. Output samples are ordered
*/
__device__ void ifftN_dit(uint32_t *z, uint32_t *x, uint32_t N, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t otherX[] = {0,0,0,0,0,0,0,0};
  uint32_t thisX[] = {0,0,0,0,0,0,0,0};
  uint32_t const *IW32 = IW32_ct;
  uint32_t const *inv_scaler = &IW32_nroots_ct[(N-1)*NWORDS_256BIT];
  uint32_t lane = threadIdx.x % warpSize;
  uint32_t i, size, root_idx, debug_tidx=0;

  movu256(thisX,x);

  if (tid == debug_tidx){
    logInfo("N : %d\n",N);
    for (i=0;i<32; i++){
       logInfoBigNumber("X in\n",&x[i*U256K_OFFSET]);
    }
  }

  //  It 0: 0,2,4,6,...    W0
  fft_butterfly(otherX, thisX, 1);
  if (lane % 2 == 0){
    addmu256(thisX, thisX, otherX, midx);
  } else {
    submu256(thisX, otherX, thisX, midx);
  }

  #pragma unroll
  for (i=1; i < N ; i++){
    size = 1 << i;    // size = 2,4,8,16
    //root_idx = ((1<< (N-i-1)) * (lane%size) * NWORDS_256BIT);
    root_idx = ((1<< (N-i-1)) * (lane%size) * NWORDS_256BIT) << (ZPOLY_FFT_32 - N);

    if (lane % (size << 1) >= size){  
       mulmontu256(thisX, thisX, &IW32[root_idx], midx);
    }
    fft_butterfly(otherX, thisX, size);
    if (lane % (size << 1) < size){
      addmu256(thisX, thisX, otherX, midx);
    } else {
      submu256(thisX, otherX, thisX, midx);
    }
 }

  // TODO . For now, I will use scaler in montgomery assuming that everything is in mongomery. Althugh this is going to change
  // depending on the format of input data, scaler must be in Montgomery or normal format.
  // If X is normal, W is montgomery => result is normal and scaler must be normal. 
  //     Scaler = 32. Inv Scaler 21204235282094297871551205565717985242031228012903033270457635305745314480129L
  // If X is Montgomery, W is Mongtgomery => Result is montgomery and scaler can be normal or Montgomery
  //    Scaler =.  Inv Scaler = 3618502788666131106986593281521497120414687020801267626233049500247285301248L => 1<< 251
 mulmontu256(z,thisX,inv_scaler, midx);
  if (tid == debug_tidx){
    for (i=0;i<32; i++){
       logInfoBigNumber("X out\n",&z[i*U256K_OFFSET]);
    }
  }
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

