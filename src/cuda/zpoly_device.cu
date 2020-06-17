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

// longer poly first
__global__ void zpoly_add_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t *x, *y, *z;
   
    if(tid >= params->padding_idx) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * NWORDS_FR];
    y = (uint32_t *) &in_vector[(params->in_length - params->padding_idx + tid) * NWORDS_FR];
    z = (uint32_t *) &out_vector[tid * NWORDS_FR];
   
    if (params->premod){
       modu256(x,x, params->midx);
       modu256(y,y, params->midx);
    }
    addmu256(z,x,y, params->midx);                

    if ( (params->in_length > 2*params->padding_idx) && (tid == 0)){
       memcpy(&z[params->padding_idx*NWORDS_FR],
              &y[params->padding_idx*NWORDS_FR],
              (params->in_length - 2*params->padding_idx)*NWORDS_FR*sizeof(uint32_t));
    }

    return;
}
// first poly longer 
__global__ void zpoly_sub_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t *x, *y, *z;

    if(tid >= params->padding_idx) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * NWORDS_FR];
    y = (uint32_t *) &in_vector[(params->in_length - params->padding_idx + tid) * NWORDS_FR];
    z = (uint32_t *) &out_vector[tid * NWORDS_FR];
    
    if (params->premod){
       modu256(x,x, params->midx);
       modu256(y,y, params->midx);
    }
    submu256(z,x,y, params->midx);                

    if ( (params->in_length > 2*params->padding_idx) && (tid == 0)){
       memcpy(&z[params->padding_idx*NWORDS_FR],
              &x[params->padding_idx*NWORDS_FR],
              (params->in_length - 2*params->padding_idx)*NWORDS_FR*sizeof(uint32_t));
    }

    return;
}

__global__ void zpoly_subprev_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t *x, *y, *z;

    if(tid >= params->padding_idx) {
      return;
    }

    x = (uint32_t *) &out_vector[tid * NWORDS_FR];
    y = (uint32_t *) &in_vector[tid * NWORDS_FR];
    z = (uint32_t *) &out_vector[tid * NWORDS_FR];
    
    if (params->premod){
       modu256(x,x, params->midx);
    }
    submu256(z,x,y, params->midx);                

    return;
}

__global__ void zpoly_mulc_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t *x, *y, *z;

    if(tid >= (params->in_length/2)) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * NWORDS_FR];
    y = (uint32_t *) &in_vector[(params->in_length/2 + tid) * NWORDS_FR];
    z = (uint32_t *) &out_vector[tid * NWORDS_FR];
    
    if (params->premod){
       modu256(x,x, params->midx);
       modu256(y,y, params->midx);
    }

    mulmontu256(z, (const uint32_t *)x,(const uint32_t *) y, params->midx);

    return;
}

__global__ void zpoly_mulcprev_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t *x, *y, *z;

    if(tid >= (params->in_length)) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * NWORDS_FR];
    y = (uint32_t *) &out_vector[tid * NWORDS_FR];
    z = (uint32_t *) &out_vector[tid * NWORDS_FR];
    
    if (params->premod){
       modu256(x,x, params->midx);
    }

    mulmontu256(z, (const uint32_t *)x,(const uint32_t *) y, params->midx);

    return;
}

__global__ void zpoly_mulK_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t *scl, *x,*z;

    if(tid >= (params->in_length - 1)) {
      return;
    }

    scl = (uint32_t *) &in_vector[0];
    x = (uint32_t *) &in_vector[tid * NWORDS_FR + NWORDS_FR];
    z = (uint32_t *) &out_vector[tid * NWORDS_FR];
    logInfoBigNumberTid(1,"SCL: \n",scl); 
    logInfoBigNumberTid(1,"X: \n",x); 
    
    if (params->premod){
       modu256(x,x, params->midx);
    }

    mulmontu256(z, (const uint32_t *)scl,(const uint32_t *) x, params->midx);
    logInfoBigNumberTid(1,"Z: \n",z); 
    return;
}

// add longest element first
__global__ void zpoly_madprev_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t *scl, *x,*z;

    if(tid >= (params->in_length - 1)) {
      return;
    }

    scl = (uint32_t *) &in_vector[0];
    x = (uint32_t *) &in_vector[tid * NWORDS_FR + NWORDS_FR];
    z = (uint32_t *) &out_vector[tid * NWORDS_FR];
    
    if (params->premod){
       modu256(x,x, params->midx);
    }

    logInfoBigNumberTid(1,"SCL: \n",scl); 
    logInfoBigNumberTid(1,"X: \n",x); 
    logInfoBigNumberTid(1,"Z: \n",z); 

    mulmontu256(x, (const uint32_t *)scl,(const uint32_t *) x, params->midx);
    logInfoBigNumberTid(1,"X: \n",x); 
    addmu256(z,z,x, params->midx);                
    logInfoBigNumberTid(1,"Z: \n",z); 

    return;
}

// add longest element first
__global__ void zpoly_addprev_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t *x,*z;

    if(tid >= params->in_length) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * NWORDS_FR];
    z = (uint32_t *) &out_vector[tid * NWORDS_FR];
    
    if (params->premod){
       modu256(x,x, params->midx);
    }

    addmu256(z,z,x, params->midx);                

    return;
}

__global__ void zpoly_divsnarks_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t *x, *y, *z;
    uint32_t nd = params->forward;
    uint32_t ne = params->padding_idx;
    // from python code:
    // me = m + nd = params->in_length
    // ne = n + nd = params->padding_idx
    // nd = params->forward
    // rem = x
   
    if(tid >= params->in_length - 2*ne + nd) {
      return;
    }

    x = (uint32_t *) &in_vector[(tid + ne) * NWORDS_FR];
    y = (uint32_t *) &in_vector[(tid + 2 * ne - nd) * NWORDS_FR];
    z = (uint32_t *) &out_vector[tid * NWORDS_FR];
   
    if (params->premod){
       modu256(x,x, params->midx);
    }

    divsnarks(z,x,y,params);

}

__global__ void zpoly_divsnarksprev_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t *x, *y, *z;
    uint32_t nd = params->forward;
    uint32_t ne = params->padding_idx;
    // from python code:
    // me = m + nd = params->in_length
    // ne = n + nd = params->padding_idx
    // nd = params->forward
    // rem = x
   
    if(tid >= params->in_length - 2*ne + nd) {
      return;
    }

    x = (uint32_t *) &in_vector[(tid + ne) * NWORDS_FR];
    y = (uint32_t *) &in_vector[(tid + 2 * ne - nd) * NWORDS_FR];
    z = (uint32_t *) &out_vector[tid * NWORDS_FR];
   
    if (params->premod){
       modu256(x,x, params->midx);
    }

    divsnarks(z,x,y,params);

}
__device__ void divsnarks(uint32_t *z_t, uint32_t *x_t, uint32_t *y_t, kernel_params_t *params)
{
    uint32_t nd = params->forward;
    uint32_t ne = params->padding_idx;
    uint32_t offset=0;
    uint32_t *x, *y;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    logInfoTid("ne : %d\n",ne);
    logInfoTid("nd : %d\n",nd);
    logInfoTid("inlength : %d\n",params->in_length);
    logInfoTid("max_tid : %d\n",params->in_length - 2*ne + nd- 1);

    logInfoBigNumberTid(1,"Z:\n",z_t);
    logInfoBigNumberTid(1,"X:\n",x_t);
    logInfoBigNumberTid(1,"Y:\n",y_t);
    addmu256(z_t,x_t,y_t, params->midx);                
    logInfoBigNumberTid(1,"Z:\n",z_t);

    // TODO : Try to have groups of 8 so that it is faster (I cannot unroll automatically)
    //for(offset=ne+1; offset<params->in_length -2*ne-1 ; offset += ne+1) { 
    for(offset=2*(ne-nd); offset<params->in_length -ne-1 ; offset += 2*(ne-nd)) { 
       logInfoTid("new_lim : %d\n",params->in_length - offset);
       logInfoTid("max_tid1 : %d\n",params->in_length - offset -ne);
       logInfoTid("max_tid2 : %d\n",params->in_length - offset -ne +nd);
       if(tid >= params->in_length - offset -ne) {
           return;
       }
       x = (uint32_t *) &x_t[(offset) * NWORDS_FR];
       logInfoTid("offset : %d\n",offset);
       logInfoBigNumberTid(1,"X:\n",x);
       addmu256(z_t,z_t,x, params->midx);                

       if(tid >= params->in_length - offset -ne+ nd) {
           return;
       }
       y = (uint32_t *) &y_t[(offset) * NWORDS_FR];
       logInfoBigNumberTid(1,"Y:\n",y);
       addmu256(z_t,z_t,y, params->midx);                
       logInfoBigNumberTid(1,"Z:\n",z_t);

       //i++;
       //if (i==3){ return;}
 
    }
    //x = (uint32_t *) &in_vector[(tid + ne + offset) * NWORDS_FR];
    //logInfoBigNumberTid(1,"Z:\n",z);
    //logInfoBigNumberTid(1,"X:\n",x);
    //addmu256(z,z,&x[offset*NWORDS_FR], params->midx);                

    //logInfoBigNumberTid(1,"Z:\n",z);

    return;
}

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
           logInfoBigNumber("x[i]: \n",&x[i*NWORDS_FR]);
        }
    }
    */
    fft32_dif(z,x, params->midx);
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

    /*
    if (tid==0){
      for (i=0; i<32 ; i++){
         logInfoBigNumber("X in \n",&x[i*U256K_OFFSET]);
      }
    }
    */
    fft32_dif(x, x, params->midx);
    /*
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
    */
    fft32_dif(y, y, params->midx);
    /*
    if (tid==0){
      for (i=0; i<32 ; i++){
         logInfoBigNumber("Y out \n",&y[i*U256K_OFFSET]);
      }
    }
    */
    mul_poly(z,x,y, 31, params->midx);
    /*
    if (tid==0){
      for (i=0; i<32 ; i++){
         logInfoBigNumber("X*Y out \n",&z[i*U256K_OFFSET]);
      }
    }
    // TODO
    // last step (IFFT) cannot be done within the same kernel. I don't understand why, but
    // this FFT is not used for anything so it is not a big deal having to call IFFT32 in a separate
    // step. See test_1fft_mul32() in test_cu_zpoly.py, f
    */
    //ifft32_dit(z,z, params->midx);
    /*
    if (tid==0){
      for (i=0; i<32 ; i++){
         logInfoBigNumber("result \n",&z[i*U256K_OFFSET]);
      }
    }
    */
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
    uint32_t const *W32 = W32_ct;
   
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
           logInfoBigNumber("x[i]: \n",&x[i*NWORDS_FR]);
        }
    }
    */
    fftN_dif(z,x, W32, N, params->midx);
}

__global__ void zpoly_ifftN_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t *x, *z;
    uint32_t N = params->fft_Nx;
    uint32_t const *W32 = IW32_ct;
   
    if(tid >= params->in_length) {
      return;
    }

    x = (uint32_t *) &in_vector[ tid * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET]; 

    if (params->premod){
      modu256(x,x, params->midx);
    }

    ifftN_dit(z, x, W32, N, params->midx);
}

__global__ void zpoly_mulN_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    uint32_t *x, *y, *z;
    uint32_t N = params->fft_Nx;
    uint32_t const *W32 = W32_ct;
    uint32_t const *IW32 = IW32_ct;
   
    if(tid >= params->in_length/2) {
      return;
    }

    x = (uint32_t *) &in_vector[ tid * U256K_OFFSET];
    y = (uint32_t *) &in_vector[ params->in_length/2 * U256K_OFFSET + tid * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET]; 

    if (params->premod){
      modu256(x,x, params->midx);
    }

    fftN_dif(x, x, W32,N, params->midx);
    fftN_dif(y, y, W32, N, params->midx);
    mul_poly(z,x,y, 31, params->midx);
    ifftN_dit(z,z, IW32, N, params->midx);
}


__global__ void zpoly_fft2DX_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    uint32_t Nx = params->fft_Nx;
    uint32_t Ny = params->fft_Ny;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t *x;

    if(tid >= params->in_length/2){
      return;
    }

    x = (uint32_t *) &in_vector[tid * U256K_OFFSET];
    
    if (params->premod){
       modu256(x,x, params->midx);
    }

    fft2Dx_dif(in_vector,in_vector, params);
 
}

__global__ void zpoly_fft2DY_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= params->in_length){
      return;
    }

    fft2Dy_dif(out_vector,in_vector, params);
 
}

__global__ void zpoly_fft3DXX_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x, *w1;
    uint32_t N1 = params->N_fftx + params->N_ffty;

    if(tid >= params->padding_idx){
      return;
    }

    x = (uint32_t *) in_vector;
    w1 = (uint32_t *) &in_vector[(1<<N1) * U256K_OFFSET];
    
    fft3Dxx_dif(x, x,  w1, params);

}

__global__ void zpoly_fft3DXXprev_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x, *z, *w1;
    uint32_t N1 = params->N_fftx + params->N_ffty;

    if(tid >= params->in_length){
      return;
    }

    x = (uint32_t *) in_vector;
    z = (uint32_t *) out_vector;
    w1 = (uint32_t *) &in_vector[(1<<N1) * U256K_OFFSET];

    fft3Dxx_dif(x, z, w1, params);
 
}


__global__ void zpoly_fft3DXY_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x, *z, *w1;
    uint32_t N1 = params->N_fftx + params->N_ffty;

    if(tid >= params->in_length){
      return;
    }

    x = (uint32_t *) in_vector;
    z = (uint32_t *) out_vector;
    w1 = (uint32_t *) &in_vector[(1<<N1) * U256K_OFFSET];
    
    fft3Dxy_dif(z, x, w1, params);
}

__global__ void zpoly_fft3DYX_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x, *z, *w1;
    uint32_t N1 = params->N_fftx + params->N_ffty;

    if(tid >= params->in_length){
      return;
    }

    x = (uint32_t *) in_vector;
    z = (uint32_t *) out_vector;
    w1 = (uint32_t *) &in_vector[(1<<N1) * U256K_OFFSET];
    
    fft3Dyx_dif(x, z, w1, params);
}

__global__ void zpoly_fft3DYY_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x, *z, *scaler;
    uint32_t N1 = params->N_fftx + params->N_ffty;

    if(tid >= params->in_length){
      return;
    } 
   
    x = (uint32_t *) in_vector;
    z = (uint32_t *) out_vector;
    scaler = (uint32_t *) &in_vector[(2<<N1) * U256K_OFFSET];

    fft3Dyy_dif(z,
                x, NULL,
                scaler, params);
}

__global__ void zpoly_fft4DXX_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t *x1, *w1, *z1;
    const uint32_t N1 = params->N_fftx + params->N_ffty;

    if(tid >= params->padding_idx){
      return;
    }

    x1 = (uint32_t *) in_vector;
    z1 = (uint32_t *) out_vector;

    if ( params->premul){
       w1 = (uint32_t *) &in_vector[2*params->padding_idx * U256K_OFFSET];
    } else {
       w1 = (uint32_t *) &in_vector[(2*params->padding_idx + params->stride - (1 << N1)) * U256K_OFFSET];
    }

    fft3Dxx_dif(z1, x1, w1, params);

}

__global__ void zpoly_fft4DXY_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x1, *z1, *w1;
    const uint32_t N1 = params->N_fftx + params->N_ffty;

    if(tid >= params->padding_idx){
      return;
    }
 
    x1 = (uint32_t *) in_vector;   
    z1 = (uint32_t *) out_vector;

    if ( params->premul){
       w1 = (uint32_t *) &in_vector[2*params->padding_idx * U256K_OFFSET];
    } else {
       w1 = (uint32_t *) &in_vector[(2*params->padding_idx + params->stride - (1 << N1)) * U256K_OFFSET];
    }

    fft3Dxy_dif(x1,z1, w1, params);
}

__global__ void zpoly_fft4DYX_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x1, *z1, *w1;
    const uint32_t N1 = params->N_fftx + params->N_ffty;

    if(tid >= params->in_length){
      return;
    }

    x1 = (uint32_t *) in_vector;   
    z1 = (uint32_t *) out_vector;

    if ( params->premul){
       w1 = (uint32_t *) &in_vector[2*params->padding_idx * U256K_OFFSET];
    } else {
       w1 = (uint32_t *) &in_vector[(2*params->padding_idx + params->stride - (1 << N1)) * U256K_OFFSET];
    }

    fft3Dyx_dif(z1, x1, w1, params);
}

__global__ void zpoly_fft4DYY_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x1, *z1, *w2, *tmp1, *scaler;

    if(tid >= params->in_length){
      return;
    } 

    x1 = (uint32_t *) in_vector;   
    z1 = (uint32_t *) out_vector;
    tmp1 = (uint32_t *) &out_vector[params->padding_idx * U256K_OFFSET];
    scaler = (uint32_t *) &in_vector[(2*params->padding_idx + params->stride) * U256K_OFFSET];   

    w2 = (uint32_t *) &in_vector[params->padding_idx * U256K_OFFSET]; // N/2 roots 
  
    fft3Dyy_dif(tmp1, z1, w2, scaler, params);
    
}

__global__ void zpoly_interp3DXX_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t *x1, *x2, *w1, *w3, *z1, *tmp1, *tmp2;

    if(tid >= params->padding_idx){
      return;
    }

    x1 = (uint32_t *) in_vector;
    x2 = (uint32_t *) &in_vector[params->padding_idx * U256K_OFFSET];
    w1 = (uint32_t *) &in_vector[2*params->padding_idx * U256K_OFFSET];
    w3 = (uint32_t *) &in_vector[(2*params->padding_idx + params->stride) * U256K_OFFSET];   

    z1 = (uint32_t *) out_vector;
    tmp1 = (uint32_t *) &out_vector[2*params->padding_idx * U256K_OFFSET];
    tmp2 = (uint32_t *) &out_vector[3*params->padding_idx * U256K_OFFSET];

    // X1 * X2
    if (!params->forward){
       mulmontu256( &z1[2*tid*U256K_OFFSET], 
                   &x1[tid*U256K_OFFSET],
                   &x2[tid*U256K_OFFSET], params->midx);
    } 

    fft3Dxx_dif(tmp1, x1, w1, params);
    fft3Dxx_dif(tmp2, x2, w1, params);

}

__global__ void zpoly_interp3DXY_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x1, *x2, *tmp1, *tmp2, *w1;

    if(tid >= params->padding_idx){
      return;
    }
 
    x1 = (uint32_t *) in_vector;   
    x2 = (uint32_t *) &in_vector[params->padding_idx * U256K_OFFSET];   
    w1 = (uint32_t *) &in_vector[2*params->padding_idx * U256K_OFFSET];   

    tmp1 = (uint32_t *) &out_vector[2*params->padding_idx * U256K_OFFSET];   
    tmp2 = (uint32_t *) &out_vector[3*params->padding_idx * U256K_OFFSET];   

    fft3Dxy_dif(x1,tmp1, w1, params);
    fft3Dxy_dif(x2,tmp2, w1, params);
}

__global__ void zpoly_interp3DYX_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x1, *x2, *tmp1, *tmp2, *w1;
    if(tid >= params->in_length){
      return;
    }

    x1 = (uint32_t *) in_vector;   
    x2 = (uint32_t *) &in_vector[params->padding_idx * U256K_OFFSET];   
    w1 = (uint32_t *) &in_vector[2*params->padding_idx * U256K_OFFSET];   

    tmp1 = (uint32_t *) &out_vector[2*params->padding_idx * U256K_OFFSET];   
    tmp2 = (uint32_t *) &out_vector[3*params->padding_idx * U256K_OFFSET];   

    fft3Dyx_dif(tmp1, x1, w1, params);
    fft3Dyx_dif(tmp2, x2, w1, params);
}

__global__ void zpoly_interp3DYY_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x1, *x2, *tmp1, *tmp2, *scaler;

    if(tid >= params->in_length){
      return;
    } 

    x1 = (uint32_t *) in_vector;   
    x2 = (uint32_t *) &in_vector[params->padding_idx * U256K_OFFSET];   
    scaler = (uint32_t *) &in_vector[(3*params->padding_idx + params->stride) * U256K_OFFSET];   

    tmp1 = (uint32_t *) &out_vector[2*params->padding_idx * U256K_OFFSET];   
    tmp2 = (uint32_t *) &out_vector[3*params->padding_idx * U256K_OFFSET];   
   
    fft3Dyy_dif(x1, tmp1, NULL, scaler,params);
    fft3Dyy_dif(x2, tmp2, NULL, scaler,params);
 
}

__global__ void zpoly_interp3Dfinish_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x1, *x2, *w3, *z1;

    if(tid >= params->in_length){
      return;
    } 

    x1 = (uint32_t *) in_vector;
    x2 = (uint32_t *) &in_vector[params->padding_idx*U256K_OFFSET];
    w3 = (uint32_t *) &in_vector[3*params->padding_idx*U256K_OFFSET];   

    z1 = (uint32_t *) out_vector;
  
    if (!params->forward){
      mulmontu256( &x1[tid*U256K_OFFSET], 
                 &x1[tid*U256K_OFFSET],
                 &w3[tid*U256K_OFFSET], params->midx);
    
      mulmontu256( &x2[tid*U256K_OFFSET], 
                 &x2[tid*U256K_OFFSET],
                 &w3[tid*U256K_OFFSET], params->midx);
    } else {
      mulmontu256( &z1[(2*tid+1)*U256K_OFFSET], 
                 &x1[tid*U256K_OFFSET],
                 &x2[tid*U256K_OFFSET], params->midx);
    }
}

__global__ void zpoly_interp4DXX_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t *x1, *x2, *w1, *z2, *tmp1, *tmp2;
    const uint32_t N1 = params->N_fftx + params->N_ffty;

    if(tid >= params->padding_idx){
      return;
    }

    x1 = (uint32_t *) in_vector;
    x2 = (uint32_t *) &in_vector[params->padding_idx * U256K_OFFSET];

    z2 = (uint32_t *) &out_vector[params->padding_idx * U256K_OFFSET];
    tmp1 = (uint32_t *) &out_vector[2*params->padding_idx * U256K_OFFSET];
    tmp2 = (uint32_t *) &out_vector[3*params->padding_idx * U256K_OFFSET];

    if ( params->premul){
       w1 = (uint32_t *) &in_vector[3*params->padding_idx * U256K_OFFSET];
    } else {
       w1 = (uint32_t *) &in_vector[(3*params->padding_idx + params->stride - (1 << N1)) * U256K_OFFSET];
    }

    // X1 * X2
    if (!params->forward && params->premul){
       mulmontu256( &z2[tid*U256K_OFFSET], 
                   &x1[tid*U256K_OFFSET],
                   &x2[tid*U256K_OFFSET], params->midx);
    } 

    fft3Dxx_dif(tmp1, x1, w1, params);
    fft3Dxx_dif(tmp2, x2, w1, params);

}

__global__ void zpoly_interp4DXY_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x1, *x2, *tmp1, *tmp2, *w1;
    const uint32_t N1 = params->N_fftx + params->N_ffty;

    if(tid >= params->padding_idx){
      return;
    }
 
    x1 = (uint32_t *) in_vector;   
    x2 = (uint32_t *) &in_vector[params->padding_idx * U256K_OFFSET];   

    tmp1 = (uint32_t *) &out_vector[2*params->padding_idx * U256K_OFFSET];   
    tmp2 = (uint32_t *) &out_vector[3*params->padding_idx * U256K_OFFSET];   

    if ( params->premul){
       w1 = (uint32_t *) &in_vector[3*params->padding_idx * U256K_OFFSET];
    } else {
       w1 = (uint32_t *) &in_vector[(3*params->padding_idx + params->stride - (1 << N1)) * U256K_OFFSET];
    }

    fft3Dxy_dif(x1,tmp1, w1, params);
    fft3Dxy_dif(x2,tmp2, w1, params);
}

__global__ void zpoly_interp4DYX_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x1, *x2, *tmp2, *tmp3, *w1;
    const uint32_t N1 = params->N_fftx + params->N_ffty;

    if(tid >= params->in_length){
      return;
    }

    x1 = (uint32_t *) in_vector;   
    x2 = (uint32_t *) &in_vector[params->padding_idx * U256K_OFFSET];   

    tmp2 = (uint32_t *) &out_vector[3*params->padding_idx * U256K_OFFSET];   
    tmp3 = (uint32_t *) &out_vector[4*params->padding_idx * U256K_OFFSET];   

    if ( params->premul){
       w1 = (uint32_t *) &in_vector[3*params->padding_idx * U256K_OFFSET];
    } else {
       w1 = (uint32_t *) &in_vector[(3*params->padding_idx + params->stride - (1 << N1)) * U256K_OFFSET];
    }

    fft3Dyx_dif(tmp2, x1, w1, params);
    fft3Dyx_dif(tmp3, x2, w1, params);
}

__global__ void zpoly_interp4DYY_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x1, *x2, *z1, *z2, *tmp1, *tmp2, *tmp3, *w2, *scaler;

    if(tid >= params->in_length){
      return;
    } 

    x1 = (uint32_t *) in_vector;   
    x2 = (uint32_t *) &in_vector[params->padding_idx * U256K_OFFSET];   
    scaler = (uint32_t *) &in_vector[(3*params->padding_idx + params->stride) * U256K_OFFSET];   

    z1 = (uint32_t *) out_vector;
    z2 = (uint32_t *) &out_vector[params->padding_idx * U256K_OFFSET];   
    tmp1 = (uint32_t *) &out_vector[2*params->padding_idx * U256K_OFFSET];   
    tmp2 = (uint32_t *) &out_vector[3*params->padding_idx * U256K_OFFSET];   
    tmp3 = (uint32_t *) &out_vector[4*params->padding_idx * U256K_OFFSET];   
    w2 = (uint32_t *) &in_vector[2*params->padding_idx * U256K_OFFSET]; // N/2 roots 
  
    if (!params->forward && params->premul){ 
        x1 = z1;
        x2 = tmp1;
    } else if (params->forward && params->premul){
        x1 = z1;
        x2 = z2;
    }

    fft3Dyy_dif(x1, tmp2, w2, scaler, params);
    fft3Dyy_dif(x2, tmp3, w2, scaler, params);
    
}

__global__ void zpoly_interp4Dfinish_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t *x1, *x2, *w3, *z1, *z2;

    if(tid >= params->in_length){
      return;
    } 

    x1 = (uint32_t *) in_vector;
    x2 = (uint32_t *) &in_vector[params->padding_idx*U256K_OFFSET];
    w3 = (uint32_t *) &in_vector[2*params->padding_idx * U256K_OFFSET];   

    z1 = (uint32_t *) out_vector;
    z2 = (uint32_t *) &out_vector[params->padding_idx*U256K_OFFSET];
  
    if (!params->forward && !params->premul){
    
      mulmontu256( &z1[tid*U256K_OFFSET], 
                 &x1[tid*U256K_OFFSET],
                 &w3[tid*U256K_OFFSET], params->midx);
    
      mulmontu256( &z2[tid*U256K_OFFSET], 
                 &x2[tid*U256K_OFFSET],
                 &w3[tid*U256K_OFFSET], params->midx);
    

    } else if (params->forward && !params->premul) {
      mulmontu256( &z1[(tid)*U256K_OFFSET], 
                 &x1[tid*U256K_OFFSET],
                 &x2[tid*U256K_OFFSET], params->midx);

    }
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
__device__ void fft2Dx_dif(uint32_t *z, uint32_t *x, kernel_params_t *params)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;


  uint32_t Nx = params->fft_Nx;
  uint32_t Ny = params->fft_Ny;
  mod_t    midx = params->midx;

  uint32_t new_ridx = (tid << Ny) & ((1 << (Nx + Ny))-1);
  uint32_t new_cidx = tid >> Nx;
  uint32_t reverse_idx;
  uint32_t *roots = &x[(1<<(Nx+Ny)) * U256K_OFFSET];
  uint32_t const *W32;

  if (params->forward) {
    W32 = W32_ct;
  } else {
    W32 = IW32_ct;
  }

  // FFT rows (Every Ny points)
  reverse_idx = (((((tid % (1<<Nx)) * 0x802 & 0x22110) | ( (tid%(1<<Nx)) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff)*(1<<Ny);
  fftN_dif(&z[(reverse_idx + (tid/(1<<Nx)))*U256K_OFFSET], &x[(new_cidx + new_ridx)*U256K_OFFSET], W32, Nx,midx);
  mul_poly(&z[(new_cidx + new_ridx)*U256K_OFFSET],&z[(new_cidx + new_ridx)*U256K_OFFSET],&roots[(tid/(1<<Nx)) * (tid % (1<<Ny))*U256K_OFFSET], 0, midx);
}
__device__ void fft2Dy_dif(uint32_t *z, uint32_t *x, kernel_params_t *params)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t Nx = params->fft_Nx;
  uint32_t Ny = params->fft_Ny;
  mod_t    midx = params->midx;

  uint32_t new_ridx = (tid << Ny) & ((1 << (Nx + Ny))-1);
  uint32_t new_cidx = tid >> Nx;
  uint32_t reverse_idx;
  uint32_t *roots = &x[(1<<(Nx+Ny)) * U256K_OFFSET];
  uint32_t *inv_scaler = &x[((1<<(Nx+Ny+1))+params->as_mont) * U256K_OFFSET];
  uint32_t const *W32;

  if (params->forward) {
    W32 = W32_ct;
  } else {
    W32 = IW32_ct;
  }
  reverse_idx = (((((tid % (1<<Nx)) * 0x802 & 0x22110) | ( (tid%(1<<Nx)) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff)*(1<<Ny);
  fftN_dif(&z[(reverse_idx + (tid/(1<<Nx)))*U256K_OFFSET], &x[tid*U256K_OFFSET],W32,Ny,midx);

  if (!params->forward){
    mulmontu256( &z[(reverse_idx + (tid/32))*U256K_OFFSET],&z[(reverse_idx + (tid/32))*U256K_OFFSET],inv_scaler ,midx);
  }
}

__device__ void fft3Dxx_dif(uint32_t *z, uint32_t *x, uint32_t *roots, kernel_params_t *params)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t Nxx = params->fft_Nx; // size of underlying FFTx
  const uint32_t Nx = params->N_fftx;  // N FFTx of size Nx
  const uint32_t Ny = params->N_ffty;  // N FFTy of size Ny
  const mod_t    midx = params->midx;
  const uint32_t offset = ZPOLY_BASE_OFFSET(Nxx);
  const uint32_t mask = ZPOLY_BASE_MASK(Nxx);
  const uint32_t Nxxxn = (1 << (Nx-Nxx)) - 1;
  const uint32_t Nxxn = (1 << Nxx) - 1;
  const uint32_t NXYN = (1 << (Nx+Ny)) - 1;

  const uint32_t new_cidx = (tid & Nxxn) << (Ny + Nx - Nxx);
  const uint32_t new_ridx = ((tid >> Nxx) & Nxxxn) << Ny;
  const uint32_t new_kidx = (tid >> Nx) & ((1 << Ny) - 1);
  const uint32_t fft_idx = (tid >> (Nx + Ny)) << (Nx + Ny);
  const uint32_t reverse_idx = ZPOLY_REVERSE_IDX(tid, Nxxn, offset, mask) << (Ny + Nx - Nxx);
  uint32_t const *W32;
  int root_idx = (((new_ridx >> Ny) * (reverse_idx >> (Ny + Nx  - Nxx))) << Ny);

  const uint32_t tida = reverse_idx + new_kidx + new_ridx + fft_idx;
  const uint32_t tidb = new_cidx    + new_kidx + new_ridx + fft_idx;

  if (params->forward) {
    W32 = W32_ct;
  } else {
    W32 = IW32_ct;
    root_idx *= -1;
    root_idx &= NXYN;
  }

  // FFT rows (Every Ny points)
  fftN_dif(&z[tida * U256K_OFFSET],
           &x[tidb * U256K_OFFSET], W32, Nxx, midx);

  mulmontu256(&z[tida*U256K_OFFSET],
              &z[tida*U256K_OFFSET],
              &roots[root_idx*NWORDS_FR], midx);
}

__device__ void fft3Dxy_dif(uint32_t *z, uint32_t *x, uint32_t *roots, kernel_params_t *params)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t Nx = params->N_fftx;  // N FFTx of size Nx
  const uint32_t Ny = params->N_ffty;  // N FFTy of size Ny
  const uint32_t Nxy = params->fft_Ny;
  const mod_t    midx = params->midx;
  const uint32_t offset = ZPOLY_BASE_OFFSET(Nxy);
  const uint32_t mask = ZPOLY_BASE_MASK(Nxy);
  const uint32_t Nxyn = (1 << Nxy) - 1;
  const uint32_t Nyn = (1 << Ny) - 1;
  const uint32_t NXYN = (1 << (Nx+Ny)) - 1;

  const uint32_t new_cidx = (tid & Nxyn) << Ny;  
  const uint32_t new_ridx = (tid >> Nxy) & Nyn;
  const uint32_t new_kidx = (tid >>(Nxy + Ny) << (Nxy + Ny)) & (NXYN);
  const uint32_t fft_idx = ((tid >> (Nxy + Ny)) >> (Nx - Nxy)) << (Nx + Ny);
  uint32_t const *W32;

  const uint32_t reverse_idx = ZPOLY_REVERSE_IDX(tid, Nxyn, offset, mask) << (Nx - Nxy + Ny) ;

  const uint32_t tida = reverse_idx + (new_kidx >> Nxy) + new_ridx + fft_idx;
  const uint32_t tidb = new_cidx    +  new_kidx         + new_ridx + fft_idx;

  int root_idx = ((tida - fft_idx) >> Ny) * ((tida - fft_idx) & Nyn);

  if (params->forward) {
    W32 = W32_ct;
  } else {
    W32 = IW32_ct;
    root_idx *= -1;
    root_idx &= NXYN;
  }
  fftN_dif(&z[tida*U256K_OFFSET],
           &x[tidb*U256K_OFFSET],W32,Nxy,midx);

  mulmontu256(&z[tida*U256K_OFFSET],
           &z[tida*U256K_OFFSET],
           &roots[root_idx * NWORDS_FR],midx);

}

__device__ void fft3Dyx_dif(uint32_t *z, uint32_t *x, uint32_t *roots, kernel_params_t *params)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t Nyx = params->fft_Nx; // size of underlying FFTx
  const uint32_t Nx = params->N_fftx;  // N FFTx of size Nx
  const uint32_t Ny = params->N_ffty;  // N FFTy of size Ny
  const mod_t    midx = params->midx;

  const uint32_t offset = ZPOLY_BASE_OFFSET(Nyx);
  const uint32_t mask = ZPOLY_BASE_MASK(Nyx);
  const uint32_t Nyxn = (1 << Nyx) - 1;
  const uint32_t Nyyxn = (1 << (Ny-Nyx)) - 1;
  const uint32_t NXYN = (1 << (Nx+Ny)) - 1;

  const uint32_t new_cidx = (tid & Nyxn) << (Ny - Nyx);  
  const uint32_t new_ridx = (tid >> Nyx) & Nyyxn;
  const uint32_t new_kidx = ((tid >> Ny) << Ny) & (NXYN);
  const uint32_t fft_idx = (tid >> (Nx + Ny)) << (Nx + Ny);
  const uint32_t reverse_idx = ZPOLY_REVERSE_IDX(tid, Nyxn, offset, mask) << (Ny - Nyx);
  const uint32_t const *W32;
  int root_idx = ((new_ridx * (reverse_idx >> (Ny - Nyx))) << Nx);

  const uint32_t tida = reverse_idx + new_kidx + new_ridx + fft_idx;
  const uint32_t tidb = new_cidx    + new_kidx + new_ridx + fft_idx;

  if (params->forward) {
    W32 = W32_ct;
  } else {
    W32 = IW32_ct;
    root_idx *= -1;
    root_idx &= NXYN;
  }

  // FFT rows (Every Ny points)
  fftN_dif(&z[tida*U256K_OFFSET],
           &x[tidb*U256K_OFFSET], W32, Nyx,midx);

  mulmontu256(&z[tida*U256K_OFFSET],
           &z[tida*U256K_OFFSET],
           &roots[root_idx * NWORDS_FR],midx);
}

__device__ void fft3Dyy_dif(uint32_t *z, uint32_t *x, uint32_t *w2, uint32_t *scaler, kernel_params_t *params)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t Nx = params->N_fftx;  // N FFTx of size Nx
  const uint32_t Ny = params->N_ffty;  // N FFTy of size Ny
  const uint32_t Nyy = params->fft_Ny;

  const mod_t    midx = params->midx;
  const uint32_t offset = ZPOLY_BASE_OFFSET(Nyy);
  const uint32_t mask = ZPOLY_BASE_MASK(Nyy);
  const uint32_t Nyyn = (1 << Nyy) - 1;
  const uint32_t Nyyyn = (1 << (Ny-Nyy)) - 1;

  const uint32_t new_cidx = tid & Nyyn;  
  const uint32_t new_ridx = ((tid >> Nyy ) & Nyyyn ) << Nx;
  const uint32_t new_kidx = (tid >> Ny) & ((1 << Nx) - 1);
  const uint32_t fft_idx = (tid >> (Nx + Ny) ) << (Nx + Ny);

  const uint32_t reverse_idx = ZPOLY_REVERSE_IDX(tid, Nyyn, offset, mask) << (Ny + Nx - Nyy);
  uint32_t const *inv_scaler = &scaler[params->as_mont * U256K_OFFSET];
  uint32_t const *W32;

  const uint32_t tida = reverse_idx + new_ridx + new_kidx + fft_idx;

  if (params->forward) {
    W32 = W32_ct;
  } else {
    W32 = IW32_ct;
  }

  fftN_dif(&z[tida*U256K_OFFSET], &x[tid*U256K_OFFSET],W32,Nyy,midx);

  if ( params->premul){
   
    mulmontu256( &z[tida * U256K_OFFSET],
                 &z[tida * U256K_OFFSET],
                 &w2[tida * U256K_OFFSET], midx);
   
  }
  else if (!params->forward){
    mulmontu256( &z[tida*U256K_OFFSET],
                 &z[tida*U256K_OFFSET],inv_scaler ,midx);
  }
 
}


/*
  Multiply 2 poly of degre d
*/
__device__ void mul_poly(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t d, mod_t midx)
{
  uint32_t i, coeff_idx;

  #pragma unroll
  for (i=0; i<= d; i++) {
    coeff_idx = i * NWORDS_FR;
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

    logInfoBigNumberTid(1,"PreB OtherX :\n",otherX);
    logInfoBigNumberTid(1,"PreB X :\n",thisX);
    logInfoTid("size : %d\n",size);

    fft_butterfly(otherX, thisX, size);

    logInfoBigNumberTid(1,"AfterB OtherX :\n",otherX);

    root_idx = (1<< (ZPOLY_FFT_32-i-1)) * (lane%size) * NWORDS_FR;
    if (lane%(size<<1) < size){  //  It 0: 0-15      W0,W1,...,W15
                                 //  It 1: 0-7, 16-23    W0,W2,W4,..W14
                                 //  It 2: 0-3, 8-11, 16-19, 24-27  W0, W4, W8, W12
                                 //  It 3: 0-1, 4-5, 8-9, 12-13,...  W0, W8

      addmu256(thisX, thisX, otherX, midx);
      logInfoTid("Op : x + otherX : lane%(size>>1) : %d\n", lane%(size<<1));
      logInfoTid("Op : x + otherX : size : %d\n",size);
    } else {
      submu256(thisX, otherX, thisX,  midx);
      mulmontu256(thisX, thisX, &W32[ root_idx], midx);
      logInfoTid("Op : otherX - x : lane%(size>>1) : %d\n", lane%(size<<1));
      logInfoTid("Op : otherX - x : size : %d\n", size);
    }
    logInfoBigNumberTid(1,"AfterB x:\n",thisX);
    logInfoBigNumberTid(1,"Root : \n",(uint32_t *)&W32[ root_idx]);
    logInfoTid("idx : %d\n", root_idx / NWORDS_FR);
  }
    logInfoBigNumberTid(1,"PreB OtherX :\n",otherX);
    logInfoBigNumberTid(1,"PreB X :\n",thisX);
    logInfoTid("size : %d\n",size);

  // I can skip mulypliying by 1 in the last iteration
  //  It 4: 0,2,4,6,...    W0

  fft_butterfly(otherX, thisX, 1);
  logInfoBigNumberTid(1,"AfterB OtherX :\n",otherX);

  if (lane % 2 == 0){  
      addmu256(z, thisX, otherX, midx);
      logInfoTid("Op : x + otherX : lane%(size>>1) : %d\n", lane%2);
  } else {
      submu256(z, otherX, thisX, midx);
      logInfoTid("Op : otherX - x : lane%(size>>1) : %d\n", lane%2);
  }
    logInfoBigNumberTid(1,"AfterB x:\n",z);
    logInfoBigNumberTid(1,"Root : \n",(uint32_t *)&W32[ (1 << (ZPOLY_FFT_32-i-1)) * (lane%size) * NWORDS_FR]);
    logInfoTid("idx : %d\n",(1 << (ZPOLY_FFT_32-i-1)) * (lane % size));
}

/*
   I32Point DIT FFT inplace. Input samples are unordered. Output samples are ordered
*/
__device__ void ifft32_dit(uint32_t *z, uint32_t *x, mod_t midx)
{
  uint32_t otherX[] = {0,0,0,0,0,0,0,0};
  uint32_t thisX[] = {0,0,0,0,0,0,0,0};
  uint32_t const *IW32 = IW32_ct;
  uint32_t const *inv_scaler = &IW32_nroots_ct[(ZPOLY_FFT_32-1)*NWORDS_FR];
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
    root_idx = (1 << (ZPOLY_FFT_32 - 1 - i)) * (lane%size) * NWORDS_FR;

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
__device__ void fftN_dif(uint32_t *z, uint32_t *x, const uint32_t *W32, uint32_t N, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t otherX[] = {0,0,0,0,0,0,0,0};
  uint32_t thisX[] =  {0,0,0,0,0,0,0,0};
  //uint32_t const *W32 = W32_ct;
  uint32_t lane = threadIdx.x % warpSize;
  //uint32_t lane = threadIdx.x % (1 << N);
  uint32_t i, size, root_idx;

  movu256(thisX,x);

  logInfoTid("N : %d\n",N);
  logInfoBigNumberTid(1,"ThisX in\n",thisX);

  #pragma unroll
  for (i=N-1; i>=1 ; i--){
    size = 1 << i;

    logInfoBigNumberTid(1,"PreB OtherX :\n",otherX);
    logInfoBigNumberTid(1,"PreB X :\n",thisX);
    logInfoTid("size : %d\n",size);

    fft_butterfly(otherX, thisX, size);

    logInfoBigNumberTid(1,"AfterB OtherX :\n",otherX);

    root_idx = ((1<< (N-i-1)) * (lane%size) * NWORDS_FR) << (ZPOLY_FFT_32 - N);
    if (lane%(size<<1) < size){  //  It 0: 0-15      W0,W1,...,W15
                                 //  It 1: 0-7, 16-23    W0,W2,W4,..W14
                                 //  It 2: 0-3, 8-11, 16-19, 24-27  W0, W4, W8, W12
                                 //  It 3: 0-1, 4-5, 8-9, 12-13,...  W0, W8

      addmu256(thisX, thisX, otherX, midx);
      logInfoTid("Op : x + otherX : lane%(size>>1) : %d\n", lane%(size<<1));
      logInfoTid("size : %d\n", lane%(size<<1));
    } else {
      submu256(thisX, otherX, thisX,  midx);
      mulmontu256(thisX, thisX, &W32[ root_idx], midx);
      logInfoTid("Op : x + otherX : lane%(size>>1) : %d\n", lane%(size<<1));
      logInfoTid("size : %d\n", lane%(size<<1));
    }
      logInfoBigNumberTid(1,"AfterB x:\n",thisX);
      logInfoBigNumberTid(1,"Root : \n",(uint32_t *)&W32[ root_idx]);
      logInfoTid("idx : %d\n", root_idx / (NWORDS_FR * (1 << (ZPOLY_FFT_32 - N))) );
  }
    logInfoBigNumberTid(1,"PreB OtherX :\n",otherX);
    logInfoBigNumberTid(1,"PreB X :\n",thisX);
    logInfoTid("size : %d\n",size);
  // I can skip mulypliying by 1 in the last iteration
  //  It 4: 0,2,4,6,...    W0
  fft_butterfly(otherX, thisX, 1);
    logInfoBigNumberTid(1,"AfterB OtherX :\n",otherX);

  if (lane % 2 == 0){  
      addmu256(z, thisX, otherX, midx);
      logInfoTid("Op : x + otherX : lane%(size>>1) : %d\n", lane%2);
  } else {
      submu256(z, otherX, thisX, midx);
      logInfoTid("Op : otherX - x : lane%(size>>1) : %d\n", lane%2);
  }
    logInfoBigNumberTid(1,"AfterB x:\n",z);
    logInfoTid("idx : %d\n",(1 << (N-i-1)) * (lane % size));
    logInfoBigNumberTid(1,"Root : \n",(uint32_t *)&W32[ (1 << (N-i-1)) * (lane%size) * NWORDS_FR]);
}

/*
   I32Point DIT FFT inplace. Input samples are unordered. Output samples are ordered
*/
__device__ void ifftN_dit(uint32_t *z, uint32_t *x, const uint32_t *IW32, uint32_t N, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t otherX[] = {0,0,0,0,0,0,0,0};
  uint32_t thisX[] = {0,0,0,0,0,0,0,0};
  uint32_t const *inv_scaler = &IW32_nroots_ct[(N-1)*NWORDS_FR];
  uint32_t lane = threadIdx.x % warpSize;
  uint32_t i, size, root_idx;
  int debug_tidx=-1;

  movu256(thisX,x);
 
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
    root_idx = ((1<< (N-i-1)) * (lane%size) * NWORDS_FR) << (ZPOLY_FFT_32 - N);

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

__forceinline__ __device__ void fft_butterfly(uint32_t *d_out, uint32_t *d_in, uint32_t srcLane )
{
    ulonglong4 in, *out;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    in = *(ulonglong4 *)d_in;
    out = (ulonglong4 *)d_out;

    #if 0
    if (tid == 0){
      logInfoBigNumber("BUT IN: \n",d_in);
      logInfoBigNumber("BUT OUT: \n",d_out);
      logInfo("lane :%d \n",srcLane);
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
    #if 0
    if (tid == 0){
      logInfo("lane :%d \n",srcLane);
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

