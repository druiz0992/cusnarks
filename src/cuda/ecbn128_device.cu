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
// File name  : ecbn128_device.cu
//
// Date       : 12/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementatoin of EC Cuda functionality
// 
// NOTE : EC Points do not require to be premoded premod. They shoould
//  already be < than prime
// ------------------------------------------------------------------

*/

#include <stdio.h>
#include <cuda_profiler_api.h>

#include "types.h"
#include "cuda.h"
#include "log.h"
#include "utils_device.h"
#include "u256_device.h"
#include "z1_device.h"
#include "z2_device.h"
#include "ecbn128_device.h"
#include "asm_device.h"

	 
/* 
  in_vector : k[0], px[0], py[0], k[1], px[1], py[1],...  Input EC points in Affine coordinates
  out vecto : px[0], py[0], pz[0], px[1], py[1],pz[1],...              Output EC points in Jacobian coordinates
*/
__global__ void addecjacaff_kernel(uint32_t   *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    //Z1_t x1, x2, xr;

    if(tid >= params->in_length/4) {
      return;
    }

    // x1 points to inPx[i]. x2 points to inPx[i+1]. xr points to outPx[i]
    Z1_t x1(&in_vector[tid * 2 * ECP_JAC_INOFFSET + ECP_JAC_INXOFFSET]);
    Z1_t x2(&in_vector[(tid * 2 + 1) * ECP_JAC_INOFFSET + ECP_JAC_INXOFFSET]);
    Z1_t xr(&out_vector[tid * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
   
    //TODO : this is not very nice, but it gets the job done. Try to come up
    // with something better 
    addecjacaff<Z1_t, uint256_t>(&xr, &x1, &x2, params->midx);

    return;

}

__global__ void addec2jacaff_kernel(uint32_t   *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid >= params->in_length/8) {
      return;
    }

    // x1 points to inPx[i]. x2 points to inPx[i+1]. xr points to outPx[i]
    Z2_t x1(&in_vector[tid * 2 * ECP2_JAC_INOFFSET + ECP2_JAC_INXOFFSET]);
    Z2_t x2(&in_vector[(tid * 2 + 1) * ECP2_JAC_INOFFSET + ECP2_JAC_INXOFFSET]);
    Z2_t xr(&out_vector[tid * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]);
   
    //TODO : this is not very nice, but it gets the job done. Try to come up
    // with something better 
    addecjacaff<Z2_t, uint512_t>(&xr, &x1, &x2, params->midx);

    return;

}

__global__ void addecjac_kernel(uint32_t   *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    //Z1_t x1, x2, xr;

    if(tid >= params->in_length/6) {
      return;
    }

    // x1 points to inPx[i]. x2 points to inPx[i+1]. xr points to outPx[i]
    Z1_t x1(&in_vector[tid *  2 * ECP_JAC_OUTOFFSET + ECP_JAC_INXOFFSET]);
    Z1_t x2(&in_vector[(tid * 2 + 1) * ECP_JAC_OUTOFFSET + ECP_JAC_INXOFFSET]);
    Z1_t xr(&out_vector[tid * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
   
    //TODO : this is not very nice, but it gets the job done. Try to come up
    // with something better 
#ifdef CU_ASM_ECADD
      addecjacz(&xr,0, &x1,0, &x2,0, params->midx);
#else
      addecjac<Z1_t, uint256_t>(&xr,0, &x1,0, &x2,0, params->midx);
#endif

    return;

}

__global__ void addec2jac_kernel(uint32_t   *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    //Z1_t x1, x2, xr;

    if(tid >= params->in_length/12) {
      return;
    }

    // x1 points to inPx[i]. x2 points to inPx[i+1]. xr points to outPx[i]
    Z2_t x1(&in_vector[tid * 2 * ECP2_JAC_OUTOFFSET + ECP2_JAC_INXOFFSET]);
    Z2_t x2(&in_vector[(tid * 2 + 1) * ECP2_JAC_OUTOFFSET + ECP2_JAC_INXOFFSET]);
    Z2_t xr(&out_vector[tid * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]);
   
    //TODO : this is not very nice, but it gets the job done. Try to come up
    // with something better 
    addecjac<Z2_t, uint512_t>(&xr,0, &x1,0, &x2,0, params->midx);

    return;

}
__global__ void doublecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // x1 points to inPx[i].  xr points to outPx[i]
    if(tid >= params->in_length/3) {
      return;
    }

    Z1_t x1(&in_vector[tid * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
    Z1_t xr(&out_vector[tid * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
    
    doublecjac<Z1_t, uint256_t>(&xr, &x1, params->midx);

    return;
}

__global__ void doublec2jac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // x1 points to inPx[i].  xr points to outPx[i]
    if(tid >= params->in_length/6) {
      return;
    }

    Z2_t x1(&in_vector[tid * ECP2_JAC_OUTOFFSET + ECP2_JAC_INXOFFSET]);
    Z2_t xr(&out_vector[tid * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]);
    
    doublecjac<Z2_t, uint512_t>(&xr, &x1, params->midx);

    return;
}
__global__ void doublecjacaff_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // x1 points to inPx[i].  xr points to outPx[i]
    if(tid >= params->in_length/2) {
      return;
    }

    Z1_t x1(&in_vector[tid * ECP_JAC_INOFFSET + ECP_JAC_INXOFFSET]);
    Z1_t xr(&out_vector[tid * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
    
    doublecjacaff<Z1_t, uint256_t>(&xr, &x1, params->midx);

    return;
}

__global__ void doublec2jacaff_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // x1 points to inPx[i].  xr points to outPx[i]
    if(tid >= params->in_length/4) {
      return;
    }

    Z2_t x1(&in_vector[tid * ECP2_JAC_INOFFSET + ECP2_JAC_INXOFFSET]);
    Z2_t xr(&out_vector[tid * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]);
    
    doublecjacaff<Z2_t, uint512_t>(&xr, &x1, params->midx);

    return;
}

#if LOG_LEVEL != LOG_LEVEL_NOLOG
__global__ void scmulecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
#else
__global__ void __launch_bounds__(256,2) scmulecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
#endif
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *scl;
 
   if(tid >= params->in_length/3) {
     return;
   }

   scl = (uint32_t *) &in_vector[tid * NWORDS_FR + ECP_SCLOFFSET];
   Z1_t x1(&in_vector[ params->in_length/3 * NWORDS_FR + tid * ECP_JAC_INOFFSET + ECP_JAC_INXOFFSET]);
   Z1_t xr(&out_vector[tid * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
  
   scmulecjac<Z1_t, uint256_t>(&xr,0, &x1,0, scl,  params);

   return;
}

__global__ void __launch_bounds__(128,5)scmulecjacopt_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t poffset = 0;

    uint32_t __restrict__ *scl = NULL;
 
    if(idx >= params->in_length/params->stride) {
      return;
    }

    Z1_t xo;
    Z1_t xr;
    scl = (uint32_t *) in_vector;
    poffset = params->in_length/3 * NWORDS_FR;
    xo.assign(&in_vector[poffset]);
    xr.assign(out_vector);

    Z1_t sumX(xr.getsingleu256(idx*3));

    Z1_t sumX2(xo.getu256(0));
      
    scmulecjac_opt<Z1_t, uint256_t>(&sumX,0,
                           &sumX2, 2*idx*DEFAULT_U256_BSELM_CUDA, 
                           &scl[idx*NWORDS_FR * DEFAULT_U256_BSELM_CUDA],  params);
   logInfoBigNumberTid(3,"XOUT :\n",&out_vector[idx*ECP_JAC_OUTOFFSET]);
}

__global__ void scmulecjac_precomputed_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t poffset = 0;

    uint32_t __restrict__ *scl = NULL;
 
    if(idx >= params->padding_idx) {
      return;
    }

    Z1_t xo;
    Z1_t xr;
    scl = (uint32_t *) in_vector;
    poffset = params->padding_idx;
    xo.assign(&in_vector[poffset]);
    xr.assign(out_vector);

    Z1_t sumX(xr.getsingleu256(idx*3));

    Z1_t sumX2(xo.getu256(0));
      
    scmulecjac_precomputed<Z1_t, uint256_t>(&sumX,0,
                           &sumX2, 2*idx<<DEFAULT_U256_BSELM_CUDA, 
                           &scl[idx*NWORDS_FR *DEFAULT_U256_BSELM_CUDA],  params);
   logInfoBigNumberTid(3,"XOUT :\n",&out_vector[idx*ECP_JAC_OUTOFFSET]);
}

__global__ void __launch_bounds__(128,4) scmulec2jacopt_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t poffset = 0;

    uint32_t __restrict__ *scl = NULL;
 
    if(idx >= params->in_length/params->stride) {
      return;
    }

    Z2_t xo;
    Z2_t xr;
    scl = (uint32_t *) in_vector;
    poffset = params->in_length/5 * NWORDS_FR;
    xo.assign(&in_vector[poffset]);
    xr.assign(out_vector);

    Z2_t sumX(xr.getu256(idx*3));
    Z2_t sumX2(xo.getu256(0));
      
    scmulecjac_opt<Z2_t, uint512_t>(&sumX,0,
                           &sumX2, 2*idx*DEFAULT_U256_BSELM_CUDA, 
                           &scl[idx*NWORDS_FR *DEFAULT_U256_BSELM_CUDA],  params);
    
    logInfoBigNumberTid(6*128,"XOUT :\n",&out_vector[idx*ECP2_JAC_OUTOFFSET]);
}


__global__ void sc1mulecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *scl;
 
   logInfoTid("Length : %d\n", params->in_length);
   logInfoTid("R Length : %d\n", params->in_length-ECP_JAC_INDIMS);
   logInfoTid("SCL O %d\n", ECP_SCLOFFSET);
   if(tid >= params->in_length-ECP_JAC_INDIMS) {
     return;
   }

   scl = (uint32_t *) &in_vector[tid * NWORDS_FR + ECP_SCLOFFSET];
   logInfoBigNumberTid(1,"SCL MONT\n", scl);

   Z1_t x1(&in_vector[(params->in_length-ECP_JAC_INDIMS)*NWORDS_256BIT + ECP_JAC_INXOFFSET]);
   Z1_t xr(&out_vector[tid * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
  
   scmulecjac<Z1_t, uint256_t>(&xr,0, &x1,0, scl,  params);

   return;
}

#if LOG_LEVEL != LOG_LEVEL_NOLOG
__global__ void scmulec2jac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
#else
__global__ void __launch_bounds__(128,2) scmulec2jac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
#endif
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *scl;
 
   if(tid >= params->in_length/5) {
     return;
   }

   scl = (uint32_t *) &in_vector[tid * NWORDS_256BIT + ECP_SCLOFFSET];
   Z2_t x1(&in_vector[ params->in_length/5 * NWORDS_256BIT+ tid * ECP2_JAC_INOFFSET + ECP2_JAC_INXOFFSET]);
   Z2_t xr(&out_vector[tid * ECP2_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
  
   scmulecjac<Z2_t, uint512_t>(&xr,0, &x1,0, scl,params);

   return;
}

__global__ void sc1mulec2jac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *scl;
 
   if(tid >= params->in_length-ECP2_JAC_INDIMS) {
     return;
   }

   scl = (uint32_t *) &in_vector[tid * NWORDS_256BIT + ECP_SCLOFFSET];

   Z2_t x1(&in_vector[(params->in_length-ECP2_JAC_INDIMS)*NWORDS_256BIT + ECP2_JAC_INXOFFSET]);
   Z2_t xr(&out_vector[tid * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]);
   logInfoBigNumberTid(1,"SCL\n", scl);
   logInfoBigNumberTid(4,"X1\n", &x1);

   scmulecjac<Z2_t, uint512_t>(&xr,0, &x1,0, scl,  params);

   return;
}

__global__ void madecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;

    uint32_t debug_idx = 0;

    extern __shared__ uint32_t smem[];
    Z1_t zsmem(smem);  // 0 .. blockDim

    uint32_t __restrict__ *scl;
   
    if(idx >= params->in_length/params->stride) {
      return;
    }
    logInfoTid("Min Padding : %d\n",params->padding_idx);
    logInfoTid("Max Padding : %d\n",params->in_length/ECP_JAC_OUTDIMS);
    if (params->padding_idx){
       uint32_t padding[] = {0,0,0,0,0,0,0,0};
       // add zeros between padding and next multiple of 32
       if (idx < params->in_length/ECP_JAC_OUTDIMS && idx >= params->padding_idx){
          movu256(&in_vector[idx * ECP_JAC_INOFFSET],padding);
          movu256(&in_vector[idx * ECP_JAC_INOFFSET + NWORDS_FP],padding);
          movu256(&in_vector[idx * ECP_JAC_INOFFSET + 2*NWORDS_FP],padding);
       }
       __syncthreads();
    }

    Z1_t xo, xr;
    if (params->premul){
      xo.assign(&in_vector[params->in_length/3 * NWORDS_256BIT + idx  * (params->stride-1) * NWORDS_256BIT + ECP_JAC_OUTXOFFSET]); // 0 .. N-1
      scl = (uint32_t *) &in_vector[idx * params->stride/3 *  NWORDS_256BIT];
      logInfoTid("LE : %d\n",params->in_length);
      logInfoTid("InVO : %d\n",(params->stride-1) * NWORDS_256BIT);
      logInfoTid("SclVO : %d\n",params->in_length/3* 2 * NWORDS_256BIT);
    } else {
      xo.assign(&in_vector[idx  * (params->stride) * NWORDS_256BIT + ECP_JAC_OUTXOFFSET]); // 0 .. N-1
      scl = NULL;
      logInfoTid("LE : %d\n",params->in_length);
      logInfoTid("InVO : %d\n",(params->stride) * NWORDS_256BIT);
      logInfoTid("SclVO : %d\n",params->in_length/3* 2 *NWORDS_256BIT);
    }
    xr.assign(&in_vector[blockIdx.x * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);  // 
    if (gridDim.x == 1){
      xr.assign(out_vector);
    }

    madecjac<Z1_t, uint256_t>(&xr, &xo, scl, &zsmem, params);
}

__global__ void madec2jac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;

    uint32_t debug_idx = 0;

    extern __shared__ uint32_t smem[];
    Z2_t zsmem(smem);  // 0 .. blockDim

    uint32_t __restrict__ *scl;
   
    if(idx >= params->in_length/params->stride) {
      return;
    }
    logInfoTid("Min Padding : %d\n",params->padding_idx);
    logInfoTid("Max Padding : %d\n",params->in_length/ECP2_JAC_OUTDIMS);
    if (params->padding_idx){
       uint32_t padding[] = {0,0,0,0,0,0,0,0};
       // add zeros between padding and next multiple of 32
       if (idx < params->in_length/ECP2_JAC_OUTDIMS && idx >= params->padding_idx){
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET+ NWORDS_FP],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 2*NWORDS_FP],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 3*NWORDS_FP],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 4*NWORDS_FP],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 5*NWORDS_FP],padding);
       }
       __syncthreads();
    }

    Z2_t xo, xr;
    if (params->premul){
      xo.assign(&in_vector[params->in_length/5 * NWORDS_256BIT + idx  * params->stride * NWORDS_256BIT + ECP2_JAC_OUTXOFFSET]); // 0 .. N-1
      scl = (uint32_t *) &in_vector[idx * params->stride/5 *  NWORDS_256BIT];
      logInfoTid("LE : %d\n",params->in_length);
      logInfoTid("InVO : %d\n",(params->stride) * NWORDS_256BIT);
      logInfoTid("SclVO : %d\n",params->in_length/5* NWORDS_256BIT);
    } else {
      xo.assign(&in_vector[idx  * (params->stride) * NWORDS_256BIT + ECP2_JAC_OUTXOFFSET]); // 0 .. N-1
      scl = NULL;
      logInfoTid("LE : %d\n",params->in_length);
      logInfoTid("InVO : %d\n",(params->stride) * NWORDS_256BIT);
      logInfoTid("SclVO : %d\n",params->in_length/6 *NWORDS_256BIT);
    }
    xr.assign(&in_vector[blockIdx.x * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]);  // 
    if (gridDim.x == 1){
      xr.assign(out_vector);
    }

    madecjac<Z2_t, uint512_t>(&xr, &xo, scl, &zsmem, params);
}

__global__ void __launch_bounds__(256,2)madecjac_shfl_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t poffset = 0;

    extern __shared__ uint32_t smem[];
    Z1_t zsmem(smem);  // 0 .. blockDim

    uint32_t __restrict__ *scl = NULL;
 
    if(idx >= params->in_length/params->stride) {
      return;
    }

    Z1_t xo;
    if (params->premul){
      scl = (uint32_t *) in_vector;
      poffset = params->in_length/3 * NWORDS_256BIT;
      xo.assign(&in_vector[poffset]);
    } else {
      xo.assign(&out_vector[idx * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]); // 0 .. N-1
    }

    Z1_t xr(&out_vector[blockIdx.x * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]); 

    madecjac_shfl<Z1_t, uint256_t>(&xr, &xo, scl, &zsmem, params);
    logInfoBigNumberTid(3,"XOUT :\n",xr.getsingleu256(0));
}

__global__ void madec2jac_shfl_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t poffset = 0;

    extern __shared__ uint32_t smem[];
    Z2_t zsmem(smem);  // 0 .. blockDim

    uint32_t __restrict__ *scl = NULL;
  
    if(idx >= params->in_length/params->stride) {
      return;
    }

    Z2_t xo;
    if (params->premul){
      scl = (uint32_t *) in_vector;
      poffset = params->in_length/5 * NWORDS_256BIT;
      xo.assign(&in_vector[poffset]);
    } else {
      xo.assign(&out_vector[idx * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]); // 0 .. N-1
    }

    Z2_t xr(&out_vector[blockIdx.x * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]);  
    madecjac_shfl<Z2_t, uint512_t>(&xr, &xo, scl, &zsmem, params);
    logInfoBigNumberTid(6,"XOUT :\n",xr.getsingleu256(0));
}

__global__ void __launch_bounds__(256,2)redecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    extern __shared__ uint32_t smem[];
    Z1_t zsmem(smem);  // 0 .. blockDim

    if(idx >= params->in_length/params->stride) {
      return;
    }

    Z1_t xo;
    Z1_t xr(&out_vector[blockIdx.x * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);  

    xo.assign(&out_vector[idx * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]); // 0 .. N-1
 
    redecjac<Z1_t, uint256_t>(&xr, &xo, &zsmem, params);
    logInfoBigNumberTid(3,"XOUT :\n",xr.getsingleu256(0));
}

__global__ void __launch_bounds__(256,2) redec2jac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    extern __shared__ uint32_t smem[];
    Z2_t zsmem(smem);  // 0 .. blockDim

    if(idx >= params->in_length/params->stride) {
      return;
    }

    Z2_t xo;
    Z2_t xr(&out_vector[blockIdx.x * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]);  

    xo.assign(&out_vector[idx * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]); // 0 .. N-1

    logInfoBigNumberTid(6,"XIN :\n",xo.getsingleu256(0));
    redecjac<Z2_t, uint512_t>(&xr, &xo, &zsmem, params);
    logInfoBigNumberTid(6,"XOUT :\n",xr.getsingleu256(0));
}

template<typename T1, typename T2>
__forceinline__ __device__ void madecjac(T1 *xr, T1 *xo, uint32_t *scl, T1 *smem_ptr, kernel_params_t *params)
{
    uint32_t i;
    //uint32_t ndbg = T1::getN();
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;

    logInfoTid("stride :%d\n",params->stride/ECP_JAC_OUTDIMS);
    // scalar multipliation
    if (params->premul){
        #pragma unroll
        for (i =0; i < params->stride/ECP_JAC_OUTDIMS; i++){
          /*
          logInfoBigNumberTid(1,"scl :\n",&scl[i*ECP_JAC_INOFFSET]);
          logInfoBigNumberTid(T1::getN(),"Xin[x]:\n",xos(i*ECP_JAC_INDIMS));
          logInfoBigNumberTid("Xin[y]:\n",&xi[i*ECP_JAC_INOFFSET + NWORDS_256BIT]);
          */

          scmulecjac<T1, T2>(xr,i*ECP_JAC_OUTDIMS, xo, i*ECP_JAC_INDIMS, &scl[i*NWORDS_256BIT],  params);
          

          /*
          logInfoBigNumberTid(1,"Xout[x]:\n",&xo[i*ECP_JAC_OUTOFFSET]);
          logInfoBigNumberTid(1,"Xout[y]:\n",&xo[i*ECP_JAC_OUTOFFSET + NWORDS_256BIT]);
          logInfoBigNumberTid(1,"Xout[z]:\n",&xo[i*ECP_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
          */
        }
    }
   
    if (params->stride/ECP_JAC_OUTDIMS > 1){ 
#ifdef CU_ASM_ECADD
      addecjacz(smem_ptr,tid*ECP_JAC_OUTDIMS, xr,0, xr,ECP_JAC_OUTDIMS, params->midx);
#else
      addecjac<T1,T2>(smem_ptr,tid*ECP_JAC_OUTDIMS, xr,0, xr,ECP_JAC_OUTDIMS, params->midx);
#endif
      /*
      logInfoBigNumberTid(1,"smem[X]\n",smem_ptr);
      logInfoBigNumberTid(1,"smem[Y]\n",&smem_ptr[NWORDS_256BIT]);
      logInfoBigNumberTid(1,"smem[Z]\n",&smem_ptr[2*NWORDS_256BIT]);
      */

      #pragma unroll
      for (i =0; i < params->stride/ECP_JAC_OUTDIMS-2; i++){
#ifdef CU_ASM_ECADD
        addecjacz(smem_ptr,tid*ECP_JAC_OUTDIMS, smem_ptr, 0,xr, (i+2)*ECP_JAC_OUTDIMS, params->midx);
#else
        addecjac<T1,T2>(smem_ptr,tid*ECP_JAC_OUTDIMS, smem_ptr, 0,xr, (i+2)*ECP_JAC_OUTDIMS, params->midx);
#endif
        /*
        logInfoBigNumberTid(1,"smem[X]\n",smem_ptr);
        logInfoBigNumberTid(1,"smem[Y]\n",&smem_ptr[NWORDS_256BIT]);
        logInfoBigNumberTid(1,"smem[Z]\n",&smem_ptr[2*NWORDS_256BIT]);
        */
      }
      __syncthreads();
    }
  
    //logDebugBigNumberTid(1,"smem[i]\n",smem_ptr);
    // reduction global mem
    if (blockDim.x >= 1024 && tid < 512){
      /*
      logInfoBigNumberTid(1,"+smem[0]\n",smem_ptr);
      logInfoBigNumberTid(1,"+smem[512]\n",&smem[(tid+512)*NWORDS_256BIT]);
      */
#ifdef CU_ASM_ECADD
      addecjacz(smem_ptr, tid*ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+512)*ECP_JAC_INDIMS, params->midx);
#else 
      addecjac<T1,T2>(smem_ptr, tid*ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+512)*ECP_JAC_INDIMS, params->midx);
#endif
      /*
      logInfoBigNumberTid(1,"smem[0]\n",smem_ptr);
      */
    }
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256){
      /*
      logInfoBigNumberTid(1,"+smem[0]\n",smem_ptr);
      logInfoBigNumberTid(1,"+smem[256]\n",&smem[(tid+256)*NWORDS_256BIT]);
      */
#ifdef CU_ASM_ECADD
      addecjacz(smem_ptr, tid * ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+256)*ECP_JAC_INDIMS, params->midx);
#else
      addecjac<T1,T2>(smem_ptr, tid * ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+256)*ECP_JAC_INDIMS, params->midx);
#endif
      /*
      logInfoBigNumberTid(1,"smem[=256]\n",smem_ptr);
      */
    }
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128){
      /*
      logInfoBigNumberTid(1,"+smem[0]\n",smem_ptr);
      logInfoBigNumberTid(1,"+smem[128]\n",&smem[(tid+128)*NWORDS_256BIT]);
      */
#ifdef CU_ASM_ECADD
      addecjacz(smem_ptr, tid * ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+128)*ECP_JAC_INDIMS, params->midx);
#else
      addecjac<T1,T2>(smem_ptr, tid * ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+128)*ECP_JAC_INDIMS, params->midx);
#endif
      /*
      logInfoBigNumberTid(1,"smem[=128+0]\n",smem_ptr);
      */
    }
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64){
      /*
      logInfoBigNumberTid(1,"+smem[0]\n",smem_ptr);
      logInfoBigNumberTid(1,"+smem[64]\n",&smem[(tid+64)*NWORDS_256BIT]);
      */
#ifdef CU_ASM_ECADD
      addecjacz(smem_ptr, tid * ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+64)*ECP_JAC_INDIMS, params->midx);
#else
      addecjac<T1,T2>(smem_ptr, tid * ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+64)*ECP_JAC_INDIMS, params->midx);
#endif
      /*
      logInfoBigNumberTid(1,"smem[=64+0]\n",smem_ptr);
      */
    }
    __syncthreads();
      
    // unrolling warp

    if (tid < 32)
    {
        //volatile uint32_t *vsmem = smem_ptr;
        uint32_t *zvsmem = smem_ptr->getu256();
        T1 vsmem(zvsmem);
 
        /*
        logInfoBigNumberTid(1,"smem[pre32X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(1,"smem[pre32Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_FP]);
        logInfoBigNumberTid(1,"smem[pre32Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_FP]);
        */
#ifdef CU_ASM_ECADD
        addecjacz(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+32)*ECP_JAC_OUTDIMS, params->midx);
#else
        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+32)*ECP_JAC_OUTDIMS, params->midx);
#endif

        /*
        logInfoBigNumberTid(1,"smem[32X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(1,"smem[32Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_FP]);
        logInfoBigNumberTid(1,"smem[32Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_FP]);

        logInfoBigNumberTid(1,"smem[pre16X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(1,"smem[pre16Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_FP]);
        logInfoBigNumberTid(1,"smem[pre16Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_FP]);
        */
#ifdef CU_ASM_ECADD
        addecjacz(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+16)*ECP_JAC_OUTDIMS, params->midx);
#else
        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+16)*ECP_JAC_OUTDIMS, params->midx);
#endif

        /*
        logInfoBigNumberTid(1,"smem[16X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(1,"smem[16Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_FP]);
        logInfoBigNumberTid(1,"smem[16Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_FP]);
        */
#ifdef CU_ASM_ECADD
        addecjacz(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+8)*ECP_JAC_OUTDIMS, params->midx);
#else
        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+8)*ECP_JAC_OUTDIMS, params->midx);
#endif

        /*
        logInfoBigNumberTid(1,"smem[8X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(1,"smem[8Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_FP]);
        logInfoBigNumberTid(1,"smem[8Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_FP]);
        */
#ifdef CU_ASM_ECADD
        addecjacz(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+4)*ECP_JAC_OUTDIMS, params->midx);
#else
        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+4)*ECP_JAC_OUTDIMS, params->midx);
#endif

        /*
        logInfoBigNumberTid(1,"smem[4X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(1,"smem[4Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_FP]);
        logInfoBigNumberTid(1,"smem[4Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_FP]);
        */
#ifdef CU_ASM_ECADD
        addecjacz(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+2)*ECP_JAC_OUTDIMS, params->midx);
#else
        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+2)*ECP_JAC_OUTDIMS, params->midx);
#endif

        /*
        logInfoBigNumberTid(1,"smem[2X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(1,"smem[2Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_FP]);
        logInfoBigNumberTid(1,"smem[2Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_FP]);
        */
#ifdef CU_ASM_ECADD
        addecjacz(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+1)*ECP_JAC_OUTDIMS, params->midx);
#else
        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+1)*ECP_JAC_OUTDIMS, params->midx);
#endif

        /*
        logInfoBigNumberTid(1,"smem[X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(1,"smem[Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_FP]);
        logInfoBigNumberTid(1,"smem[Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_FP]);
        */

        if (tid==0) {
           xr->setu256(0,smem_ptr,0);
        }
    }

  return;
}

template<typename T1, typename T2>
__device__ void madecjac_shfl(T1 *xr, T1 *xo, uint32_t *scl, T1 *smem_ptr, kernel_params_t *params)
{
    uint32_t i,  size2;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t __align__(16) zsumX[ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint32_t)];
    uint32_t __align__(16) zsumY[ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint32_t)];
    uint32_t laneIdx = tid % warpSize;
    uint32_t warpIdx = tid / warpSize;
    uint32_t opt_ec_offset = 0;
    T1 sumX(zsumX);
    T1 sumY(zsumY);

    //logInfoTid("opt_ec_offset : %d\n",opt_ec_offset);
    size2 = blockDim.x >> 6;
    // scalar multiplication
    if (params->premul){
        sumX.setu256(0,xo,idx*2,1);
        sumX.setu256(1,xo,idx*2+1,1);
        if (idx < opt_ec_offset || params->padding_idx == 0){
	  logInfoBigNumberTid(1,"scl :\n",&scl[idx * NWORDS_FR]);
          logInfoBigNumberTid(2*T1::getN(),"Xin[x,y,z]:\n",&sumX);
          scmulecjac<T1, T2>(&sumX,0, &sumX, 0, &scl[idx * NWORDS_FR],  params);
	  logInfoBigNumberTid(3*T1::getN(),"Xout[x,y,z]:\n",&sumX);
        } else {
           T1 sumX2(xo->getu256(0));
           //logInfoBigNumberTid(2*T1::getN(),"Xo[x,y]:\n",&sumX2);
      
           scmulecjac_opt<T1, T2>(&sumX,0,
                                  &sumX2, 2*(idx*DEFAULT_U256_BSELM_CUDA - opt_ec_offset), 
                                  &scl[idx*NWORDS_FR *DEFAULT_U256_BSELM_CUDA],  params);
           //logInfoBigNumberTid(3*T1::getN(),"Final R : \n",&sumX);
        }
        
    } else {
        sumX.setu256(0,xo,0);
    }

    __syncthreads();
 
    // block wide warp reduce
    #pragma unroll
    for (i = WARP_HALF_SIZE; i > 0; i >>= 1){
      shflxoruecc<T1,T2>(&sumY, &sumX, i);
      logInfoTid("idx:%d\n",i);
      logInfoBigNumberTid(3*T1::getN(),"sumX1\n",&sumX);
      logInfoBigNumberTid(3*T1::getN(),"sumY1\n",&sumY);
    
   #ifdef CU_ASM_ECADD
      addecjacz(&sumX,0, &sumX,0, &sumY,0, params->midx);
   #else
      addecjac<T1,T2>(&sumX,0, &sumX,0, &sumY,0, params->midx);
   #endif
      logInfoBigNumberTid(3*T1::getN(),"sumX1+\n",&sumX);

    }

    __syncthreads();
    if (laneIdx == 0) {
       smem_ptr->setu256(warpIdx*ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint256_t), &sumX,0);
       //logInfoTid("save idx:%d\n",warpIdx);
       //logInfoBigNumberTid(3*T1::getN(),"val\n",&sumX);
    }

    __syncthreads();
  
    if (size2){

      logInfoBigNumberTid(3*T1::getN(),"Smem\n",smem_ptr);
      if (tid < size2*2) {
        //logInfoTid("blockDim :%d\n",blockDim.x);
        //logInfoTid("LaneIdx :%d\n",laneIdx);
        //logInfoTid("Size :%d\n",size2);
  
        sumX.setu256(0,smem_ptr,laneIdx*ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint256_t));
      } else {
        T1 _inf;
        infz(&_inf, params->midx);
        sumX.setu256(0,&_inf,0);
      }
      logInfoBigNumberTid(3*T1::getN(),"Second\n",&sumX);
  
      #pragma unroll
      for (i=size2; i > 0; i >>=1){
        shflxoruecc<T1,T2>(&sumY, &sumX, i);
        logInfoTid("2idx:%d\n",i);
        logInfoBigNumberTid(3*T1::getN(),"sumY\n",&sumY);
        logInfoBigNumberTid(3*T1::getN(),"sumX\n",&sumX);

     #ifdef CU_ASM_ECADD
        addecjacz(&sumX,0, &sumX,0, &sumY,0, params->midx);
     #else
        addecjac<T1,T2>(&sumX,0, &sumX,0, &sumY,0, params->midx);
     #endif
        logInfoBigNumberTid(3*T1::getN(),"sumX+\n",&sumX);
      }
    }

    if (tid==0) {
     xr->setu256(0,&sumX,0);
     //logInfoBigNumberTid(3*T1::getN(),"Z-sumX : \n",&sumX);
    } 

  return;
}

template<typename T1, typename T2>
__device__ void redecjac(T1 *xr, T1 *xo, T1 *smem_ptr, kernel_params_t *params)
{
  uint32_t i,  size2;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t __align__(16) zsumX[ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint32_t)];
    uint32_t __align__(16) zsumY[ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint32_t)];
    uint32_t laneIdx = tid % warpSize;
    uint32_t warpIdx = tid / warpSize;

    //T1 sumX(xo->getu256(0));
    T1 sumX(zsumX);
    T1 sumY(zsumY);
    sumX.setu256(0,xo,0);
    //logInfoTid("opt_ec_offset : %d\n",opt_ec_offset);
    size2 = blockDim.x >> 6;
 
    // block wide warp reduce
    #pragma unroll
    for (i = WARP_HALF_SIZE; i > 0; i >>= 1){
      shflxoruecc<T1,T2>(&sumY, &sumX, i);
      logInfoTid("idx:%d\n",i);
      logInfoBigNumberTid(3*T1::getN(),"sumX1\n",&sumX);
      logInfoBigNumberTid(3*T1::getN(),"sumY1\n",&sumY);
    
   #ifdef CU_ASM_ECADD
      addecjacz(&sumX,0, &sumX,0, &sumY,0, params->midx);
   #else
      addecjac<T1,T2>(&sumX,0, &sumX,0, &sumY,0, params->midx);
   #endif
      logInfoBigNumberTid(3*T1::getN(),"sumX1+\n",&sumX);

    }

    __syncthreads();
    if (laneIdx == 0) {
       smem_ptr->setu256(warpIdx*ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint256_t), &sumX,0);
       //logInfoTid("save idx:%d\n",warpIdx);
       //logInfoBigNumberTid(3*T1::getN(),"val\n",&sumX);
    }

    __syncthreads();
  
    if (size2){

      logInfoBigNumberTid(3*T1::getN(),"Smem\n",smem_ptr);
      if (tid < size2*2) {
        //logInfoTid("blockDim :%d\n",blockDim.x);
        //logInfoTid("LaneIdx :%d\n",laneIdx);
        //logInfoTid("Size :%d\n",size2);
  
        sumX.setu256(0,smem_ptr,laneIdx*ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint256_t));
      } else {
        T1 _inf;
        infz(&_inf, params->midx);
        sumX.setu256(0,&_inf,0);
      }
      //logInfoBigNumberTid(3*T1::getN(),"Second\n",&sumX);
  
      #pragma unroll
      for (i=size2; i > 0; i >>=1){
        shflxoruecc<T1,T2>(&sumY, &sumX, i);
        //logInfoTid("idx:%d\n",i);
        logInfoBigNumberTid(3*T1::getN(),"sumY\n",&sumY);
        logInfoBigNumberTid(3*T1::getN(),"sumX\n",&sumX);

     #ifdef CU_ASM_ECADD
        addecjacz(&sumX,0, &sumX,0, &sumY,0, params->midx);
     #else
        addecjac<T1,T2>(&sumX,0, &sumX,0, &sumY,0, params->midx);
     #endif
        logInfoBigNumberTid(3*T1::getN(),"sumX+\n",&sumX);
      }
    }

    if (tid==0) {
     xr->setu256(0,&sumX,0);
     //logInfoBigNumberTid(3*T1::getN(),"Z-sumX : \n",&sumX);
    } 

  return;
}



/*
  EC point addition
  
  Algorithm (https://en.wikibooks.org/wiki/Cryptography/Prime_Curve/Jacobian_Coordinates):
  IN : P1(X1,Y1,Z1), P2(X2,Y2,Z2)
  OUT: P3(X3,Y3,Z3)

    U1 = X1*Z2^2
    U2 = X2*Z1^2
    S1 = Y1*Z2^3
    S2 = Y2*Z1^3
    if (U1 == U2)
      if (S1 != S2)
        return POINT_AT_INFINITY
      else 
        return POINT_DOUBLE(X1, Y1, Z1)
    H = U2 - U1
    R = S2 - S1
    X3 = R^2 - H^3 - 2*U1*H^2
    Y3 = R*(U1*H^2 - X3) - S1*H^3
    Z3 = H*Z1*Z2
    return (X3, Y3, Z3)

    TODO . check if I can remove
    NOTE X1 cannot be 0. X2 can from two sources: 
      - reduction -> when there is not enough input data, i append with 0 and put it in second addition term
      - scalar multiplication -> if first bit of scalar is 0, I add 0
*/

template<typename T1, typename T2>
__device__ void addecjac(T1 *zxr, uint32_t zoffset, T1 *zx1, uint32_t x1offset, T1 *zx2, uint32_t x2offset, mod_t midx)
{
  T1 x1(zx1->getu256(0+x1offset)), y1(zx1->getu256(1+x1offset)), z1(zx1->getu256(2+x1offset));
  T1 x2(zx2->getu256(0+x2offset)), y2(zx2->getu256(1+x2offset)), z2(zx2->getu256(2+x2offset));
  T1 xr(zxr->getu256(0+zoffset)),  yr(zxr->getu256(1+zoffset)), zr(zxr->getu256(2+zoffset));
 
  uint32_t __restrict__ ztmp[5*sizeof(T2)/sizeof(uint32_t)];
  T1 tmp1(ztmp), tmp3(&ztmp[1*sizeof(T2)/sizeof(uint32_t)]),
                 tmp_x(&ztmp[2*sizeof(T2)/sizeof(uint32_t)]),
                 tmp_y(&ztmp[3*sizeof(T2)/sizeof(uint32_t)]),
                 tmp_z(&ztmp[4*sizeof(T2)/sizeof(uint32_t)]);

  // TODO : Change definition of inf to 0, 1, 0 instead of 1,0,1 as it is now
  /*
  logInfoBigNumberTid(T1::getN(),"x1\n",x1.getu256());
  logInfoBigNumberTid(T1::getN(),"y1\n",y1.getu256());
  logInfoBigNumberTid(T1::getN(),"z1\n",z1.getu256());
  logInfoBigNumberTid(T1::getN(),"x2\n",x2.getu256());
  logInfoBigNumberTid(T1::getN(),"y2\n",y2.getu256());
  logInfoBigNumberTid(T1::getN(),"z2\n",z2.getu256());
  */

  if (eq0z(&z1)){ 
      zxr->setu256(T1::getN()*zoffset,zx2,T1::getN()*x2offset);
      return;  
  }
  if (eq0z(&z2)){ 
      zxr->setu256(T1::getN()*zoffset,zx1,T1::getN()*x1offset);
      return;  
  }

  squarez(&tmp_x, &z1,         midx);  // tmp_x = z1sq 
  squarez(&tmp_y, &z2,        midx);  // tmp_y = z2sq

  mulz(&tmp_z, &tmp_x, &x2, midx);  // tmp_z = u2 = x2 * z1sq
  mulz(&tmp1, &x1, &tmp_y, midx);  // tmp1 = u1 = x1 * z2sq

  mulz(&tmp_x, &tmp_x, &z1, midx);  // tmp_x = z1cube
  mulz(&tmp_y, &tmp_y, &z2, midx);  // tmp_y = z2cube

  mulz(&tmp_x, &tmp_x, &y2, midx);  // tmp_x = s2 = z1cube * y2
  mulz(&tmp_y, &tmp_y, &y1, midx);  // tmp_y = s1 = z2cube * y1

  //  if U1 == U2 and S1 == S2 => P1 = P2 -> double
  //  if U1 == U2 and S1 != S2 => P1 = -P2 -> return 0
  //  instead of calling double,  i proceed. It is better to avoid warp divergence
  if (eqz(&tmp1, &tmp_z)) {    // u1 == u2
      if (!eqz( &tmp_y, &tmp_x)){  // s1 != s2
          T1 _inf;
          infz(&_inf, midx);
          zxr->setu256(zoffset,&_inf,x1offset);
	  return;  
      }
      doublecjac<T1, T2>(&xr,&x1, midx);
      return;
  }

  subz(&tmp_z, &tmp_z, &tmp1, midx);     // H = tmp2 = u2 - u1
  mulz(&zr, &z1, &z2, midx);      // tmp_z = z1 * z2
  squarez(&tmp3, &tmp_z,        midx);     // Hsq = tmp3 = H * H 
  mulz(&zr, &zr, &tmp_z, midx);       // zr = z1 * z2  * h

  /*
  logInfoBigNumberTid(T1::getN(),"H\n",&tmp2);
  logInfoBigNumberTid(T1::getN(),"z1 * z2\n",&tmp_z);
  logInfoBigNumberTid(T1::getN(),"z1 * z2  * h\n",&zr);
  */

  mulz(&tmp_z, &tmp_z, &tmp3, midx);     // Hcube = tmp2 = Hsq * H 
  mulz(&tmp1, &tmp1, &tmp3, midx);     // tmp1 = u1 * Hsq

  /*
  logInfoBigNumberTid(T1::getN(),"Hsq\n",&tmp3);
  logInfoBigNumberTid(T1::getN(),"H3\n",&tmp2);
  logInfoBigNumberTid(T1::getN(),"Hsq * u1\n",&tmp1);
  */

  subz(&tmp3, &tmp_x, &tmp_y, midx);        // R = tmp3 = S2 - S1 tmp1=u1*Hsq, tmp2=Hcube, tmp_x=free, tmp_y=s1, zr=zr
  mulz(&tmp_y, &tmp_y, &tmp_z, midx);     // tmp_y = Hcube * s1
  squarez(&tmp_x, &tmp3, midx);     // tmp_x = R * R

  /*
  logInfoBigNumberTid(T1::getN(),"R\n",&tmp3);
  logInfoBigNumberTid(T1::getN(),"Hcube* s1\n",&tmp_y);
  logInfoBigNumberTid(T1::getN(),"Rsq * u1\n",&tmp_x);
  */

  mul2z(&xr, &tmp1, midx);     // tmp4 = u1*hsq *_2
  subz(&tmp_x, &tmp_x, &tmp_z, midx);        // tmp_x = x3= (R*R)-Hcube, tmp_y = Hcube * S1, zr=zr, tmp1=u1*Hsq, tmp2 = Hcube, tmp3 = R


  /*
  logInfoBigNumberTid(T1::getN(),"Rsq - H3\n",&tmp_x);
  logInfoBigNumberTid(T1::getN(),"Hsq * 2 * u1\n",&tmp4);
  */

  subz(&xr, &tmp_x, &xr, midx);               // x3 = xr
  subz(&tmp1, &tmp1, &xr, midx);       // tmp1 = u1*hs1 - x3
  mulz(&tmp1, &tmp1, &tmp3, midx);  // tmp1 = r * (u1 * hsq - x3)
  subz(&yr, &tmp1, &tmp_y, midx);

  /*
  logInfoBigNumberTid(T1::getN(),"X : \n",&xr);
  logInfoBigNumberTid(T1::getN(),"Y : \n",&yr);
  logInfoBigNumberTid(T1::getN(),"Z : \n",&zr);
  */
}

template<typename T1, typename T2>
__device__ void toaff(T1 *zxr,  T1 *zx1, mod_t midx)
{
  T1 xr(zxr->getu256(0)),  yr(zxr->getu256(1));
  T1 x(zx1->getu256(0)), y(zx1->getu256(1)), z(zx1->getu256(2));
 
  uint32_t __restrict__ ztmp[2*sizeof(T2)/sizeof(uint32_t)];
  T1 tmp1(ztmp), tmp2(&ztmp[1*sizeof(T2)/sizeof(uint32_t)]);

  // if x == 0 => return 0;	
  invz(&tmp1, &z, midx);  
  squarez(&tmp2, &tmp1, midx);
  mulz(&tmp1, &tmp1, &tmp2, midx);

  mulz(&xr, &x, &tmp2, midx);
  mulz(&yr, &y, &tmp1, midx);

}


template<typename T1, typename T2>
__forceinline__ __device__ void addecjacmixed(T1 *zxr, uint32_t zoffset, T1 *zx1, uint32_t x1offset, T1 *zx2, uint32_t x2offset, mod_t midx)
{
  T1 x1(zx1->getu256(0+x1offset)), y1(zx1->getu256(1+x1offset));
  T1 x2(zx2->getu256(0+x2offset)), y2(zx2->getu256(1+x2offset)), z2(zx2->getu256(2+x2offset));
  T1 xr(zxr->getu256(0+zoffset)),  yr(zxr->getu256(1+zoffset)), zr(zxr->getu256(2+zoffset));
 
  uint32_t __restrict__ ztmp[4*sizeof(T2)/sizeof(uint32_t)];
  T1 tmp1(&ztmp[0*sizeof(T2)/sizeof(uint32_t)]), 
     tmp3(&ztmp[1*sizeof(T2)/sizeof(uint32_t)]),
     tmp_x(&ztmp[2*sizeof(T2)/sizeof(uint32_t)]),
     tmp_z(&ztmp[3*sizeof(T2)/sizeof(uint32_t)]);

  // TODO : Change definition of inf to 0, 1, 0 instead of 1,0,1 as it is now
  /*
  logInfoBigNumberTid(T1::getN(),"x1\n",x1.getu256());
  logInfoBigNumberTid(T1::getN(),"y1\n",y1.getu256());
  logInfoBigNumberTid(T1::getN(),"x2\n",x2.getu256());
  logInfoBigNumberTid(T1::getN(),"y2\n",y2.getu256());
  logInfoBigNumberTid(T1::getN(),"z2\n",z2.getu256());
  */

  if (eq0z(&z2)){ 
      zxr->setu256(T1::getN()*zoffset,zx1,T1::getN()*x1offset);
      logInfoTid("R1=inf %d\n", midx);
      return;  
  }
  squarez(&tmp_x, &z2,         midx);  // tmp_x = z2sq 
  mulz(&tmp_z, &tmp_x, &x1, midx);  // tmp_z = u1 = x1 * z2sq
  mulz(&tmp_x, &tmp_x, &z2, midx);  // tmp_x = z2cube
  mulz(&tmp_x, &tmp_x, &y1, midx);  // tmp_x = s1 = z2cube * y1

  //  if U1 == U2 and S1 == S2 => P1 = P2 -> double
  //  if U1 == U2 and S1 != S2 => P1 = -P2 -> return 0
  //  instead of calling double,  i proceed. It is better to avoid warp divergence
  if (eqz(&x2, &tmp_z)){    // u1 == u2
       if (!eqz( &y2, &tmp_x)){  // s1 != s2
          T1 _inf;
          infz(&_inf, midx);
          zxr->setu256(T1::getN()*zoffset,&_inf,0);
          logInfoTid("R2=inf %d\n", midx);
	  return;  
        } 
        //logInfoTid("R4=D %d\n", midx);
        doublecjac<T1, T2>(&xr,&x2, midx);
        return;
  }
  subz(&tmp1, &x2, &tmp_z, midx);     // H = tmp1 = u2 - u1
  mulz(&zr, &z2, &tmp1, midx);       // zr = z1 * z2  * h
  squarez(&tmp3, &tmp1,        midx);     // Hsq = tmp3 = H * H 

  /*
  logInfoBigNumberTid(T1::getN(),"H\n",&tmp1);
  logInfoBigNumberTid(T1::getN(),"z2 * h\n",&zr);
  logInfoBigNumberTid(T1::getN(),"Hsq\n",&tmp3);
  */

  mulz(&tmp1, &tmp1, &tmp3, midx);     // Hcube = tmp1= Hsq * H 
  mulz(&tmp3, &tmp3, &tmp_z, midx);     // tmp3 = u1 * Hsq

  /*
  logInfoBigNumberTid(T1::getN(),"H3\n",&tmp1);
  logInfoBigNumberTid(T1::getN(),"Hsq * u1\n",&tmp3);
  */

  subz(&tmp_z, &y2, &tmp_x, midx);        // R = tmp_z = S2 - S1 
  squarez(&xr, &tmp_z, midx);     // xr = Rsq

  /*
  logInfoBigNumberTid(T1::getN(),"R\n",&tmp_z);
  logInfoBigNumberTid(T1::getN(),"Rsq\n",&xr);
  */

  subz(&xr, &xr, &tmp1, midx);     // xr = Rsq - Hcube

  //logInfoBigNumberTid(T1::getN(),"Rsq - Hcube\n",&xr);

  subz(&xr, &xr, &tmp3, midx);     // xr = Rsq - Hcube - u1*Hsq
  subz(&xr, &xr, &tmp3, midx);     // xr = Rsq - Hcube - 2*u1*Hsq

  //logInfoBigNumberTid(T1::getN(),"X\n",&xr);

  subz(&yr, &tmp3, &xr, midx);          // yr = u1*Hsq - xr
  mulz(&yr, &yr, &tmp_z, midx);        //  yr = R*(u1*Hsq - xr)

  mulz(&tmp1, &tmp1, &tmp_x, midx);     // tmp1 = Hcube * s1
  subz(&yr, &yr, &tmp1, midx);

  /*
  logInfoBigNumberTid(T1::getN(),"X : \n",&xr);
  logInfoBigNumberTid(T1::getN(),"Y : \n",&yr);
  logInfoBigNumberTid(T1::getN(),"Z : \n",&zr);
  */
}

/*
  input is in affine coordinates -> P(Z) = 1
  I can do Q = Q+Y or Q = Y + Q
    NOTE X1, X2 cannot be 0
*/

template <typename T1, typename T2>
__forceinline__ __device__ void addecjacaff(T1  *zxr, T1 *zx1, T1 *zx2, mod_t midx)
{
  T1 y1(zx1->getu256(ECP_JAC_YOFFSET_BASE)), y2(zx2->getu256(ECP_JAC_YOFFSET_BASE)),
     xr(zxr->getu256(ECP_JAC_XOFFSET_BASE)),
     yr(zxr->getu256(ECP_JAC_YOFFSET_BASE)), zr(zxr->getu256(ECP_JAC_ZOFFSET_BASE));

  uint32_t __restrict__ ztmp[4*sizeof(T2)/sizeof(uint32_t)];
  T1 tmp1(ztmp), tmp2(&ztmp[sizeof(T2)/sizeof(uint32_t)]),
                 tmp3(&ztmp[2*sizeof(T2)/sizeof(uint32_t)]), 
                 tmp4(&ztmp[3*sizeof(T2)/sizeof(uint32_t)]);

  // TODO Check if I can call add to compute x + x (instead of double)
  //  if not, I should call double below. I don't want to to avoid warp divergnce
  if (eqz(zx1, zx2)){   // u1 == u2
      if (!eqz( &y1,  &y2)){  // s1 != s2
          T1 _inf;
          infz(&_inf, midx);
          zxr->setu256(0,&_inf,0);
	  return;  //  if U1 == U2 and S1 == S2 => P1 == P2 (call double)
     }
     doublecjacaff<T1, T2>(zxr,zx1, midx);
     return;
  }

  logInfoBigNumberTid(T1::getN(),"x1\n",zx1);
  logInfoBigNumberTid(T1::getN(),"y1\n",&y1);
  logInfoBigNumberTid(T1::getN(),"x2\n",zx2);
  logInfoBigNumberTid(T1::getN(),"y2\n",&y2);

  subz(&zr, zx2, zx1, midx);     // H = tmp2 = u2 - u1
  
  logInfoBigNumberTid(T1::getN(),"H\n",&zr);

  squarez(&tmp3, &zr,        midx);     // Hsq = tmp3 = H * H 
  mulz(&tmp2, &tmp3, &zr, midx);     // Hcube = tmp2 = Hsq * H 
  mulz(&tmp1, zx1, &tmp3, midx);     // tmp1 = u1 * Hsq

  logInfoBigNumberTid(T1::getN(),"Hsq\n",&tmp3);
  logInfoBigNumberTid(T1::getN(),"Hcube\n",&tmp2);
  logInfoBigNumberTid(T1::getN(),"u1 * Hsq\n",&tmp1);

  subz(&tmp3, &y2, &y1, midx);        // R = tmp3 = S2 - S1 tmp1=u1*Hsq, tmp2=Hcube, xr=free, yr=s1, zr=zr
  mulz(&yr, &y1, &tmp2, midx);     // yr = Hcube * s1
  squarez(zxr, &tmp3, midx);     // xr = R * R

  logInfoBigNumberTid(T1::getN(),"R\n",&tmp3);
  logInfoBigNumberTid(T1::getN(),"s1\n",&yr);
  logInfoBigNumberTid(T1::getN(),"Rsq\n",&xr);

  subz(zxr, zxr, &tmp2, midx);        // xr = x3= (R*R)-Hcube, yr = Hcube * S1, zr=zr, tmp1=u1*Hsq, tmp2 = Hcube, tmp3 = R

  // TODO muluk256
  mul2z(&tmp4, &tmp1, midx);     // tmp4 = u1*hsq *_2

  logInfoBigNumberTid(T1::getN(),"Rsq - Hcube\n",&xr);
  logInfoBigNumberTid(T1::getN(),"u1 * Hsq * 2\n",&tmp4);

  subz(zxr, &xr, &tmp4, midx);               // x3 = xr
  subz(&tmp1, &tmp1, zxr, midx);       // tmp1 = u1*hs1 - x3

  logInfoBigNumberTid(T1::getN(),"u1*hsq - x3\n",&tmp1);

  mulz(&tmp1, &tmp1, &tmp3, midx);  // tmp1 = r * (u1 * hsq - x3)

  logInfoBigNumberTid(T1::getN(),"r * (u1*hsq - x3)\n",&tmp1);

  subz(&yr, &tmp1, &yr, midx);

  logInfoBigNumberTid(T1::getN(),"X3\n",&xr);
  logInfoBigNumberTid(T1::getN(),"Y3\n",&yr);
  logInfoBigNumberTid(T1::getN(),"Z3\n",&zr);
}

/*
  EC point doubling
  
  Algorithm (https://en.wikibooks.org/wiki/Cryptography/Prime_Curve/Jacobian_Coordinates):
  IN : P1(X1,Y1,Z1)
  OUT: P'(X',Y',Z')

   if (Y == 0)
      return POINT_AT_INFINITY
   S = 4*X*Y^2
   M = 3*X^2 + a*Z^4
   X' = M^2 - 2*S
   Y' = M*(S - X') - 8*Y^4
   Z' = 2*Y*Z
   return (X', Y', Z')
*/
template<typename T1, typename T2>
 __device__ void doublecjac(T1 *zxr, T1 *zx1, mod_t midx)
{
  T1 y1(zx1->getu256(1)), z1(zx1->getu256(2));
  T1 yr(zxr->getu256(1)), zr(zxr->getu256(2));

  uint32_t __restrict__ ztmp[2*sizeof(T2)/sizeof(uint32_t)];
  T1 tmp_y(&ztmp[0*sizeof(T2)/sizeof(uint32_t)]), 
     tmp_z(&ztmp[1*sizeof(T2)/sizeof(uint32_t)]);

  // TODO : review this comparison, and see if I can do better. or where I should put it
  // as i check this in several places
  if (eq0z(&z1)){ 
      T1 _inf;
      infz(&_inf,midx);
      zxr->setu256(0,&_inf,0);
      //logInfoTid("R3 Inf : %d\n",midx);
      //memcpy(xr, _inf, 3 * NWORDS_FP * sizeof(uint32_t));
      return;  
  }
  squarez(&tmp_z, &y1,            midx);  // tmp_z = ysq
  mulz(&zr, &y1, &z1, midx);     //  Z3 = Y * Z
  squarez(&tmp_y, &tmp_z, midx);  // tmp_y = ysqsq
  
  mulz(&tmp_z, &tmp_z, zx1, midx);  
  addz(&tmp_y, &tmp_y, &tmp_y, midx);  // tmp_y = ysqsq + ysqsq
  addz(&tmp_z, &tmp_z, &tmp_z, midx);  
  addz(&zr, &zr, &zr, midx);
  squarez(&yr, zx1, midx);           
  addz(&tmp_y, &tmp_y, &tmp_y, midx);  // tmp_y = 2ysqsq + 2ysqsq
  addz(&tmp_z, &tmp_z, &tmp_z, midx);  // S = tmp_z = 2X1Ysq + 2X1Ysq
  addz(zxr, &yr, &yr, midx);       
  addz(&tmp_y, &tmp_y, &tmp_y, midx);  // tmp_y = 4ysqsq + 4ysqsq
  addz(&yr, zxr, &yr, midx);       // M = yr = 3Xsq


  squarez(zxr, &yr, midx);       // X3 = Msq

  subz(zxr, zxr, &tmp_z, midx);   // X3 = Msq - S
  subz(zxr, zxr, &tmp_z, midx);      // X3 = Msq - 2S

  subz(&tmp_z, &tmp_z, zxr, midx);   //  tmp_z = S - X3
  mulz(&yr, &yr, &tmp_z, midx);     //  Y3 = M * (S - X3)
  subz(&yr, &yr, &tmp_y, midx);    // Y3 = M * (S - X3) - 8ysqsq


  /*
  logInfoBigNumberTid(T1::getN(),"X : \n",zxr);
  logInfoBigNumberTid(T1::getN(),"Y : \n",&yr);
  logInfoBigNumberTid(T1::getN(),"Z : \n",&zr);
  */
}

/*
  EC point doubling
  
  Algorithm (https://en.wikibooks.org/wiki/Cryptography/Prime_Curve/Jacobian_Coordinates):
  IN : P1(X1,Y1,Z1)
  OUT: P'(X',Y',Z')

   if (Y == 0)
      return POINT_AT_INFINITY
   S = 4*X*Y^2
   M = 3*X^2 + a*Z^4
   X' = M^2 - 2*S
   Y' = M*(S - X') - 8*Y^4
   Z' = 2*Y*Z
   return (X', Y', Z')
*/
template<typename T1, typename T2>
 __device__ void doublecjac(T1 *zxr, T1 *zx1, uint32_t nrep, mod_t midx)
{
  T1 y1(zx1->getu256(1)), z1(zx1->getu256(2));
  T1 yr(zxr->getu256(1)), zr(zxr->getu256(2));

  uint32_t __restrict__ ztmp[2*sizeof(T2)/sizeof(uint32_t)];
  T1 tmp2(&ztmp[0*sizeof(T2)/sizeof(uint32_t)]),
     tmp3(&ztmp[1*sizeof(T2)/sizeof(uint32_t)]); 

  // TODO : review this comparison, and see if I can do better. or where I should put it
  // as i check this in several places
  if (eq0z(&z1)){ 
      T1 _inf;
      infz(&_inf,midx);
      zxr->setu256(0,&_inf,0);
      //logInfoTid("R3 Inf : %d\n",midx);
      //memcpy(xr, _inf, 3 * NWORDS_FP * sizeof(uint32_t));
      return;  
  }
  zxr->setu256(0,zx1,0);
  mul2z(&yr, &y1, midx);              // Y = 2*Y
  while(nrep--){
    squarez(&tmp2, zx1, midx);        // tmp2 = X^2 
    squarez(&tmp3, &yr, midx);        // tmp3 = Y^2
    mul3z(&tmp2, &tmp2, midx);        // tmp2 = A = 3*X^2
    mulz(&tmp3, &tmp3, zxr, midx);    // tmp3 = B = X * Y^2
    squarez(zxr, &tmp2, midx);        // X = A^2
    mulz(&zr, &zr, &yr, midx);        // Z = Z*Y
    subz(zxr, zxr, &tmp3, midx);       // X = A^2 - B
    squarez(&yr, &yr,midx);           
    subz(zxr, zxr, &tmp3, midx);       // X = A^2 - 2*B
    squarez(&yr, &yr,midx);            // Y = Y^4
    subz(&tmp3, &tmp3, zxr, midx);     // tmp3 = B - X

    mulz(&tmp3, &tmp3, &tmp2, midx);  // tmp3 = A * (B - X)
    mul2z(&tmp3, &tmp3, midx);        // tmp3 = 2*A*(B - X)
    subz(&yr, &tmp3, &yr, midx);       // Y = 2 * A*(B -X) - Y^4
  }

  div2z(&yr, &yr);

  /*
  logInfoBigNumberTid(T1::getN(),"X : \n",zxr);
  logInfoBigNumberTid(T1::getN(),"Y : \n",&yr);
  logInfoBigNumberTid(T1::getN(),"Z : \n",&zr);
  */
}

/* 
   X1 cannot be 0
 */
template<typename T1, typename T2>
__forceinline__ __device__ void doublecjacaff(T1 *zxr, T1 *zx1, mod_t midx)
{
  uint32_t ztmp[3*sizeof(T2)/sizeof(uint32_t)];
 
  T1 y1(zx1->getu256(1)); 
  T1 yr(zxr->getu256(1)), zr(zxr->getu256(2)); 

  T1 tmp1(ztmp), tmp2(&ztmp[sizeof(T2)/sizeof(uint32_t)]),
                 tmp_y(&ztmp[2*sizeof(T2)/sizeof(uint32_t)]);

  /*
  logInfoBigNumberTid(T1::getN(),"x1\n",zx1->getu256());
  logInfoBigNumberTid(T1::getN(),"y1\n",y1.getu256());
  */
  squarez(&zr, &y1, midx);  // zr = ysq
  squarez(&tmp_y, &zr, midx);  // yr = ysqsq

  /*
  logInfoBigNumberTid(T1::getN(),"ysq\n",zr.getu256());
  logInfoBigNumberTid(T1::getN(),"Yqsq\n",tmp_y.getu256());
  */
  // TODO muluk256
  mul8z(&tmp_y, &tmp_y, midx);  // tmp_y = ysqsq *_8
  mulz(&zr, &zr, zx1, midx);  // S = zr = x * ysq

  /*
  logInfoBigNumberTid(T1::getN(),"8*Ysqsq\n",tmp_y.getu256());
  logInfoBigNumberTid(T1::getN(),"S\n",zr.getu256());
  */
  // TODO muluk256
  mul4z(&zr, &zr, midx);  // S = zr = S * _4

  //logInfoBigNumberTid(T1::getN(),"S*4\n",zr.getu256());

  squarez(zxr, zx1, midx);  // M1 = xr = x * x
  // TODO muluk256
  mul3z(&tmp1, zxr, midx);  // M = tmp1 = M1 * _3

  /*
  logInfoBigNumberTid(T1::getN(),"Xsq\n",zxr->getu256());
  logInfoBigNumberTid(T1::getN(),"M\n",tmp1.getu256());
  */
  squarez(zxr, &tmp1, midx);  // X3 = xr = M * M,  tmp_y = Ysqsq * _8, zr = S; tmp1 = M
  // TODO muluk256
  mul2z(&tmp2, &zr, midx);   // tmp2 = S * _2

  /* 
  logInfoBigNumberTid(T1::getN(),"M*M\n",zxr->getu256());
  logInfoBigNumberTid(T1::getN(),"S*2\n",tmp2.getu256());
  */

  subz(zxr, zxr, &tmp2, midx);      // X3 = xr; tmp_y = Ysqsq * _8, zr = S, tmp1 = M, 
  subz(&tmp2, &zr, zxr, midx);   //  tmp2 = S - X3

  /*
  logInfoBigNumberTid(T1::getN(),"X3\n",zxr->getu256());
  logInfoBigNumberTid(T1::getN(),"S-X3\n",tmp2.getu256());
  */

  mulz(&tmp2, &tmp2, &tmp1, midx); // tmp2 = M * (S - X3)
  //logInfoBigNumberTid(T1::getN(),"M * (S-X3)\n",tmp2.getu256());
  // TODO muluk256
  mul2z(&zr, &y1, midx);
  subz(&yr, &tmp2, &tmp_y, midx);

  /*
  logInfoBigNumberTid(T1::getN(),"y3\n",yr.getu256());
  logInfoBigNumberTid(T1::getN(),"z3\n",zxr->getu256());
  */
}

template<typename T1, typename T2>
__forceinline__ __device__ void scmulecjac(T1 *zxr, uint32_t zoffset, T1 *zx1, uint32_t xoffset, uint32_t *scl, kernel_params_t *params)
{
  uint32_t i;
  mod_t midx = params->midx;

  uint32_t __restrict__ zN[3*sizeof(T2)/sizeof(uint32_t)]; // N = P
  uint32_t *_1 = G2One_ct;
  T1 N(zN);
  T1 Q(zxr->getu256(zoffset));

  T1 _inf;
  infz(&_inf, midx);

  N.setu256(0,zx1,xoffset,1);
  N.setu256(1,zx1,xoffset+1,1);
  setkz(&N,2,_1);

  T1 x1(N.getu256(xoffset+0));
  T1 y1(N.getu256(xoffset+1));

  zxr->setu256(0,&_inf,0); 

  //logInfoBigNumberTid(3*T1::getN(),"X: \n",&N);
  // TODO : revew this comparison
  if ( (eq0z(&x1) && eqz(&y1,_1)) || eq0u256(scl)){ 
      logInfoBigNumberTid(T1::getN(),"X1 : \n",&x1);
      logInfoBigNumberTid(T1::getN(),"Y1 : \n",&y1);
      logInfoBigNumberTid(3*T1::getN(),"ZX1 : \n",zx1);
      logInfoBigNumberTid(1,"scl : \n",scl);
      logInfoTid("eq0z(&x1) : %d\n",eq0z(&x1));
      logInfoTid("eqz(&y1, _1) : %d\n",eqz(&y1,_1));
      logInfoTid("eq0u256(&scl) : %d\n",eq0u256(scl));
      //zxr->setu256(zoffset,&_inf,0);
      logInfoBigNumberTid(3*T1::getN(),"Inf: \n",zxr);
      return;  
  }


  //Q.setu256(0,&_inf, 0);

  // TODO : Either implement left to right, or count where msb is and substitute while by unrolled
  // loop

  // TODO : MAD several numbers at once using shamir's trick

  logInfoBigNumberTid(1,"SCL mul: \n",scl);
  logInfoBigNumberTid(3*T1::getN(),"Q: \n",&Q);
  logInfoBigNumberTid(3*T1::getN(),"N: \n",&N);

  #if 0
    uint32_t __restrict__ scl_cpy[NWORDS_FR];
    movu256(scl_cpy, scl);
    for (i=0; i< 32; i++){
        scmulecjac_step_r2l<T1, T2>(&Q,&N, scl_cpy, midx);
        scmulecjac_step_r2l<T1, T2>(&Q,&N, scl_cpy, midx);
        scmulecjac_step_r2l<T1, T2>(&Q,&N, scl_cpy, midx);
        scmulecjac_step_r2l<T1, T2>(&Q,&N, scl_cpy, midx);
        scmulecjac_step_r2l<T1, T2>(&Q,&N, scl_cpy, midx);
        scmulecjac_step_r2l<T1, T2>(&Q,&N, scl_cpy, midx);
        scmulecjac_step_r2l<T1, T2>(&Q,&N, scl_cpy, midx);
        scmulecjac_step_r2l<T1, T2>(&Q,&N, scl_cpy, midx);
    }
  #else
    uint32_t offset;
    uint32_t msb = clzu256(scl);

    logInfoTid("msb : %d \n",msb);
    //#pragma unroll
    for (i=msb; i< 1 << (NWORDS_FR); i++){
        offset = i;
        scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset,   midx);
        //scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+1, midx);
        //scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+2, midx);
        //scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+3, midx);
        //scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+4, midx);
        //scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+5, midx);
        //scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+6, midx);
        //scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+7, midx);
     }
  #endif

  logInfoBigNumberTid(3*T1::getN(),"R-N: \n",&N);
  logInfoBigNumberTid(3*T1::getN(),"R-Q: \n",&Q);

  return;
}



template<typename T1, typename T2>
__device__ void scmulecjac_opt(T1 *zxr, uint32_t zoffset, T1 *zx1, uint32_t xoffset, uint32_t *scl, kernel_params_t *params)
{
  uint32_t i;

  uint32_t __restrict__ EC_table[((1<<DEFAULT_U256_BSELM_CUDA))*3*sizeof(T2)/sizeof(uint32_t)]; // N = P
  uint32_t offset;
  uint32_t msb = clzMu256(scl);
  uint32_t  b;

  T1 _inf;
  infz(&_inf, params->midx);

  T1 Q(zxr->getu256(zoffset));
  T1 T(EC_table);

  Q.setu256(0,&_inf, 0);
  //logInfoBigNumberTid(3*T1::getN(),"Initial Q : \n",&Q);

  //logInfoTid("msb : %d\n",msb);

  build_ec_table<T1, T2>(&T, zx1, xoffset, scl, params);
  //logInfoBigNumberTid(3*T1::getN(),"T[10]: \n",T.getsingleu256(0));

  scmulecjac_l2r2<T1,T2>(&Q, &T, scl, msb, params->midx);
  logInfoBigNumberTid(3*T1::getN(),"Final Q : \n",&Q);

  return;
}

template<typename T1, typename T2>
__device__ void scmulecjac_precomputed(T1 *zxr, uint32_t zoffset, T1 *zx1, uint32_t xoffset, uint32_t *scl, kernel_params_t *params)
{
  uint32_t i;

  uint32_t msb = clzMu256(scl);
  uint32_t  b;

  T1 _inf;
  infz(&_inf, params->midx);

  T1 Q(zxr->getu256(zoffset));
  T1 T(zx1->getu256(xoffset));

  Q.setu256(0,&_inf, 0);
  //logInfoBigNumberTid(3*T1::getN(),"Initial Q : \n",&Q);

  //logInfoTid("msb : %d\n",msb);

  scmulecjac_l3r3<T1,T2>(&Q, &T, scl, msb, params->midx);
  logInfoBigNumberTid(3*T1::getN(),"Final Q : \n",&Q);

  return;
}



template<typename T1, typename T2>
__device__ void scmulecjac_l2r2(T1 *Q,T1 *N, uint32_t *scl, uint32_t msb,  mod_t midx )
{
  uint32_t b;
  for (uint32_t i=msb; i< (1 << (NWORDS_FR)); i++){
      b = bselMu256(scl,NWORDS_FR*NBITS_WORD-1-i);
      doublecjac<T1, T2>(Q,Q, midx);

      if (b){
#ifdef CU_ASM_ECADD
       addecjacz (Q,0, N,b*3, Q,0, midx);
#else
       addecjac<T1, T2>(Q,0, N,b*3, Q,0, midx);
#endif
   }
  }
}

template<typename T1, typename T2>
__device__ void scmulecjac_l3r3(T1 *Q,T1 *N, uint32_t *scl, uint32_t msb,  mod_t midx )
{
  uint32_t b;
  for (uint32_t i=msb; i< (1 << (NWORDS_FR)); i++){
      b = bselMu256(scl,NWORDS_FR*NBITS_WORD-1-i);
      doublecjac<T1, T2>(Q,Q, midx);

      if (b){
       addecjacmixed<T1, T2>(Q,0, N,b*2, Q,0, midx);
   }
  }
}
template<typename T1, typename T2>
__device__ void scmulecjac_step_r2l(T1 *Q,T1 *N, uint32_t *scl, mod_t midx )
{
   uint32_t  b0 = shr1u256(scl);
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   logInfoTid("B0 : %d\n",b0);
   if (b0) {
#ifdef CU_ASM_ECADD
      addecjacz (Q,0, N,0, Q,0, midx);
#else
      addecjac<T1, T2> (Q,0, N,0, Q,0, midx);
#endif
   }
   doublecjac<T1, T2>(N,N, midx);
}

template<typename T1, typename T2>
__device__ void scmulecjac_step_l2r(T1 *Q,T1 *N, uint32_t *scl, uint32_t offset, mod_t midx )
{
   uint32_t  b0 = bselu256(scl,NWORDS_FR * NBITS_WORD -1 -offset);
   //int tid = threadIdx.x + blockDim.x * blockIdx.x;
   logInfoTid("B0 : %d\n",b0);
   doublecjac<T1, T2>(Q,Q, midx);
   //logInfoBigNumberTid(3*T1::getN(),"Q-D : \n",Q);
   if (b0) {
      addecjacmixed<T1, T2> (Q,0, N,0, Q,0, midx);
      //logInfoBigNumberTid(3*T1::getN(),"Q-A : \n",Q);
      //addecjac<T1, T2> (Q,0, N,0, Q,0, midx);
   }
   
}

template<typename T1, typename T2>
__device__ void build_ec_table(T1 *d_out,T1 *d_in, uint32_t din_offset, uint32_t *scl, kernel_params_t *params)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t j, last_pow2, k=0;
  T1 _inf;
  uint32_t __restrict__ *_1 = G2One_ct;
  uint32_t indims = 5;
  if (sizeof(T2) == sizeof(uint256_t)){
	    indims = 3;
  }
  
  infz(&_inf, params->midx);

  d_out->setu256(0,&_inf,0);

  //logInfoTid("din_offset : %d\n",din_offset);
  //logInfoTid("stride : %d\n",params->stride);
  //logInfoTid("in_length : %d\n",params->in_length);

  for (j=1; j< (1<<DEFAULT_U256_BSELM_CUDA); j++){
      // check if power of 2
      if ((j & (j-1)) == 0) {
	  last_pow2 = j;
          // check if no more numbers, ecp is inf or scl 0. In all these cases, set input to inf
          //if  ((din_offset + 2*k >= 2*params->in_length/params->stride) ||
          if  ((din_offset + 2*k >= 2*params->in_length/indims) ||
               eq0u256(&scl[k*NWORDS_FR]) ||
               (eq0z(d_in, din_offset + k*2) && eqz(d_in, din_offset + k*2 + 1,_1))) {
                d_out->setu256(j*3 * T1::getN(),&_inf,0);
          }
          // else , add number to table
          else {
                d_out->setu256(j* 3,
                            d_in, din_offset + k*2,
			    1);
                d_out->setu256(j*3+1,
                            d_in, din_offset + (k*2+1),
			    2);
                setkz(d_out,j*3+2,_1);
         }
         k++;
      } else {
#ifdef CU_ASM_ECADD
         addecjacz(d_out,j*3,
                  d_out, last_pow2*3,
                  d_out,(j-last_pow2)*3, params->midx);
#else
         addecjac<T1,T2>(d_out,j*3,
                         d_out, last_pow2*3,
                         d_out,(j-last_pow2)*3, params->midx);
#endif

      }
  }
}

template<typename T1, typename T2>
__forceinline__ __device__ void shflxoruecc(T1 *d_out,T1 *d_in, uint32_t srcLane )
{
    ulonglong4 *in, *out;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t i;

    #pragma unroll
    for (i=0; i<ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint256_t);i++){
    
      in = (ulonglong4 *)d_in->getsingleu256(i);
      out = (ulonglong4 *)d_out->getsingleu256(i);

      out->x = __shfl_xor_sync(0xffffffff, in->x, srcLane);
      out->y = __shfl_xor_sync(0xffffffff, in->y, srcLane);
      out->z = __shfl_xor_sync(0xffffffff, in->z, srcLane);
      out->w = __shfl_xor_sync(0xffffffff, in->w, srcLane);
    }
}


