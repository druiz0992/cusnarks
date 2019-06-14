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

#include "types.h"
#include "cuda.h"
#include "log.h"
#include "utils_device.h"
#include "u256_device.h"
#include "z1_device.h"
#include "z2_device.h"
#include "ecbn128_device.h"

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
    Z1_t x1(&in_vector[tid * 2 * ECP_JAC_OUTOFFSET + ECP_JAC_INXOFFSET]);
    Z1_t x2(&in_vector[(tid * 2 + 1) * ECP_JAC_OUTOFFSET + ECP_JAC_INXOFFSET]);
    Z1_t xr(&out_vector[tid * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
   
    //TODO : this is not very nice, but it gets the job done. Try to come up
    // with something better 
    addecjac<Z1_t, uint256_t>(&xr,0, &x1,0, &x2,0, params->midx);

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


__global__ void scmulecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *scl;
 
   if(tid >= params->in_length/3) {
     return;
   }

   scl = (uint32_t *) &in_vector[tid * NWORDS_256BIT + ECP_SCLOFFSET];
   Z1_t x1(&in_vector[ params->in_length/3 * NWORDS_256BIT+ tid * ECP_JAC_INOFFSET + ECP_JAC_INXOFFSET]);
   Z1_t xr(&out_vector[tid * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
  
   scmulecjac<Z1_t, uint256_t>(&xr,0, &x1,0, scl,  params->midx);

   return;
}

__global__ void sc1mulecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *scl;
 
   if(tid >= (params->in_length-1)/2) {
     return;
   }

   scl = (uint32_t *) &in_vector[ECP_SCLOFFSET];
   Z1_t x1(&in_vector[NWORDS_256BIT + tid * ECP_JAC_INOFFSET + ECP_JAC_INXOFFSET]);
   Z1_t xr(&out_vector[tid * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
  
   scmulecjac<Z1_t, uint256_t>(&xr,0, &x1,0, scl,  params->midx);

   return;
}

__global__ void scmulec2jac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *scl;
 
   if(tid >= params->in_length/5) {
     return;
   }

   scl = (uint32_t *) &in_vector[tid * NWORDS_256BIT + ECP_SCLOFFSET];
   Z2_t x1(&in_vector[ params->in_length/5 * NWORDS_256BIT+ tid * ECP2_JAC_INOFFSET + ECP2_JAC_INXOFFSET]);
   Z2_t xr(&out_vector[tid * ECP2_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
  
   scmulecjac<Z2_t, uint512_t>(&xr,0, &x1,0, scl,  params->midx);

   return;
}

__global__ void sc1mulec2jac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *scl;
 
   if(tid >= (params->in_length-1)/4) {
     return;
   }

   scl = (uint32_t *) &in_vector[ECP_SCLOFFSET];
   Z2_t x1(&in_vector[NWORDS_256BIT+ tid * ECP2_JAC_INOFFSET + ECP2_JAC_INXOFFSET]);
   Z2_t xr(&out_vector[tid * ECP2_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);
  
   scmulecjac<Z2_t, uint512_t>(&xr,0, &x1,0, scl,  params->midx);

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
    logInfoTid(idx,"Min Padding : %d\n",params->padding_idx);
    logInfoTid(idx,"Max Padding : %d\n",params->in_length/ECP_JAC_OUTDIMS);
    if (params->padding_idx){
       uint32_t padding[] = {0,0,0,0,0,0,0,0};
       // add zeros between padding and next multiple of 32
       if (idx < params->in_length/ECP_JAC_OUTDIMS && idx >= params->padding_idx){
          movu256(&in_vector[idx * ECP_JAC_INOFFSET],padding);
          movu256(&in_vector[idx * ECP_JAC_INOFFSET + NWORDS_256BIT],padding);
          movu256(&in_vector[idx * ECP_JAC_INOFFSET + 2*NWORDS_256BIT],padding);
       }
       __syncthreads();
    }

    Z1_t xo, xr;
    if (params->premul){
      xo.assign(&in_vector[params->in_length/3 * NWORDS_256BIT + idx  * (params->stride-1) * NWORDS_256BIT + ECP_JAC_OUTXOFFSET]); // 0 .. N-1
      scl = (uint32_t *) &in_vector[idx * params->stride/3 *  NWORDS_256BIT];
      logInfoTid(idx,"LE : %d\n",params->in_length);
      logInfoTid(idx,"InVO : %d\n",(params->stride-1) * NWORDS_256BIT);
      logInfoTid(idx,"SclVO : %d\n",params->in_length/3* 2 * NWORDS_256BIT);
    } else {
      xo.assign(&in_vector[idx  * (params->stride) * NWORDS_256BIT + ECP_JAC_OUTXOFFSET]); // 0 .. N-1
      scl = NULL;
      logInfoTid(idx,"LE : %d\n",params->in_length);
      logInfoTid(idx,"InVO : %d\n",(params->stride) * NWORDS_256BIT);
      logInfoTid(idx,"SclVO : %d\n",params->in_length/3* 2 *NWORDS_256BIT);
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
    logInfoTid(idx,"Min Padding : %d\n",params->padding_idx);
    logInfoTid(idx,"Max Padding : %d\n",params->in_length/ECP2_JAC_OUTDIMS);
    if (params->padding_idx){
       uint32_t padding[] = {0,0,0,0,0,0,0,0};
       // add zeros between padding and next multiple of 32
       if (idx < params->in_length/ECP2_JAC_OUTDIMS && idx >= params->padding_idx){
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET+ NWORDS_256BIT],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 2*NWORDS_256BIT],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 3*NWORDS_256BIT],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 4*NWORDS_256BIT],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 5*NWORDS_256BIT],padding);
       }
       __syncthreads();
    }

    Z2_t xo, xr;
    if (params->premul){
      xo.assign(&in_vector[params->in_length/5 * NWORDS_256BIT + idx  * params->stride * NWORDS_256BIT + ECP2_JAC_OUTXOFFSET]); // 0 .. N-1
      scl = (uint32_t *) &in_vector[idx * params->stride/5 *  NWORDS_256BIT];
      logInfoTid(idx,"LE : %d\n",params->in_length);
      logInfoTid(idx,"InVO : %d\n",(params->stride) * NWORDS_256BIT);
      logInfoTid(idx,"SclVO : %d\n",params->in_length/5* NWORDS_256BIT);
    } else {
      xo.assign(&in_vector[idx  * (params->stride) * NWORDS_256BIT + ECP2_JAC_OUTXOFFSET]); // 0 .. N-1
      scl = NULL;
      logInfoTid(idx,"LE : %d\n",params->in_length);
      logInfoTid(idx,"InVO : %d\n",(params->stride) * NWORDS_256BIT);
      logInfoTid(idx,"SclVO : %d\n",params->in_length/6 *NWORDS_256BIT);
    }
    xr.assign(&in_vector[blockIdx.x * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]);  // 
    if (gridDim.x == 1){
      xr.assign(out_vector);
    }

    madecjac<Z2_t, uint512_t>(&xr, &xo, scl, &zsmem, params);
}
__global__ void madecjac_shfl_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
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
    logInfoTid(idx,"Min Padding : %d\n",params->padding_idx);
    logInfoTid(idx,"Max Padding : %d\n",params->in_length/ECP_JAC_OUTDIMS);
    logInfoTid(idx,"OUt : %d\n",params->in_length/params->stride);
    if (params->padding_idx){
       uint32_t __align__(16) padding[] = {0,0,0,0,0,0,0,0};
       // add zeros between padding and next multiple of 32
       if (idx < params->in_length/ECP_JAC_OUTDIMS && idx >= params->padding_idx){
          movu256(&in_vector[idx * ECP_JAC_OUTOFFSET],padding);
          movu256(&in_vector[idx * ECP_JAC_OUTOFFSET + NWORDS_256BIT],padding);
          movu256(&in_vector[idx * ECP_JAC_OUTOFFSET + 2*NWORDS_256BIT],padding);
       }
       __syncthreads();
    }

    Z1_t xo;
    if (params->premul){
      scl = (uint32_t *) &in_vector[idx *  NWORDS_256BIT];
      poffset = params->in_length/3 * NWORDS_256BIT;
      xo.assign(&in_vector[poffset + idx * ECP_JAC_INOFFSET + ECP_JAC_INXOFFSET]); // 0 .. N-1
      logInfoBigNumberTid(idx,1,"SCL in \n",scl);
    } else {
      xo.assign(&out_vector[idx * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]); // 0 .. N-1
    }

    Z1_t xr(&out_vector[blockIdx.x * ECP_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]);  // 
  
    //if (gridDim.x == 1){
      //xr.assign(out_vector);
    //} 

    logInfoBigNumberTid(idx,2,"X in \n",&xo);
    //logInfoBigNumberTid(idx,32*3,"In \n",in_vector);
    madecjac_shfl<Z1_t, uint256_t>(&xr, &xo, scl, &zsmem, params);
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
    logInfoTid(idx,"Min Padding : %d\n",params->padding_idx);
    logInfoTid(idx,"OUt : %d\n",params->in_length/params->stride);
    if (params->padding_idx){
       uint32_t padding[] = {0,0,0,0,0,0,0,0};
       // add zeros between padding and next multiple of 32
       if (idx < params->in_length/ECP2_JAC_OUTDIMS && idx >= params->padding_idx){
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + NWORDS_256BIT],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 2*NWORDS_256BIT],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 3*NWORDS_256BIT],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 4*NWORDS_256BIT],padding);
          movu256(&in_vector[idx * ECP2_JAC_OUTOFFSET + 5*NWORDS_256BIT],padding);
       }
       __syncthreads();
    }

    Z2_t xo;
    if (params->premul){
      scl = (uint32_t *) &in_vector[idx *  NWORDS_256BIT];
      poffset = params->in_length/5 * NWORDS_256BIT;
      xo.assign(&in_vector[poffset + idx * ECP2_JAC_INOFFSET + ECP2_JAC_INXOFFSET]); // 0 .. N-1
    } else {
      xo.assign(&in_vector[idx * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]); // 0 .. N-1
    }

    Z2_t xr(&in_vector[blockIdx.x * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET]);  // 
  
    if (gridDim.x == 1){
      xr.assign(out_vector);
    } 

    logInfoBigNumberTid(idx,2,"X in \n",&xo);
    //logInfoBigNumberTid(idx,32*3,"In \n",in_vector);
    madecjac_shfl<Z2_t, uint512_t>(&xr, &xo, scl, &zsmem, params);
}


template<typename T1, typename T2>
__forceinline__ __device__ void madecjac(T1 *xr, T1 *xo, uint32_t *scl, T1 *smem_ptr, kernel_params_t *params)
{
    uint32_t i;
    uint32_t ndbg = T1::getN();
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;

    logInfoTid(idx,"stride :%d\n",params->stride/ECP_JAC_OUTDIMS);
    // scalar multipliation
    if (params->premul){
        #pragma unroll
        for (i =0; i < params->stride/ECP_JAC_OUTDIMS; i++){
          /*
          logInfoBigNumberTid(idx,1,"scl :\n",&scl[i*ECP_JAC_INOFFSET]);
          logInfoBigNumberTid(idx,ndbg,"Xin[x]:\n",xos(i*ECP_JAC_INDIMS));
          logInfoBigNumberTid(idx,,"Xin[y]:\n",&xi[i*ECP_JAC_INOFFSET + NWORDS_256BIT]);
          */

          scmulecjac<T1, T2>(xr,i*ECP_JAC_OUTDIMS, xo, i*ECP_JAC_INDIMS, &scl[i*NWORDS_256BIT],  params->midx);
          

          /*
          logInfoBigNumberTid(idx,1,"Xout[x]:\n",&xo[i*ECP_JAC_OUTOFFSET]);
          logInfoBigNumberTid(idx,1,"Xout[y]:\n",&xo[i*ECP_JAC_OUTOFFSET + NWORDS_256BIT]);
          logInfoBigNumberTid(idx,1,"Xout[z]:\n",&xo[i*ECP_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
          */
        }
    }
   
    if (params->stride/ECP_JAC_OUTDIMS > 1){ 
      addecjac<T1,T2>(smem_ptr,tid*ECP_JAC_OUTDIMS, xr,0, xr,ECP_JAC_OUTDIMS, params->midx);
      /*
      logInfoBigNumberTid(idx,1,"smem[X]\n",smem_ptr);
      logInfoBigNumberTid(idx,1,"smem[Y]\n",&smem_ptr[NWORDS_256BIT]);
      logInfoBigNumberTid(idx,1,"smem[Z]\n",&smem_ptr[2*NWORDS_256BIT]);
      */

      #pragma unroll
      for (i =0; i < params->stride/ECP_JAC_OUTDIMS-2; i++){
        addecjac<T1,T2>(smem_ptr,tid*ECP_JAC_OUTDIMS, smem_ptr, 0,xr, (i+2)*ECP_JAC_OUTDIMS, params->midx);
        /*
        logInfoBigNumberTid(idx,1,"smem[X]\n",smem_ptr);
        logInfoBigNumberTid(idx,1,"smem[Y]\n",&smem_ptr[NWORDS_256BIT]);
        logInfoBigNumberTid(idx,1,"smem[Z]\n",&smem_ptr[2*NWORDS_256BIT]);
        */
      }
      __syncthreads();
    }
  
    //logDebugBigNumberTid(idx,1,"smem[i]\n",smem_ptr);
    // reduction global mem
    if (blockDim.x >= 1024 && tid < 512){
      /*
      logInfoBigNumberTid(idx,1,"+smem[0]\n",smem_ptr);
      logInfoBigNumberTid(idx,1,"+smem[512]\n",&smem[(tid+512)*NWORDS_256BIT]);
      */
      
      addecjac<T1,T2>(smem_ptr, tid*ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+512)*ECP_JAC_INDIMS, params->midx);
      /*
      logInfoBigNumberTid(idx,1,"smem[0]\n",smem_ptr);
      */
    }
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256){
      /*
      logInfoBigNumberTid(idx,1,"+smem[0]\n",smem_ptr);
      logInfoBigNumberTid(idx,1,"+smem[256]\n",&smem[(tid+256)*NWORDS_256BIT]);
      */
      addecjac<T1,T2>(smem_ptr, tid * ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+256)*ECP_JAC_INDIMS, params->midx);
      /*
      logInfoBigNumberTid(idx,1,"smem[=256]\n",smem_ptr);
      */
    }
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128){
      /*
      logInfoBigNumberTid(idx,1,"+smem[0]\n",smem_ptr);
      logInfoBigNumberTid(idx,1,"+smem[128]\n",&smem[(tid+128)*NWORDS_256BIT]);
      */
      addecjac<T1,T2>(smem_ptr, tid * ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+128)*ECP_JAC_INDIMS, params->midx);
      /*
      logInfoBigNumberTid(idx,1,"smem[=128+0]\n",smem_ptr);
      */
    }
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64){
      /*
      logInfoBigNumberTid(idx,1,"+smem[0]\n",smem_ptr);
      logInfoBigNumberTid(idx,1,"+smem[64]\n",&smem[(tid+64)*NWORDS_256BIT]);
      */
      addecjac<T1,T2>(smem_ptr, tid * ECP_JAC_INDIMS,
               smem_ptr,tid * ECP_JAC_INDIMS,
               smem_ptr, (tid+64)*ECP_JAC_INDIMS, params->midx);
      /*
      logInfoBigNumberTid(idx,1,"smem[=64+0]\n",smem_ptr);
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
        logInfoBigNumberTid(idx,1,"smem[pre32X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(idx,1,"smem[pre32Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_256BIT]);
        logInfoBigNumberTid(idx,1,"smem[pre32Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
        */

        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+32)*ECP_JAC_OUTDIMS, params->midx);

        /*
        logInfoBigNumberTid(idx,1,"smem[32X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(idx,1,"smem[32Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_256BIT]);
        logInfoBigNumberTid(idx,1,"smem[32Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_256BIT]);

        logInfoBigNumberTid(idx,1,"smem[pre16X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(idx,1,"smem[pre16Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_256BIT]);
        logInfoBigNumberTid(idx,1,"smem[pre16Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
        */

        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+16)*ECP_JAC_OUTDIMS, params->midx);

        /*
        logInfoBigNumberTid(idx,1,"smem[16X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(idx,1,"smem[16Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_256BIT]);
        logInfoBigNumberTid(idx,1,"smem[16Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
        */

        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+8)*ECP_JAC_OUTDIMS, params->midx);

        /*
        logInfoBigNumberTid(idx,1,"smem[8X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(idx,1,"smem[8Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_256BIT]);
        logInfoBigNumberTid(idx,1,"smem[8Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
        */

        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+4)*ECP_JAC_OUTDIMS, params->midx);

        /*
        logInfoBigNumberTid(idx,1,"smem[4X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(idx,1,"smem[4Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_256BIT]);
        logInfoBigNumberTid(idx,1,"smem[4Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
        */

        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+2)*ECP_JAC_OUTDIMS, params->midx);

        /*
        logInfoBigNumberTid(idx,1,"smem[2X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(idx,1,"smem[2Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_256BIT]);
        logInfoBigNumberTid(idx,1,"smem[2Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
        */

        addecjac<T1, T2>(&vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,tid * ECP_JAC_OUTDIMS,
                 &vsmem,(tid+1)*ECP_JAC_OUTDIMS, params->midx);

        /*
        logInfoBigNumberTid(idx,1,"smem[X]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET]);
        logInfoBigNumberTid(idx,1,"smem[Y]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET+NWORDS_256BIT]);
        logInfoBigNumberTid(idx,1,"smem[Z]\n",(uint32_t *)&vsmem[tid * ECP_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
        */

        if (tid==0) {
           xr->setu256(0,smem_ptr,0);
        }
    }

  return;
}

template<typename T1, typename T2>
__forceinline__ __device__ void madecjac_shfl(T1 *xr, T1 *xo, uint32_t *scl, T1 *smem_ptr, kernel_params_t *params)
{
    uint32_t i, size1, size2;
    uint32_t ndbg = T1::getN();
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t __align__(16) zsumX[ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint32_t)];
    uint32_t __align__(16) zsumY[ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint32_t)];
    uint32_t laneIdx = tid % warpSize;
    uint32_t warpIdx = tid / warpSize;
    T1 sumX(zsumX);
    T1 sumY(zsumY);
    T1 _inf;
    infz(&_inf, params->midx);

    size1 = 16;
    // ECP_JAC_INOFFSET = 3 * NWORDS_256BIT
    // ECP_JAC_INXOFFSET = 1 * NWORDS_256BIT
    // scalar multipliation
    if (params->premul){
        sumX.setu256(0,xo,0,1);
        sumX.setu256(1,xo,1,1);

        logInfoBigNumberTid(idx,1,"scl :\n",scl);
 
        scmulecjac<T1, T2>(&sumX,0, &sumX, 0, scl,  params->midx);
          
        logInfoBigNumberTid(idx,3*ndbg,"Xout[x,y,z]:\n",&sumX);

        size2 = blockDim.x >> 6;
    } else {
        size2 = blockDim.x >> 6;
        sumX.setu256(0,xo,0);
    }
   
    __syncthreads();

    // block wide warp reduce
    #pragma unroll
    for (i = size1; i > 0; i >>= 1){
      shflxoruecc<T1,T2>(&sumY, &sumX, i);
      logInfoTid(idx,"idx:%d\n",i);
      logInfoBigNumberTid(idx,3*ndbg,"sumX\n",&sumX);
      logInfoBigNumberTid(idx,3*ndbg,"sumY\n",&sumY);

      addecjac<T1,T2>(&sumX,0, &sumX,0, &sumY,0, params->midx);

      logInfoBigNumberTid(idx,3*ndbg,"sumX+\n",&sumX);
    }

    __syncthreads();
    if (laneIdx == 0) {
       smem_ptr->setu256(warpIdx*ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint256_t), &sumX,0);
       logInfoTid(idx,"save idx:%d\n",warpIdx);
       logInfoBigNumberTid(idx,ndbg*3,"val\n",&sumX);
    }

    __syncthreads();
  
    if (size2){

      logInfoBigNumberTid(idx,ndbg*3,"Smem\n",smem_ptr);
      if (tid < size2*2) {
        logInfoTid(idx,"blockDim :%d\n",blockDim.x);
        logInfoTid(idx,"LaneIdx :%d\n",laneIdx);
        logInfoTid(idx,"Size :%d\n",size2);
  
        sumX.setu256(0,smem_ptr,laneIdx*ECP_JAC_OUTDIMS*sizeof(T2)/sizeof(uint256_t));
      } else {
        sumX.setu256(0,&_inf,0);
      }
      logInfoBigNumberTid(idx,ndbg*3,"Second\n",&sumX);
  
      #pragma unroll
      // last warp reduce
      for (i=size2; i > 0; i >>=1){
        shflxoruecc<T1,T2>(&sumY, &sumX, i);
        logInfoTid(idx,"idx:%d\n",i);
        logInfoBigNumberTid(idx,ndbg*3,"sumY\n",&sumY);
        logInfoBigNumberTid(idx,ndbg*3,"sumX\n",&sumX);
        addecjac<T1,T2>(&sumX,0, &sumX,0, &sumY,0, params->midx);
        logInfoBigNumberTid(idx,3*ndbg,"sumX+\n",&sumX);
      }
    }

    __syncthreads();
    if (tid==0) {
     //TODO change be movu256
     xr->setu256(0,&sumX,0);
     logInfoBigNumberTid(idx,ndbg*3,"Z-sumX : \n",&sumX);
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
__forceinline__ __device__ void addecjac(T1 *zxr, uint32_t zoffset, T1 *zx1, uint32_t x1offset, T1 *zx2, uint32_t x2offset, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  T1 x1(zx1->getu256(0+x1offset)), y1(zx1->getu256(1+x1offset)), z1(zx1->getu256(2+x1offset));
  T1 x2(zx2->getu256(0+x2offset)), y2(zx2->getu256(1+x2offset)), z2(zx2->getu256(2+x2offset));
  T1 xr(zxr->getu256(0+zoffset)),  yr(zxr->getu256(1+zoffset)), zr(zxr->getu256(2+zoffset));
  T1 _inf;

  infz(&_inf, midx);
 
  uint32_t ndbg=T1::getN();
  uint32_t __restrict__ ztmp[7*sizeof(T2)/sizeof(uint32_t)];
  T1 tmp1(ztmp), tmp2(&ztmp[sizeof(T2)/sizeof(uint32_t)]),
                 tmp3(&ztmp[2*sizeof(T2)/sizeof(uint32_t)]),
                 tmp4(&ztmp[3*sizeof(T2)/sizeof(uint32_t)]),
                 tmp_x(&ztmp[4*sizeof(T2)/sizeof(uint32_t)]),
                 tmp_y(&ztmp[5*sizeof(T2)/sizeof(uint32_t)]),
                 tmp_z(&ztmp[6*sizeof(T2)/sizeof(uint32_t)]);

  // TODO : Change definition of inf to 0, 1, 0 instead of 1,0,1 as it is now
  /*
  logInfoBigNumberTid(tid,ndbg,"x1\n",x1.getu256());
  logInfoBigNumberTid(tid,ndbg,"y1\n",y1.getu256());
  logInfoBigNumberTid(tid,ndbg,"z1\n",z1.getu256());
  logInfoBigNumberTid(tid,ndbg,"x2\n",x2.getu256());
  logInfoBigNumberTid(tid,ndbg,"y2\n",y2.getu256());
  logInfoBigNumberTid(tid,ndbg,"z2\n",z2.getu256());
  */

  if (eq0z(&y2)){ 
      zxr->setu256(zoffset,zx1,x1offset);
      //logInfoTid(tid,"R1=inf\n",tid);
      return;  
  }
  squarez(&tmp_x, &z1,         midx);  // tmp_x = z1sq 
  mulz(&tmp_z, &tmp_x, &x2, midx);  // tmp_z = u2 = x2 * z1sq
  mulz(&tmp_x, &tmp_x, &z1, midx);  // tmp_x = z1cube
  mulz(&tmp_x, &tmp_x, &y2, midx);  // tmp_x = s2 = z1cube * y2
  squarez(&tmp_y, &z2,        midx);  // tmp_y = z2sq
  mulz(&tmp1, &x1, &tmp_y, midx);  // tmp1 = u1 = x1 * z2sq
  mulz(&tmp_y, &tmp_y, &z2, midx);  // tmp_y = z2cube
  mulz(&tmp_y, &tmp_y, &y1, midx);  // tmp_y = s1 = z2cube * y1

  //  if U1 == U2 and S1 == S2 => P1 = P2 -> double
  //  if U1 == U2 and S1 != S2 => P1 = -P2 -> return 0
  //  instead of calling double,  i proceed. It is better to avoid warp divergence
  if (eqz(&tmp1, &tmp_z) &&   // u1 == u2
       !eqz( &tmp_y, &tmp_x)){  // s1 != s2
          zxr->setu256(zoffset,&_inf,x1offset);
          //logInfoTid(tid,"R2=inf\n",tid);
	  return;  

  }

  subz(&tmp2, &tmp_z, &tmp1, midx);     // H = tmp2 = u2 - u1
  mulz(&tmp_z, &z1, &z2, midx);      // tmp_z = z1 * z2
  mulz(&zr, &tmp_z, &tmp2, midx);       // zr = z1 * z2  * h

  /*
  logInfoBigNumberTid(tid,ndbg,"H\n",&tmp2);
  logInfoBigNumberTid(tid,ndbg,"z1 * z2\n",&tmp_z);
  logInfoBigNumberTid(tid,ndbg,"z1 * z2  * h\n",&zr);
  */

  squarez(&tmp3, &tmp2,        midx);     // Hsq = tmp3 = H * H 
  mulz(&tmp2, &tmp3, &tmp2, midx);     // Hcube = tmp2 = Hsq * H 
  mulz(&tmp1, &tmp1, &tmp3, midx);     // tmp1 = u1 * Hsq

  /*
  logInfoBigNumberTid(tid,ndbg,"Hsq\n",&tmp3);
  logInfoBigNumberTid(tid,ndbg,"H3\n",&tmp2);
  logInfoBigNumberTid(tid,ndbg,"Hsq * u1\n",&tmp1);
  */

  subz(&tmp3, &tmp_x, &tmp_y, midx);        // R = tmp3 = S2 - S1 tmp1=u1*Hsq, tmp2=Hcube, tmp_x=free, tmp_y=s1, zr=zr
  mulz(&tmp_y, &tmp_y, &tmp2, midx);     // tmp_y = Hcube * s1
  squarez(&tmp_x, &tmp3, midx);     // tmp_x = R * R

  /*
  logInfoBigNumberTid(tid,ndbg,"R\n",&tmp3);
  logInfoBigNumberTid(tid,ndbg,"Hcube* s1\n",&tmp_y);
  logInfoBigNumberTid(tid,ndbg,"Rsq * u1\n",&tmp_x);
  */

  subz(&tmp_x, &tmp_x, &tmp2, midx);        // tmp_x = x3= (R*R)-Hcube, tmp_y = Hcube * S1, zr=zr, tmp1=u1*Hsq, tmp2 = Hcube, tmp3 = R

  // TODO muluk256
  mul2z(&tmp4, &tmp1, midx);     // tmp4 = u1*hsq *_2

  /*
  logInfoBigNumberTid(tid,ndbg,"Rsq - H3\n",&tmp_x);
  logInfoBigNumberTid(tid,ndbg,"Hsq * 2 * u1\n",&tmp4);
  */

  subz(&xr, &tmp_x, &tmp4, midx);               // x3 = xr
  subz(&tmp1, &tmp1, &xr, midx);       // tmp1 = u1*hs1 - x3
  mulz(&tmp1, &tmp1, &tmp3, midx);  // tmp1 = r * (u1 * hsq - x3)
  subz(&yr, &tmp1, &tmp_y, midx);

  /*
  logInfoBigNumberTid(tid,ndbg,"X : \n",&xr);
  logInfoBigNumberTid(tid,ndbg,"Y : \n",&yr);
  logInfoBigNumberTid(tid,ndbg,"Z : \n",&zr);
  */
}

template<typename T1, typename T2>
__forceinline__ __device__ void addecjacmixed(T1 *zxr, uint32_t zoffset, T1 *zx1, uint32_t x1offset, T1 *zx2, uint32_t x2offset, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  T1 x1(zx1->getu256(0+x1offset)), y1(zx1->getu256(1+x1offset));
  T1 x2(zx2->getu256(0+x2offset)), y2(zx2->getu256(1+x2offset)), z2(zx2->getu256(2+x2offset));
  T1 xr(zxr->getu256(0+zoffset)),  yr(zxr->getu256(1+zoffset)), zr(zxr->getu256(2+zoffset));
  T1 _inf;

  infz(&_inf, midx);
 
  uint32_t ndbg=T1::getN();
  uint32_t __restrict__ ztmp[4*sizeof(T2)/sizeof(uint32_t)];
  T1 tmp1(ztmp), tmp3(&ztmp[sizeof(T2)/sizeof(uint32_t)]),
                 tmp_x(&ztmp[2*sizeof(T2)/sizeof(uint32_t)]),
                 tmp_z(&ztmp[3*sizeof(T2)/sizeof(uint32_t)]);

  // TODO : Change definition of inf to 0, 1, 0 instead of 1,0,1 as it is now
  /*
  logInfoBigNumberTid(tid,ndbg,"x1\n",x1.getu256());
  logInfoBigNumberTid(tid,ndbg,"y1\n",y1.getu256());
  logInfoBigNumberTid(tid,ndbg,"x2\n",x2.getu256());
  logInfoBigNumberTid(tid,ndbg,"y2\n",y2.getu256());
  logInfoBigNumberTid(tid,ndbg,"z2\n",z2.getu256());
  */

  if (eq0z(&y2)){ 
      zxr->setu256(zoffset,zx1,x1offset);
      //logInfoTid(tid,"R1=inf\n",tid);
      return;  
  }
  squarez(&tmp_x, &z2,         midx);  // tmp_x = z2sq 
  mulz(&tmp_z, &tmp_x, &x1, midx);  // tmp_z = u1 = x1 * z2sq
  mulz(&tmp_x, &tmp_x, &z2, midx);  // tmp_x = z2cube
  mulz(&tmp_x, &tmp_x, &y1, midx);  // tmp_x = s1 = z2cube * y1

  //  if U1 == U2 and S1 == S2 => P1 = P2 -> double
  //  if U1 == U2 and S1 != S2 => P1 = -P2 -> return 0
  //  instead of calling double,  i proceed. It is better to avoid warp divergence
  if (eqz(&x2, &tmp_z) &&   // u1 == u2
       !eqz( &y2, &tmp_x)){  // s1 != s2
          zxr->setu256(zoffset,&_inf,x1offset);
          //logInfoTid(tid,"R2=inf\n",tid);
	  return;  

  }

  subz(&tmp1, &x2, &tmp_z, midx);     // H = tmp1 = u2 - u1
  mulz(&zr, &z2, &tmp1, midx);       // zr = z1 * z2  * h
  squarez(&tmp3, &tmp1,        midx);     // Hsq = tmp3 = H * H 

  /*
  logInfoBigNumberTid(tid,ndbg,"H\n",&tmp1);
  logInfoBigNumberTid(tid,ndbg,"z2 * h\n",&zr);
  logInfoBigNumberTid(tid,ndbg,"Hsq\n",&tmp3);
  */

  mulz(&tmp1, &tmp3, &tmp1, midx);     // Hcube = tmp1= Hsq * H 
  mulz(&tmp3, &tmp_z, &tmp3, midx);     // tmp3 = u1 * Hsq

  /*
  logInfoBigNumberTid(tid,ndbg,"H3\n",&tmp1);
  logInfoBigNumberTid(tid,ndbg,"Hsq * u1\n",&tmp3);
  */

  subz(&tmp_z, &y2, &tmp_x, midx);        // R = tmp_z = S2 - S1 
  squarez(&xr, &tmp_z, midx);     // xr = Rsq

  /*
  logInfoBigNumberTid(tid,ndbg,"R\n",&tmp_z);
  logInfoBigNumberTid(tid,ndbg,"Rsq\n",&xr);
  */

  subz(&xr, &xr, &tmp1, midx);     // xr = Rsq - Hcube

  /*
  logInfoBigNumberTid(tid,ndbg,"Rsq - Hcube\n",&xr);
  */

  subz(&xr, &xr, &tmp3, midx);     // xr = Rsq - Hcube - u1*Hsq
  subz(&xr, &xr, &tmp3, midx);     // xr = Rsq - Hcube - 2*u1*Hsq

  /*
  logInfoBigNumberTid(tid,ndbg,"X\n",&xr);
  */

  subz(&yr, &tmp3, &xr, midx);          // yr = u1*Hsq - xr
  mulz(&yr, &yr, &tmp_z, midx);        //  yr = R*(u1*Hsq - xr)

  mulz(&tmp1, &tmp1, &tmp_x, midx);     // tmp1 = Hcube * s1
  subz(&yr, &yr, &tmp1, midx);

  /*
  logInfoBigNumberTid(tid,ndbg,"X : \n",&xr);
  logInfoBigNumberTid(tid,ndbg,"Y : \n",&yr);
  logInfoBigNumberTid(tid,ndbg,"Z : \n",&zr);
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
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  T1 y1(zx1->getu256(ECP_JAC_YOFFSET_BASE)), y2(zx2->getu256(ECP_JAC_YOFFSET_BASE)),
     xr(zxr->getu256(ECP_JAC_XOFFSET_BASE)),
     yr(zxr->getu256(ECP_JAC_YOFFSET_BASE)), zr(zxr->getu256(ECP_JAC_ZOFFSET_BASE));

  T1 _inf;
  uint32_t ndbg = T1::getN();

  infz(&_inf, midx);

 
  uint32_t __restrict__ ztmp[4*sizeof(T2)/sizeof(uint32_t)];
  T1 tmp1(ztmp), tmp2(&ztmp[sizeof(T2)/sizeof(uint32_t)]),
                 tmp3(&ztmp[2*sizeof(T2)/sizeof(uint32_t)]), 
                 tmp4(&ztmp[3*sizeof(T2)/sizeof(uint32_t)]);

  // TODO Check if I can call add to compute x + x (instead of double)
  //  if not, I should call double below. I don't want to to avoid warp divergnce
  if (eqz(zx1, zx2) &&   // u1 == u2
       !eqz( &y1,  &y2)){  // s1 != s2
          zxr->setu256(0,&_inf,0);
	  return;  //  if U1 == U2 and S1 == S2 => P1 == P2 (call double)
  }

  /*
  logInfoBigNumberTid(tid,ndbg,"x1\n",zx1);
  logInfoBigNumberTid(tid,ndbg,"y1\n",&y1);
  logInfoBigNumberTid(tid,ndbg,"x2\n",zx2);
  logInfoBigNumberTid(tid,ndbg,"y2\n",&y2);

  logInfoBigNumberTid(tid,1,"x22\n",zx2->get2u256());
  logInfoBigNumberTid(tid,1,"x12\n",zx1->get2u256());
  */

  subz(&zr, zx2, zx1, midx);     // H = tmp2 = u2 - u1
  
  /*
  logInfoBigNumberTid(tid,ndbg,"H\n",&zr);
  */

  squarez(&tmp3, &zr,        midx);     // Hsq = tmp3 = H * H 
  mulz(&tmp2, &tmp3, &zr, midx);     // Hcube = tmp2 = Hsq * H 
  mulz(&tmp1, zx1, &tmp3, midx);     // tmp1 = u1 * Hsq

  /*
  logInfoBigNumberTid(tid,ndbg,"Hsq\n",&tmp3);
  logInfoBigNumberTid(tid,ndbg,"Hcube\n",&tmp2);
  logInfoBigNumberTid(tid,ndbg,"u1 * Hsq\n",&tmp1);
  */

  subz(&tmp3, &y2, &y1, midx);        // R = tmp3 = S2 - S1 tmp1=u1*Hsq, tmp2=Hcube, xr=free, yr=s1, zr=zr
  mulz(&yr, &y1, &tmp2, midx);     // yr = Hcube * s1
  squarez(zxr, &tmp3, midx);     // xr = R * R

  /*
  logInfoBigNumberTid(tid,ndbg,"R\n",&tmp3);
  logInfoBigNumberTid(tid, ndbg,"s1\n",&yr);
  logInfoBigNumberTid(tid,ndbg,"Rsq\n",&xr);
  */
  subz(zxr, zxr, &tmp2, midx);        // xr = x3= (R*R)-Hcube, yr = Hcube * S1, zr=zr, tmp1=u1*Hsq, tmp2 = Hcube, tmp3 = R

  // TODO muluk256
  mul2z(&tmp4, &tmp1, midx);     // tmp4 = u1*hsq *_2

  /*
  logInfoBigNumberTid(tid,ndbg,"Rsq - Hcube\n",&xr);
  logInfoBigNumberTid(tid,ndbg,"u1 * Hsq * 2\n",&tmp4);
  */

  subz(zxr, &xr, &tmp4, midx);               // x3 = xr
  subz(&tmp1, &tmp1, zxr, midx);       // tmp1 = u1*hs1 - x3
  //logInfoBigNumberTid(tid,ndbg,"u1*hsq - x3\n",&tmp1);
  mulz(&tmp1, &tmp1, &tmp3, midx);  // tmp1 = r * (u1 * hsq - x3)
  //logInfoBigNumberTid(tid,ndbg,"r * (u1*hsq - x3)\n",&tmp1);
  subz(&yr, &tmp1, &yr, midx);

  /*
  logInfoBigNumberTid(tid,ndbg,"X3\n",&xr);
  logInfoBigNumberTid(tid,ndbg,"Y3\n",&yr);
  logInfoBigNumberTid(tid,ndbg,"Z3\n",&zr);
  */
}

/*
  EC point addition
  
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
__forceinline__ __device__ void doublecjac(T1 *zxr, T1 *zx1, mod_t midx)
{
  #if 0
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  T1 y1(zx1->getu256(1)), z1(zx1->getu256(2));
  T1 yr(zxr->getu256(1)), zr(zxr->getu256(2));
  T1 _inf;
  uint32_t ndbg = T1::getN();

  infz(&_inf,midx);


  uint32_t __restrict__ ztmp[5*sizeof(T2)/sizeof(uint32_t)];
  T1 tmp1(ztmp), tmp2(&ztmp[sizeof(T2)/sizeof(uint32_t)]),
                 tmp_x(&ztmp[2*sizeof(T2)/sizeof(uint32_t)]),
                 tmp_y(&ztmp[3*sizeof(T2)/sizeof(uint32_t)]),
                 tmp_z(&ztmp[4*sizeof(T2)/sizeof(uint32_t)]);

  // TODO : review this comparison, and see if I can do better. or where I should put it
  // as i check this in several places
  if (eq0z(&y1)){ 
      zxr->setu256(0,&_inf,0);
      //memcpy(xr, _inf, 3 * NWORDS_256BIT * sizeof(uint32_t));
      return;  
  }
  squarez(&tmp_z, &y1,            midx);  // tmp_z = ysq
  squarez(&tmp_y, &tmp_z, midx);  // tmp_y = ysqsq
  // TODO muluk256
  mul8z(&tmp_y, &tmp_y, midx);  // tmp_y = ysqsq *_8
  mulz(&tmp_z, &tmp_z, zx1, midx);  // S = tmp_z = x * ysq
  // TODO muluk256
  mul4z(&tmp_z, &tmp_z, midx);  // S = tmp_z = S * _4

  squarez(&tmp_x, zx1, midx);  // M1 = tmp_x = x * x
  // TODO muluk256
  mul3z(&tmp1, &tmp_x, midx);  // M = tmp1 = M1 * _3
  squarez(&tmp_x, &tmp1, midx);  // X3 = tmp_x = M * M,  tmp_y = Ysqsq * _8, tmp_z = S; tmp1 = M
  // TODO muluk256
  mul2z(&tmp2, &tmp_z, midx);   // tmp2 = S * _2
  subz(zxr, &tmp_x, &tmp2, midx);      // X3 = tmp_x; tmp_y = Ysqsq * _8, tmp_z = S, tmp1 = M, 
  subz(&tmp2, &tmp_z, zxr, midx);   //  tmp2 = S - X3
  mulz(&tmp2, &tmp2, &tmp1, midx); // tmp2 = M * (S - X3)
  mulz(&tmp_z, &y1, &z1, midx);
  // TODO muluk256
  mul2z(&zr, &tmp_z, midx);
  subz(&yr, &tmp2, &tmp_y, midx);

  /*
  logInfoBigNumberTid(tid,ndbg,"X : \n",zxr);
  logInfoBigNumberTid(tid,ndbg,"Y : \n",&yr);
  logInfoBigNumberTid(tid,ndbg,"Z : \n",&zr);
  */
  #else

 int tid = threadIdx.x + blockDim.x * blockIdx.x;

  T1 y1(zx1->getu256(1)), z1(zx1->getu256(2));
  T1 yr(zxr->getu256(1)), zr(zxr->getu256(2));
  T1 _inf;
  uint32_t ndbg = T1::getN();

  infz(&_inf,midx);


  uint32_t __restrict__ ztmp[2*sizeof(T2)/sizeof(uint32_t)];
  T1 tmp_y(ztmp), tmp_z(&ztmp[sizeof(T2)/sizeof(uint32_t)]);

  // TODO : review this comparison, and see if I can do better. or where I should put it
  // as i check this in several places
  if (eq0z(&y1)){ 
      zxr->setu256(0,&_inf,0);
      //memcpy(xr, _inf, 3 * NWORDS_256BIT * sizeof(uint32_t));
      return;  
  }
  squarez(&tmp_z, &y1,            midx);  // tmp_z = ysq
  squarez(&tmp_y, &tmp_z, midx);  // tmp_y = ysqsq

  addz(&tmp_y, &tmp_y, &tmp_y, midx);  // tmp_y = ysqsq + ysqsq
  addz(&tmp_y, &tmp_y, &tmp_y, midx);  // tmp_y = 2ysqsq + 2ysqsq
  addz(&tmp_y, &tmp_y, &tmp_y, midx);  // tmp_y = 4ysqsq + 4ysqsq

  mulz(&tmp_z, &tmp_z, zx1, midx);  
  addz(&tmp_z, &tmp_z, &tmp_z, midx);  
  addz(&tmp_z, &tmp_z, &tmp_z, midx);  // S = tmp_z = 2X1Ysq + 2X1Ysq

  mulz(&zr, &y1, &z1, midx);     //  Z3 = Y * Z
  addz(&zr, &zr, &zr, midx);

  squarez(&yr, zx1, midx);           
  addz(zxr, &yr, &yr, midx);       
  addz(&yr, zxr, &yr, midx);       // M = yr = 3Xsq

  squarez(zxr, &yr, midx);       // X3 = Msq

  subz(zxr, zxr, &tmp_z, midx);   // X3 = Msq - S
  subz(zxr, zxr, &tmp_z, midx);      // X3 = Msq - 2S

  subz(&tmp_z, &tmp_z, zxr, midx);   //  tmp_z = S - X3
  mulz(&yr, &yr, &tmp_z, midx);     //  Y3 = M * (S - X3)
  subz(&yr, &yr, &tmp_y, midx);    // Y3 = M * (S - X3) - 8ysqsq


  /*
  logInfoBigNumberTid(tid,ndbg,"X : \n",zxr);
  logInfoBigNumberTid(tid,ndbg,"Y : \n",&yr);
  logInfoBigNumberTid(tid,ndbg,"Z : \n",&zr);
  */
  #endif
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

  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  T1 _inf;
  uint32_t ndbg = T1::getN();

  infz(&_inf,midx);

  /*
  logInfoBigNumberTid(tid, ndbg,"x1\n",zx1->getu256());
  logInfoBigNumberTid(tid,ndbg,"y1\n",y1.getu256());
  */
  squarez(&zr, &y1, midx);  // zr = ysq
  squarez(&tmp_y, &zr, midx);  // yr = ysqsq

  /*
  logInfoBigNumberTid(tid,ndbg,"ysq\n",zr.getu256());
  logInfoBigNumberTid(tid,ndbg,"Yqsq\n",tmp_y.getu256());
  */
  // TODO muluk256
  mul8z(&tmp_y, &tmp_y, midx);  // tmp_y = ysqsq *_8
  mulz(&zr, &zr, zx1, midx);  // S = zr = x * ysq

  /*
  logInfoBigNumberTid(tid,ndbg,"8*Ysqsq\n",tmp_y.getu256());
  logInfoBigNumberTid(tid,ndbg,"S\n",zr.getu256());
  */
  // TODO muluk256
  mul4z(&zr, &zr, midx);  // S = zr = S * _4

  //logInfoBigNumberTid(tid,ndbg,"S*4\n",zr.getu256());

  squarez(zxr, zx1, midx);  // M1 = xr = x * x
  // TODO muluk256
  mul3z(&tmp1, zxr, midx);  // M = tmp1 = M1 * _3

  /*
  logInfoBigNumberTid(tid,ndbg,"Xsq\n",zxr->getu256());
  logInfoBigNumberTid(tid,ndbg,"M\n",tmp1.getu256());
  */
  squarez(zxr, &tmp1, midx);  // X3 = xr = M * M,  tmp_y = Ysqsq * _8, zr = S; tmp1 = M
  // TODO muluk256
  mul2z(&tmp2, &zr, midx);   // tmp2 = S * _2

  /* 
  logInfoBigNumberTid(tid,ndbg,"M*M\n",zxr->getu256());
  logInfoBigNumberTid(tid,ndbg,"S*2\n",tmp2.getu256());
  */

  subz(zxr, zxr, &tmp2, midx);      // X3 = xr; tmp_y = Ysqsq * _8, zr = S, tmp1 = M, 
  subz(&tmp2, &zr, zxr, midx);   //  tmp2 = S - X3

  /*
  logInfoBigNumberTid(tid,ndbg,"X3\n",zxr->getu256());
  logInfoBigNumberTid(tid,ndbg,"S-X3\n",tmp2.getu256());
  */

  mulz(&tmp2, &tmp2, &tmp1, midx); // tmp2 = M * (S - X3)
  //logInfoBigNumberTid(tid,ndbg,"M * (S-X3)\n",tmp2.getu256());
  // TODO muluk256
  mul2z(&zr, &y1, midx);
  subz(&yr, &tmp2, &tmp_y, midx);

  /*
  logInfoBigNumberTid(tid,ndbg,"y3\n",yr.getu256());
  logInfoBigNumberTid(tid,ndbg,"z3\n",zxr->getu256());
  */
}

template<typename T1, typename T2>
__forceinline__ __device__ void scmulecjac(T1 *zxr, uint32_t zoffset, T1 *zx1, uint32_t xoffset, uint32_t *scl, mod_t midx)
{
  uint32_t b0;
  uint32_t ndbg = T1::getN();
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t i, iter=0;

  uint32_t __restrict__ zN[3*sizeof(T2)/sizeof(uint32_t)]; // N = P
  uint32_t *_1 = misc_const_ct[midx]._1;
  T1 _inf;
  T1 N(zN);
  T1 Q(zxr->getu256(zoffset));
  T1 y1(zx1->getu256(xoffset+1));

  infz(&_inf, midx);

  // TODO : review this comparison
  if (eq0z(&y1)){ 
      zxr->setu256(zoffset,&_inf,0);
      return;  
  }

  //N.setu256(0,zx1,xoffset);
  N.setu256(0,zx1,xoffset,1);
  N.setu256(1,zx1,xoffset+1,1);
  setkz(&N,2,_1);


  Q.setu256(0,&_inf, 0);
  
  if (eq0u256(scl)) { return; }

  // TODO : Either implement left to right, or count where msb is and substitute while by unrolled
  // loop

  // TODO : MAD several numbers at once using shamir's trick

  logInfoBigNumberTid(tid,1,"SCL mul: \n",scl);
  logInfoBigNumberTid(tid,3*ndbg,"Q: \n",&Q);
  logInfoBigNumberTid(tid,3*ndbg,"N: \n",&N);

  #if 0
    uint32_t __restrict__ scl_cpy[NWORDS_256BIT];
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
    uint32_t msb = clzMu256(scl);
    logInfoTid(tid,"msb : %d \n",msb);
    //#pragma unroll
    for (i=msb>>U256_MBSCLUSTER; i< (1 << (NWORDS_256BIT - U256_MBSCLUSTER)); i++){
        offset = i<<U256_MBSCLUSTER;
        scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset,   midx);
        scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+1, midx);
        scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+2, midx);
        scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+3, midx);
        //scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+4, midx);
        //scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+5, midx);
        //scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+6, midx);
        //scmulecjac_step_l2r<T1, T2>(&Q,&N, scl, offset+7, midx);
     }
  #endif

  logInfoBigNumberTid(tid,3*ndbg,"R-N: \n",&N);
  logInfoBigNumberTid(tid,3*ndbg,"R-Q: \n",&Q);
  return;
}

template<typename T1, typename T2>
__device__ void scmulecjac_step_r2l(T1 *Q,T1 *N, uint32_t *scl, mod_t midx )
{
   uint32_t  b0 = shr1u256(scl);
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   logInfoTid(tid,"B0 : %d\n",b0);
   if (b0) {
      addecjac<T1, T2> (Q,0, N,0, Q,0, midx);
   }
   doublecjac<T1, T2>(N,N, midx);
}

template<typename T1, typename T2>
__device__ void scmulecjac_step_l2r(T1 *Q,T1 *N, uint32_t *scl, uint32_t offset, mod_t midx )
{
   uint32_t  b0 = bselMu256(scl,255-offset);
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   logInfoTid(tid,"B0 : %d\n",b0);
   doublecjac<T1, T2>(Q,Q, midx);
   if (b0) {
      addecjacmixed<T1, T2> (Q,0, N,0, Q,0, midx);
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
/////////
// Temporary implemenation of future functionality. Leave here for now...


#if 0
__forceinline__ __device__
 void addecjacaff(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, const uint32_t *x2, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const uint32_t __restrict__ *y1 = &x1[NWORDS_256BIT];
  const uint32_t __restrict__ *y2 = &x2[NWORDS_256BIT];
  uint32_t __restrict__ *yr = &xr[NWORDS_256BIT];
  uint32_t __restrict__ *zr = &xr[NWORDS_256BIT*2];
  uint32_t __restrict__ *_inf = misc_const_ct[midx]._inf;
  uint32_t __restrict__ *_2 = misc_const_ct[midx]._2;
 
  uint32_t __restrict__ tmp1[NWORDS_256BIT], tmp2[NWORDS_256BIT], tmp3[NWORDS_256BIT], tmp4[NWORDS_256BIT];

  // TODO Check if I can call add to compute x + x (instead of double)
  //  if not, I should call double below. I don't want to to avoid warp divergnce
  if (eqz((const uint32_t *)x1, (const uint32_t *)x2) &&   // u1 == u2
       !eqz( (const uint32_t *) y1, (const uint32_t *) y2)){  // s1 != s2
          memcpy(xr, _inf, 3 * NWORDS_256BIT * sizeof(uint32_t));
	  return;  //  if U1 == U2 and S1 == S2 => P1 == P2 (call double)
  }

  /*
  if (tid == 0){
     logInfoBigNumber("x1\n",(uint32_t *)x1);
     logInfoBigNumber("y1\n",(uint32_t *)y1);
     logInfoBigNumber("x2\n",(uint32_t *)x2);
     logInfoBigNumber("y2\n",(uint32_t *)y2);
  }
  */
  subz(zr, x2, x1, midx);     // H = tmp2 = u2 - u1
  if (tid == 0){
    logInfoBigNumber("H\n",(uint32_t *)zr);
  }

  squarez(tmp3, zr,        midx);     // Hsq = tmp3 = H * H 
  mulz(tmp2, tmp3, zr, midx);     // Hcube = tmp2 = Hsq * H 
  mulz(tmp1, x1, tmp3, midx);     // tmp1 = u1 * Hsq

  /*
  if (tid == 0){
    logInfoBigNumber("Hsq\n",(uint32_t *)tmp3);
    logInfoBigNumber("Hcube\n",(uint32_t *)tmp2);
    logInfoBigNumber("u1 * Hsq\n",(uint32_t *)tmp1);
  }
  */

  subz(tmp3, y2, y1, midx);        // R = tmp3 = S2 - S1 tmp1=u1*Hsq, tmp2=Hcube, xr=free, yr=s1, zr=zr
  mulz(yr, y1, tmp2, midx);     // yr = Hcube * s1
  squarez(xr, tmp3, midx);     // xr = R * R

  /*
  if (tid == 0){
    logInfoBigNumber("R\n",(uint32_t *)tmp3);
    logInfoBigNumber("s1\n",(uint32_t *)yr);
    logInfoBigNumber("Rsq\n",(uint32_t *)xr);
  }
  */
  subz(xr, xr, tmp2, midx);        // xr = x3= (R*R)-Hcube, yr = Hcube * S1, zr=zr, tmp1=u1*Hsq, tmp2 = Hcube, tmp3 = R

  // TODO muluk256
  mul2z(tmp4, tmp1, midx);     // tmp4 = u1*hsq *_2

  /*
  if (tid == 0){
    logInfoBigNumber("Rsq - Hcube\n",(uint32_t *)xr);
    logInfoBigNumber("u1 * Hsq * 2\n",(uint32_t *)tmp4);
  }
  */
  subz(xr, xr, tmp4, midx);               // x3 = xr
  subz(tmp1, tmp1, xr, midx);       // tmp1 = u1*hs1 - x3
  mulz(tmp1, tmp1, tmp3, midx);  // tmp1 = r * (u1 * hsq - x3)
  subz(yr, tmp1, yr, midx);

  /*
  if (tid == 0){
    logInfoBigNumber("X3\n",(uint32_t *)xr);
    logInfoBigNumber("u1 * hsq - x3\n",(uint32_t *)tmp1);
    logInfoBigNumber("Y3\n",(uint32_t *)yr);
  }
  */
}
#endif

#if 0
__global__ void addecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t __restrict__ *x1, *x2, *xr;
 
    if(tid >= params->in_length/6) {
      return;
    }

    x1 = (uint32_t *) &in_vector[tid * 2 * ECK_LDR_INOFFSET + ECP_LDR_INXOFFSET];
    x2 = (uint32_t *) &in_vector[(tid * 2 + 1) * ECK_LDR_INOFFSET + ECP_LDR_INXOFFSET];

    xr = (uint32_t *) &out_vector[tid * ECK_LDR_OUTOFFSET + ECP_LDR_OUTXOFFSET];
    
    addecldr(xr, x1, x2, x1, params->midx);

    return;
}
__global__ void doublecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t __restrict__ *x1,*xr;
 
    if(tid >= params->in_length/3) {
      return;
    }

    x1 = (uint32_t *) &in_vector[tid * ECK_LDR_INOFFSET + ECP_LDR_INXOFFSET];

    xr = (uint32_t *) &out_vector[tid * ECK_LDR_OUTOFFSET + ECP_LDR_OUTXOFFSET];
    
    doublecldr(xr, x1, params->midx);

  return;
}
__global__ void scmulecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *x1, *scl, *xr;
 
   if(tid >= params->in_length/3) {
     return;
   }

   x1  = (uint32_t *) &in_vector[tid * ECK_LDR_INOFFSET + ECP_LDR_INXOFFSET];
   scl = (uint32_t *) &in_vector[tid * ECK_LDR_INOFFSET + ECP_SCLOFFSET];

   xr = (uint32_t *) &out_vector[tid * ECK_LDR_OUTOFFSET + ECP_LDR_OUTXOFFSET];

   
   ldrstep(xr, x1, scl,  params->midx);

   return;
}

__global__ void madecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
  return;
}

    
__forceinline__ __device__
 void addecldr(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, 
            const uint32_t __restrict__ *x2, const uint32_t __restrict__ *xp, mod_t midx)
{
   // Xr = -4*b Z1 * Z2 * (X1 * Z2 + X2 * Z1) + (X1 * X2)^2 
   // Zr = xp * (X1 * Z2 - X2 * Z1)^2

   // 7 M, 2 SQ, 3 ADD
   const uint32_t __restrict__ *z1 = &x1[NWORDS_256BIT];
   const uint32_t __restrict__ *z2 = &x2[NWORDS_256BIT];
   uint32_t __restrict__ *zr =&zr[NWORDS_256BIT];

   uint32_t tmp1[NWORDS_256BIT];
   uint32_t __restrict__ *_4b = misc_const_ct[midx]._4b;
   

   mulz(tmp1, x2  , z1  , midx);      
   mulz(xr  , x1  , z2  , midx);      
   subz(   zr  , xr  , tmp1, midx);
   addz(   tmp1, tmp1, xr  , midx);
   mulz(tmp1, tmp1, z2  , midx);    
   mulz(tmp1, tmp1, z1  , midx);    
   mulz(xr  , x1  , x2  , midx);      
   squarez(xr  , xr         , midx);    
   // multiply by 12. 
   //  Using Montgomery: 136 mul + 346 add.
   //  Chaining 12 additions : 0 mul + 84 adds + modulus!!!
   // TODO : Use muluk256 function
   mulkz(tmp1, tmp1,_4b  , midx);  
   subz(   xr,   tmp1, xr  , midx);
   squarez(zr,   zr         , midx);     
   mulz(zr,   zr  , xp  , midx);   

  return;
}

__forceinline__ __device__
 void doublecldr(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, mod_t midx)
{
  // Xr = X1^4 - 8*b * X1*Z1^3
  // Zr = 4*Z1 * (X1^3 + b*Z1^3) 

  // 7 M, 3 SQ, 2 Add
  const uint32_t __restrict__ *z1 = &x1[NWORDS_256BIT];
  uint32_t __restrict__ *zr =&zr[NWORDS_256BIT];

  uint32_t tmp1[NWORDS_256BIT], tmp2[NWORDS_256BIT];
  uint32_t __restrict__ *_4 = misc_const_ct[midx]._4;
  uint32_t __restrict__ *b = ecbn128_params_ct[midx].b;
  uint32_t __restrict__ *_8b = misc_const_ct[midx]._8b;

  squarez(xr,  z1,           midx);    
  mulz(zr,  xr,   z1,   midx);      // Zr = Z1^3
  mulz(xr,  zr,   x1,   midx);      
  // TODO muluk256
  mulkz(xr,  xr,  _8b,   midx);      // Xr = 8b * X1 * Z1^3
  squarez(tmp1, x1,         midx);      
  squarez(tmp2, tmp1,       midx);    
  subz(   xr,  tmp2, xr,   midx);

  // TODO muluk256
  mulkz(zr,  zr,   b,    midx);      // Zr = b*Z1^3
  mulz(tmp1, tmp1,  x1,   midx);     
  addz(   zr, tmp1,   zr,   midx);
  // TODO muluk256
  mulkz(zr, zr,   _4,    midx);
  mulz(zr, zr,   z1,    midx); 

  return;
}

// NOTE : EC points are in affine coordinates => Pz = 1 (in montgomery someting else)
// NOTE : EC points in montgomery, scl normal 
__forceinline__ __device__
 void ldrstep(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, uint32_t *scl, mod_t midx)
{
  uint32_t b0, idxn, idxp;
  uint32_t __restrict__ *_1 = misc_const_ct[midx]._1;

  uint32_t __restrict__ R[4*NWORDS_256BIT];

  //R[] = {[1,0],[X,1]} 
  memcpy(R, _1, NWORDS_256BIT * sizeof(uint32_t));
  memcpy(&R[3 * NWORDS_256BIT], _1, NWORDS_256BIT * sizeof(uint32_t));
  memcpy(&R[2 * NWORDS_256BIT], x1, NWORDS_256BIT * sizeof(uint32_t));

  while (!eq0u256(scl)){
     b0 = shr1u256(scl);
     idxn = ~b0 * 2 * NWORDS_256BIT;
     idxp =  b0 * 2 * NWORDS_256BIT;
     /*
     if (b0) { R0 = R0 + R1; R1 = R1 + R1;
     else {    R1 = R0 + R1; R0 = R0 + R0}
     */
     addecldr(&R[idxn], &R[idxn], &R[idxp], x1, midx);
     doublecldr(&R[idxp], &R[idxp], midx);
  }
   // TODO
   // Retrieve y(P) . Not sure if i need to convert to affine now. If I don't,
   // then I have three coordinates and it doesn't fit in my allocated space
   //
   // P = (x1, y1) , Q = (x2, y2), P-Q = (x,y)
   // Q = k P => x(R0) = X(Q), x(R1) = X(P-Q)
   //
   // y(P) = y1 = [2b + (a + x * x1) * (x + x1) - x2(x - x1) ^ 2] / (2*y)

  memcpy(xr,R,2 * NWORDS_256BIT * sizeof(uint32_t));

  return;
}

#endif


