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
#include "ecbn128_device.h"

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


/* 
  in_vector : k[0], px[0], py[0], k[1], px[1], py[1],...  Input EC points in Affine coordinates
  out vecto : px[0], py[0], pz[0], px[1], py[1],pz[1],...              Output EC points in Jacobian coordinates
*/
__global__ void addecjac_kernel(uint32_t   *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t __restrict__ *x1, *x2, *xr;
 
    if(tid >= params->in_length/6) {
      return;
    }

    // x1 points to inPx[i]. x2 points to inPx[i+1]. xr points to outPx[i]
    x1 = (uint32_t *) &in_vector[tid * 2 * ECK_JAC_INOFFSET + ECP_JAC_INXOFFSET];
    x2 = (uint32_t *) &in_vector[(tid * 2 + 1) * ECK_JAC_INOFFSET + ECP_JAC_INXOFFSET];
    xr = (uint32_t *) &out_vector[tid * ECK_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET];
    
    addecjacaff(xr, x1, x2, params->midx);

    return;

}

__global__ void doublecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t __restrict__ *x1,*xr;
 
    // x1 points to inPx[i].  xr points to outPx[i]
    if(tid >= params->in_length/3) {
      return;
    }

    x1 = (uint32_t *) &in_vector[tid * ECK_JAC_INOFFSET + ECP_JAC_INXOFFSET];
    xr = (uint32_t *) &out_vector[tid * ECK_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET];
    
    doublecjacaff(xr, x1, params->midx);

    return;
}

__global__ void scmulecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *x1, *scl, *xr;
 
   if(tid >= params->in_length/3) {
     return;
   }

   x1  = (uint32_t *) &in_vector[tid * ECK_JAC_INOFFSET + ECP_JAC_INXOFFSET];
   scl = (uint32_t *) &in_vector[tid * ECK_JAC_INOFFSET + ECP_SCLOFFSET];

   xr = (uint32_t *) &out_vector[tid * ECK_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET];
  
   scmulecjac(xr, x1, scl,  params->midx);

   return;
}

__global__ void madecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    uint32_t i;

    uint32_t debug_idx = 0;

    extern __shared__ uint32_t smem[];
    uint32_t *smem_ptr = &smem[tid*NWORDS_256BIT*ECK_JAC_OUTDIMS];  // 0 .. blockDim

    uint32_t __restrict__ *xo, *xr;
   
    if(idx >= params->in_length/params->stride) {
      return;
    }

    xo = (uint32_t *) &in_vector[idx  * params->stride * ECK_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET]; // 0 .. N-1

    xr = (uint32_t *) &out_vector[blockIdx.x * ECK_JAC_OUTOFFSET + ECP_JAC_OUTXOFFSET];  // 
    //memset(smem, 0, blockDim.x * NWORDS_256BIT*ECK_JAC_OUTOFFSET * sizeof(uint32_t));
    
    /*
    if (idx == debug_idx){
       logDebugBigNumber("smem[0]\n",smem_ptr);
      for (i =0; i < params->stride; i++){
       logDebugBigNumber("X[0]\n",&x[i * U256K_OFFSET]);
      }
    }
    */
    // scalar multipliation
    if (params->premul){
        uint32_t __restrict__ *xi, *scl;
        xi = (uint32_t *) &in_vector[idx  * params->stride * ECK_JAC_INOFFSET + ECP_JAC_INXOFFSET]; // 0 .. N-1
        scl = (uint32_t *) &in_vector[idx * params->stride * ECK_JAC_INOFFSET + ECP_SCLOFFSET];
        #pragma unroll
        for (i =0; i < params->stride; i++){
          if (idx == 0){
             logInfoBigNumber("scl :\n",&scl[i*ECK_JAC_INOFFSET]);
             logInfoBigNumber("Xin[x]:\n",&xi[i*ECK_JAC_INOFFSET]);
             logInfoBigNumber("Xin[y]:\n",&xi[i*ECK_JAC_INOFFSET + NWORDS_256BIT]);
          }
          scmulecjac(&xo[i*ECK_JAC_OUTOFFSET], &xi[i*ECK_JAC_INOFFSET], &scl[i*ECK_JAC_INOFFSET],  params->midx);
          if (idx == 0){
             logInfoBigNumber("scl :\n",&scl[i*ECK_JAC_INOFFSET]);
             logInfoBigNumber("Xout[x]:\n",&xo[i*ECK_JAC_OUTOFFSET]);
             logInfoBigNumber("Xout[y]:\n",&xo[i*ECK_JAC_OUTOFFSET + NWORDS_256BIT]);
             logInfoBigNumber("Xout[z]:\n",&xo[i*ECK_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
          }
        }
    }
    
    addecjac(smem_ptr, (const uint32_t *)xo, (const uint32_t *)&xo[ECK_JAC_OUTOFFSET], params->midx);

    if (idx == 0){
       logInfoBigNumber("smem[X]\n",smem_ptr);
       logInfoBigNumber("smem[Y]\n",&smem_ptr[NWORDS_256BIT]);
       logInfoBigNumber("smem[Z]\n",&smem_ptr[2*NWORDS_256BIT]);
    }

    #pragma unroll
    for (i =0; i < params->stride-2; i++){
      addecjac(smem_ptr, (const uint32_t *)smem_ptr, (const uint32_t *)&xo[(i+2)*ECK_JAC_OUTOFFSET], params->midx);
      if (idx == 0){  
       logInfoBigNumber("smem[X]\n",smem_ptr);
       logInfoBigNumber("smem[Y]\n",&smem_ptr[NWORDS_256BIT]);
       logInfoBigNumber("smem[Z]\n",&smem_ptr[2*NWORDS_256BIT]);
      }
    }
    __syncthreads();

    /*
    if (idx == debug_idx){
       logDebugBigNumber("smem[i]\n",smem_ptr);
    }
    */
    // reduction global mem
    if (blockDim.x >= 1024 && tid < 512){
      /*
      if (idx == debug_idx){
        logDebugBigNumber("+smem[0]\n",smem_ptr);
        logDebugBigNumber("+smem[512]\n",&smem[(tid+512)*NWORDS_256BIT]);
      }
      */
      addecjac(smem_ptr, 
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+512)*ECK_JAC_INOFFSET], params->midx);
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
      addecjac(smem_ptr, 
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+256)*ECK_JAC_INOFFSET], params->midx);
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
      addecjac(smem_ptr, 
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+128)*ECK_JAC_INOFFSET], params->midx);
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
      addecjac(smem_ptr, 
               (const uint32_t *)smem_ptr,
               (const uint32_t *)&smem[(tid+64)*ECK_JAC_INOFFSET], params->midx);
      /*
      if (idx == debug_idx){
        logDebugBigNumber("smem[=64+0]\n",smem_ptr);
      }
      */
    }
    __syncthreads();
      
    // unrolling warp
    if (tid < 32)
    {
        volatile uint32_t *vsmem = smem;
        /*
        if (idx == debug_idx){
          logDebugBigNumber("+smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("+smem[32]\n",&smem[(tid+32)*NWORDS_256BIT]);
        }
        */
      if (idx == 0){  
       logInfoBigNumber("smem[pre32X]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET]);
       logInfoBigNumber("smem[pre32Y]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET+NWORDS_256BIT]);
       logInfoBigNumber("smem[pre32Z]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
      }
        addecjac((uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[(tid+32)*ECK_JAC_OUTOFFSET], params->midx);
      if (idx == 0){  
       logInfoBigNumber("smem[32X]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET]);
       logInfoBigNumber("smem[32Y]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET+NWORDS_256BIT]);
       logInfoBigNumber("smem[32Z]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
      }
        /*
        if (idx == debug_idx){
        logDebugBigNumber("smem[=32+0]\n",(uint32_t *)vsmem);
        }
       
        if (idx == debug_idx){
          logDebugBigNumber("+smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("+smem[16]\n",&smem[(tid+16)*NWORDS_256BIT]);
        }
        */
      if (idx == 0){  
       logInfoBigNumber("smem[pre16X]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET]);
       logInfoBigNumber("smem[pre16Y]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET+NWORDS_256BIT]);
       logInfoBigNumber("smem[pre16Z]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
      }
        addecjac((uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[(tid+16)*ECK_JAC_OUTOFFSET], params->midx);
      if (idx == 0){  
       logInfoBigNumber("smem[16X]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET]);
       logInfoBigNumber("smem[16Y]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET+NWORDS_256BIT]);
       logInfoBigNumber("smem[16Z]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
      }
        /*
        if (idx == debug_idx){
        logDebugBigNumber("smem[=16+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("+smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("+smem[8]\n",&smem[(tid+8)*NWORDS_256BIT]);
        }
        */
        addecjac((uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[(tid+8)*ECK_JAC_OUTOFFSET], params->midx);
      if (idx == 0){  
       logInfoBigNumber("smem[8X]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET]);
       logInfoBigNumber("smem[8Y]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET+NWORDS_256BIT]);
       logInfoBigNumber("smem[8Z]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
      }
        /*
        if (idx == debug_idx){
          logDebugBigNumber("smem[=8+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("smem[4]\n",&smem[(tid+4)*NWORDS_256BIT]);
        }
        */
        addecjac((uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[(tid+4)*ECK_JAC_OUTOFFSET], params->midx);
      if (idx == 0){  
       logInfoBigNumber("smem[4X]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET]);
       logInfoBigNumber("smem[4Y]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET+NWORDS_256BIT]);
       logInfoBigNumber("smem[4Z]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
      }
        /*
        if (idx == debug_idx){
          logDebugBigNumber("smem[=4+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("smem[2]\n",&smem[(tid+2)*NWORDS_256BIT]);
        }
        */
        addecjac((uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[(tid+2)*ECK_JAC_OUTOFFSET], params->midx);
      if (idx == 0){  
       logInfoBigNumber("smem[2X]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET]);
       logInfoBigNumber("smem[2Y]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET+NWORDS_256BIT]);
       logInfoBigNumber("smem[2Z]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
      }
        /*
        if (idx == debug_idx){
          logDebugBigNumber("smem[=2+0]\n",(uint32_t *)vsmem);
        }
        if (idx == debug_idx){
          logDebugBigNumber("smem[0]\n",(uint32_t *)vsmem);
        logDebugBigNumber("smem[1]\n",&smem[(tid+1)*NWORDS_256BIT]);
        }
        */
        addecjac((uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET],
                 (const uint32_t *)&vsmem[(tid+1)*ECK_JAC_OUTOFFSET], params->midx);
      if (idx == 0){  
       logInfoBigNumber("smem[X]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET]);
       logInfoBigNumber("smem[Y]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET+NWORDS_256BIT]);
       logInfoBigNumber("smem[Z]\n",(uint32_t *)&vsmem[tid * ECK_JAC_OUTOFFSET + 2*NWORDS_256BIT]);
      }
        /*
        if (idx == debug_idx){
          logDebugBigNumber("smem[=0+1]\n",(uint32_t *)vsmem);
        }
        */

        if (tid==0) {
           memcpy(xr, smem_ptr, sizeof(uint32_t) * NWORDS_256BIT * ECK_JAC_OUTDIMS);
        }
    }

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
   

   mulmontu256(tmp1, x2  , z1  , midx);      
   mulmontu256(xr  , x1  , z2  , midx);      
   submu256(   zr  , xr  , tmp1, midx);
   addmu256(   tmp1, tmp1, xr  , midx);
   mulmontu256(tmp1, tmp1, z2  , midx);    
   mulmontu256(tmp1, tmp1, z1  , midx);    
   mulmontu256(xr  , x1  , x2  , midx);      
   sqmontu256(xr  , xr         , midx);    
   // multiply by 12. 
   //  Using Montgomery: 136 mul + 346 add.
   //  Chaining 12 additions : 0 mul + 84 adds + modulus!!!
   // TODO : Use muluk256 function
   mulmontu256(tmp1, tmp1,_4b  , midx);  
   submu256(   xr,   tmp1, xr  , midx);
   sqmontu256(zr,   zr         , midx);     
   mulmontu256(zr,   zr  , xp  , midx);   

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

  sqmontu256(xr,  z1,           midx);    
  mulmontu256(zr,  xr,   z1,   midx);      // Zr = Z1^3
  mulmontu256(xr,  zr,   x1,   midx);      
  // TODO muluk256
  mulmontu256(xr,  xr,  _8b,   midx);      // Xr = 8b * X1 * Z1^3
  sqmontu256(tmp1, x1,         midx);      
  sqmontu256(tmp2, tmp1,       midx);    
  submu256(   xr,  tmp2, xr,   midx);

  // TODO muluk256
  mulmontu256(zr,  zr,   b,    midx);      // Zr = b*Z1^3
  mulmontu256(tmp1, tmp1,  x1,   midx);     
  addmu256(   zr, tmp1,   zr,   midx);
  // TODO muluk256
  mulmontu256(zr, zr,   _4,    midx);
  mulmontu256(zr, zr,   z1,    midx); 

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
__forceinline__ __device__
 void addecjac(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, const uint32_t *x2, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const uint32_t __restrict__ *y1 = &x1[NWORDS_256BIT];
  const uint32_t __restrict__ *z1 = &x1[NWORDS_256BIT*2];
  const uint32_t __restrict__ *y2 = &x2[NWORDS_256BIT];
  const uint32_t __restrict__ *z2 = &x2[NWORDS_256BIT*2];
  uint32_t __restrict__ *yr = &xr[NWORDS_256BIT];
  uint32_t __restrict__ *zr = &xr[NWORDS_256BIT*2];
  uint32_t __restrict__ *_inf = misc_const_ct[midx]._inf;
  uint32_t __restrict__ *_2 = misc_const_ct[midx]._2;
 
  uint32_t __restrict__ tmp1[NWORDS_256BIT], tmp2[NWORDS_256BIT], tmp3[NWORDS_256BIT], tmp4[NWORDS_256BIT];
  uint32_t __restrict__ tmp_x[NWORDS_256BIT], tmp_y[NWORDS_256BIT], tmp_z[NWORDS_256BIT];

  // TODO : Change definition of inf to 0, 1, 0 instead of 1,0,1 as it is now
  /*
  if (tid == 0){
     logInfoBigNumber("x1\n",(uint32_t *)x1);
     logInfoBigNumber("y1\n",(uint32_t *)y1);
     logInfoBigNumber("z1\n",(uint32_t *)z1);
     logInfoBigNumber("x2\n",(uint32_t *)x2);
     logInfoBigNumber("y2\n",(uint32_t *)y2);
     logInfoBigNumber("z2\n",(uint32_t *)z2);
  }
  */

  if (eq0u256(y2)){ 
      memcpy(xr, x1, 3 * NWORDS_256BIT * sizeof(uint32_t));
      return;  
  }
  sqmontu256(tmp_x, z1,         midx);  // tmp_x = z1sq 
  mulmontu256(tmp_z, tmp_x, x2, midx);  // tmp_z = u2 = x2 * z1sq
  mulmontu256(tmp_x, tmp_x, z1, midx);  // tmp_x = z1cube
  mulmontu256(tmp_x, tmp_x, y2, midx);  // tmp_x = s2 = z1cube * y2
  sqmontu256(tmp_y, z2,        midx);  // tmp_y = z2sq
  mulmontu256(tmp1, x1, tmp_y, midx);  // tmp1 = u1 = x1 * z2sq
  mulmontu256(tmp_y, tmp_y, z2, midx);  // tmp_y = z2cube
  mulmontu256(tmp_y, tmp_y, y1, midx);  // tmp_y = s1 = z2cube * y1

  //  if U1 == U2 and S1 == S2 => P1 = P2 -> double
  //  if U1 == U2 and S1 != S2 => P1 = -P2 -> return 0
  //  instead of calling double,  i proceed. It is better to avoid warp divergence
  if (equ256((const uint32_t *)tmp1, (const uint32_t *)tmp_z) &&   // u1 == u2
       !equ256( (const uint32_t *) tmp_y, (const uint32_t *) tmp_x)){  // s1 != s2
          memcpy(xr, _inf, 3 * NWORDS_256BIT * sizeof(uint32_t));
	  return;  

  }

  submu256(tmp2, tmp_z, tmp1, midx);     // H = tmp2 = u2 - u1
  mulmontu256(tmp_z, z1, z2, midx);      // tmp_z = z1 * z2
  mulmontu256(zr, tmp_z, tmp2, midx);       // zr = z1 * z2  * h
  
  /*
  if (tid == 0){
     logInfoBigNumber("H\n",(uint32_t *)tmp2);
     logInfoBigNumber("z1 * z2\n",(uint32_t *)tmp_z);
     logInfoBigNumber("z1 * z2  * h\n",(uint32_t *)zr);
  }
  */
  sqmontu256(tmp3, tmp2,        midx);     // Hsq = tmp3 = H * H 
  mulmontu256(tmp2, tmp3, tmp2, midx);     // Hcube = tmp2 = Hsq * H 
  mulmontu256(tmp1, tmp1, tmp3, midx);     // tmp1 = u1 * Hsq

  /*
  if (tid == 0){
     logInfoBigNumber("Hsq\n",(uint32_t *)tmp3);
     logInfoBigNumber("H3\n",(uint32_t *)tmp2);
     logInfoBigNumber("Hsq * u1\n",(uint32_t *)tmp1);
  }
  */
  submu256(tmp3, tmp_x, tmp_y, midx);        // R = tmp3 = S2 - S1 tmp1=u1*Hsq, tmp2=Hcube, tmp_x=free, tmp_y=s1, zr=zr
  mulmontu256(tmp_y, tmp_y, tmp2, midx);     // tmp_y = Hcube * s1
  sqmontu256(tmp_x, tmp3, midx);     // tmp_x = R * R

  /*
  if (tid == 0){
     logInfoBigNumber("R\n",(uint32_t *)tmp3);
     logInfoBigNumber("Hcube* s1\n",(uint32_t *)tmp_y);
     logInfoBigNumber("Rsq * u1\n",(uint32_t *)tmp_x);
  }
  */
  submu256(tmp_x, tmp_x, tmp2, midx);        // tmp_x = x3= (R*R)-Hcube, tmp_y = Hcube * S1, zr=zr, tmp1=u1*Hsq, tmp2 = Hcube, tmp3 = R

  // TODO muluk256
  mulmontu256(tmp4, tmp1, _2, midx);     // tmp4 = u1*hsq *_2

  /*
  if (tid == 0){
     logInfoBigNumber("Rsq - H3\n",(uint32_t *)tmp_x);
     logInfoBigNumber("Hsq * 2 * u1\n",(uint32_t *)tmp4);
  }
  */
  submu256(xr, tmp_x, tmp4, midx);               // x3 = xr
  submu256(tmp1, tmp1, xr, midx);       // tmp1 = u1*hs1 - x3
  mulmontu256(tmp1, tmp1, tmp3, midx);  // tmp1 = r * (u1 * hsq - x3)
  submu256(yr, tmp1, tmp_y, midx);
  /*
  if (tid == 0){
    logInfoBigNumber("X : \n",xr);
    logInfoBigNumber("Y : \n",yr);
    logInfoBigNumber("Z : \n",zr);
  }
  */
}
/*
  input is in affine coordinates -> P(Z) = 1
  I can do Q = Q+Y or Q = Y + Q
    NOTE X1, X2 cannot be 0
*/
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
  if (equ256((const uint32_t *)x1, (const uint32_t *)x2) &&   // u1 == u2
       !equ256( (const uint32_t *) y1, (const uint32_t *) y2)){  // s1 != s2
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
  submu256(zr, x2, x1, midx);     // H = tmp2 = u2 - u1
  if (tid == 0){
    logInfoBigNumber("H\n",(uint32_t *)zr);
  }

  sqmontu256(tmp3, zr,        midx);     // Hsq = tmp3 = H * H 
  mulmontu256(tmp2, tmp3, zr, midx);     // Hcube = tmp2 = Hsq * H 
  mulmontu256(tmp1, x1, tmp3, midx);     // tmp1 = u1 * Hsq

  /*
  if (tid == 0){
    logInfoBigNumber("Hsq\n",(uint32_t *)tmp3);
    logInfoBigNumber("Hcube\n",(uint32_t *)tmp2);
    logInfoBigNumber("u1 * Hsq\n",(uint32_t *)tmp1);
  }
  */

  submu256(tmp3, y2, y1, midx);        // R = tmp3 = S2 - S1 tmp1=u1*Hsq, tmp2=Hcube, xr=free, yr=s1, zr=zr
  mulmontu256(yr, y1, tmp2, midx);     // yr = Hcube * s1
  sqmontu256(xr, tmp3, midx);     // xr = R * R

  /*
  if (tid == 0){
    logInfoBigNumber("R\n",(uint32_t *)tmp3);
    logInfoBigNumber("s1\n",(uint32_t *)yr);
    logInfoBigNumber("Rsq\n",(uint32_t *)xr);
  }
  */
  submu256(xr, xr, tmp2, midx);        // xr = x3= (R*R)-Hcube, yr = Hcube * S1, zr=zr, tmp1=u1*Hsq, tmp2 = Hcube, tmp3 = R

  // TODO muluk256
  mulmontu256(tmp4, tmp1, _2, midx);     // tmp4 = u1*hsq *_2

  /*
  if (tid == 0){
    logInfoBigNumber("Rsq - Hcube\n",(uint32_t *)xr);
    logInfoBigNumber("u1 * Hsq * 2\n",(uint32_t *)tmp4);
  }
  */
  submu256(xr, xr, tmp4, midx);               // x3 = xr
  submu256(tmp1, tmp1, xr, midx);       // tmp1 = u1*hs1 - x3
  mulmontu256(tmp1, tmp1, tmp3, midx);  // tmp1 = r * (u1 * hsq - x3)
  submu256(yr, tmp1, yr, midx);

  /*
  if (tid == 0){
    logInfoBigNumber("X3\n",(uint32_t *)xr);
    logInfoBigNumber("u1 * hsq - x3\n",(uint32_t *)tmp1);
    logInfoBigNumber("Y3\n",(uint32_t *)yr);
  }
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
__forceinline__ __device__
 void doublecjac(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const uint32_t __restrict__ *y1 = &x1[NWORDS_256BIT];
  const uint32_t __restrict__ *z1 = &x1[NWORDS_256BIT*2];
  uint32_t __restrict__ *yr = &xr[NWORDS_256BIT];
  uint32_t __restrict__ *zr = &xr[NWORDS_256BIT*2];
  uint32_t __restrict__ *_inf = misc_const_ct[midx]._inf;
  uint32_t __restrict__ *_8 = misc_const_ct[midx]._8;
  uint32_t __restrict__ *_4 = misc_const_ct[midx]._4;
  uint32_t __restrict__ *_3 = misc_const_ct[midx]._3;
  uint32_t __restrict__ *_2 = misc_const_ct[midx]._2;
 
  uint32_t __restrict__ tmp1[NWORDS_256BIT], tmp2[NWORDS_256BIT];
  uint32_t __restrict__ tmp_x[NWORDS_256BIT], tmp_y[NWORDS_256BIT], tmp_z[NWORDS_256BIT];

  // TODO : review this comparison, and see if I can do better. or where I should put it
  // as i check this in several places
  if (eq0u256(y1)){ 
      memcpy(xr, _inf, 3 * NWORDS_256BIT * sizeof(uint32_t));
      return;  
  }
  sqmontu256(tmp_z, y1,            midx);  // tmp_z = ysq
  sqmontu256(tmp_y, tmp_z, midx);  // tmp_y = ysqsq
  // TODO muluk256
  mulmontu256(tmp_y, tmp_y, _8, midx);  // tmp_y = ysqsq *_8
  mulmontu256(tmp_z, tmp_z, x1, midx);  // S = tmp_z = x * ysq
  // TODO muluk256
  mulmontu256(tmp_z, tmp_z, _4, midx);  // S = tmp_z = S * _4

  sqmontu256(tmp_x, x1, midx);  // M1 = tmp_x = x * x
  // TODO muluk256
  mulmontu256(tmp1, tmp_x, _3, midx);  // M = tmp1 = M1 * _3
  sqmontu256(tmp_x, tmp1, midx);  // X3 = tmp_x = M * M,  tmp_y = Ysqsq * _8, tmp_z = S; tmp1 = M
  // TODO muluk256
  mulmontu256(tmp2, tmp_z, _2, midx);   // tmp2 = S * _2
  submu256(xr, tmp_x, tmp2, midx);      // X3 = tmp_x; tmp_y = Ysqsq * _8, tmp_z = S, tmp1 = M, 
  submu256(tmp2, tmp_z, xr, midx);   //  tmp2 = S - X3
  mulmontu256(tmp2, tmp2, tmp1, midx); // tmp2 = M * (S - X3)
  mulmontu256(tmp_z, y1, z1, midx);
  // TODO muluk256
  mulmontu256(zr, tmp_z, _2, midx);
  submu256(yr, tmp2, tmp_y, midx);

  /*
  if (tid == 0){
    logInfoBigNumber("X : \n",xr);
    logInfoBigNumber("Y : \n",yr);
    logInfoBigNumber("Z : \n",zr);
  }
  */
}

/* 
   X1 cannot be 0
 */
__forceinline__ __device__
 void doublecjacaff(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, mod_t midx)
{
  const uint32_t __restrict__ *y1 = &x1[NWORDS_256BIT];
  uint32_t __restrict__ *yr = &xr[NWORDS_256BIT];
  uint32_t __restrict__ *zr = &xr[NWORDS_256BIT*2];
  uint32_t __restrict__ *_inf = misc_const_ct[midx]._inf;
  uint32_t __restrict__ *_8 = misc_const_ct[midx]._8;
  uint32_t __restrict__ *_4 = misc_const_ct[midx]._4;
  uint32_t __restrict__ *_3 = misc_const_ct[midx]._3;
  uint32_t __restrict__ *_2 = misc_const_ct[midx]._2;
 
  uint32_t __restrict__ tmp1[NWORDS_256BIT], tmp2[NWORDS_256BIT];
  uint32_t __restrict__ tmp_y[NWORDS_256BIT];

  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  /*
  if (tid == 0){
     logInfoBigNumber("x1\n",(uint32_t *)x1);
     logInfoBigNumber("y1\n",(uint32_t *)y1);
  }
  */
  sqmontu256(zr, y1, midx);  // zr = ysq
  sqmontu256(tmp_y, zr, midx);  // yr = ysqsq

  /*
  if (tid == 0){
    logInfoBigNumber("ysq\n",(uint32_t *)zr);
    logInfoBigNumber("Yqsq\n",(uint32_t *)tmp_y);
  }
  */
  // TODO muluk256
  mulmontu256(tmp_y, tmp_y, _8, midx);  // tmp_y = ysqsq *_8
  mulmontu256(zr, zr, x1, midx);  // S = zr = x * ysq

  /*
  if (tid == 0){
    logInfoBigNumber("8*Ysqsq\n",(uint32_t *)tmp_y);
    logInfoBigNumber("S\n",(uint32_t *)zr);
  }
  */
  // TODO muluk256
  mulmontu256(zr, zr, _4, midx);  // S = zr = S * _4

  /*
  if (tid == 0){
    logInfoBigNumber("S*4\n",(uint32_t *)zr);
  }
  */

  sqmontu256(xr, x1, midx);  // M1 = xr = x * x
  // TODO muluk256
  mulmontu256(tmp1, xr, _3, midx);  // M = tmp1 = M1 * _3

  /*
  if (tid == 0){
    logInfoBigNumber("Xsq\n",(uint32_t *)xr);
    logInfoBigNumber("M\n",(uint32_t *)tmp1);
  }
  */
  sqmontu256(xr, tmp1, midx);  // X3 = xr = M * M,  tmp_y = Ysqsq * _8, zr = S; tmp1 = M
  // TODO muluk256
  mulmontu256(tmp2, zr, _2, midx);   // tmp2 = S * _2
  
  /*
  if (tid == 0){
    logInfoBigNumber("M*M\n",(uint32_t *)xr);
    logInfoBigNumber("S*2\n",(uint32_t *)tmp2);
  }
  */

  submu256(xr, xr, tmp2, midx);      // X3 = xr; tmp_y = Ysqsq * _8, zr = S, tmp1 = M, 
  submu256(tmp2, zr, xr, midx);   //  tmp2 = S - X3
  /*
  if (tid == 0){
    logInfoBigNumber("X3\n",(uint32_t *)xr);
    logInfoBigNumber("S-X3\n",(uint32_t *)tmp2);
  }
  */
  mulmontu256(tmp2, tmp2, tmp1, midx); // tmp2 = M * (S - X3)
  /*
  if (tid == 0){
    logInfoBigNumber("M * (S-X3)\n",(uint32_t *)tmp2);
  }
  */
  // TODO muluk256
  mulmontu256(zr, y1, _2, midx);
  submu256(yr, tmp2, tmp_y, midx);
  /*
  if (tid == 0){
    logInfoBigNumber("y3\n",(uint32_t *)yr);
    logInfoBigNumber("z3\n",(uint32_t *)zr);
  }
  */

}

__forceinline__ __device__
 void scmulecjac(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, uint32_t *scl, mod_t midx)
{
  uint32_t b0;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  uint32_t __restrict__ N[3*NWORDS_256BIT]; // N = P
  uint32_t __restrict__ *Q = xr; // Q = 0
  uint32_t __restrict__ *_inf = misc_const_ct[midx]._inf;
  uint32_t __restrict__ *_1 = misc_const_ct[midx]._1;
  uint32_t __restrict__ scl_cpy[NWORDS_256BIT];

  if (tid==0){
    logInfo("SCMULEC JAC\n");
  }

  // TODO : review this comparison
  if (eq0u256(&x1[2*NWORDS_256BIT])){ 
      memcpy(xr, _inf, 3 * NWORDS_256BIT * sizeof(uint32_t));
      return;  
  }

  memcpy(N, x1, 2 * NWORDS_256BIT * sizeof(uint32_t));
  memcpy(&N[2*NWORDS_256BIT], _1, NWORDS_256BIT * sizeof(uint32_t));

  memcpy(scl_cpy, scl, NWORDS_256BIT * sizeof(uint32_t));
  memcpy(Q, _inf, 3* NWORDS_256BIT * sizeof(uint32_t));

  // TODO : Either implement left to right, or count where msb is and substitute while by unrolled
  // loop

  // TODO : MAD several numbers at once using shamir's trick

  // TODO : During MAD operation, first time I call kernel input is in affine, but 
  // subsequent times input is not in affine. Thus, I remove below code
  // Think of how to include below code just for first time kernel is called. For now,
  // i remove it
  #if 0
  // first iteration, take advantage that input is in affine
  if (!eq0u256(scl_cpy)){
     /*
     if (tid == 0){
       logInfoBigNumber("scl\n",scl);
     }
     */
     b0 = shr1u256(scl_cpy);
     /*
     if (tid == 0){
       logInfoBigNumber("scl >> 1\n",scl);
     }
     */
     /*
     if (b0) { Q = Q + N }
     N = N + N
     */
     if (b0) {
       //addecjacaff(Q, Q, N, midx);
       memcpy(Q,N,3*NWORDS_256BIT * sizeof(uint32_t));
       /*
       if (tid == 0){
         logInfoBigNumber("Q\n",Q);
       }
       */
     }
     doublecjacaff(N,N, midx);
  }
  #endif
  // ret of iterations, input are not affine anymore
  while (!eq0u256(scl_cpy)){
     b0 = shr1u256(scl_cpy);
     /*
     if (tid == 0){
     logInfoBigNumber("scl >> 1\n",scl);
     }
     */
     /*
     if (b0) { Q = Q + N }
     N = N + N
     */
     if (b0) {
       addecjac(Q, N, Q, midx);
     }
     doublecjac(N,N, midx);
  }

  return;
}

