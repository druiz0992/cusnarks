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
// ------------------------------------------------------------------

*/

#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "ecbn128_device.h"
#include "u256_device.h"

__global__ void addecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t __restrict__ *x1, *x2, *xr, *z1, *z2, *zr;
 
    if(tid >= params->length/6) {
      return;
    }

    x1 = (uint32_t *) &in_vector[tid * 2 * ECK_OFFSET + ECP_XOFFSET];
    //z1 = (uint32_t *) &in_vector[tid * 2 * ECK_OFFSET + ECP_ZOFFSET];
    x2 = (uint32_t *) &in_vector[(tid * 2 + 1) * ECK_OFFSET + ECP_XOFFSET];
    //z2 = (uint32_t *) &in_vector[(tid * 2 + 1) * ECK_OFFSET + ECP_ZOFFSET];

    xr = (uint32_t *) &out_vector[tid * ECK_OFFSET + ECP_XOFFSET];
    //zr = (uint32_t *) &out_vector[tid * ECK_OFFSET + ECP_ZOFFSET];
    
    if (params->premod){
      modu256(x1,x1, params->midx);
      modu256(&x1[NWORDS_256BIT],&x1[NWORDS_256BIT], params->midx);
      //modu256(z1,z1);
      modu256(x2,x2, params->midx);
      modu256(&x2[NWORDS_256BIT],&x2[NWORDS_256BIT], params->midx);
      //modu256(z2,z2);
    }

    addecldr(xr, x1, x2, x1, params->midx);

    return;
}
__global__ void doublecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
  return;
}
__global__ void scmulecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
  return;
}
__global__ void addecldr_reduce_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
  return;
}
__global__ void scmulecldr_reduce_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
  return;
}
    
__forceinline__ __device__
 void addecldr(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, 
            const uint32_t __restrict__ *x2, const uint32_t __restrict__ *xp, mod_t midx)
{
   // Xr = -12 Z1 * Z2 * (X1 * Z2 + X2 * Z1) + (X1 * X2)^2 
   // Zr = xp * (X1 * Z2 - X2 * Z1)^2
   const uint32_t __restrict__ *z1 = &x1[NWORDS_256BIT];
   const uint32_t __restrict__ *z2 = &x2[NWORDS_256BIT];
   uint32_t __restrict__ *zr =&zr[NWORDS_256BIT];

   uint32_t tmp1[NWORDS_256BIT];
   uint32_t __restrict__ *twelve = misc_const_ct[midx].twelve;
   

   mulmontu256(tmp1, x2  , z1  , midx);      
   mulmontu256(xr  , x1  , z2  , midx);      
   submu256(   zr  , xr  , tmp1, midx);
   addmu256(   tmp1, tmp1, xr  , midx);
   mulmontu256(tmp1, tmp1, z2  , midx);    
   mulmontu256(tmp1, tmp1, z1  , midx);    
   mulmontu256(xr  , x1  , x2  , midx);      
   mulmontu256(xr  , xr  , xr  , midx);          // TODO : implement squaring
   mulmontu256(tmp1, tmp1,twelve   ,midx);
   submu256(   xr,   tmp1, xr      ,midx);
   mulmontu256(zr,   zr  , zr      ,midx);     // TODO : implement squaring
   mulmontu256(zr,   zr  , xp      ,midx);   

  return;
}

__forceinline__ __device__
 void doublecldr(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, mod_t midx)
{
  // Xr = X1^4 - 24 * X1*Z1^3
  // Zr = 4*Z1 * (X1^3 + 3*Z1^3)
}
