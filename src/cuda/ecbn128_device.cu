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

__global__ void addecc_kernel(uint32_t *out_vector, uint32_t *in_vector, const uint32_t *p, uint32_t np, uint32_t len, uint32_t premod)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t *x1, *x2, *xr, *z1, *z2, *zr;
 
    if(tid >= len/6) {
      return;
    }

    x1 = (uint32_t *) &in_vector[tid * 2 * ECK_OFFSET + ECP_XOFFSET];
    //z1 = (uint32_t *) &in_vector[tid * 2 * ECK_OFFSET + ECP_ZOFFSET];
    x2 = (uint32_t *) &in_vector[(tid * 2 + 1) * ECK_OFFSET + ECP_XOFFSET];
    //z2 = (uint32_t *) &in_vector[(tid * 2 + 1) * ECK_OFFSET + ECP_ZOFFSET];

    xr = (uint32_t *) &out_vector[tid * ECK_OFFSET + ECP_XOFFSET];
    //zr = (uint32_t *) &out_vector[tid * ECK_OFFSET + ECP_ZOFFSET];
    
    if (premod){
      modu256(x1,x1,p);
      modu256(&x1[NWORDS_256BIT],&x1[NWORDS_256BIT],p);
      //modu256(z1,z1,p);
      modu256(x2,x2,p);
      modu256(&x2[NWORDS_256BIT],&x2[NWORDS_256BIT],p);
      //modu256(z2,z2,p);
    }

    addecc(xr, x1, x2, p, np);

    return;
}
__global__ void doublecc_kernel(uint32_t *out_vector, uint32_t *in_vector, const uint32_t *p, uint32_t len, uint32_t premod)
{
  return;
}
__global__ void scmulecc_kernel(uint32_t *out_vector, uint32_t *in_vector, const uint32_t *p, uint32_t len, uint32_t premod)
{
  return;
}
__global__ void addecc_reduce_kernel(uint32_t *out_vector, uint32_t *in_vector, const uint32_t *p, uint32_t len, uint32_t premod)
{
  return;
}
__global__ void scmulecc_reduce_kernel(uint32_t *out_vector, uint32_t *in_vector const uint32_t *p, uint32_t len, uint32_t premod)
{
  return;
}
    
__forceinline__ __device__ void addecc(uint32_t *xr, const uint32_t *x1, const uint32_t *x2, const uint32_t *p, const uint32_t *np)
{
   // Xr = -12 Z1 * Z2 * (X1 * Z2 + X2 * Z1) + (X1 * X2)^2 
   // Zr = x * (X1 * Z2 - X2 * Z1)^2
   const uint32_t *z1 = &x1[NWORDS_256BIT];
   const uint32_t *z2 = &x2[NWORDS_256BIT];
   uint32_t *zr =&zr[NWORDS_256BIT];

   uint32_t tmp1[NWORDS_256BIT];

   mulmontu256(tmp1, x2, z1, p, np);      
   mulmontu256(xr, x1, z2, p, np);      
   submu256(zr, xr, tmp1,  p);
   addmu256(tmp1, tmp1, xr, p);
   mulmontu256(tmp1, tmp1, z2, p, np);    
   mulmontu256(tmp1, tmp1, z1, p, np);    
   mulmontu256(xr, x1, x2, p, np);      
   mulmontu256(xr, xr, xr, p, np);     // TODO : implement squaring
   // TODO Multiply xr by -12
   addmu256(xr, tmp1, xr, p);
   mulmontu256(zr, zr, zr, p, np);     // TODO : implement squaring
   // TODO multiply zr by x

  return;
}
