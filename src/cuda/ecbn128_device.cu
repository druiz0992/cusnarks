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
    x2 = (uint32_t *) &in_vector[(tid * 2 + 1) * ECK_OFFSET + ECP_XOFFSET];

    xr = (uint32_t *) &out_vector[tid * ECK_OFFSET + ECP_XOFFSET];
    
    addecldr(xr, x1, x2, x1, params->midx);

    return;
}
__global__ void doublecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t __restrict__ *x1,*xr, *z1,*zr;
 
    if(tid >= params->length/3) {
      return;
    }

    x1 = (uint32_t *) &in_vector[tid * ECK_OFFSET + ECP_XOFFSET];

    xr = (uint32_t *) &out_vector[tid * ECK_OFFSET + ECP_XOFFSET];
    
    if (params->premod){
      modu256(x1,x1, params->midx);
      modu256(&x1[NWORDS_256BIT],&x1[NWORDS_256BIT], params->midx);
    }

    doublecldr(xr, x1, params->midx);

  return;
}
__global__ void scmulecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *x1, *z1, *scl, *xr, *zr;
 
   if(tid >= params->length/3) {
     return;
   }

   x1  = (uint32_t *) &in_vector[tid * ECK_OFFSET + ECP_XOFFSET];
   scl = (uint32_t *) &in_vector[tid * ECK_OFFSET + ECP_SCLOFFSET];

   xr = (uint32_t *) &out_vector[tid * ECK_OFFSET + ECP_XOFFSET];
   
   ldrstep(xr, x1, scl,  params->midx);

   return;
}

__global__ void madecldr_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
  return;
}



__global__ void addecjac_kernel(uint32_t   *out_vector_data, uint32_t *in_vector_data, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t __restrict__ *x1, *x2, *xr, *z1, *z2, *zr;
 
    if(tid >= params->length/6) {
      return;
    }

    x1 = (uint32_t *) &in_vector[tid * 2 * ECK_OFFSET + ECP_XOFFSET];
    x2 = (uint32_t *) &in_vector[(tid * 2 + 1) * ECK_OFFSET + ECP_XOFFSET];

    xr = (uint32_t *) &out_vector[tid * ECK_OFFSET + ECP_XOFFSET];
    
    addecjac(xr, x1, x2, params->midx);

    return;

}

__global__ void doublecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t __restrict__ *x1,*xr, *z1,*zr;
 
    if(tid >= params->length/3) {
      return;
    }

    x1 = (uint32_t *) &in_vector[tid * ECK_OFFSET + ECP_XOFFSET];

    xr = (uint32_t *) &out_vector[tid * ECK_OFFSET + ECP_XOFFSET];
    
    if (params->premod){
      modu256(x1,x1, params->midx);
      modu256(&x1[NWORDS_256BIT],&x1[NWORDS_256BIT], params->midx);
    }

    doublecjac(xr, x1, params->midx);

    return;
}

__global__ void scmulecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *x1, *z1, *scl, *xr, *zr;
 
   if(tid >= params->length/3) {
     return;
   }

   x1  = (uint32_t *) &in_vector[tid * ECK_OFFSET + ECP_XOFFSET];
   scl = (uint32_t *) &in_vector[tid * ECK_OFFSET + ECP_SCLOFFSET];

   xr = (uint32_t *) &out_vector[tid * ECK_OFFSET + ECP_XOFFSET];
   
   scmulecjac(xr, x1, scl,  params->midx);

   return;
}

__global__ void madecjac_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
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
   

   mulmontu256(tmp1, x2  , z1  , midx);      
   mulmontu256(xr  , x1  , z2  , midx);      
   submu256(   zr  , xr  , tmp1, midx);
   addmu256(   tmp1, tmp1, xr  , midx);
   mulmontu256(tmp1, tmp1, z2  , midx);    
   mulmontu256(tmp1, tmp1, z1  , midx);    
   mulmontu256(xr  , x1  , x2  , midx);      
   mulmontu256(xr  , xr  , xr  , midx);          // TODO : implement squaring
   mulmontu256(tmp1, tmp1,_4b  , midx);
   submu256(   xr,   tmp1, xr  , midx);
   mulmontu256(zr,   zr  , zr  , midx);     // TODO : implement squaring
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

  mulmontu256(xr,  z1,   z1,   midx);      // TODO squaring
  mulmontu256(zr,  xr,   z1,   midx);      // Zr = Z1^3
  mulmontu256(xr,  zr,   x1,   midx);      
  mulmontu256(xr,  xr,  _b8,   midx);      // Xr = 8b * X1 * Z1^3
  mulmontu256(tmp1, x1,   x1,   midx);      // TODO squaring
  mulmontu256(tmp2,tmp,  tmp,  midx);     // TODO squaring
  submu256(   xr,  tmp2, xr,   midx);

  mulmontu256(zr,  zr,   b,    midx);      // Zr = b*Z1^3
  mulmontu256(tmp, tmp,  x1,   midx);     
  addmu256(   zr, tmp,   zr,   midx);
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
  cudaMemset(&R[1 * NWORDS_256BIT], 0, NWORDS_256BIT * sizeof(uint32_t));

  while (!eq0u256(scl)){
     b0 = shr1u256(scl);
     idxn = ~b0 * 2 * NWORDS_256BIT;
     idxp =  b0 * 2 * NWORDS_256BIT;
     /*
     if (b0) { R0 = R0 + R1; R1 = R1 + R1;
     else {    R1 = R0 + R1; R0 = R0 + R0}
     */
     addecldr(R[idxn], R[idxn], R[idxp], x1, midx);
     doublecldr(R[idxp], R[idxp], midx);
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

__forceinline__ __device__
 void addecjac(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, const uint32_t *x2, mod_t midx)
{
  const uint32_t __restrict__ *y1 = &x1[NWORDS_256BIT];
  const uint32_t __restrict__ *z1 = &x1[NWORDS_256BIT*2];
  const uint32_t __restrict__ *y2 = &x2[NWORDS_256BIT];
  const uint32_t __restrict__ *z2 = &x2[NWORDS_256BIT*2];
  uint32_t __restrict__ *yr = &xr[NWORDS_256BIT];
  uint32_t __restrict__ *zr = &xr2[NWORDS_256BIT*2];
 
  uint32_t __restrict__ tmp1[NWORDS_256BIT], tmp2[NWORDS_256BIT], tmp3[NWORDS_256BIT], tmp4[NWORDS_256BIT];

  mulmontu256(xr, z1, z1, midx);  // xr = z1sq -> double
  mulmontu256(zr, xr, x2, midx);  // zr = u2 = x2 * z1sq
  mulmontu256(xr, xr, z1, midx);  // xr = z1cube
  mulmontu256(xr, xr, y2, midx);  // xr = s2 = z1cube * y2
  mulmontu256(yr, z2, z2, midx);  // yr = z2sq
  mulmontu256(tmp1, x1, yr, midx);  // tmp1 = u1 = x1 * z2sq
  mulmontu256(yr, yr, z2, midx);  // yr = z2cube
  mulmontu256(yr, yr, y1, midx);  // yr = s1 = z2cube * y1

  // TODO Check if I can call add to compute x + x (instead of double)
  //  if not, I should call doublw below. I don't to avoid warp divergnce
  if (equ256((const uint32_t *)tmp, (const uint32_t *)zr)) &&   // u1 == u2
       (!eq256( (const uint32_t *) yr, (const uint32_t *) xr))  // s1 != s2
	  return point_at_inf;  //  if U1 == U2 and S1 == S2 => P1 == P2 (call double)

  submu256(tmp2, zr, tmp1, midx);     // H = tmp2 = u2 - u1
  mulmontu256(zr, z1, z2, midx);      // zr = z1 * z2
  mulmontu256(zr, zr, h, midx);       // zr = z1 * z2  * h

  mulmontu256(tmp3, tmp2, tmp2, midx);     // Hsq = tmp3 = H * H 
  mulmontu256(tmp2, tmp3, tmp2, midx);     // Hcube = tmp2 = Hsq * H 
  mulmontu256(tmp1, tmp1, tmp3, midx);     // tmp1 = u1 * Hsq

  submu256(tmp3, xr, zr, midx);        // R = tmp3 = S2 - S1 tmp1=u1*Hsq, tmp2=Hcube, xr=free, yr=s1, zr=zr
  mulmontu256(yr, yr, tmp2, midx);     // yr = Hcube * s1
  mulmontu256(xr, tmp3, tmp3, midx);     // xr = R * R
  submu256(xr, xr, tmp2, midx);        // xr = x3= (R*R)-Hcube, yr = Hcube * S1, zr=zr, tmp1=u1*Hsq, tmp2 = Hcube, tmp3 = R

  mulmontu256(tmp4, tmp1 _2, midx);     // tmp4 = u1*hsq *_2
  submu256(xr, xr, tmp4);               // x3 = xr
  submu256(tmp1, tmp1, xr, midx);       // tmp1 = u1*hs1 - x3
  mulmontu256(tmp1, tmp1, tmp3, midx);  // tmp1 = r * (u1 * hsq - x3)
  submu256(yr, tmp1, yr, midx);

}

__forceinline__ __device__
 void doublecjac(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, mod_t midx)
{
  const uint32_t __restrict__ *y1 = &x1[NWORDS_256BIT];
  const uint32_t __restrict__ *z1 = &x1[NWORDS_256BIT*2];
  uint32_t __restrict__ *yr = &xr[NWORDS_256BIT];
  uint32_t __restrict__ *zr = &xr2[NWORDS_256BIT*2];
 
  uint32_t __restrict__ tmp1[NWORDS_256BIT], tmp2[NWORDS_256BIT];

  mulmontu256(zr, y1, y1, midx);  // zr = ysq
  mulmontu256(yr, zr, zr, midx);  // yr = ysqsq
  mulmontu256(yr, yr, _8, midx);  // yr = ysqsq *_8
  mulmontu256(zr, zr, x1, midx);  // S = zr = x * ysq
  mulmontu256(zr, zr, _4, midx);  // S = zr = S * _4

  mulmontu256(xr, x1, x1, midx);  // M1 = xr = x * x
  mulmontu256(tmp1, xr, _3, midx);  // M = tmp1 = M1 * _3
  mulmontu256(xr, tmp1, tmp1, midx);  // X3 = xr = M * M,  yr = Ysqsq * _8, zr = S; tmp1 = M
  mulmontu256(tmp2, zr, _2, midx);   // tmp2 = S * _2
  submu256(xr, xr, tmp2, midx);      // X3 = xr; yr = Ysqsq * _8, zr = S, tmp1 = M, 
  submu256(tmp2, zr, xr, midx);   //  tmp2 = S - X3
  mulmontu256(tmp2, tmp2, tmp1, midx); // tmp2 = M * (S - X3)
  submu256(yr, tmp2, yr, midx);
  mulmontu256(zr, y1, x1, midx);
  mulmontu256(zr, zr, _2, midx)

}

 void scmulecjac(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, uint32_t *scl, mod_t midx)
{
  uint32_t b0;

  uint32_t __restrict__ N[2*NWORDS_256BIT]; // N = P
  uint32_t __restrict__ *Q = xr; // Q = 0

  memcpy(N, x1, 2 * NWORDS_256BIT * sizeof(uint32_t));
  memcpy(Q, _1, NWORDS_256BIT * sizeof(uint32_t));
  cudaMemset(&Q[NWORDS_256BIT], 0, NWORDS_256BIT * sizeof(uint32_t));

  while (!eq0u256(scl)){
     b0 = shr1u256(scl);
     /*
     if (b0) { Q = Q + N }
     N = N + N
     */
     if (b0) {
       addecjac(Q, Q, N, midx);
     }
     doublecjac(N,N, midx);
  }

  return;
}
