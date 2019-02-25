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
// File name  : ecbn128_2_device.cu
//
// Date       : 22/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementatoin of EC Cuda functionality for extended fields
// 
// NOTE : EC Points do not require to be premoded premod. They shoould
//  already be < than prime
// ------------------------------------------------------------------

*/

#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "log.h"
#include "ecbn128_2_device.h"
#include "u256_device.h"

/* 
  in_vector  :  px0[0], px1[0], py[0], py[1],..,
    (affine)             px0[N-1], px1[N-1], py0[N-1], py1[N-1]   
  out vector :  px0[0], px1[0], py[0], py[1], pz[0], pz[1],..., 
    (jacobian)         px0[N-1], px1[N-1], py0[N-1], py1[N-1], pz0[N-1], pz1[N-1]
*/
__global__ void addecjac2_kernel(uint32_t   *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t __restrict__ *x1, *x2, *xr;
 
    if(tid >= params->in_length/8) {
      return;
    }

    // x1 points to inPx[i]. x2 points to inPx[i+1]. xr points to outPx[i]
    x1 = (uint32_t *) &in_vector[tid * 8 * ECP2_AFF_INOFFSET + ECP2_AFF_INXOFFSET];
    x2 = (uint32_t *) &in_vector[(tid * 8 + 1) * ECP2_AFF_INOFFSET + ECP2_AFF_INXOFFSET];
    xr = (uint32_t *) &out_vector[tid * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET];
    
    addecjacaff2(xr, x1, x2, params->midx);

    return;

}

/* 
  in_vector  :  px0[0], px1[0], py[0], py[1],..,
    (affine)             px0[N-1], px1[N-1], py0[N-1], py1[N-1]   
  out vector :  px0[0], px1[0], py[0], py[1], pz[0], pz[1],..., 
    (jacobian)         px0[N-1], px1[N-1], py0[N-1], py1[N-1], pz0[N-1], pz1[N-1]
*/
__global__ void doublecjac2_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t __restrict__ *x1,*xr;
 
    // x1 points to inPx[i].  xr points to outPx[i]
    if(tid >= params->in_length/4) {
      return;
    }

    x1 = (uint32_t *) &in_vector[tid * ECP2_AFF_INOFFSET + ECP2_AFF_INXOFFSET];
    xr = (uint32_t *) &out_vector[tid * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET];
    
    doublecjacaff2(xr, x1, params->midx);

    return;
}

/* 
  in_vector  :  k[0], k[1],..,k[n-1], px0[0], px1[0], py[0], py[1],..,
    (affine)             px0[N-1], px1[N-1], py0[N-1], py1[N-1]   
  out vector :  px0[0], px1[0], py[0], py[1], pz[0], pz[1],..., 
    (jacobian)         px0[N-1], px1[N-1], py0[N-1], py1[N-1], pz0[N-1], pz1[N-1]
*/
__global__ void scmulecjac2_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
   int tid = threadIdx.x + blockDim.x * blockIdx.x;

   uint32_t __restrict__ *x1, *scl, *xr;
   uint32_t ecp2_offset = params->in_length/5 * NWORDS_256BIT;
 
   if(tid >= params->in_length/3) {
     return;
   }

   x1  = (uint32_t *) &in_vector[ecp2_offset + tid * ECP2_AFF_INOFFSET + ECP2_AFF_INXOFFSET];
   scl = (uint32_t *) &in_vector[tid * U256K_OFFSET];

   xr = (uint32_t *) &out_vector[tid * ECP2_JAC_OUTOFFSET + ECP2_JAC_OUTXOFFSET];
  
   scmulecjac2(xr, x1, scl,  params->midx);

   return;
}

__global__ void madecjac2_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params)
{
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
*/
__forceinline__ __device__
 void addecjac2(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, const uint32_t *x2, mod_t midx)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  const uint32_t __restrict__ *ax0 = x1;
  const uint32_t __restrict__ *ax1 = &x1[1*NWORDS_256BIT];
  const uint32_t __restrict__ *ay0 = &x1[2*NWORDS_256BIT];
  const uint32_t __restrict__ *ay1 = &x1[3*NWORDS_256BIT];
  const uint32_t __restrict__ *az0 = &x1[4*NWORDS_256BIT];
  const uint32_t __restrict__ *az1 = &x1[5*NWORDS_256BIT];

  const uint32_t __restrict__ *bx0 = x2;
  const uint32_t __restrict__ *bx1 = &x2[1*NWORDS_256BIT];
  const uint32_t __restrict__ *by0 = &x2[2*NWORDS_256BIT];
  const uint32_t __restrict__ *by1 = &x2[3*NWORDS_256BIT];
  const uint32_t __restrict__ *bz0 = &x2[4*NWORDS_256BIT];
  const uint32_t __restrict__ *bz1 = &x2[5*NWORDS_256BIT];

  uint32_t __restrict__ *rx0 = xr;
  uint32_t __restrict__ *rx1 = &xr[1*NWORDS_256BIT];
  uint32_t __restrict__ *ry0 = &xr[2*NWORDS_256BIT];
  uint32_t __restrict__ *ry1 = &xr[3*NWORDS_256BIT];
  uint32_t __restrict__ *rz0 = &xr[4*NWORDS_256BIT];
  uint32_t __restrict__ *rz1 = &xr[5*NWORDS_256BIT];

}
void scmulecjac2(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, uint32_t *scl, mod_t midx)
{
  return;
}

/*
  input is in affine coordinates -> P(Z) = 1
  I can do Q = Q+Y or Q = Y + Q
*/
__forceinline__ __device__
 void addecjacaff2(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, const uint32_t *x2, mod_t midx)
{
   return;
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
   return;
}
__forceinline__ __device__
 void doublecjacaff2(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, mod_t midx)
{
    return;
}

__forceinline__ __device__
 void scmulecjac_2(uint32_t __restrict__ *xr, const uint32_t __restrict__ *x1, uint32_t *scl, mod_t midx)
{
  return;
}

