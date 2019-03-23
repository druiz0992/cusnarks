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
// File name  : z1_device.cu
//
// Date       : 20/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementatoin of Finite Field arithmetic
// 
// ------------------------------------------------------------------

*/

#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "utils_device.h"
#include "u256_device.h"
#include "z1_device.h"


__device__ uint32_t eq0z(Z1_t *x)
{ 
   return eq0u256(x->getu256());
}

__device__ uint32_t eqz(Z1_t *x, Z1_t *y)
{
  return equ256(x->getu256(), y->getu256());
}
__device__  void squarez(Z1_t *z, Z1_t *x, mod_t midx)
{
  sqmontu256(z->getu256(), x->getu256(),         midx);  
}

__device__ void mulz(Z1_t *z,  Z1_t *x, Z1_t *y, mod_t midx)
{
  mulmontu256(z->getu256(), x->getu256(), y->getu256(), midx);  
}

__device__ void subz(Z1_t *z, Z1_t *x, Z1_t *y, mod_t midx)
{
  submu256(z->getu256(), x->getu256(), y->getu256(), midx);    
}
__device__ void addz(Z1_t *z, Z1_t *x, Z1_t *y, mod_t midx)
{
  addmu256(z->getu256(), x->getu256(), y->getu256(), midx);    
}

__device__ void movz(Z1_t *x, uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t size)
{
  x->setu256(xoffset,y,yoffset,size);
}
__device__ void movz(uint32_t *y, uint32_t yoffset, Z1_t *x, uint32_t xoffset, uint32_t size)
{
  uint32_t i;
  
  #pragma unroll
  for (i=0; i< size; i++){
    movu256(&y[yoffset],x->getu256(xoffset));
    xoffset += NWORDS_256BIT;
    yoffset += NWORDS_256BIT;
  }
}

__device__ void xeccz(Z1_t *z, Z1_t *x)
{
  z->assign(x->getu256());
}
__device__ void yeccz(Z1_t *z, Z1_t *x)
{
  z->assign(x->getu256(1));
}
__device__ void zeccz(Z1_t *z, Z1_t *x)
{
  z->assign(x->getu256(2));
}

__device__ void infz(Z1_t *z, mod_t midx)
{
  z->assign(misc_const_ct[midx]._inf);
}
__device__ void _1z(Z1_t *z, mod_t midx)
{
  z->assign(misc_const_ct[midx]._1);
}
__device__ void _2z(Z1_t *z, mod_t midx)
{
  z->assign( misc_const_ct[midx]._2);
}
__device__ void _3z(Z1_t *z, mod_t midx)
{
  z->assign(misc_const_ct[midx]._3);
}
__device__ void _4z(Z1_t *z, mod_t midx)
{
  z->assign(misc_const_ct[midx]._4);
}
__device__ void _8z(Z1_t *z, mod_t midx)
{
  z->assign(misc_const_ct[midx]._8);
}
