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
#include "asm_device.h"

__device__ Z1_t::Z1_t() {}
__device__ Z1_t::Z1_t(uint32_t *x) : el(x) {}

__device__ uint32_t * Z1_t::getu256()
{
  return el;
}
__device__ uint32_t * Z1_t::get2u256()
{
  return el;
}

__device__ uint32_t  * Z1_t::getu256(uint32_t offset)
{
    return &el[offset*NWORDS_256BIT];
}
__device__ uint32_t  * Z1_t::getsingleu256(uint32_t offset)
{
    return &el[offset*NWORDS_256BIT];
}

__device__ void Z1_t::setu256(uint32_t xoffset, Z1_t *y, uint32_t yoffset)
{ 
   movu256(&el[xoffset*NWORDS_256BIT],&y->el[yoffset*NWORDS_256BIT]);
   movu256(&el[(xoffset+1)*NWORDS_256BIT],&y->el[(yoffset+1)*NWORDS_256BIT]);
   movu256(&el[(xoffset+2)*NWORDS_256BIT],&y->el[(yoffset+2)*NWORDS_256BIT]);
   //memcpy(&el[xoffset*NWORDS_256BIT],&y->el[yoffset*NWORDS_256BIT],ysize * NWORDS_256BIT * sizeof(uint32_t));
}
__device__ void  Z1_t::setu256(uint32_t xoffset, uint32_t *y, uint32_t yoffset)
{ 
    movu256(&el[xoffset*NWORDS_256BIT],&y[yoffset*NWORDS_256BIT]);
    movu256(&el[(xoffset+1)*NWORDS_256BIT],&y[(yoffset+1)*NWORDS_256BIT]);
    movu256(&el[(xoffset+2)*NWORDS_256BIT],&y[(yoffset+2)*NWORDS_256BIT]);
    //memcpy(&el[xoffset*NWORDS_256BIT],&y[yoffset*NWORDS_256BIT],ysize * NWORDS_256BIT * sizeof(uint32_t));
}
__device__ void Z1_t::setu256(uint32_t xoffset, Z1_t *y, uint32_t yoffset, uint32_t ysize)
{ 
   movu256(&el[(xoffset)*NWORDS_256BIT],&y->el[(yoffset)*NWORDS_256BIT]);
}
__device__ void Z1_t::setsingleu256(uint32_t xoffset, Z1_t *y, uint32_t yoffset)
{ 
   movu256(&el[(xoffset)*NWORDS_256BIT],&y->el[(yoffset)*NWORDS_256BIT]);
}
__device__ void  Z1_t::setu256(uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t ysize)
{ 
   movu256(&el[(xoffset)*NWORDS_256BIT],&y[(yoffset)*NWORDS_256BIT]);
}
__device__ void Z1_t::assign(uint32_t *y)
{ 
   el = y;
}
__device__ uint32_t Z1_t::getN()
{
    return ECP_JAC_N256W;
}


/////

__device__ uint32_t eq0z(Z1_t *x)
{ 
   return eq0u256(x->getu256());
}
__device__ uint32_t eq0z(Z1_t *x, uint32_t offset)
{ 
   return eq0u256(x->getu256(offset));
}

__device__ uint32_t eqz(Z1_t *x, Z1_t *y)
{
  return equ256(x->getu256(), y->getu256());
}
__device__ uint32_t eqz(Z1_t *x, uint32_t xoffset, uint32_t *y )
{
  return equ256(x->getu256(xoffset), y);
}
__device__ uint32_t eqz(Z1_t *x, uint32_t *y)
{
  return equ256(x->getu256(), y);
}


__device__  void squarez(Z1_t *z, Z1_t *x, mod_t midx)
{
  sqmontu256(z->getu256(), x->getu256(),         midx);  
}

__device__ void mulz(Z1_t *z,  Z1_t *x, Z1_t *y, mod_t midx)
{
  mulmontu256(z->getu256(), x->getu256(), y->getu256(), midx);  
}

__device__ void invz(Z1_t *z,  Z1_t *x,  mod_t midx)
{
  invmontu256(z->getu256(), x->getu256(), midx );
}

__device__ void mul2z(Z1_t *z,  Z1_t *x, mod_t midx)
{
  addmu256(z->getu256(), x->getu256(), x->getu256(), midx);    
}
__device__ void div2z(Z1_t *z,  Z1_t *x)
{
  div2u256(z->getu256(), x->getu256());
}
__device__ void mul3z(Z1_t *z,  Z1_t *x, mod_t midx)
{
  addmu256(z->getu256(), x->getu256(), x->getu256(), midx);    
  addmu256(z->getu256(), z->getu256(), x->getu256(), midx);    
}
__device__ void mul4z(Z1_t *z,  Z1_t *x, mod_t midx)
{
  addmu256(z->getu256(), x->getu256(), x->getu256(), midx);    

  addmu256(z->getu256(), z->getu256(), z->getu256(), midx);    
}
__device__ void mul8z(Z1_t *z,  Z1_t *x, mod_t midx)
{
  addmu256(z->getu256(), x->getu256(), x->getu256(), midx);    
  addmu256(z->getu256(), z->getu256(), z->getu256(), midx);    
  addmu256(z->getu256(), z->getu256(), z->getu256(), midx);    
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

__device__ void setkz(Z1_t *z, uint32_t offset, uint32_t *x)
{
  z->setu256(offset,x,0,1);
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

__device__ void addecjacz(Z1_t *zxr, uint32_t zoffset, Z1_t *zx1, uint32_t x1offset, Z1_t *zx2, uint32_t x2offset, mod_t midx)
{
  uint32_t *xr, *x1, *x2;
  uint32_t *_1 = misc_const_ct[midx]._1;
  uint32_t const __restrict__ *P_u256 = &N_ct[ModOffset_ct[midx]];
  uint32_t const __restrict__ *PN_u256 = &NPrime_ct[ModOffset_ct[midx]];

  xr = zxr->getu256(zoffset);
  x1 = zx1->getu256(x1offset);
  x2 = zx2->getu256(x2offset);

  asm(ASM_ADDECJAC_INIT
      ASM_ECJACADD
      ASM_ECFINSH 
      ASM_ECJACADD_PACK );

  return;
}
