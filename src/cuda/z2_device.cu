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
// File name  : z2_device.cu
//
// Date       : 23/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementatoin of Extended (2) Finite Field arithmetic
// 
// ------------------------------------------------------------------

*/

#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "utils_device.h"
#include "u256_device.h"
#include "z2_device.h"

__device__ Z2_t::Z2_t() {}
__device__ Z2_t::Z2_t(uint32_t *x) : el(x) {}

__device__ uint32_t * Z2_t::getu256()
{
  return el;
}

__device__ uint32_t * Z2_t::get2u256()
{
  return &el[NWORDS_256BIT];
}

__device__ uint32_t * Z2_t::getsingleu256(uint32_t offset)
{
  return &el[offset*NWORDS_256BIT];
}
__device__ uint32_t * Z2_t::getu256(uint32_t offset)
{
  return &el[offset*ECP2_JAC_N256W*NWORDS_256BIT];
}

__device__ uint32_t * Z2_t::get2u256(uint32_t offset)
{
  return &el[offset*(ECP2_JAC_N256W+1)*NWORDS_256BIT];
}

__device__ void Z2_t::setu256(uint32_t xoffset, Z2_t *y, uint32_t yoffset)
{ 
   //memcpy(&el[xoffset*ECP2_JAC_N256W*NWORDS_256BIT],
         //&y->el[yoffset*ECP2_JAC_N256W*NWORDS_256BIT],
         //3 * ECP2_JAC_N256W* NWORDS_256BIT * sizeof(uint32_t));
   movu256x6(&el[xoffset*NWORDS_256BIT],&y->el[yoffset*NWORDS_256BIT]);
   //movu256(&el[(xoffset+1)*NWORDS_256BIT],&y->el[(yoffset+1)*NWORDS_256BIT]);
   //movu256(&el[(xoffset+2)*NWORDS_256BIT],&y->el[(yoffset+2)*NWORDS_256BIT]);
   //movu256(&el[(xoffset+3)*NWORDS_256BIT],&y->el[(yoffset+3)*NWORDS_256BIT]);
   //movu256(&el[(xoffset+4)*NWORDS_256BIT],&y->el[(yoffset+4)*NWORDS_256BIT]);
   //movu256(&el[(xoffset+5)*NWORDS_256BIT],&y->el[(yoffset+5)*NWORDS_256BIT]);
}

__device__ void Z2_t::setu256(uint32_t xoffset, uint32_t *y, uint32_t yoffset)
{ 
    //memcpy(&el[xoffset*ECP2_JAC_N256W*NWORDS_256BIT],
           //&y[yoffset*ECP2_JAC_N256W*NWORDS_256BIT],
           //3* ECP2_JAC_N256W * NWORDS_256BIT * sizeof(uint32_t));
   movu256x6(&el[xoffset*NWORDS_256BIT],&y[yoffset*NWORDS_256BIT]);
   //movu256(&el[(xoffset+1)*NWORDS_256BIT],&y[(yoffset+1)*NWORDS_256BIT]);
   //movu256(&el[(xoffset+2)*NWORDS_256BIT],&y[(yoffset+2)*NWORDS_256BIT]);
   //movu256(&el[(xoffset+3)*NWORDS_256BIT],&y[(yoffset+3)*NWORDS_256BIT]);
   //movu256(&el[(xoffset+4)*NWORDS_256BIT],&y[(yoffset+4)*NWORDS_256BIT]);
   //movu256(&el[(xoffset+5)*NWORDS_256BIT],&y[(yoffset+5)*NWORDS_256BIT]);
}

__device__ void Z2_t::setu256(uint32_t xoffset, Z2_t *y, uint32_t yoffset, uint32_t ysize)
{ 
       //memcpy(&el[xoffset*ECP2_JAC_N256W*NWORDS_256BIT],
              //&y->el[yoffset*ECP2_JAC_N256W*NWORDS_256BIT],
              //ysize * ECP2_JAC_N256W* NWORDS_256BIT * sizeof(uint32_t));
   movu256(&el[(xoffset)*2*NWORDS_256BIT],&y->el[(yoffset)*2*NWORDS_256BIT]);
   movu256(&el[(xoffset)*2*NWORDS_256BIT+NWORDS_256BIT],&y->el[(yoffset)*2*NWORDS_256BIT+NWORDS_256BIT]);
}

__device__ void Z2_t::setu256(uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t ysize)
{ 
    //memcpy(&el[xoffset*ECP2_JAC_N256W*NWORDS_256BIT],
           //&y[yoffset*ECP2_JAC_N256W*NWORDS_256BIT],
           //ysize * ECP2_JAC_N256W * NWORDS_256BIT * sizeof(uint32_t));
   movu256(&el[(xoffset)*2*NWORDS_256BIT],&y[(yoffset)*2*NWORDS_256BIT]);
   movu256(&el[(xoffset)*2*NWORDS_256BIT+NWORDS_256BIT],&y[(yoffset)*2*NWORDS_256BIT+NWORDS_256BIT]);
}
/*
__device__ void Z2_t::set2u256(uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t ysize)
{ 
    //memcpy(&el[xoffset*ECP2_JAC_N256W*NWORDS_256BIT],
           //&y[yoffset*ECP2_JAC_N256W*NWORDS_256BIT],
           //ysize * NWORDS_256BIT * sizeof(uint32_t));
   movu256(&el[(xoffset+1)*NWORDS_256BIT],&y[(yoffset+1)*NWORDS_256BIT]);
}
*/

__device__ void Z2_t::assign(uint32_t *y)
{ 
    el = y;
}

__device__  uint32_t Z2_t::getN()
{
    return ECP2_JAC_N256W;
}

////

__device__ uint32_t eq0z(Z2_t *x)
{ 
   return (eq0u256(x->getu256()) && eq0u256(x->get2u256()));
}

__device__ uint32_t eq1z(Z2_t *x)
{ 
   return (eq1u256(x->getu256()) && eq0u256(x->get2u256()));
}

__device__ uint32_t eqz(Z2_t *x, Z2_t *y)
{
  return (equ256(x->getu256(), y->getu256()) && equ256(x->get2u256(), y->get2u256()));
}
__device__ uint32_t eqz(Z2_t *x, uint32_t *y)
{
  return (equ256(x->getu256(), y) && equ256(x->get2u256(),  &y[NWORDS_256BIT]));
}

__device__  void squarez(Z2_t *z, Z2_t *x, mod_t midx)
{
  sqmontu256_2(z->getu256(), x->getu256(),         midx);  
}

__device__ void mulz(Z2_t *z,  Z2_t *x, Z2_t *y, mod_t midx)
{
  mulmontu256_2(z->getu256(), x->getu256(), y->getu256(), midx);  
}

__device__ void mul2z(Z2_t *z,  Z2_t *x, mod_t midx)
{
  addmu256(z->getu256(), x->getu256(), x->getu256(), midx);    
  addmu256(z->get2u256(), x->get2u256(), x->get2u256(), midx);    
}
__device__ void mul3z(Z2_t *z,  Z2_t *x, mod_t midx)
{
   addmu256(z->getu256(), x->getu256(), x->getu256(), midx);    
   addmu256(z->get2u256(), x->get2u256(), x->get2u256(), midx);    

   addmu256(z->getu256(), z->getu256(), x->getu256(), midx);    
   addmu256(z->get2u256(), z->get2u256(), x->get2u256(), midx);    
}
__device__ void mul4z(Z2_t *z,  Z2_t *x, mod_t midx)
{
  addmu256(z->getu256(), x->getu256(), x->getu256(), midx);    
  addmu256(z->get2u256(), x->get2u256(), x->get2u256(), midx);    

  addmu256(z->getu256(), z->getu256(), z->getu256(), midx);    
  addmu256(z->get2u256(), z->get2u256(), z->get2u256(), midx);    
}
__device__ void mul8z(Z2_t *z,  Z2_t *x, mod_t midx)
{
  addmu256(z->getu256(), x->getu256(), x->getu256(), midx);    
  addmu256(z->get2u256(), x->get2u256(), x->get2u256(), midx);    

  addmu256(z->getu256(), z->getu256(), z->getu256(), midx);    
  addmu256(z->get2u256(), z->get2u256(), z->get2u256(), midx);    
    
  addmu256(z->getu256(), z->getu256(), z->getu256(), midx);    
  addmu256(z->get2u256(), z->get2u256(), z->get2u256(), midx);    
}

__device__ void subz(Z2_t *z, Z2_t *x, Z2_t *y, mod_t midx)
{
  submu256(z->getu256(), x->getu256(), y->getu256(), midx);    
  submu256(z->get2u256(), x->get2u256(), y->get2u256(), midx);    
}
__device__ void addz(Z2_t *z, Z2_t *x, Z2_t *y, mod_t midx)
{
  addmu256(z->getu256(), x->getu256(), y->getu256(), midx);    
  addmu256(z->get2u256(), x->get2u256(), y->get2u256(), midx);    
}

__device__ void movz(Z2_t *x, uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t size)
{
  x->setu256(xoffset,y,yoffset,size);
}
__device__ void movz(uint32_t *y, uint32_t yoffset, Z2_t *x, uint32_t xoffset, uint32_t size)
{
  uint32_t i;
  
  #pragma unroll
  for (i=0; i< 2*size; i++){
    movu256(&y[yoffset],x->getu256(xoffset));
    xoffset += NWORDS_256BIT;
    yoffset += NWORDS_256BIT;
  }
}

__device__ void setkz(Z2_t *z, uint32_t offset, uint32_t *x)
{
  z->setu256(offset,x,0,1);
}
__device__ void xeccz(Z2_t *z, Z2_t *x)
{
  z->assign(x->getu256());
}
__device__ void yeccz(Z2_t *z, Z2_t *x)
{
  z->assign(x->getu256(1));
}
__device__ void zeccz(Z2_t *z, Z2_t *x)
{
  z->assign(x->getu256(2));
}

__device__ void infz(Z2_t *z, mod_t midx)
{
  z->assign(misc_const_ct[midx]._inf2);
}
