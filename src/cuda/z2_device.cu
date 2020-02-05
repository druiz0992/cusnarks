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
#include "asm_device.h"

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
  return &el[offset*ECP2_JAC_N256W*NWORDS_256BIT+NWORDS_256BIT];
}

__device__ void Z2_t::setu256(uint32_t xoffset, Z2_t *y, uint32_t yoffset)
{ 
   //memcpy(&el[xoffset*ECP2_JAC_N256W*NWORDS_256BIT],
         //&y->el[yoffset*ECP2_JAC_N256W*NWORDS_256BIT],
         //3 * ECP2_JAC_N256W* NWORDS_256BIT * sizeof(uint32_t));
   movu256x6(&el[xoffset*NWORDS_256BIT],&y->el[yoffset*NWORDS_256BIT]);
   //movu256x6(&el[xoffset*NWORDS_256BIT],&y->el[yoffset*NWORDS_256BIT]);
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
__device__ uint32_t eq0z(Z2_t *x, uint32_t offset)
{ 
   return (eq0u256(x->getu256(offset)) && eq0u256(x->get2u256(offset)));
}

__device__ uint32_t eqz(Z2_t *x, Z2_t *y)
{
  return (equ256(x->getu256(), y->getu256()) && equ256(x->get2u256(), y->get2u256()));
}
__device__ uint32_t eqz(Z2_t *x, uint32_t xoffset, uint32_t *y)
{
  return (equ256(x->getu256(xoffset), y) && equ256(x->get2u256(xoffset), &y[NWORDS_256BIT]));
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

__device__ void invz(Z2_t *z,  Z2_t *x,  mod_t midx)
{
  invmontu256_2(z->getu256(), x->getu256(), midx );
}

__device__ void div2z(Z2_t *z,  Z2_t *x)
{
  div2u256(z->getu256(), x->getu256());
  div2u256(z->get2u256(), x->get2u256());
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


__device__ void addecjacz(Z2_t *zxr, uint32_t zoffset, Z2_t *zx1, uint32_t x1offset, Z2_t *zx2, uint32_t x2offset, mod_t midx)
{
  Z2_t x1(zx1->getu256(0+x1offset)), y1(zx1->getu256(1+x1offset)), z1(zx1->getu256(2+x1offset));
  Z2_t x2(zx2->getu256(0+x2offset)), y2(zx2->getu256(1+x2offset)), z2(zx2->getu256(2+x2offset));
  Z2_t xr(zxr->getu256(0+zoffset)),  yr(zxr->getu256(1+zoffset)), zr(zxr->getu256(2+zoffset));
 
  uint32_t __restrict__ ztmp[5*NWORDS_256BIT*2];
  Z2_t tmp1(ztmp), tmp3(&ztmp[1*NWORDS_256BIT*2]),
                 tmp_x(&ztmp[2*NWORDS_256BIT*2]),
                 tmp_y(&ztmp[3*NWORDS_256BIT*2]),
                 tmp_z(&ztmp[4*NWORDS_256BIT*2]);

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
      zxr->setu256(2*zoffset,zx2,2*x2offset);
      return;  
  }
  if (eq0z(&z2)){ 
      zxr->setu256(2*zoffset,zx1,2*x1offset);
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
          Z2_t _inf;
          infz(&_inf, midx);
          zxr->setu256(zoffset,&_inf,x1offset);
	  return;  
      }
      doublecjacz(&xr,&x1, midx);
      return;
  }

  subz(&tmp_z, &tmp_z, &tmp1, midx);     // H = tmp2 = u2 - u1
  mulz(&zr, &z1, &z2, midx);      // tmp_z = z1 * z2
  squarez(&tmp3, &tmp_z,        midx);     // Hsq = tmp3 = H * H 
  mulz(&zr, &zr, &tmp_z, midx);       // zr = z1 * z2  * h

  mulz(&tmp_z, &tmp_z, &tmp3, midx);     // Hcube = tmp2 = Hsq * H 
  mulz(&tmp1, &tmp1, &tmp3, midx);     // tmp1 = u1 * Hsq

  subz(&tmp3, &tmp_x, &tmp_y, midx);        // R = tmp3 = S2 - S1 tmp1=u1*Hsq, tmp2=Hcube, tmp_x=free, tmp_y=s1, zr=zr
  mulz(&tmp_y, &tmp_y, &tmp_z, midx);     // tmp_y = Hcube * s1
  squarez(&tmp_x, &tmp3, midx);     // tmp_x = R * R

  mul2z(&xr, &tmp1, midx);     // tmp4 = u1*hsq *_2
  subz(&tmp_x, &tmp_x, &tmp_z, midx);        // tmp_x = x3= (R*R)-Hcube, tmp_y = Hcube * S1, zr=zr, tmp1=u1*Hsq, tmp2 = Hcube, tmp3 = R

  subz(&xr, &tmp_x, &xr, midx);               // x3 = xr
  subz(&tmp1, &tmp1, &xr, midx);       // tmp1 = u1*hs1 - x3
  mulz(&tmp1, &tmp1, &tmp3, midx);  // tmp1 = r * (u1 * hsq - x3)
  subz(&yr, &tmp1, &tmp_y, midx);

}

__device__ void doublecjacz(Z2_t *zxr, Z2_t *zx1, mod_t midx)
{
  Z2_t y1(zx1->getu256(1)), z1(zx1->getu256(2));
  Z2_t yr(zxr->getu256(1)), zr(zxr->getu256(2));

  uint32_t __restrict__ ztmp[2*NWORDS_256BIT*2];
  Z2_t tmp_y(&ztmp[0*NWORDS_256BIT*2]),
     tmp_z(&ztmp[1*NWORDS_256BIT*2]);

  // TODO : review this comparison, and see if I can do better. or where I should put it
  // as i check this in several places
  if (eq0z(&z1)){ 
      Z2_t _inf;
      infz(&_inf,midx);
      zxr->setu256(0,&_inf,0);
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

}

#if 0
__device__ void scmulec_stepz(Z2_t *Q,Z2_t *N, uint32_t *scl, uint32_t msb,  mod_t midx )
{
  uint32_t b;
  for (uint32_t i=msb; i< (1 << (NWORDS_256BIT)); i++){
      b = bselMu256(scl,255-i);
      doublecjacz(Q,Q, midx);

      if (b){
       addecjacz(Q,0, N,b*3, Q,0, midx);
     }
  }
}
#endif
