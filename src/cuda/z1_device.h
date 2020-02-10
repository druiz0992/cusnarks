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
// File name  : z1_device.h
//
// Date       : 20/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of Finite Field arithmetic
// ------------------------------------------------------------------

*/
#ifndef _Z1_H
#define _Z1_H

class Z1_t {
   private:
     uint32_t __align__(16) *el;

   public:
     __device__  Z1_t();
     __device__  Z1_t(uint32_t *x);

     __device__ uint32_t *getu256();
     __device__ uint32_t *getu256(uint32_t offset);
     __device__ uint32_t *get2u256();
     __device__ uint32_t  *getsingleu256(uint32_t offset);
     __device__ void      setsingleu256(uint32_t xoffset, Z1_t *y, uint32_t yoffset);
     __device__ void setu256(uint32_t xoffset, Z1_t *y, uint32_t yoffset);
     __device__ void setu256(uint32_t xoffset, uint32_t *y, uint32_t yoffset);
     __device__ void setu256(uint32_t xoffset, Z1_t *y, uint32_t yoffset, uint32_t ysize);
     __device__ void setu256(uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t ysize);
     __device__ void assign(uint32_t *y);
     __device__ static uint32_t getN();
};

__device__ uint32_t eq0z(Z1_t *x);
__device__ uint32_t eq0z(Z1_t *x, uint32_t offset);
__device__ uint32_t eqz(Z1_t *x, Z1_t *y);
__device__ uint32_t eqz(Z1_t *x, uint32_t *y);
__device__ uint32_t eqz(Z1_t *x, uint32_t xoffset, uint32_t *y );
__device__ void squarez(Z1_t *z, Z1_t *x, mod_t midx);
__device__ void mulz(Z1_t *z, Z1_t *x, Z1_t *y, mod_t midx);
__device__ void mul2z(Z1_t *z, Z1_t *x, mod_t midx);
__device__ void mul3z(Z1_t *z, Z1_t *x, mod_t midx);
__device__ void mul4z(Z1_t *z, Z1_t *x, mod_t midx);
__device__ void mul8z(Z1_t *z, Z1_t *x, mod_t midx);
__device__ void subz(Z1_t *z, Z1_t *x, Z1_t *y, mod_t midx);
__device__ void addz(Z1_t *z, Z1_t *x, Z1_t *y, mod_t midx);
__device__ void movz(Z1_t *x, uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t size);
__device__ void movz(uint32_t *y, uint32_t yoffset, Z1_t *x, uint32_t xoffset, uint32_t size);
__device__ void setkz(Z1_t *z, uint32_t offset, uint32_t *x);
__device__ void invz(Z1_t *z, Z1_t *x, mod_t midx);
__device__ void div2z(Z1_t *z,  Z1_t *x);

__device__ void xeccz(Z1_t *z, Z1_t *x);
__device__ void yeccz(Z1_t *z, Z1_t *x);
__device__ void zeccz(Z1_t *z, Z1_t *x);
__device__ void infz(Z1_t *z, mod_t midx);
__device__ void addecjacz(Z1_t *zxr, uint32_t zoffset, Z1_t *zx1, uint32_t x1offset, Z1_t *zx2, uint32_t x2offset, mod_t midx);
__device__ void scmulec_stepz(Z1_t *Q,Z1_t *N, uint32_t *scl, uint32_t msb,  mod_t midx );


#endif
