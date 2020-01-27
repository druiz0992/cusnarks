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
// File name  : z2_device.h
//
// Date       : 24/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of Extended (2) Finite Field arithmetic
// ------------------------------------------------------------------

*/
#ifndef _Z2_H
#define _Z2_H

class Z2_t {
   private:
     uint32_t __align__(16) *el;

   public:
     __device__  Z2_t();
     __device__  Z2_t(uint32_t *x);

     __device__ uint32_t *getu256();
     __device__ uint32_t *get2u256();
     __device__ uint32_t *getu256(uint32_t offset);
     __device__ uint32_t *get2u256(uint32_t offset);
     __device__ uint32_t *getsingleu256(uint32_t offset);
     __device__ void setu256(uint32_t xoffset, Z2_t *y, uint32_t yoffset);
     __device__ void setu256(uint32_t xoffset, uint32_t *y, uint32_t yoffset);
     __device__ void setu256(uint32_t xoffset, Z2_t *y, uint32_t yoffset, uint32_t ysize);
     __device__ void setu256(uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t ysize);
     __device__ void set2u256(uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t ysize);
     __device__ void assign(uint32_t *y);
     __device__ static  uint32_t getN();
};

__device__ uint32_t eq0z(Z2_t *x);
__device__ uint32_t eq0z(Z2_t *x, uint32_t offset);
__device__ uint32_t eqz(Z2_t *x, Z2_t *y);
__device__ uint32_t eqz(Z2_t *x, uint32_t *y);
__device__ uint32_t eqz(Z2_t *x, uint32_t xoffset, uint32_t *y );;
__device__ void squarez(Z2_t *z, Z2_t *x, mod_t midx);
__device__ void mulz(Z2_t *z, Z2_t *x, Z2_t *y, mod_t midx);
__device__ void mul2z(Z2_t *z, Z2_t *x, mod_t midx);
__device__ void mul3z(Z2_t *z, Z2_t *x, mod_t midx);
__device__ void mul4z(Z2_t *z, Z2_t *x, mod_t midx);
__device__ void mul8z(Z2_t *z, Z2_t *x, mod_t midx);
__device__ void subz(Z2_t *z, Z2_t *x, Z2_t *y, mod_t midx);
__device__ void addz(Z2_t *z, Z2_t *x, Z2_t *y, mod_t midx);
__device__ void movz(Z2_t *x, uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t size);
__device__ void movz(uint32_t *y, uint32_t yoffset, Z2_t *x, uint32_t xoffset, uint32_t size);
__device__ void setkz(Z2_t *z, uint32_t offset, uint32_t *x);
__device__ void invz(Z2_t *z,  Z2_t *x,  mod_t midx);
__device__ void div2z(Z2_t *z, Z2_t *x);

__device__ void xeccz(Z2_t *z, Z2_t *x);
__device__ void yeccz(Z2_t *z, Z2_t *x);
__device__ void zeccz(Z2_t *z, Z2_t *x);
__device__ void infz(Z2_t *z, mod_t midx);


#endif
