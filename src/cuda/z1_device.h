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
     uint32_t *el;

   public:
     __device__  Z1_t() {}
     __device__  Z1_t(uint32_t *x) : el(x) {}

     __device__ uint32_t *getu256()
     {
       return el;
     }
     __device__ uint32_t *getu256(uint32_t offset)
     {
       return &el[offset*NWORDS_256BIT];
     }
     __device__ void setu256(uint32_t xoffset, Z1_t *y, uint32_t yoffset, uint32_t ysize)
     { 
       memcpy(&el[xoffset*NWORDS_256BIT],&y->el[yoffset*NWORDS_256BIT],ysize * NWORDS_256BIT * sizeof(uint32_t));
     }
     __device__ void setu256(uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t ysize)
     { 
       memcpy(&el[xoffset*NWORDS_256BIT],&y[yoffset*NWORDS_256BIT],ysize * NWORDS_256BIT * sizeof(uint32_t));
     }
     __device__ void assign(uint32_t *y)
     { 
       el = y;
     }
     __device__ static  uint32_t getN()
     {
        return ECP_JAC_N256W;
     }
};

__device__ uint32_t eq0z(Z1_t *x);
__device__ uint32_t eqz(Z1_t *x, Z1_t *y);
__device__ void squarez(Z1_t *z, Z1_t *x, mod_t midx);
__device__ void mulz(Z1_t *z, Z1_t *x, Z1_t *y, mod_t midx);
__device__ void mulkz(Z1_t *z, Z1_t *x, uint32_t *y, mod_t midx);
__device__ void subz(Z1_t *z, Z1_t *x, Z1_t *y, mod_t midx);
__device__ void addz(Z1_t *z, Z1_t *x, Z1_t *y, mod_t midx);
__device__ void movz(Z1_t *x, uint32_t xoffset, uint32_t *y, uint32_t yoffset, uint32_t size);
__device__ void movz(uint32_t *y, uint32_t yoffset, Z1_t *x, uint32_t xoffset, uint32_t size);
__device__ void setkz(Z1_t *z, uint32_t offset, uint32_t *x);

__device__ void xeccz(Z1_t *z, Z1_t *x);
__device__ void yeccz(Z1_t *z, Z1_t *x);
__device__ void zeccz(Z1_t *z, Z1_t *x);
__device__ void infz(Z1_t *z, mod_t midx);
/*
__device__ void _1z(Z1_t *z, mod_t midx);
__device__ void _2z(Z1_t *z, mod_t midx);
__device__ void _3z(Z1_t *z, mod_t midx);
__device__ void _4z(Z1_t *z, mod_t midx);
__device__ void _8z(Z1_t *z, mod_t midx);
*/


#endif
