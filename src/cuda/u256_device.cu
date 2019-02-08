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
// File name  : u256_kernel.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of uint256 kernel and device functions
// ------------------------------------------------------------------

*/

/*
   addition of two 256 bit number modulo p Z[i] = X[i] + Y[i] (mod p)

   Input vector contains intercalated X, Y and Z numbers (X[0], Y[0], Z[0], X[1], Y[1], Z[1],..
    X[N-1], Y[N-1], Z[N-1]) where X, Y and Z are 256 bit numbers represented as an array of uint32_t
   
*/

#include <stdio.h>

#include "types.h"
#include "u256_device.h"


/*
    256 bit addition kernel

    Arguments :
      in_vector : Input vector of up to N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      len : number of elements in output vector to be xferred. 
          Cannot be greater than half amount reseved during constructor, but not checked
*/
__global__ void addu256_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    const uint32_t *x;
    const uint32_t *y;
    uint32_t * z;
 
    if(tid >= len) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + XOFFSET];
    y = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + YOFFSET * NWORDS_256BIT];
    z = (uint32_t *) &out_vector[tid * NWORDS_256BIT];
    
    addu256(x, y, z);

    return;
}

/*
    256 bit sub kernel

    Arguments :
      in_vector : Input vector of up to N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      len : number of elements in output vector to be xferred. 
          Cannot be greater than half amount reseved during constructor, but not checked
*/
__global__ void subu256_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    const uint32_t *x;
    const uint32_t *y;
    uint32_t * z;
 
    if(tid >= len) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + XOFFSET];
    y = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + YOFFSET * NWORDS_256BIT];
    z = (uint32_t *) &out_vector[tid * NWORDS_256BIT];
    
    subu256(x, y, z);

    return;
}

/*
    Modular addition kernel

    Arguments :
      in_vector : Input vector of up to N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      p : 256 bit module in 8 word uint32 array
      len : number of elements in output vector to be xferred. 
          Cannot be greater than half amount reseved during constructor, but not checked
*/
__global__ void addmu256_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    const uint32_t *x;
    const uint32_t *y;
    uint32_t * z;
 
    if(tid >= len) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + XOFFSET];
    y = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + YOFFSET * NWORDS_256BIT];
    z = (uint32_t *) &out_vector[tid * NWORDS_256BIT];
    
    addmu256(x, y, z, p);
}

/*
    Modular Sub kernel

    Arguments :
      in_vector : Input vector of up to N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      p : 256 bit module in 8 word uint32 array
      len : number of elements in output vector to be xferred. 
          Cannot be greater than half amount reseved during constructor, but not checked
*/
__global__ void submu256_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    const uint32_t *x;
    const uint32_t *y;
    uint32_t * z;
 
    if(tid >= len) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + XOFFSET];
    y = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + YOFFSET * NWORDS_256BIT];
    z = (uint32_t *) &out_vector[tid * NWORDS_256BIT];
    
    submu256(x, y, z, p);
}

/*
    Modulo

    Arguments :
      in_vector : Input vector of up to N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      p : 256 bit module in 8 word uint32 array
      len : number of elements in output vector to be xferred. 
          Cannot be greater than half amount reseved during constructor, but not checked
*/
__global__ void modu256_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    const uint32_t *x;
    uint32_t * z;
 
    if(tid >= len) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * NWORDS_256BIT];
    z = (uint32_t *) &out_vector[tid * NWORDS_256BIT];
    
    modu256(x, z, p);
}


__global__ void mulmontu256_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector)
{
    return;
}

__forceinline__ __device__ void addu256(const uint32_t *x, const uint32_t *y, uint32_t *z)
{
  // z[i] = x[i] + y[i] for 8x32 bit words
  asm("add.cc.u32        %0, %8,  %9;\n\t"              // sum with carry out
      "addc.cc.u32       %1, %10, %11;\n\t"             // sum with carry in and carry out
      "addc.cc.u32       %2, %12, %13;\n\t"
      "addc.cc.u32       %3, %14, %15;\n\t"
      "addc.cc.u32       %4, %16, %17;\n\t"
      "addc.cc.u32       %5, %18, %19;\n\t"
      "addc.cc.u32       %6, %20, %21;\n\t"
      "addc.u32          %7, %22, %23;\n\t"            // sum with carry in
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(x[0]), "r"(y[0]), "r"(x[1]), "r"(y[1]),
        "r"(x[2]), "r"(y[2]), "r"(x[3]), "r"(y[3]),
        "r"(x[4]), "r"(y[4]), "r"(x[5]), "r"(y[5]),
        "r"(x[6]), "r"(y[6]), "r"(x[7]), "r"(y[7]));
}

__forceinline__ __device__ void subu256(const uint32_t *x, const uint32_t *y, uint32_t *z)
{
  // z[i] = x[i] - y[i] for 8x32 bit words
  asm("sub.cc.u32        %0, %8,  %9;\n\t"              // sub with borrow out
      "subc.cc.u32       %1, %10, %11;\n\t"             // sub with borrow out and borrow in
      "subc.cc.u32       %2, %12, %13;\n\t"
      "subc.cc.u32       %3, %14, %15;\n\t"
      "subc.cc.u32       %4, %16, %17;\n\t"
      "subc.cc.u32       %5, %18, %19;\n\t"
      "subc.cc.u32       %6, %20, %21;\n\t"
      "subc.u32          %7, %22, %23;\n\t"            // sum with carry in
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(x[0]), "r"(y[0]), "r"(x[1]), "r"(y[1]),
        "r"(x[2]), "r"(y[2]), "r"(x[3]), "r"(y[3]),
        "r"(x[4]), "r"(y[4]), "r"(x[5]), "r"(y[5]),
        "r"(x[6]), "r"(y[6]), "r"(x[7]), "r"(y[7]));

}


__forceinline__ __device__ void addmu256(const uint32_t *x, const uint32_t *y, uint32_t *z, const uint32_t *p)
{
  uint32_t do_modf;
  uint32_t z_tmp[NWORDS_256BIT];

  // z[i] = x[i] + y[i] 
  addu256(x, y, z);

  // z_tmp[i] = z[i] - p[i]
  subu256(z, p, z_tmp);
  
  // do_modf = most significant bit of z_tmp is 1
  asm("bfe.u32	%0, %1, 31, 1;\n\t"              
      : "=r"(do_modf)
      : "r"(z_tmp[7]));

  // if do_modf, return z_tmp. Else, return <
  if (do_modf){
     asm("mov.u32     %0,  %8;\n\t"
         "mov.u32     %1,  %9;\n\t"
         "mov.u32     %2,  %10;\n\t"
         "mov.u32     %3,  %11;\n\t"
         "mov.u32     %4,  %12;\n\t"
         "mov.u32     %5,  %13;\n\t"
         "mov.u32     %6,  %14;\n\t"
         "mov.u32     %7,  %15;\n\t"
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(z_tmp[0]), "r"(z_tmp[1]), "r"(z_tmp[2]), "r"(z_tmp[3]),
        "r"(z_tmp[4]), "r"(z_tmp[5]), "r"(z_tmp[6]), "r"(z_tmp[7]));
  }    
  
}
__forceinline__ __device__ void submu256(const uint32_t *x, const uint32_t *y, uint32_t *z, const uint32_t *p)
{
  uint32_t do_modf;
  uint32_t z_tmp[NWORDS_256BIT];

  // z[i] = x[i] - y[i] 
  subu256(x, y, z);

  // z_tmp[i] = z[i] + p[i]
  addu256(z, p, z_tmp);
  
  // do_modf = most significant bit of z_tmp is 1
  asm("bfe.u32	%0, %1, 31, 1;\n\t"              
      : "=r"(do_modf)
      : "r"(z_tmp[7]));

  // if do_modf, return z_tmp. Else, return <
  if (do_modf){
     asm("mov.u32     %0,  %8;\n\t"
         "mov.u32     %1,  %9;\n\t"
         "mov.u32     %2,  %10;\n\t"
         "mov.u32     %3,  %11;\n\t"
         "mov.u32     %4,  %12;\n\t"
         "mov.u32     %5,  %13;\n\t"
         "mov.u32     %6,  %14;\n\t"
         "mov.u32     %7,  %15;\n\t"
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(z_tmp[0]), "r"(z_tmp[1]), "r"(z_tmp[2]), "r"(z_tmp[3]),
        "r"(z_tmp[4]), "r"(z_tmp[5]), "r"(z_tmp[6]), "r"(z_tmp[7]));
  }    
  
}

__forceinline__ __device__ void modu256(const uint32_t *x, uint32_t *z, const uint32_t *p)
{
  uint32_t do_modf;
  uint32_t z_tmp[NWORDS_256BIT];

  // z_tmp[i] = z[i] + p[i]
  addu256(x, p, z_tmp);
  
  // do_modf = most significant bit of z_tmp is 1
  asm("bfe.u32	%0, %1, 31, 1;\n\t"              
      : "=r"(do_modf)
      : "r"(z_tmp[7]));

  // if do_modf, return z_tmp. Else, return <
  if (do_modf){
     asm("mov.u32     %0,  %8;\n\t"
         "mov.u32     %1,  %9;\n\t"
         "mov.u32     %2,  %10;\n\t"
         "mov.u32     %3,  %11;\n\t"
         "mov.u32     %4,  %12;\n\t"
         "mov.u32     %5,  %13;\n\t"
         "mov.u32     %6,  %14;\n\t"
         "mov.u32     %7,  %15;\n\t"
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(z_tmp[0]), "r"(z_tmp[1]), "r"(z_tmp[2]), "r"(z_tmp[3]),
        "r"(z_tmp[4]), "r"(z_tmp[5]), "r"(z_tmp[6]), "r"(z_tmp[7]));
  }    
  
}



/*
   (z,c) = x * y + a
   z is 2 x uint32_t
   c is carry
*/
__forceinline__ __device__ void madcu32(uint32_t x, uint32_t y, uint32_t a, uint32_t *z, uint32_t *c)
{

   asm(".reg .u64      %prod;                 \n\t"
       ".reg .u64      %sum;                  \n\t"
       ".cvt.u64.u32   %sum   %4;             \n\t"
       "mad.wide.u32   %prod, %2,    %3, %sum;\n\t"
       "cvt.u32.u64    %0,    %prod;          \n\t"
       "shr.u64        %prod, %prod, 32;      \n\t"
       "cvt.u32.u64    %1 %prod;              \n\t"
       : "=r"(z[0]), "=r"(z[1]) 
       : "r"(x), "r"(y), "r"(a));

   // (C,S) = t[0] + a[0] * b[i] -> No carry in
   asm("mad.lo.cc.u32  %0, %3, %4, %5;      \n\t"
       "madc.hi.cc.u32 %1, %3, %4, 0;       \n\t"
       "addc.u32       %tmp, %1, 0;         \n\t"
       "set.lt.u32     %2, %1, %tmp;        \n\t"
       : "=r"(z[0]), "=r"(z[1]), "=r"(c[0]) 
       : "r"(x), "r"(y), "r"(a));
}

__forceinline__ __device__ void propcu32(uint32_t *x, uint32_t *c)
{
   asm("move.u32    %cin, %3;  \n\t"
       "move.u32    %tmp, %2;  \n\t"
       "add.u32     %0, %cin"
       "set.lt.u32  %1, %0, %tmp;        \n\t"
       : "=r"(x[0]), "=r"(c[0]) 
       : "r"(x[0]), "r"(c[0]));
}

