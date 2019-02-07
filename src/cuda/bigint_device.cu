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
// File name  : bigint_kernel.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of biginteger kernel and device functions
// ------------------------------------------------------------------

*/

/*
   addition of two 256 bit number modulo p Z[i] = X[i] + Y[i] (mod p)

   Input vector contains intercalated X, Y and Z numbers (X[0], Y[0], Z[0], X[1], Y[1], Z[1],..
    X[N-1], Y[N-1], Z[N-1]) where X, Y and Z are 256 bit numbers represented as an array of uint32_t
   
*/

#include <stdio.h>

#include "types.h"
#include "bigint_device.h"

/*
    Modular addition kernel

    Arguments :
      in_vector : Input vector of up to N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      p : 256 bit module in 8 word uint32 array
      len : number of elements in output vector to be xferred. 
          Cannot be greater than half amount reseved during constructor, but not checked
*/
__global__ void addm_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    const uint32_t *x;
    const uint32_t *y;
    uint32_t * z;
    uint32_t c = 0;
    uint32_t i;
 
    if(tid >= len) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + XOFFSET];
    y = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + YOFFSET];
    z = (uint32_t *) &out_vector[tid * NWORDS_256BIT];
    
    uaddm256(x, y, z, p);

    return;
}

__global__ void montmul_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector)
{
    return;
}

/*
   (z,c) = x * y + a
   z is 2 x uint32_t
   c is carry
*/
__forceinline__ __device__ void umadc32(uint32_t x, uint32_t y, uint32_t a, uint32_t *z, uint32_t *c)
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

__forceinline__ __device__ void ucprop32(uint32_t *x, uint32_t *c)
{
   asm("move.u32    %cin, %3;  \n\t"
       "move.u32    %tmp, %2;  \n\t"
       "add.u32     %0, %cin"
       "set.lt.u32  %1, %0, %tmp;        \n\t"
       : "=r"(x[0]), "=r"(c[0]) 
       : "r"(x[0]), "r"(c[0]));
}
__forceinline__ __device__ void uadd256(const uint32_t *x, const uint32_t *y, uint32_t *z)
{
  // z[i] = x[i] + y[i] for 8x32 bit words
  asm("add.cc.u32        %0, %8, %12;\n\t"              // sum with carry out
      "addc.cc.u32       %1, %9,  %13;\n\t"             // sum with carry in and carry out
      "addc.cc.u32       %2, %10, %14;\n\t"
      "addc.cc.u32       %3, %11, %15;\n\t"
      "addc.cc.u32       %4, %12, %16;\n\t"
      "addc.cc.u32       %5, %13, %17;\n\t"
      "addc.cc.u32       %6, %14, %18;\n\t"
      "addc.u32          %7, %15, %19;\n\t"            // sum with carry in
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(x[0]), "r"(y[0]), "r"(x[1]), "r"(y[1]),
        "r"(x[2]), "r"(y[2]), "r"(x[3]), "r"(y[3]),
        "r"(x[4]), "r"(y[4]), "r"(x[5]), "r"(y[5]),
        "r"(x[6]), "r"(y[6]), "r"(x[7]), "r"(y[7]));
}

__forceinline__ __device__ void uaddm256(const uint32_t *x, const uint32_t *y, uint32_t *z, const uint32_t *p)
{
  uint32_t do_modf;
  uint32_t z_tmp[NWORDS_256BIT];

  // z[i] = x[i] + y[i] 
  uadd256(x, y, z);

  // z_tmp[i] = z[i] - p[i]
  usub256(z, p, z_tmp);
  
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

__forceinline__ __device__ void usub256(const uint32_t *x, const uint32_t *y, uint32_t *z)
{
  // z[i] = x[i] - y[i] for 8x32 bit words
  asm("sub.cc.u32        %0, %8, %12;\n\t"              // sub with borrow out
      "subc.cc.u32       %1, %9,  %13;\n\t"             // sub with borrow out and borrow in
      "subc.cc.u32       %2, %10, %14;\n\t"
      "subc.cc.u32       %3, %11, %15;\n\t"
      "subc.cc.u32       %4, %12, %16;\n\t"
      "subc.cc.u32       %5, %13, %17;\n\t"
      "subc.cc.u32       %6, %14, %18;\n\t"
      "subc.u32          %7, %15, %19;\n\t"            // sum with carry in
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(x[0]), "r"(y[0]), "r"(x[1]), "r"(y[1]),
        "r"(x[2]), "r"(y[2]), "r"(x[3]), "r"(y[3]),
        "r"(x[4]), "r"(y[4]), "r"(x[5]), "r"(y[5]),
        "r"(x[6]), "r"(y[6]), "r"(x[7]), "r"(y[7]));

}


/*
__global__ void ciosV2(KernelArray<unsigned int>d_a1, KernelArray<unsigned int>d_b1, KernelArray<unsigned int>d_ans, KernelArray<unsigned int>d_n, KernelArray<unsigned int>d_n1, int d_s, int blkSize)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int t[33] = { 0 };
    unsigned long long temp;
    __shared__ unsigned int shared_n[32], shared_n1[32], shared_s;
    shared_s = d_s;
    for (int i = 0; i < shared_s; i++){
        shared_n[i] = d_n._array[i];
        shared_n1[i] = d_n1._array[i];
    }
    __syncthreads();
    for (int i = 0; i < shared_s; i++){
        unsigned long long c = 0;
        for (int j = 0; j < shared_s; j++){
            temp = t[j] + (unsigned long long)d_a1._array[j * (1024 *
            blkSize) + idx] * (unsigned long long)d_b1._array[i * (1024 * blkSize) + idx] +
            c;
            t[j] = temp & 4294967295;
            c = temp >> 32;
        }
        temp = (unsigned long long)t[shared_s] + c;
        t[shared_s] = temp & 4294967295;
        t[shared_s + 1] = temp >> 32;
        unsigned long long m = ((unsigned long long)t[0] * (unsigned long
        long)shared_n1[0]) & 4294967295;
        temp = (unsigned long long)t[0] + m*(unsigned long
        long)shared_n[0];
        c = temp >> 32;
        for (int j = 1; j < shared_s; j++){
            temp = (unsigned long long)t[j] + m*(unsigned long
            long)shared_n[j] + c;
            t[j - 1] = temp & 4294967295;
            c = temp >> 32;
        }
        temp = (unsigned long long)t[shared_s] + c;
        t[shared_s - 1] = temp & 4294967295;
        c = temp >> 32;
        t[shared_s] = t[shared_s + 1] + c;
    }
    unsigned int u[33];
    for (int j = 0; j < shared_s + 1; j++){
        u[j] = t[j];
    }
    int b = 0;
    long long sub;
    for (int i = 0; i < shared_s; i++){
        sub = (long long)u[i] - shared_n[i] - b;
        if (sub < 0){
            t[i] = sub + 4294967296;
            b = 1;
        }
        else{
            t[i] = sub;
            b = 0;
        }
    }
    sub = (long long)u[shared_s] - b;
    u[shared_s] = sub;
    if (sub >= 0){
        int counter = 0;
        for (int i = 0; i < 32; i++){
            d_ans._array[i * 1024 * blkSize + idx] = t[counter++];
        }
    }
    else{
        int counter = 0;
        for (int i = 0; i < 32; i++){
            d_ans._array[i * 1024 * blkSize + idx] = u[counter++];
        }
    }
}

__global__ void BigInt_MontMul(KernelArray<unsigned int>d_a1, KernelArray<unsigned int>d_b1, KernelArray<unsigned int>d_ans, KernelArray<unsigned int>d_n, KernelArray<unsigned int>d_n1, int d_s, int blkSize)
{
    for (int i = 0; i < shared_s; i++){
        unsigned long long c = 0;
        for (int j = 0; j < shared_s; j++){
            temp = t[j] + (unsigned long long)d_a1._array[j * (1024 *
            blkSize) + idx] * (unsigned long long)d_b1._array[i * (1024 * blkSize) + idx] +
            c;
            t[j] = temp & 4294967295;
            c = temp >> 32;
        }
        temp = (unsigned long long)t[shared_s] + c;
        t[shared_s] = temp & 4294967295;
        t[shared_s + 1] = temp >> 32;
        unsigned long long m = ((unsigned long long)t[0] * (unsigned long
        long)shared_n1[0]) & 4294967295;
        temp = (unsigned long long)t[0] + m*(unsigned long
        long)shared_n[0];
        c = temp >> 32;
        for (int j = 1; j < shared_s; j++){
            temp = (unsigned long long)t[j] + m*(unsigned long
            long)shared_n[j] + c;
            t[j - 1] = temp & 4294967295;
            c = temp >> 32;
        }
        temp = (unsigned long long)t[shared_s] + c;
        t[shared_s - 1] = temp & 4294967295;
        c = temp >> 32;
        t[shared_s] = t[shared_s + 1] + c;
    }
    unsigned int u[33];
    for (int j = 0; j < shared_s + 1; j++){
        u[j] = t[j];
    }
    int b = 0;
    long long sub;
    for (int i = 0; i < shared_s; i++){
        sub = (long long)u[i] - shared_n[i] - b;
        if (sub < 0){
            t[i] = sub + 4294967296;
            b = 1;
        }
        else{
            t[i] = sub;
            b = 0;
        }
    }
    sub = (long long)u[shared_s] - b;
    u[shared_s] = sub;
    if (sub >= 0){
        int counter = 0;
        for (int i = 0; i < 32; i++){
            d_ans._array[i * 1024 * blkSize + idx] = t[counter++];
        }
    }
    else{
        int counter = 0;
        for (int i = 0; i < 32; i++){
            d_ans._array[i * 1024 * blkSize + idx] = u[counter++];
        }
    }
}

*/
