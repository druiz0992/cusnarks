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
// File name  : u256_device.h
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of U256 integer arithmetic
// ------------------------------------------------------------------

*/
#ifndef _U256_DEVICE_H_
#define _U256_DEVICE_H_

#define U256_MAX_SMALLK (32)

#include "log.h"

typedef struct{
  uint32_t *x0;
  uint32_t *x1;
  uint32_t *x2;
  uint32_t *x3;
  uint32_t *x4;
}inv_t;


__global__ void addmu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void addmu256_reduce_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void addmu256_reduce_shfl_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void submu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void modu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void mulmontu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void mulmontu256_2_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void shr1u256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void shl1u256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void shlu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);
__global__ void almmontinvu256_kernel(uint32_t *out_vector, uint32_t *in_vector, kernel_params_t *params);

__forceinline__ __device__ uint32_t subgtu256(uint32_t __restrict__ *x, const uint32_t __restrict__ *y);
__forceinline__ __device__ void shflxoru256(uint32_t *d_out, uint32_t *d_in, uint32_t srcLane );
__forceinline__ __device__ void almmontinv_step_h(inv_t *table);
__forceinline__ __device__ void init_invtable(inv_t *data_table, uint32_t *u, uint32_t *v, uint32_t *s, uint32_t *r1);

extern __device__ void modu256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, mod_t midx);
extern __device__ void modu255(uint32_t __restrict__ *z, const uint32_t __restrict__ *x, mod_t midx);
extern __device__ void mulmontu256(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, const uint32_t __restrict__ *B, mod_t midx);
extern __device__ void sqmontu256(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, mod_t midx);
extern __device__ void mulmontu256_2(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, const uint32_t __restrict__ *B, mod_t midx);
extern __device__ void mulmontu256_2_asm(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, const uint32_t __restrict__ *B, mod_t midx);
extern __device__ void sqmontu256_2(uint32_t __restrict__ *U, const uint32_t __restrict__ *A, mod_t midx);
extern __device__ uint32_t almmontinvu256(uint32_t __restrict__ *y, const uint32_t __restrict__ *x, mod_t midx);
extern __device__ uint32_t invmontu256(uint32_t __restrict__ *y, const uint32_t __restrict__ *x, mod_t midx);
extern __device__ uint32_t invmontu256_2(uint32_t __restrict__ *y, const uint32_t __restrict__ *x, mod_t midx);
extern __device__ uint32_t shr1u256(uint32_t __restrict__ *x);
extern __device__ uint32_t shl1u256(uint32_t __restrict__ *x);
extern __device__ void shlu256(uint32_t *x, uint32_t count);
extern __device__ uint32_t bselMu256(const uint32_t __restrict__ *x, uint32_t bsel);
extern __device__ uint32_t bselu256(const uint32_t __restrict__ *x, uint32_t bsel);
extern __device__ uint32_t clzu256(const uint32_t __restrict__ *x);
extern __device__ uint32_t clzMu256(const uint32_t __restrict__ *x);
extern __device__ void div2u256(uint32_t __restrict__ *z, const uint32_t __restrict__ *x);


// Implementation of tamplate functions, and functions used by template functions.

/*
  z = x + y, x and y are 256 bit numbers
*/
template <typename T1, typename T2, typename T3>
__forceinline__ __device__ void addu256(T1 *z, T2 *x, T3 *y)
{
  // z[i] = x[i] + y[i] for 8x32 bit words
  asm ("{                             \n\t"
      ".reg .u32         %x_;        \n\t"
      ".reg .u32         %y_;        \n\t"
      "mov.u32           %x_,%8;     \n\t"
      "mov.u32           %y_,%9;     \n\t"
      "add.cc.u32        %0, %x_, %y_;\n\t"             // sum with carry out
      "mov.u32           %x_,%10;     \n\t"
      "mov.u32           %y_,%11;     \n\t"
      "addc.cc.u32       %1, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%12;     \n\t"
      "mov.u32           %y_,%13;     \n\t"
      "addc.cc.u32       %2, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%14;     \n\t"
      "mov.u32           %y_,%15;     \n\t"
      "addc.cc.u32       %3, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%16;     \n\t"
      "mov.u32           %y_,%17;     \n\t"
      "addc.cc.u32       %4, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%18;     \n\t"
      "mov.u32           %y_,%19;     \n\t"
      "addc.cc.u32       %5, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%20;     \n\t"
      "mov.u32           %y_,%21;     \n\t"
      "addc.cc.u32       %6, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%22;     \n\t"
      "mov.u32           %y_,%23;     \n\t"
      "addc.u32          %7, %x_, %y_;\n\t"             // sum with carry in 
      "}                             \n\n"
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(x[0]), "r"(y[0]), "r"(x[1]), "r"(y[1]),
        "r"(x[2]), "r"(y[2]), "r"(x[3]), "r"(y[3]),
        "r"(x[4]), "r"(y[4]), "r"(x[5]), "r"(y[5]),
        "r"(x[6]), "r"(y[6]), "r"(x[7]), "r"(y[7]));
}
__forceinline__ __device__ void addu256(uint32_t *z, const uint32_t *x, const uint32_t *y)
{
  // z[i] = x[i] + y[i] for 8x32 bit words
  asm ("{                             \n\t"
      ".reg .u32         %x_;        \n\t"
      ".reg .u32         %y_;        \n\t"
      "mov.u32           %x_,%8;     \n\t"
      "mov.u32           %y_,%9;     \n\t"
      "add.cc.u32        %0, %x_, %y_;\n\t"             // sum with carry out
      "mov.u32           %x_,%10;     \n\t"
      "mov.u32           %y_,%11;     \n\t"
      "addc.cc.u32       %1, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%12;     \n\t"
      "mov.u32           %y_,%13;     \n\t"
      "addc.cc.u32       %2, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%14;     \n\t"
      "mov.u32           %y_,%15;     \n\t"
      "addc.cc.u32       %3, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%16;     \n\t"
      "mov.u32           %y_,%17;     \n\t"
      "addc.cc.u32       %4, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%18;     \n\t"
      "mov.u32           %y_,%19;     \n\t"
      "addc.cc.u32       %5, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%20;     \n\t"
      "mov.u32           %y_,%21;     \n\t"
      "addc.cc.u32       %6, %x_, %y_;\n\t"             // sum with carry in and carry out
      "mov.u32           %x_,%22;     \n\t"
      "mov.u32           %y_,%23;     \n\t"
      "addc.u32          %7, %x_, %y_;\n\t"             // sum with carry in 
      "}                             \n\n"
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(x[0]), "r"(y[0]), "r"(x[1]), "r"(y[1]),
        "r"(x[2]), "r"(y[2]), "r"(x[3]), "r"(y[3]),
        "r"(x[4]), "r"(y[4]), "r"(x[5]), "r"(y[5]),
        "r"(x[6]), "r"(y[6]), "r"(x[7]), "r"(y[7]));
}

/*
  z = x - y, x and y are 256 bit numbers
*/
template <typename T1, typename T2, typename T3>
__forceinline__ __device__ void subu256(T1 *z, T2 *x, T3 *y)
{
  // z[i] = x[i] - y[i] for 8x32 bit words
  asm ("{                             \n\t"
      ".reg .u32          %x_;        \n\t"
      ".reg .u32          %y_;        \n\t"
      "mov.u32           %x_,%8;     \n\t"
      "mov.u32           %y_,%9;     \n\t"
      "sub.cc.u32        %0, %x_, %y_;\n\t"             // sum with borrow out
      "mov.u32           %x_,%10;     \n\t"
      "mov.u32           %y_,%11;     \n\t"
      "subc.cc.u32       %1, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%12;     \n\t"
      "mov.u32           %y_,%13;     \n\t"
      "subc.cc.u32       %2, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%14;     \n\t"
      "mov.u32           %y_,%15;     \n\t"
      "subc.cc.u32       %3, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%16;     \n\t"
      "mov.u32           %y_,%17;     \n\t"
      "subc.cc.u32       %4, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%18;     \n\t"
      "mov.u32           %y_,%19;     \n\t"
      "subc.cc.u32       %5, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%20;     \n\t"
      "mov.u32           %y_,%21;     \n\t"
      "subc.cc.u32       %6, %x_, %y_;\n\t"             // sum with borrow in and borrow out
      "mov.u32           %x_,%22;     \n\t"
      "mov.u32           %y_,%23;     \n\t"
      "subc.u32          %7, %x_, %y_;\n\t"             // sum with borrow in 
      "}                             \n\t"
      : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
        "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
      : "r"(x[0]), "r"(y[0]), "r"(x[1]), "r"(y[1]),
        "r"(x[2]), "r"(y[2]), "r"(x[3]), "r"(y[3]),
        "r"(x[4]), "r"(y[4]), "r"(x[5]), "r"(y[5]),
        "r"(x[6]), "r"(y[6]), "r"(x[7]), "r"(y[7]));

}

/*
   x == 0 for 256 bit numbers
*/
template <typename T1>
__forceinline__ __device__ uint32_t eq0u256(T1 *x)
{
 #if 0
  if (x[0] == 0 && x[1] ==  0 && x[2] == 0 && x[3] == 0 && x[4] == 0 && x[5] == 0 && x[6] == 0 && x[7] == 0){
    return 1;
  } else { 
    return 0;
  }
 #else
  ulonglong4 *x4 = (ulonglong4 *)x;
  if (x4->x == 0 && x4->y == 0 && x4->z==0 && x4->w == 0){
     return 1;
  } else {
     return 0;
  }
  #endif
  
}

/*
   x == 1 for 256 bit numbers
*/
template <typename T1>
__forceinline__ __device__ uint32_t eq1u256(T1 *x)
{
 #if 1
  if (x[0] == 1 && x[1] ==  0 && x[2] == 0 && x[3] == 0 && x[4] == 0 && x[5] == 0 && x[6] == 0 && x[7] == 0){
    return 1;
  } else { 
    return 0;
  }
 #else
  ulonglong4 *x4 = (ulonglong4 *)x;
  if (x4->x == 1 && x4->y == 0 && x4->z==0 && x4->w == 0){
     return 1;
  } else {
     return 0;
  }
  #endif
}  

/*
   x <= y

   NOTE : The procedure is :
    z = y - x
    r = number of leading zeros in z
    if r == 0 return false (x >= y). 
    else return true (x <= y)
*/
template <typename T1, typename T2>
__forceinline__ __device__ uint32_t lteu256(T1 *x, T2 *y)
{
   uint32_t z[NWORDS_256BIT];
   uint32_t r, flag;

   subu256(z,y,x); 

   asm("clz.b32    %0,%1;\n\t"
       :"=r"(r) : "r"(z[NWORDS_256BIT-1]));
   flag = r > 0;
   return r;

}

/*
   x < y

   NOTE : The procedure is :
    z = y - x
    r = number of leading zeros in z
    if z is 0 or r == 0, return false (x >= y). 
    else return true (x < y)
*/
template <typename T1, typename T2>
__forceinline__ __device__ uint32_t ltu256(T1 *x, T2 *y)
{
   uint32_t z[NWORDS_256BIT];
   uint32_t r, flag;

   subu256(z,y,x); 

   asm("clz.b32    %0,%1;\n\t"
       :"=r"(r) : "r"(z[NWORDS_256BIT-1]));
   flag = r > 0;
   if ((r == 32) && eq0u256(z)){
     flag = 0;
   }
   return r;

}

__forceinline__ __device__ void negu256(uint32_t *d_out, uint32_t *d_in)
{
    uint32_t carry;
    ulonglong4 *out = (ulonglong4 *)d_out;
    ulonglong4 *in =  (ulonglong4 *)d_in;

    out->w = -in->w+1;
    asm("clz.b32    %0,%1;\n\t"
       :"=r"(carry) : "r"(d_in[1]));
    
    out->z = -in->z+(carry == 0);
    asm("clz.b32    %0,%1;\n\t"
       :"=r"(carry) : "r"(d_in[3]));

    out->y = -in->y+(carry == 0);
    asm("clz.b32    %0,%1;\n\t"
       :"=r"(carry) : "r"(d_in[5]));

    out->x = -in->x+(carry==0);
}
__forceinline__ __device__ void movu256(uint32_t *d_out, uint32_t *d_in)
{
  #if 0
   asm("mov.u32     %0,  %8;\n\t"
       "mov.u32     %1,  %9;\n\t"
       "mov.u32     %2,  %10;\n\t"
       "mov.u32     %3,  %11;\n\t"
       "mov.u32     %4,  %12;\n\t"
       "mov.u32     %5,  %13;\n\t"
       "mov.u32     %6,  %14;\n\t"
       "mov.u32     %7,  %15;\n\t"
    : "=r"(d_out[0]), "=r"(d_out[1]), "=r"(d_out[2]), "=r"(d_out[3]),
      "=r"(d_out[4]), "=r"(d_out[5]), "=r"(d_out[6]), "=r"(d_out[7])
    : "r"(d_in[0]), "r"(d_in[1]), "r"(d_in[2]), "r"(d_in[3]),
      "r"(d_in[4]), "r"(d_in[5]), "r"(d_in[6]), "r"(d_in[7]));
  #endif

    ulonglong4 *out = (ulonglong4 *)d_out;
    ulonglong4 *in =  (ulonglong4 *)d_in;
    out->x = in->x;
    out->y = in->y;
    out->z = in->z;
    out->w = in->w;
    /*
    uint4 *out = (uint4 *)d_out;
    uint4 *in =  (uint4 *)d_in;
    out->x = in->x;
    out->y = in->y;
    out->z = in->z;
    out->w = in->w;
    */
}

__forceinline__ __device__ void movu256x6(uint32_t *d_out, uint32_t *d_in)
{
    uint32_t i;
    ulonglong4 *out = (ulonglong4 *)d_out;
    ulonglong4 *in =  (ulonglong4 *)d_in;

    #pragma unroll
    for(i=0; i < 6; i++){
      out->x = in->x;
      out->y = in->y;
      out->z = in->z;
      out->w = in->w;
      out++;
      in++;
    }
}

__forceinline__ __device__ void movu256x2(uint32_t *d_out, uint32_t *d_in)
{
    uint32_t i;
    ulonglong4 *out = (ulonglong4 *)d_out;
    ulonglong4 *in =  (ulonglong4 *)d_in;

    #pragma unroll
    for(i=0; i < 2; i++){
      out->x = in->x;
      out->y = in->y;
      out->z = in->z;
      out->w = in->w;
      out++;
      in++;
    }
}

__forceinline__ __device__ void set0u256(uint32_t *d_out)
{
  #if 0
   asm("mov.u32     %0,  %8;\n\t"
       "mov.u32     %1,  %9;\n\t"
       "mov.u32     %2,  %10;\n\t"
       "mov.u32     %3,  %11;\n\t"
       "mov.u32     %4,  %12;\n\t"
       "mov.u32     %5,  %13;\n\t"
       "mov.u32     %6,  %14;\n\t"
       "mov.u32     %7,  %15;\n\t"
    : "=r"(d_out[0]), "=r"(d_out[1]), "=r"(d_out[2]), "=r"(d_out[3]),
      "=r"(d_out[4]), "=r"(d_out[5]), "=r"(d_out[6]), "=r"(d_out[7])
    : "r"(d_in[0]), "r"(d_in[1]), "r"(d_in[2]), "r"(d_in[3]),
      "r"(d_in[4]), "r"(d_in[5]), "r"(d_in[6]), "r"(d_in[7]));
  #endif

    ulonglong4 *out = (ulonglong4 *)d_out;
    out->x = 0;
    out->y = 0;
    out->z = 0;
    out->w = 0;
}

template <typename T1, typename T2, typename T3>
__device__ void addmu256(T1 *z, T2 *x, T3 *y, mod_t midx)
{
  uint32_t const __restrict__ *p = mod_info_ct[midx].p;

   addu256(z,x,y);
    if (ltu256(p,z)){
      subu256(z,z,p);
   } 

}


/*
   x - y (mod N) for 256 bit numbers

   NOTE. Function requires that x, y < N
*/

template <typename T1, typename T2, typename T3>
__device__ void submu256(T1 *z, T2 *x, T3 *y, mod_t midx)
{

  uint32_t const __restrict__ *p = mod_info_ct[midx].p;

  //logInfoBigNumberTid(1,"y:\n",(uint32_t *)y);
  subu256(z,x,y);
  if (z[NWORDS_256BIT-1] > p[NWORDS_256BIT-1]){
      addu256(z, z, p);
  } 


}

#endif
