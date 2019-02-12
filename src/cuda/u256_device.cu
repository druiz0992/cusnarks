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

#include <stdio.h>

#include "types.h"
#include "cuda.h"
#include "u256_device.h"

__device__ void printNumber(uint32_t *n);
/*
    Modular addition kernel

    Arguments :
      in_vector : Input vector of up to N 256 bit elements X[0], X[1], X[2] ... X[N-1].
      out_vector : Results of addition operation Y[0] = X[0] + X[1] mod p, Y[1] = X[2] + X[3] mod p...
      p : 256 bit module in 8 word uint32 array
      len : number of elements in output vector to be xferred. 
          Cannot be greater than half amount reseved during constructor, but not checked
*/
__global__ void addmu256_kernel(uint32_t *out_vector, uint32_t *in_vector, const uint32_t *p, uint32_t len, uint32_t premod)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t *x;
    uint32_t *y;
    uint32_t *z;
 
    if(tid >= len/2) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * 2 * U256K_OFFSET + U256_XOFFSET];
    y = (uint32_t *) &in_vector[tid * 2 * U256K_OFFSET + U256_YOFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    
    if (premod){
      modu256(x,x,p);
      modu256(y,y,p);
    }

    addmu256(z,(const uint32_t *)x, (const uint32_t *)y, p);
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
__global__ void submu256_kernel(uint32_t *out_vector, uint32_t *in_vector, const uint32_t *p, uint32_t len, uint32_t premod)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t *x;
    uint32_t *y;
    uint32_t * z;
 
    if(tid >= len/2) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * 2 * U256K_OFFSET + U256_XOFFSET];
    y = (uint32_t *) &in_vector[tid * 2 * U256K_OFFSET + U256_YOFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    
    if (premod){
      modu256(x,x,p);
      modu256(y,y,p);
    }

    submu256(z,(const uint32_t *)x, (const uint32_t *)y, p);
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
__global__ void modu256_kernel(uint32_t *out_vector, const uint32_t *in_vector, const uint32_t *p, uint32_t len)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    const uint32_t *x;
    uint32_t * z;
 
    if(tid >= len) {
      return;
    }

    x = (const uint32_t *) &in_vector[tid * U256K_OFFSET];
    z = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    
    modu256(z, x, p);
}


__global__ void mulmontu256_kernel(uint32_t *out_vector, uint32_t *in_vector, const uint32_t *p,  const uint32_t np, uint32_t len, uint32_t premod)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid >= len/2) {
      return;
    }

    uint32_t *A, *B, *U;
    const uint32_t *P;
    uint32_t i,j, NP;
 
    A = (uint32_t *) &in_vector[tid * 2 * U256K_OFFSET + U256_XOFFSET];
    B = (uint32_t *) &in_vector[tid * 2 * U256K_OFFSET + U256_YOFFSET];
    U = (uint32_t *) &out_vector[tid * U256K_OFFSET];
    P = p;
    NP = np;
   
    // ensure A, B < p 
    if (premod){
      modu256(A,A,p);
      modu256(B,B,p);
    }

    mulmontu256(U, (const uint32_t *)A, (const uint32_t *) B, P, NP);

   return;
}

__device__ void printNumber(uint32_t *n)
{
  uint32_t i;
  
  for (i=0; i < NWORDS_256BIT; i++){
    printf("%u ",n[i]);
  }
  printf("\n");
}

__forceinline__ __device__ void addu256(uint32_t *z, const uint32_t *x, const uint32_t *y)
{
  // z[i] = x[i] + y[i] for 8x32 bit words
  asm("{                             \n\t"
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

__forceinline__ __device__ void subu256(uint32_t *z, const uint32_t *x, const uint32_t *y)
{
  // z[i] = x[i] - y[i] for 8x32 bit words
  asm("{                             \n\t"
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


__forceinline__ __device__ uint32_t eq0u256(const uint32_t *x)
{
  if (x[0] == 0 && x[1] ==  0 && x[2] == 0 && x[3] == 0 && x[4] == 0 && x[5] == 0 && x[6] == 0 && x[7] == 0){
    return 1;
  } else { 
    return 0;
  }
}

__forceinline__ __device__ uint32_t ltu256(const uint32_t *x, const uint32_t *y)
{
   if (x[7] > y[7]) return 0;
   else if (x[7] < y[7]) return 1;
   else if (x[6] > y[6]) return 0;
   else if (x[6] < y[6]) return 1;
   else if (x[5] > y[5]) return 0;
   else if (x[5] < y[5]) return 1;
   else if (x[4] > y[4]) return 0;
   else if (x[4] < y[4]) return 1;
   else if (x[3] > y[3]) return 0;
   else if (x[3] < y[3]) return 1;
   else if (x[2] > y[2]) return 0;
   else if (x[2] < y[2]) return 1;
   else if (x[1] > y[1]) return 0;
   else if (x[1] < y[1]) return 1;
   else if (x[0] >= y[0]) return 0;
   else return 1;
}
__forceinline__ __device__ void addmu256(uint32_t *z, const uint32_t *x, const uint32_t *y, const uint32_t *p)
{
   if (eq0u256(y)) {
      //z[0] = 0; z[1] = 0; z[2] = 0; z[3] = 0; z[4] = 0; z[5] = 0; z[6] = 0; z[7] = 0;
      asm("mov.u32     %0,  0;\n\t"
          "mov.u32     %1,  0;\n\t"
          "mov.u32     %2,  0;\n\t"
          "mov.u32     %3,  0;\n\t"
          "mov.u32     %4,  0;\n\t"
          "mov.u32     %5,  0;\n\t"
          "mov.u32     %6,  0;\n\t"
          "mov.u32     %7,  0;\n\t"
       : "=r"(z[0]), "=r"(z[1]), "=r"(z[2]), "=r"(z[3]),
         "=r"(z[4]), "=r"(z[5]), "=r"(z[6]), "=r"(z[7])
       :);
   } else {
      subu256(z,p,y);
      submu256(z,x,z,p);
   }
}

__forceinline__ __device__ void submu256(uint32_t *z, const uint32_t *x, const uint32_t *y, const uint32_t *p)
{
  if (ltu256(x,y)){
    subu256(z,p,y);
    addu256(z,x,z);

  } else {
    subu256(z,x,y);
  }
  
}

__forceinline__ __device__ void mulmontu256(uint32_t *U, const uint32_t *A, const uint32_t *B, const uint32_t *P, const uint32_t *NP)
{ 
    uint32_t i,j;
    uint32_t S, C=0, C1, C2, M[2], X[2];
    uint32_t T[]={0,0,0,0,0,0,0,0,0,0,0};

    #pragma unroll
    for(i=0; i<NWORDS_256BIT; i++)
    {
      // (C,S) = t[0] + a[0]*b[i], worst case 2 words
      madcu32(&C,&S,A[0],B[i],T[0]);

      // ADD(t[1],C)
      propcu32(T, C, 1);

     // m = S*n'[0] mod W, where W=2^32
     // Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
     mulu32(M, S, NP);

     // (C,S) = S + m*n[0], worst case 2 words
     madcu32(&C,&S,M[0],P[0],S);

     #pragma unroll
     for(j=1; j<NWORDS_256BIT; j++)
     {
       // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
       mulu32(X, A[j], B[i]);
       addcu32(&C1, &S, T[j], C);
       addcu32(&C2, &S, S, X[0]);
       addcu32(&C, &X[0], C1, X[1]);
       addcu32(&C, &X[0], C, C2);

       // ADD(t[j+1],C)
       propcu32(T,C,j+1);

       // (C,S) = S + m*n[j]
       madcu32(&C,&S,M[0], P[j],S);

       // t[j-1] = S
       T[j-1] = S;
     }

     // (C,S) = t[s] + C
     addcu32(&C,&S, T[NWORDS_256BIT], C);
     // t[s-1] = S
     T[NWORDS_256BIT-1] = S;
     // t[s] = t[s+1] + C
     addcu32(&C,&T[NWORDS_256BIT], T[NWORDS_256BIT+1], C);
     // t[s+1] = 0
     T[NWORDS_256BIT+1] = 0;
   }

   /* Step 3: if(u>=n) return u-n else return u */
   if (ltu256(P,T)){
      subu256(U,T,P);
   } else {
      memcpy(U, T, sizeof(uint32_t) * NWORDS_256BIT);
   }

   return;
}
/*
   Assumes p is at least 253 bits
   */
__forceinline__ __device__ void modu256(uint32_t *z, const uint32_t *x, const uint32_t *p)
{
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
    : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
      "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));

  // x(255 bit number worst case) - p (253 bit number) = z (255 bit number) : ex 31(5 b) - 4(3 b) =27 (5 b) 
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }
  // x(255 bit ) - p (253 bit number) = z (255 bit number) : ex 27(5 b) - 4(3 b) =23 (5 b) 
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }
  // x(255 bit ) - p (253 bit number) = z (255 bit number) : ex 23(5 b) - 4(3 b) = 19 (5 b) 
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }
  // x(255 bit ) - p (253 bit number) = z (254 bit number) : ex 19(5 b) - 4(3 b) = 15 (4 b) 
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }
  // x(254 bit ) - p (253 bit number) = z (254 bit number) : ex 15(5 b) - 4(3 b) = 11 (4 b) 
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }
  // x(254 bit ) - p (253 bit number) = z (254 bit number) : ex 11(5 b) - 4(3 b) = 7 (3 b) 
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }
  // x(254 bit ) - p (253 bit number) = z (254 bit number) : ex 7(5 b) - 4(3 b) = 3 (3 b) 
  if (!ltu256(z,p)){
     subu256(z,z,p);
  } else { return; }

  assert(0);

}

/*
   z = x >> 1
*/
__forceinline__ __device__ void shr1u256(const uint32_t *x)
{
    
   asm("{                                    \n\t"
       ".reg .u32          %tmp;             \n\t"
       "shr.u32            %0,   %8,  1;     \n\t"  // x[0] = x[0] >> 1
       "and.b32            %tmp, %9,  1;     \n\t"  // x[0] |= (x[1] & 1) << 31
       "shl.u32            %tmp, %tmp, 31;   \n\t"
       "or.u32             %0,   %0,  %tmp;  \n\t"
       "shr.u32            %1,   %9,   1;     \n\t"  // x[1] = x[1] >> 1
       "and.b32            %tmp, %10,  1;     \n\t"  // x[1] |= (x[2] & 1) << 31
       "shl.u32            %tmp, %tmp, 31;   \n\t"
       "or.u32             %1,   %1,  %tmp;   \n\t"
       "shr.u32            %2,   %10,  1;     \n\t"  // x[2] = x[2] >> 1
       "and.b32            %tmp, %11,  1;     \n\t"  // x[2] |= (x[3] & 1) << 31
       "shl.u32            %tmp, %tmp, 31;   \n\t"
       "or.u32             %2,   %2,  %tmp;   \n\t"
       "shr.u32            %3,   %11,  1;     \n\t"  // x[3] = x[3] >> 1
       "and.b32            %tmp, %12,  1;     \n\t"  // x[3] |= (x[4] & 1) << 31
       "shl.u32            %tmp, %tmp, 31;   \n\t"
       "or.u32             %3,   %3,  %tmp;   \n\t"
       "shr.u32            %4,   %12,  1;     \n\t"  // x[4] = x[4] >> 1
       "and.b32            %tmp, %13,  1;     \n\t"  // x[4] |= (x[5] & 1) << 31
       "shl.u32            %tmp, %tmp, 31;   \n\t"
       "or.u32             %4,   %4,  %tmp;   \n\t"
       "shr.u32            %5,   %13,  1;     \n\t"  // x[5] = x[5] >> 1
       "and.b32            %tmp, %14,  1;     \n\t"  // x[5] |= (x[6] & 1) << 31
       "shl.u32            %tmp, %tmp, 31;   \n\t"
       "or.u32             %5,   %5,  %tmp;   \n\t"
       "shr.u32            %6,   %14,  1;     \n\t"  // x[6] = x[6] >> 1
       "and.b32            %tmp, %15,  1;     \n\t"  // x[6] |= (x[7] & 1) << 31
       "shl.u32            %tmp, %tmp, 31;   \n\t"
       "or.u32             %6,   %6,  %tmp;   \n\t"
       "shr.u32            %7,   %15,  1;     \n\t"  // x[7] = x[7] >> 1
       "}                               \n\t"
       : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]), 
         "=r"(x[4]), "=r"(x[5]), "=r"(x[6]), "=r"(x[7])
       : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), 
         "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]));
}
__forceinline__ __device__ void mulu32(uint32_t *z, const uint32_t x, const uint32_t y)
{
  // z[i] = x * y for 32 bit words
  asm("{                                    \n\t"
      ".reg .u64 %prod;                     \n\t"
      "mul.wide.u32        %prod, %2,    %3;\n\t"           
      "cvt.u32.u64         %0,    %prod;    \n\t"
      "shr.u64             %prod, %prod, 32;\n\t"
      "cvt.u32.u64         %1,    %prod;    \n\t"
      "}                                    \n\t"
      : "=r"(z[0]), "=r"(z[1])
      : "r"(x), "r"(y));
}


/*
   (z,c) = x * y + a
   z is 2 x uint32_t
   c is carry
*/
__forceinline__ __device__ void madcu32(uint32_t *c, uint32_t *s, uint32_t x, uint32_t y, uint32_t a)
{
   // (C,S) = t[0] + a[0] * b[i] -> No carry in
   asm("{                                   \n\t"
       //".reg .u32      %tmp;                \n\t"
       //"mad.lo.cc.u32  %0, %3, %4, %5;      \n\t"
       //"madc.hi.cc.u32 %1, %3, %4, 0;       \n\t"
       "mad.lo.cc.u32  %0, %2, %3, %4;      \n\t"
       "madc.hi.u32    %1, %2, %3, 0;       \n\t"
       //"addc.u32       %tmp, %1, 0;         \n\t"
       //"set.lt.u32.u32 %2, %1, %tmp;        \n\t"
       "}                                   \n\t"
       //: "=r"(s[0]), "=r"(s[1]), "=r"(c[0]) 
       : "=r"(s[0]), "=r"(c[0])
       : "r"(x), "r"(y), "r"(a));
}
__forceinline__ __device__ void addcu32(uint32_t *c, uint32_t *s, uint32_t x, uint32_t y)
{
   // (C,S) = t[0] + a[0] * b[i] -> No carry in
   asm("{                               \n\t"
       ".reg .u32          %tmp;          \n\t"
       "add.cc.u32         %tmp, %2, %3;    \n\t"
       "set.lt.u32.u32     %1, %tmp, %2; \n\t"
       "mov.u32            %0, %tmp;        \n\t"
       "and.b32            %1, %1,1;     \n\t" 
       "}                               \n\t"
       : "=r"(s[0]), "=r"(c[0]) 
       : "r"(x), "r"(y));
}

__forceinline__ __device__ void propcu32(uint32_t *x, uint32_t c, uint32_t digit)
{
   while ((digit < NWORDS_256BIT_FIOS) && (c))
   {
     asm("{                                   \n\t"
         //".reg .u32       %cin;               \n\t"
         ".reg .u32       %tmp;               \n\t"
         //"mov.u32         %cin, %3;           \n\t"
         "mov.u32         %tmp, %2;           \n\t"
         "add.cc.u32      %0,   %1, %tmp;   \n\t"
         "set.lt.u32.u32  %1,   %0,   %tmp;   \n\t"
         "and.b32         %1,   %1,   1;      \n\t"
         "}                                   \n\t"
         : "=r"(x[digit]), "=r"(c) 
         : "r"(x[digit]), "r"(c));
     digit++;
   }
}

