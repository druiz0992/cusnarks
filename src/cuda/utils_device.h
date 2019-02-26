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
// File name  : utils_device.h
//
// Date       : 26/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of small utils functions
// ------------------------------------------------------------------

*/
#ifndef _UTILS_DEVICE_H_
#define _UTILS_DEVICE_H_


/*
   x == 0 for 256 bit numbers
*/
__forceinline__ __device__ uint32_t eq0u256(const uint32_t __restrict__ *x)
{
  if (x[0] == 0 && x[1] ==  0 && x[2] == 0 && x[3] == 0 && x[4] == 0 && x[5] == 0 && x[6] == 0 && x[7] == 0){
    return 1;
  } else { 
    return 0;
  }
}

/*
   x == y for 256 bit numbers
*/
__forceinline__ __device__ uint32_t equ256(const uint32_t __restrict__ *x, const uint32_t __restrict__ *y)
{
  if (x[0] == y[0] && x[1] ==  y[1] && x[2] == y[2] && x[3] == y[3] &&
		  x[4] == y[4] && x[5] == y[5] && x[6] == y[6] && x[7] == y[7]){
    return 1;
  } else { 
    return 0;
  }
}


/*
   x < y for 256 bit numbers
*/
__forceinline__ __device__ uint32_t ltu256(const uint32_t __restrict__ *x, const uint32_t __restrict__ *y)
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

__forceinline__ __device__ void movu256(uint32_t __restrict__ *d_out, uint32_t __restrict__ *d_in)
{
   asm("mov.u32     %0,  %8;\n\t"
       "mov.u32     %1,  %9;\n\t"
       "mov.u32     %2,  %10;\n\t"
       "mov.u32     %3,  %11;\n\t"
       "mov.u32     %4,  %12;\n\t"
       "mov.u32     %5,  %13;\n\t"
       "mov.u32     %6,  %14;\n\t"
       "mov.u32     %7,  %15;\n\t"
    : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
      "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
    : "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]),
      "r"(in[4]), "r"(in[5]), "r"(in[6]), "r"(in[7]));

  #if 0
    for (uint32_t i=0; i< NWORDS_256BIT; i+=2){
      reinterpret_cast <ulonglong4 *>(d_out)[i] = reinterpret_cast <ulonglong4 *>(d_in)[i];
    }
  #endif
}

/*
   x * y for 32 bit number. Returns 64 bit number
 */
__device__ void mulu32(uint32_t __restrict__ *z, uint32_t x, uint32_t y)
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
   (c,s) = x + y * a for 32 bit number. Returns 32 bit number and 32 bit carry
*/
__forceinline__ __device__ void madcu32(uint32_t *c, uint32_t *s, uint32_t x, uint32_t y, uint32_t a)
{
   // (C,S) =  x * y + a -> No carry in
   asm("{                                   \n\t"
       "mad.lo.cc.u32  %0, %2, %3, %4;      \n\t"
       "madc.hi.u32    %1, %2, %3, 0;       \n\t"
       "}                                   \n\t"
       : "=r"(s[0]), "=r"(c[0])
       : "r"(x), "r"(y), "r"(a));
}
/*
   (c.s) = x + y for 32 bit numbers. Returns 32 bit number and 1 bit carry
*/
__forceinline__ __device__ void addcu32(uint32_t *c, uint32_t *s, uint32_t x, uint32_t y)
{
   asm("{                               \n\t"
       "add.cc.u32         %0, %2, %3;    \n\t"
       "set.lt.u32.u32     %1, %0, %2; \n\t"
       "and.b32            %1, %1,1;     \n\t" 
       "}                               \n\t"
       : "=r"(s[0]), "=r"(c[0]) 
       : "r"(x), "r"(y));
}
/*
   Propagate carry bit across a 256 bit number starting in 32 bit word indexed by digit
*/
__forceinline__ __device__ void propcu32(uint32_t *x, uint32_t c, uint32_t digit)
{
   #pragma unroll
   for (; digit < NWORDS_256BIT_FIOS-1 ; digit++)
   {
     asm("{                                   \n\t"
         "add.cc.u32      %0,   %3, %2;   \n\t"
         "set.lt.u32.u32  %1,   %0, %2;   \n\t"
         "and.b32         %1,   %1,  1;      \n\t"
         "}                                   \n\t"
         : "=r"(x[digit]), "=r"(c) 
         : "r"(x[digit]), "r"(c));
   }
}

#endif
