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
// File name  : bigint.h
//
// Date       : 06/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of big integer arithmetic
//   sub
//   add
//   sort
//   setRandom
//   comp
//   lt
//   eq
//   shlr
//   shll
//   msb
//   setbit
//   getbit
//   swap
//   print
//
//   mul64
//   lt32
//   add64
// ------------------------------------------------------------------

*/
#ifndef _BIGINT_H_
#define _BIGINT_H_

#define NEG64(X) (~(X))
/*
template<typename T>
inline void sub(T *c, const T *a, const T*b)
{
  uint32_t carry=0;
  const t_uint64 *dA = (t_uint64 *)a;
  const t_uint64 *dB = (t_uint64 *)b;
  t_uint64 *dC = (t_uint64 *)c;
  t_uint64 tmp;
  uint32_t n = getNwords(c);

  tmp = NEG64(dB[0])+1;
  carry = (tmp < 1);
  dC[0] = dA[0] + tmp;
  carry += (dC[0] < tmp);

  for (uint32_t i=1; i< n; i++){
    tmp = NEG64(dB[i]);
    dC[i] = dA[i] + carry;
    carry = (dC[i] < carry);
    dC[i] += tmp;
    carry += (dC[i] < tmp);
  }

}
*/
/* 
   Substract 256 bit integers X and Y.

   uint32_t *x : 256 bit integer x
   uint32_t *y : 256 bit integer y
   returns x - y
*/
inline void subu256_h(uint32_t *c, const uint32_t *a, const uint32_t *b)
{
  uint32_t carry=0;
  const t_uint64 *dA = (t_uint64 *)a;
  const t_uint64 *dB = (t_uint64 *)b;
  t_uint64 *dC = (t_uint64 *)c;
  t_uint64 tmp;


  tmp = NEG64(dB[0])+1;
  carry = (tmp < 1);
  dC[0] = dA[0] + tmp;
  carry += (dC[0] < tmp);

  tmp = NEG64(dB[1]);
  dC[1] = dA[1] + carry;
  carry = (dC[1] < carry);
  dC[1] += tmp;
  carry += (dC[1] < tmp);

  tmp = NEG64(dB[2]);
  dC[2] = dA[2] + carry;
  carry = (dC[2] < carry);
  dC[2] += tmp;
  carry += (dC[2] < tmp);

  tmp = NEG64(dB[3]);
  dC[3] = dA[3] + carry;
  carry = (dC[3] < carry);
  dC[3] += tmp;
  carry += (dC[3] < tmp);

}
inline void subu256_h(uint32_t *x, const uint32_t *y)
{
   subu256_h(x, x, y);
} 
  

/* 
   Add 256 bit integers X and Y.

   uint32_t *x : 256 bit integer x
   uint32_t *y : 256 bit integer y
   returns x + y
*/
inline void addu256_h(uint32_t *c, const uint32_t *a, const uint32_t *b)
{
  uint32_t carry=0;
  const t_uint64 *dA = (t_uint64 *)a;
  const t_uint64 *dB = (t_uint64 *)b;
  t_uint64 *dC = (t_uint64 *)c;

  t_uint64 tmp = dA[0];

  dC[0] = dA[0] + dB[0];
  carry = (dC[0] < tmp);

  tmp = dB[1];
  dC[1] = dA[1] + carry;
  carry = (dC[1] < carry);
  dC[1] += tmp;
  carry += (dC[1] < tmp);

  tmp = dB[2];
  dC[2] = dA[2] + carry;
  carry = (dC[2] < carry);
  dC[2] += tmp;
  carry += (dC[2] < tmp);

  tmp = dB[3];
  dC[3] = dA[3] + carry;
  carry = (dC[3] < carry);
  dC[3] += tmp;
  carry += (dC[3] < tmp);
 
}

inline void addu256_h(uint32_t *x, const uint32_t *y)
{
   addu256_h(x, x, y);
}   

void setRandom(uint32_t *x, const uint32_t);
void sortu256_idx_h(uint32_t *idx, const uint32_t *v, uint32_t len, uint32_t sort_en);
/* 
   Compare 256 bit integers X and Y.

   uint32_t *x : 256 bit integer x
   uint32_t *y : 256 bit integer y
   returns 
      0          : x == y
      pos number : x > y
      neg number : x < y
*/
inline int32_t compu256_h(const uint32_t *a, const uint32_t *b)
{
  uint32_t gt=0, lt=0;
  uint32_t idx = NWORDS_256BIT/2-1;

  const t_uint64 *dA = (const t_uint64 *)a;
  const t_uint64 *dB = (const t_uint64 *)b;
  // idx = 3
  gt = (dA[idx] > dB[idx]);
  lt = (dA[idx] < dB[idx]);
  if (gt) return 1;
  if (lt) return -1;

  // idx = 2
  idx--;
  gt = (dA[idx] > dB[idx]);
  lt = (dA[idx] < dB[idx]);
  if (gt) return 1;
  if (lt) return -1;

  // idx = 1
  idx--;
  gt = (dA[idx] > dB[idx]);
  lt = (dA[idx] < dB[idx]);
  if (gt) return 1;
  if (lt) return -1;

  // idx =0
  idx--;
  gt = (dA[idx] > dB[idx]);
  lt = (dA[idx] < dB[idx]);
  if (gt) return 1;
  if (lt) return -1;

  return 0;

}

/* 
   Compare 256 bit integers X and Y.

   uint32_t *x : 256 bit integer x
   uint32_t *y : 256 bit integer y

   returns 
      1          : x < y
      0         : x >= y
*/
inline int32_t ltu256_h(const uint32_t *x, const uint32_t *y)
{
  return (compu256_h(x, y) < 0);
}

inline int32_t ltu32_h(const uint32_t *x, const uint32_t *y)
{
  return *x < *y;
}

/* 
   Compare 256 bit integers X and Y.

   uint32_t *x : 256 bit integer x
   uint32_t *y : 256 bit integer y

   returns 
      1          : x == y
      0         : x != y
*/
inline int32_t equ256_h(const uint32_t *x, const uint32_t *y)
{
  return (compu256_h(x, y) == 0);
}
uint32_t shlru256_h(uint32_t *y, uint32_t *x, uint32_t count);
uint32_t shllu256_h(uint32_t *y, uint32_t *x, uint32_t count);
uint32_t msbu256_h(uint32_t *x);
void setbitu256_h(uint32_t *x, uint32_t n);
uint32_t getbitu256_h(uint32_t *x, uint32_t n);
uint32_t getbitu256_h(uint32_t *x, uint32_t n, uint32_t group_size);
uint32_t getbitu32_h(uint32_t *x, uint32_t n, uint32_t group_size);

inline void mulu64_h(t_uint64 p[2], const t_uint64 *x, const t_uint64 *y)
{
 __int128 *r = (__int128 *) p;
 *r = (__int128) y[0] * x[0];
}

inline t_uint64 addu64_h(t_uint64 *c, t_uint64 *a, t_uint64 *b)
{
  t_uint64 carry=0;

  const t_uint64 *dA = (t_uint64 *)a;
  const t_uint64 *dB = (t_uint64 *)b;
  t_uint64 *dC = (t_uint64 *)c;
  t_uint64 tmp = dA[0];

  dC[0] = dA[0] + dB[0];
  carry = (dC[0] < tmp);

 return carry;
}

/*
   Swaps two 256 bit variables x,y
*/
inline void swapu256_h(uint32_t *x, uint32_t *y)
{
  t_uint64 *dX = (t_uint64 *) x;
  t_uint64 *dY = (t_uint64 *) y;
  t_uint64 tmp = dX[0];

  dX[0] = dY[0];
  dY[0] = tmp;
  
  tmp = dX[1]; 
  dX[1] = dY[1];
  dY[1] = tmp;

  tmp = dX[2]; 
  dX[2] = dY[2];
  dY[2] = tmp;

  tmp = dX[3]; 
  dX[3] = dY[3];
  dY[3] = tmp;
}

void printU256Number(const uint32_t *x);
void printU256Number(const char *, const uint32_t *x);
#endif
