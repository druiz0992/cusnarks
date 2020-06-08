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
   Substract BI bit integers X and Y.

   uint32_t *x : BI bit integer x
   uint32_t *y : BI bit integer y
   returns x - y
*/
inline void subuBI_h(uint32_t *c, const uint32_t *a, const uint32_t *b, const uint32_t biSize)
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

  for (uint32_t i=1; i < biSize/2; i++) {
    tmp = NEG64(dB[i]);
    dC[i] = dA[i] + carry;
    carry = (dC[i] < carry);
    dC[i] += tmp;
    carry += (dC[i] < tmp);
  }
}

inline void subuBI_h(uint32_t *x, const uint32_t *y, const uint32_t biSize)
{
   subuBI_h(x, x, y, biSize);
} 
  
/* 
   Add BI bit integers X and Y.

   uint32_t *x : BI bit integer x
   uint32_t *y : BI bit integer y
   returns x + y
*/
inline void adduBI_h(uint32_t *c, const uint32_t *a, const uint32_t *b, const uint32_t biSize)
{
  uint32_t carry=0;
  const t_uint64 *dA = (t_uint64 *)a;
  const t_uint64 *dB = (t_uint64 *)b;
  t_uint64 *dC = (t_uint64 *)c;

  t_uint64 tmp = dA[0];

  dC[0] = dA[0] + dB[0];
  carry = (dC[0] < tmp);

  for (uint32_t i=1; i < biSize/2; i++){
    tmp = dB[i];
    dC[i] = dA[i] + carry;
    carry = (dC[i] < carry);
    dC[i] += tmp;
    carry += (dC[i] < tmp);
  }

}

inline void adduBI_h(uint32_t *x, const uint32_t *y, const uint32_t biSize)
{
   adduBI_h(x, x, y, biSize);
}   

/* 
   Compare BI bit integers X and Y.

   uint32_t *x : BI bit integer x
   uint32_t *y : BI bit integer y
   returns 
      0          : x == y
      pos number : x > y
      neg number : x < y
*/
inline int32_t compuBI_h(const uint32_t *a, const uint32_t *b, const uint32_t biSize)
{
  uint32_t gt=0, lt=0;
  int idx = biSize/2-1;

  const t_uint64 *dA = (const t_uint64 *)a;
  const t_uint64 *dB = (const t_uint64 *)b;

  for (idx=biSize/2-1; idx >= 0; idx--){
    gt = (dA[idx] > dB[idx]);
    lt = (dA[idx] < dB[idx]);
    if (gt) return 1;
    if (lt) return -1;
  }
  return 0;
}

/* 
   Compare BI bit integers X and Y.

   uint32_t *x : BI bit integer x
   uint32_t *y : BI bit integer y

   returns 
      1          : x < y
      0         : x >= y
*/
inline int32_t ltuBI_h(const uint32_t *x, const uint32_t *y, const uint32_t biSize)
{
  return (compuBI_h(x, y, biSize) < 0);
}

inline int32_t ltu32_h(const uint32_t *x, const uint32_t *y)
{
  return *x < *y;
}

/* 
   Compare BI bit integers X and Y.

   uint32_t *x : BI bit integer x
   uint32_t *y : BI bit integer y

   returns 
      1          : x == y
      0         : x != y
*/
inline int32_t equBI_h(const uint32_t *x, const uint32_t *y, const uint32_t biSize)
{
  return (compuBI_h(x, y, biSize) == 0);
}

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
   Swaps two BI bit variables x,y
*/
inline void swapuBI_h(uint32_t *x, uint32_t *y, const uint32_t biSize)
{
  t_uint64 *dX = (t_uint64 *) x;
  t_uint64 *dY = (t_uint64 *) y;
  t_uint64 tmp;

  for (uint32_t i=0; i < biSize/2; i++){
    tmp = dX[i];
    dX[i] = dY[i];
    dY[i] = tmp;
  }
}


void setRandom(uint32_t *x, const uint32_t);
void sortuBI_idx_h(uint32_t *idx, const uint32_t *v, uint32_t len, uint32_t biSize, uint32_t sort_en);
uint32_t shlruBI_h(uint32_t *y, uint32_t *x, uint32_t count, uint32_t biSize);
uint32_t shlluBI_h(uint32_t *y, uint32_t *x, uint32_t count, uint32_t biSize);
uint32_t msbuBI_h(uint32_t *x, uint32_t biSize);
void setbituBI_h(uint32_t *x, uint32_t n);
uint32_t getbituBI_h(uint32_t *x, uint32_t n);
uint32_t getbituBI_h(uint32_t *x, uint32_t n, uint32_t group_size, uint32_t biSize);
void printUBINumber(const uint32_t *x, uint32_t biSize);
void printUBINumber(const char *, const uint32_t *x, uint32_t biSize);

#endif
