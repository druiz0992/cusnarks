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
*/

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : bigint.cpp
//
// Date       : 6/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//
//  Big Integer Arithmetic
//
// ------------------------------------------------------------------

#include <stdio.h>
#include <omp.h>

#include "types.h"
#include "rng.h"
#include "log.h"
//#include "uint256.h"
#include "constants.h"
#include "bigint.h"


#ifdef PARALLEL_EN
static  uint32_t parallelism_enabled =  1;
#else
static  uint32_t parallelism_enabled =  0;
#endif

/*
   Generate N 32 bit random samples

   uint32_t *x      : output vector containing 32 bit samples. Vector is of length nsamples
   uint32_t ndigits : Number of samples to generate
   
*/
void setRandom(uint32_t *x, const uint32_t nsamples)
{
  int i;
  _RNG* rng = _RNG::get_instance(nsamples);
 
  rng->randu32(x,nsamples); 
}


/*
   Sort 256-bit samples in ascending order.  Input samples indexes are actually sorted. Samples are
    left unsorted. 

   uint32_t *idx  : output vector containing sorted indexed. Size of idx is len.
   uint32_t *v    : input vector of size len 256 bit samples
   uint32_t len   : number of samples to sort 
  
*/
void sortuBI_idx_h(uint32_t *idx, const uint32_t *v, uint32_t len, uint32_t biSize,  uint32_t sort_en)
{
  uint32_t i;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0;i < len; i++){  
    idx[i] = i;
  }

  if (sort_en){
     std::sort(idx, idx+len, 
       [&v,biSize](uint32_t i1, uint32_t i2){ 
         return (ltu32_h((const uint32_t*)&v[i1*biSize+biSize-1],(const uint32_t *)&v[i2*biSize+biSize-1]));});
  }
}


uint32_t shlluBI_h(uint32_t *y, uint32_t *x, uint32_t count, uint32_t biSize)
{
 uint64_t t,carry;
 int i, places=0, sh;
 uint32_t out;

 out = x[biSize-1] & ( ((1 << count) -1) << (NBITS_WORD - count));
 sh = count - count/NBITS_WORD * NBITS_WORD;
 /* Shift bits. */
 for(i = carry = 0; i < biSize; i++)
 {
  t = ((uint64_t)(x[i]) << sh ) | carry;
  y[i] = t & 0xFFFFFFFF;
  carry = t >> NBITS_WORD;
 }

 if(count >= NBITS_WORD) {
  places = count / NBITS_WORD;

  for(i = biSize-1; i >=  places; i--) {
   y[i] = y[i - places];
  }

  for(; i >= 0; i--){
   y[i] = 0;
  }

  if(sh == 0) {
   return out >> (NBITS_WORD - count);
  }
 }

 return out >> (NBITS_WORD - count);
}


uint32_t shlruBI_h(uint32_t *y, uint32_t *x, uint32_t count, uint32_t biSize)
{
  uint64_t t, carry;
  int i;
  uint32_t places=0, sh;
  uint32_t out = x[0] &  ((1 << count) -1);

  sh = count - count / NBITS_WORD * NBITS_WORD;
  
  if (count >= biSize * NBITS_WORD) { 
    memset(y,0,biSize*sizeof(uint32_t));
  }

  /* Shift any remaining bits. */
  for(i = biSize - 1, carry = 0; i >= 0; i--)
  {
   t = (uint64_t)(x[i]) << NBITS_WORD;
   t >>= sh;
   t |= carry;
   carry = (t & 0xFFFFFFFF) << NBITS_WORD;
   y[i] = t >> NBITS_WORD;
  }

  if(count >= NBITS_WORD) {
    places = count / NBITS_WORD;

    if(places > biSize) {
      memset(y, 0, biSize * sizeof(uint32_t));
      return 0;
    }

    for(i = 0; i < (int) (biSize - places); i++){
      y[i] = y[i + places];
    }
    for(; i < biSize; i++) {
      y[i] = 0;
    }

  }
  return out; 
}

void setbituBI_h(uint32_t *x, uint32_t n)
{
  uint32_t w, b;
 
  w = n / NBITS_WORD;
  b = n % NBITS_WORD;

  x[w] |=  (1 << b);
}

uint32_t getbituBI_h(uint32_t *x, uint32_t n)
{
  uint32_t w, b;
  
  w = n >> NBITS_WORD_LOG2;
  b = n & NBITS_WORD_MOD;

  return ( (x[w] >> b) & 0x1);
}
uint32_t getbituBI_h(uint32_t *x, uint32_t n, uint32_t group_size, uint32_t biSize)
{
  uint32_t w, b,i, val=0;
  
  w = n >> NBITS_WORD_LOG2;
  b = n & NBITS_WORD_MOD;

  for (i = 0; i < group_size; i++){
    val |= (( (x[w+biSize*i] >> b) & 0x1) << i);
  }

  return val;
}

// __builtin_clz only defined for 32 bits
uint32_t msbuBI_h(uint32_t *x, uint32_t biSize)
{
  int i,j;
  uint32_t count=0, n=0; 
  for(i=biSize-1; i >= 0; i--){
    if (x[i] == 0){
      n = NBITS_WORD; 
    } else {
       n = __builtin_clz(x[i]);
    }
    count += n;
    if (n != NBITS_WORD) return count;
  }
  return biSize*NBITS_WORD-1;
}

/*
  Display u256 samples
 
  TODO - substitute by log function
*/
void printUBINumber(const uint32_t *x, uint32_t biSize)
{
  for (uint32_t i=0; i < biSize; i++){
    //printf("%8x ",x[i]);
    printf("%u ",x[i]);
  }
  printf ("\n");
}

void printUBINumber(const char *s, const uint32_t *x, uint32_t biSize)
{
  printf("%s",s);
  for (uint32_t i=0; i < biSize; i++){
    printf("%x ",x[i]);
  }
  printf ("\n");
}

