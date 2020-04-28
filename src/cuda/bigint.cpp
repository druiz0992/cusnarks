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
void sortu256_idx_h(uint32_t *idx, const uint32_t *v, uint32_t len, uint32_t sort_en)
{
  uint32_t i;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0;i < len; i++){  
    idx[i] = i;
  }

  if (sort_en){
     //std::sort(idx, idx+len, [&v](uint32_t i1, uint32_t i2){ return (v[i1*NWORDS_256BIT] < v[i2*NWORDS_256BIT]);});
     std::sort(idx, idx+len, 
       [&v](uint32_t i1, uint32_t i2){ 
         //return (ltu256_h((const uint32_t*)&v[i1*NWORDS_256BIT],(const uint32_t *)&v[i2*NWORDS_256BIT]));});
         return (ltu32_h((const uint32_t*)&v[i1*NWORDS_256BIT+NWORDS_256BIT-1],(const uint32_t *)&v[i2*NWORDS_256BIT+NWORDS_256BIT-1]));});
  }
}


uint32_t shllu256_h(uint32_t *y, uint32_t *x, uint32_t count)
{
 uint64_t t,carry;
 int i, places=0, sh;
 uint32_t out;

 out = x[NWORDS_256BIT-1] & ( ((1 << count) -1) << (32 - count));
 sh = count - count/NBITS_WORD * NBITS_WORD;
 /* Shift bits. */
 for(i = carry = 0; i < NWORDS_256BIT; i++)
 {
  t = ((uint64_t)(x[i]) << sh ) | carry;
  y[i] = t & 0xFFFFFFFF;
  carry = t >> NBITS_WORD;
 }

 if(count >= NBITS_WORD) {
  places = count / NBITS_WORD;

  for(i = NWORDS_256BIT-1; i >=  places; i--) {
   y[i] = y[i - places];
  }

  for(; i >= 0; i--){
   y[i] = 0;
  }

  if(sh == 0) {
   return out >> (32 - count);
  }
 }

 return out >> (32 - count);
}


uint32_t shlru256_h(uint32_t *y, uint32_t *x, uint32_t count)
{
  uint64_t t, carry;
  int i;
  uint32_t places=0, sh;
  uint32_t out = x[0] &  ((1 << count) -1);

  sh = count - count / NBITS_WORD * NBITS_WORD;
  
  if (count >= NWORDS_256BIT * NBITS_WORD) { 
    memset(y,0,NWORDS_256BIT*sizeof(uint32_t));
  }

  /* Shift any remaining bits. */
  for(i = NWORDS_256BIT - 1, carry = 0; i >= 0; i--)
  {
   t = (uint64_t)(x[i]) << NBITS_WORD;
   t >>= sh;
   t |= carry;
   carry = (t & 0xFFFFFFFF) << NBITS_WORD;
   y[i] = t >> NBITS_WORD;
  }

  if(count >= NBITS_WORD) {
    places = count / NBITS_WORD;

    if(places > NWORDS_256BIT) {
      memset(y, 0, NWORDS_256BIT * sizeof(uint32_t));
      return 0;
    }

    for(i = 0; i < (int) (NWORDS_256BIT - places); i++){
      y[i] = y[i + places];
    }
    for(; i < NWORDS_256BIT; i++) {
      y[i] = 0;
    }

  }
  return out; 
}

void setbitu256_h(uint32_t *x, uint32_t n)
{
  uint32_t w, b;
 
  w = n / NBITS_WORD;
  b = n % NBITS_WORD;

  x[w] |=  (1 << b);
}

uint32_t getbitu256_h(uint32_t *x, uint32_t n)
{
  uint32_t w, b;
  
  w = n >> NBITS_WORD_LOG2;
  b = n & NBITS_WORD_MOD;

  return ( (x[w] >> b) & 0x1);
}
uint32_t getbitu256_h(uint32_t *x, uint32_t n, uint32_t group_size)
{
  uint32_t w, b,i, val=0;
  
  w = n >> NBITS_WORD_LOG2;
  b = n & NBITS_WORD_MOD;

  for (i = 0; i < group_size; i++){
    val |= (( (x[w+NWORDS_256BIT*i] >> b) & 0x1) << i);
  }

  return val;
}

uint32_t msbu256_h(uint32_t *x)
{
  int i,j;
  uint32_t count=0, n; 
  for(i=NWORDS_256BIT-1; i >= 0; i--){
    n = __builtin_clz(x[i]);
    count += n;
    if (n != 32) return count;
  }
}

/*
  Display u256 samples
 
  TODO - substitute by log function
*/
void printU256Number(const uint32_t *x)
{
  for (uint32_t i=0; i < NWORDS_256BIT; i++){
    printf("%8x ",x[i]);
  }
  printf ("\n");
}

void printU256Number(const char *s, const uint32_t *x)
{
  printf("%s",s);
  for (uint32_t i=0; i < NWORDS_256BIT; i++){
    printf("%x ",x[i]);
  }
  printf ("\n");
}

