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
// File name  : utils_host.cpp
//
// Date       : 6/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Util functions for host. Functions mostly implement 256 bit arithmetic and polynomial of finite field elementss
//
//   256 bit samples are represented by 8 32-bit words (NWORDS_256BIT), where word 0 is the least significant word
//
//   A polynomial of degree N is represented by N+1 256 bit samples ((N+1) * NWORDS_256BIT), where degree 0 coefficient
//   is stored in the first NWORDS_256BIT words
//
// ------------------------------------------------------------------

// NOTE Signigicant parts of this code have been taken from :
//
// https://github.com/Xilinx/embeddedsw/blob/master/XilinxProcessorIPLib/drivers/hdcp22_rx/src/xhdcp22_rx_crypt.c
// https://github.com/Xilinx/embeddedsw/blob/master/XilinxProcessorIPLib/drivers/hdcp22_common/src/bigdigits.c
//
// [1] in function headers mean function was at least partially obtained from this site

/******************************************************************************
*
* Copyright (C) 2015 - 2016 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/***** BEGIN LICENSE BLOCK *****
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2001-15 David Ireland, D.I. Management Services Pty Limited
 * <http://www.di-mgt.com.au/bigdigits.html>. All rights reserved.
 *
 ***** END LICENSE BLOCK *****/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <algorithm>
#include "types.h"
#include "constants.h"
#include "rng.h"
#include "utils_host.h"


#define MAX_DIGIT 0xFFFFFFFFUL
#define MAX(X,Y)  ((X)>=(Y) ? (X) : (Y))

// Internal functions
// single/multiple precision
void mpAddWithCarryProp(uint32_t *A, uint32_t C, int SDigit, int max_digit);
uint32_t mpAdd(uint32_t w[], const uint32_t u[], const uint32_t v[], size_t ndigits);
int spMultiply(uint32_t p[2], uint32_t x, uint32_t y);
int mpMultiply(uint32_t p[3], uint32_t x[2], uint32_t y);
int mpCompare(const uint32_t a[], const uint32_t b[], size_t ndigits);
uint32_t mpSubtract(uint32_t w[], const uint32_t u[], const uint32_t v[], size_t ndigits);

// FFT helper functions
uint32_t reverse(uint32_t x, uint32_t bits);
inline void swap(uint32_t *x, uint32_t *y);


//////

/****************************************************************************/
/**  [1]
*
* This function performs a carry propagation adding C to the input
* array A of size NDigits, given by the first argument starting from
* the first element SDigit, and propagates it until no further carry
* is generated.
*
* ADD(A[i],C)
*
* Reference:
* Analyzing and Comparing Montgomery Multiplication Algorithms
* IEEE Micro, 16(3):26-33,June 1996
* By: Cetin Koc, Tolga Acar, and Burton Kaliski
*
* @param A is an input array of size NDigits
* @param C is the value being added to the input A
* @param SDigit is the start digit
* @param NDigits is the integer precision of the arguments (A)
*
* @return None.
*
* @note  None.
*****************************************************************************/
void mpAddWithCarryProp(uint32_t *A, uint32_t C, int SDigit, int max_digit)
{
 int i;
 int j=0;

 for(i=SDigit; i<max_digit; i++) {
   C = mpAdd(A+i, A+i, &C, 1);

   if(C == 0) {
     //if (j > 0) {
           //printf("%d\n",j);
     //}
     return;
   }
   j++;
 }
 //if (j > 0) { printf("%d\n",j);}
}

/* [1]

  Calculates w = u + v
  where w, u, v are multiprecision integers of ndigits each
  Returns carry if overflow. Carry = 0 or 1.
  Ref: Knuth Vol 2 Ch 4.3.1 p 266 Algorithm A.
*/
uint32_t mpAdd(uint32_t w[], const uint32_t u[], const uint32_t v[], size_t ndigits)
{

 uint32_t k;
 size_t j;

 /* Step A1. Initialise */
 k = 0;

 for (j = 0; j < ndigits; j++) {
  /* Step A2. Add digits w_j = (u_j + v_j + k)
   Set k = 1 if carry (overflow) occurs
  */
  w[j] = u[j] + k;
  if (w[j] < k) k = 1; 
  else k = 0;

  w[j] += v[j];
  if (w[j] < v[j]) k++;

 } /* Step A3. Loop on j */

 return k; /* w_n = k */
}

/* [1]
  
  Computes multiplication of 2 32 bit numbers x and y and stores it in 2x32 bit array p

  uint32_t p[2] : output result
  uint32_t x    : input factor 1
  uint32_t y    : input factor 2
*/
int spMultiply(uint32_t p[2], uint32_t x, uint32_t y)
{
 /* Use a 64-bit temp for product */
 uint64_t t = (uint64_t)x * (uint64_t)y;
 /* then split into two parts */
 p[1] = (uint32_t)(t >> 32);
 p[0] = (uint32_t)(t & 0xFFFFFFFF);

 return 0;
}

/* [1]
  
  Computes multiplication of one 32 bit numbers y times a 64 bit number x and stores it in 3x32 bit array p

  uint32_t p[3] : output result
  uint32_t x[2]  : 64 bit input factor 1
  uint32_t y    : 32 bit input factor 2
*/

int mpMultiply(uint32_t p[3], uint32_t x[2], uint32_t y)
{
 uint64_t t1 = (uint64_t)x[0] * (uint64_t)y;
 uint64_t t2 = (uint64_t)x[1] * (uint64_t)y;
 uint32_t c;

 /* then split into two parts */
 p[0] = (uint32_t)(t1 & 0xFFFFFFFF);
 p[1] = (uint32_t)(t1 >> 32) + (uint32_t)(t2 & 0xFFFFFFFF);
 c = p[1] < (uint32_t)(t2 & 0xFFFFFFFF);
 p[2] = (uint32_t)(t2 >> 32) + c;

 return 0;
}

/* 
   Compare multi precision  integers X and Y.

   uint32_t a[] : integer x
   uint32_t b[] : integer y
   size_t ndigits : number of 32-bit words in x and y

   returns 
      0          : x == y
      pos number : x > y
      neg number : x < y
*/
int mpCompare(const uint32_t a[], const uint32_t b[], size_t ndigits)
{
 /* All these vars are either 0 or 1 */
 unsigned int gt = 0;
 unsigned int lt = 0;
 unsigned int mask = 1; /* Set to zero once first inequality found */
 unsigned int c;

 while (ndigits--) {
  gt |= (a[ndigits] > b[ndigits]) & mask;
  lt |= (a[ndigits] < b[ndigits]) & mask;
  c = (gt | lt);
  mask &= (c-1); /* Unchanged if c==0 or mask==0, else mask=0 */
 }

 return (int)gt - (int)lt; /* EQ=0 GT=+1 LT=-1 */
}

/*  [1]

  Calculates w = u - v where u >= v
  w, u, v are multiprecision integers of ndigits each
  Returns 0 if OK, or 1 if v > u.
  Ref: Knuth Vol 2 Ch 4.3.1 p 267 Algorithm S.
*/
uint32_t mpSubtract(uint32_t w[], const uint32_t u[], const uint32_t v[], size_t ndigits)
{

 uint32_t k;
 size_t j;

 /* Step S1. Initialise */
 k = 0;

 for (j = 0; j < ndigits; j++)
 {
  /* Step S2. Subtract digits w_j = (u_j - v_j - k)
   Set k = 1 if borrow occurs.
  */
  w[j] = u[j] - k;
  if (w[j] > MAX_DIGIT - k) k = 1;
  else k = 0;

  w[j] -= v[j];
  if (w[j] > MAX_DIGIT - v[j]) k++;

 } /* Step S3. Loop on j */

 return k; /* Should be zero if u >= v */
}

/*
  Bit reverse 32 bit number

  uint32_t x : input number
  uint32_t bits : number of bits
  
  returns bit reversed input number
*/
uint32_t reverse(uint32_t x, uint32_t bits)
{
  uint32_t y = 0;
  for (uint32_t i=0; i<bits; i++){
     y = (y << 1) | (x & 1);
     x >>= 1;
  }
  return y;
}

/*
   Swaps two 256 bit variables x,y
*/
inline void swap(uint32_t *x, uint32_t *y)
{
  uint32_t tmp[NWORDS_256BIT];

  memcpy(tmp, x, sizeof(uint32_t)*NWORDS_256BIT);
  memcpy(x,y, sizeof(uint32_t)*NWORDS_256BIT);
  memcpy(y,tmp, sizeof(uint32_t)*NWORDS_256BIT);
}


/*
  Transpose matrix of 256 bit coefficients
 
  uint32_t *mout : Output transposed matrix
  uint32_t *min : Input matrix
  uint32_t in_nrows : Number of rows in input matrix
  uint32_t in_ncols : Nuimber of columns in input matrix 
*/
void transpose_h(uint32_t *mout, const uint32_t *min, uint32_t in_nrows, uint32_t in_ncols)
{
  uint32_t i,j,k;

  for (i=0; i<in_nrows; i++){
    for(j=0; j<in_ncols; j++){
      for (k=0; k<NWORDS_256BIT; k++){
        //printf("OUT: %d, IN : %d\n",(j*in_nrows+i)*NWORDS_256BIT+k, (i*in_ncols+j)*NWORDS_256BIT+k);
        mout[(j*in_nrows+i)*NWORDS_256BIT+k] = min[(i*in_ncols+j)*NWORDS_256BIT+k];
      }
    }
  }
}


///////////////////
/*
  TODO
*/
void mpoly_eval_h(uint32_t *pout, const uint32_t *scalar, uint32_t *pin, uint32_t ncoeff, uint32_t last_idx, uint32_t pidx)
{
  uint32_t n_zpoly = pin[0];
  uint32_t zcoeff_d_offset = 1 + n_zpoly;
  uint32_t zcoeff_v_offset;
  uint32_t n_zcoeff;
  uint32_t scl[NWORDS_256BIT];
  uint32_t i,j;
  uint32_t *zcoeff_v_in, *zcoeff_v_out, zcoeff_d;
  uint32_t prev_n_zcoeff = 0, accum_n_zcoeff;

  /*
  printf("N zpoly: %d\n",n_zpoly);
  printf("Zcoeff D Offset : %d\n",zcoeff_d_offset);
  */
  for (i=0; i<last_idx; i++){
    to_montgomery_h(scl, &scalar[i*NWORDS_256BIT], pidx);
    /*
    printf("In Scalar : \n");
    printU256Number(&scalar[i*NWORDS_256BIT]);
    printf("Out Scalar : \n");
    printU256Number(scl);
    */
    
    accum_n_zcoeff = pin[1+i];
    n_zcoeff = accum_n_zcoeff - prev_n_zcoeff;
    prev_n_zcoeff = accum_n_zcoeff;
    zcoeff_v_offset = zcoeff_d_offset + n_zcoeff;

    /*
    if ((i< 5) || (i > last_idx-5)){
      printf("N Zcoeff[%d] : %d\n", i, n_zcoeff);
      printf("Accum N Zcoeff[%d] : %d\n", i, accum_n_zcoeff);
      printf("Zcoeff D Offset : %d\n",zcoeff_d_offset);
      printf("ZCoeff_v_offset[%d] : %d\n", i , zcoeff_v_offset);
    }   
    */
    
    for (j=0; j< n_zcoeff; j++){
       zcoeff_d = pin[zcoeff_d_offset+j];
       zcoeff_v_in = &pin[zcoeff_v_offset+j*NWORDS_256BIT];
       zcoeff_v_out = &pout[zcoeff_d*NWORDS_256BIT];
       /*
       if ( ((i<5) || (i > last_idx-5)) && ((j<5) || (j>n_zcoeff-5))){
         printf("V[%d] in \n", zcoeff_d);
         printU256Number(zcoeff_v_in);
       }
       */
       montmult_h(zcoeff_v_in, zcoeff_v_in, scl, pidx);
       /*
       if ( ((i<5) || (i > last_idx-5)) && ((j<5) || (j>n_zcoeff-5))){
         printf("V[%d] in after mult \n", zcoeff_d);
         printU256Number(zcoeff_v_in);
         printf("V[%d] out before add \n", zcoeff_d);
         printU256Number(zcoeff_v_out);
       }
       */
       addm_h(zcoeff_v_out, zcoeff_v_out, zcoeff_v_in, pidx);
       /*
       if ( ((i<5) || (i > last_idx-5)) && ((j<5) || (j>n_zcoeff-5))){
         printf("V[%d] out after add \n", zcoeff_d);
         printU256Number(zcoeff_v_out);
       }
       */
    }
    zcoeff_d_offset = accum_n_zcoeff*(NWORDS_256BIT+1) +1 + n_zpoly;
  }
}

int r1cs_to_mpoly_h(uint32_t *pout, uint32_t *cin, cirbin_hfile_t *header, uint32_t extend)
{
  uint32_t **tmp_poly;
  uint32_t i,j,k;
  uint32_t poly_idx, const_offset, n_coeff,prev_n_coeff, coeff_offset, coeff_idx;
  uint32_t final_poly_size=0;
  const uint32_t *One = CusnarksOneGet();

  tmp_poly = (uint32_t **) calloc(header->nVars,sizeof(uint32_t *));

  for (i=0; i < header->nVars; i++){
     tmp_poly[i] = (uint32_t *)calloc(MAX_R1CSPOLYTMP_NWORDS * NWORDS_256BIT, sizeof(uint32_t));
  }
   
  const_offset = cin[0]+1;
  prev_n_coeff = 0;

  for (i=0; i < header->nConstraints; i++){
     n_coeff = cin[1+i];
     coeff_offset = const_offset + n_coeff - prev_n_coeff;
     for (j=0; j < n_coeff - prev_n_coeff ;j++){
       poly_idx = cin[const_offset+j];
       coeff_idx = tmp_poly[poly_idx][0]++;
       final_poly_size++;
       if (coeff_idx >= (MAX_R1CSPOLYTMP_NWORDS*(NWORDS_256BIT-1))/NWORDS_256BIT - 1){
           for(k=0;k < header->nVars;k++){
              free(tmp_poly[k]);
           }
           free(tmp_poly);
           return coeff_idx*NWORDS_256BIT/(NWORDS_256BIT-1);
       }
       tmp_poly[poly_idx][1+coeff_idx]=i;
       memcpy(&tmp_poly[poly_idx][1+MAX_R1CSPOLYTMP_NWORDS+coeff_idx*NWORDS_256BIT], &cin[coeff_offset] ,NWORDS_256BIT * sizeof(uint32_t));
       coeff_offset += NWORDS_256BIT;
     }
     const_offset += ((n_coeff - prev_n_coeff) * (NWORDS_256BIT+1));
     prev_n_coeff = n_coeff;
  }

  if (extend){
    for (i=0; i < header->nPubInputs + header->nOutputs + 1; i++){
       coeff_idx = tmp_poly[i][0]++;
       final_poly_size++;
       if (coeff_idx >= MAX_R1CSPOLYTMP_NWORDS*(NWORDS_256BIT-1)/NWORDS_256BIT - 1){
           for(k=0;k < header->nVars;k++){
              free(tmp_poly[k]);
           }
           free(tmp_poly);
           return coeff_idx*NWORDS_256BIT/(NWORDS_256BIT-1);
       }
       tmp_poly[i][1+coeff_idx]=i + header->nConstraints;
       memcpy(&tmp_poly[i][1+MAX_R1CSPOLYTMP_NWORDS+coeff_idx*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);
    }
  }
  #if 0
  for (j=0; j < header->nVars; j++){
    printf("\n\nPoly : %d\n",j);
    for (i=0; i <= tmp_poly[j][0]; i++){
      printf("%d\n",tmp_poly[j][i]);
      //if (i % 200 == 0) { printf("\n");}
    }
  }
  printf("\n\nEND\n");
  #endif

  final_poly_size = final_poly_size*(NWORDS_256BIT+1)+ header->nVars +1;
 
  if (MAX_R1CSPOLY_NWORDS*NWORDS_256BIT <= final_poly_size){
      for(k=0;k < header->nVars;k++){
         free(tmp_poly[k]);
      }
      free(tmp_poly);
      return -(final_poly_size/NWORDS_256BIT);
  } 

  poly_idx = 0;
  for (i=0; i<header->nVars;i++){
    coeff_idx = tmp_poly[i][0];
    if (coeff_idx == 0) {
      continue;
    }
    pout[1+poly_idx++] = coeff_idx;
  }
  pout[0] = poly_idx;

  const_offset = 0;
  for (i=0; i<header->nVars;i++){
    coeff_idx = tmp_poly[i][0];
    if (coeff_idx == 0) {
      continue;
    }
    memcpy(&pout[1+poly_idx+const_offset],&tmp_poly[i][1],sizeof(uint32_t)*coeff_idx);
    const_offset += coeff_idx;
    memcpy(&pout[1+poly_idx+const_offset],&tmp_poly[i][1+MAX_R1CSPOLYTMP_NWORDS],sizeof(uint32_t)*NWORDS_256BIT*coeff_idx);
    const_offset += (NWORDS_256BIT*coeff_idx);
    free(tmp_poly[i]);
  }

  free(tmp_poly);

  #if 0 
  for (i=0; i< final_poly_size; i++){
    //if (i % 200 == 0) { printf("\n");}
    //if (i== pout[0]+1) { printf("\nVals\n");}
    printf("%d\n",r1cs[i]);
  }
  #endif
  return 0;
}
/*
  Read header circuit binary file

  char * filename      : location of file to be written

  circuit bin header file format:
*/
void readU256CircuitFileHeader_h(cirbin_hfile_t *hfile, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");
  fread(&hfile->nWords, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->nPubInputs, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->nOutputs, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->nVars, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->nConstraints, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->R1CSA_nWords, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->R1CSB_nWords, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->R1CSC_nWords, sizeof(uint32_t), 1, ifp); 
  fclose(ifp);

}
/*
  Read circuit binary file

       
*/
void readU256CircuitFile_h(uint32_t *samples, const char *filename, uint32_t nwords=0)
{
  FILE *ifp = fopen(filename,"rb");
  uint32_t i=0;
  if (!nwords){
    while (!feof(ifp)){
      fread(&samples[i++], sizeof(uint32_t), 1, ifp); 
    }
  } else {
      fread(samples, sizeof(uint32_t), nwords, ifp); 
  }
  fclose(ifp);

}


/*
  Write circuit binary file

  t_uint32_t * samples : input vector containing samples. Vector is of length nwords 
  char * filename      : location of file to be written
  uint32_t nwords      : Number of samples to write.
*/
void writeU256CircuitFile_h(uint32_t *samples, const char *filename, uint32_t nwords)
{
  FILE *ifp = fopen(filename,"wb");
  fwrite(samples, sizeof(uint32_t), nwords, ifp); 
  fclose(ifp);

}

/*
  Read u256 data binary file and optionally decimate samples

  t_uint32_t * samples : output vector containing samples. Vector is of length outsize
  char * filename      : location of file containing samples
  uint32_t insize      : Number of samples from file to read. 
  uint32_t outsize     : Number of output samples. Samples are stored in vector with a 
                         insize/outsize ratio 
*/
void readU256DataFile_h(uint32_t *samples, const char *filename, uint32_t insize, uint32_t outsize)
{
  uint32_t i, j=0,k=0;
  uint32_t r[NWORDS_256BIT];
  FILE *ifp = fopen(filename,"rb");

  uint32_t count = insize/outsize;
  for (i=0;i<insize; i++){
    fread(r,sizeof(uint32_t),NWORDS_256BIT,ifp);
    if (j % count == 0){
      memcpy(&samples[k*NWORDS_256BIT], r, sizeof(uint32_t)*NWORDS_256BIT);
      k++;
    }
    j++;
  }
  
  fclose(ifp);
}
/*
  Display u256 samples
 
  TODO - substitute by log function
*/
void printU256Number(const uint32_t *x)
{
  for (uint32_t i=0; i < NWORDS_256BIT; i++){
    printf("%u ",x[i]);
  }
  printf ("\n");
}

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
   Generate N 256 bit random samples

   uint32_t *x       : output vector containing 256 bit samples. Vector is of length nsamples
   uint32_t nsamples : Number of 256 bit samples to generate
   uint32_t *p       : If different from null, samples will be less than p (p is a 256 bit number)
   
*/
void setRandom256(uint32_t *x, const uint32_t nsamples, const uint32_t *p)
{
  int j;
  _RNG* rng = _RNG::get_instance(x[0]);
  uint32_t nwords, nbits;


  memset(x,0,NWORDS_256BIT*sizeof(uint32_t));

  for (j=0; j < nsamples; j++){
    rng->randu32(&nwords,1);
    rng->randu32(&nbits,1);

    nwords %= NWORDS_256BIT;
    nbits %= 32;

    rng->randu32(&x[j*NWORDS_256BIT],nwords+1); 

    x[j*NWORDS_256BIT+nwords] &= ((1 << nbits)-1);
    if ((p!= NULL) && (nwords==NWORDS_256BIT) && (compu256_h(&x[j*NWORDS_256BIT], p) >= 0)){
         do{
           subu256_h(&x[j*NWORDS_256BIT], p);
         }while(compu256_h(&x[j*NWORDS_256BIT],p) >=0);
    }
  }
}
/* 
   Compare 256 bit integers X and Y.

   uint32_t *x : 256 bit integer x
   uint32_t *y : 256 bit integer y

   returns 
      0          : x == y
      pos number : x > y
      neg number : x < y
*/
int compu256_h(const uint32_t *x, const uint32_t *y)
{
  return mpCompare(x, y, NWORDS_256BIT);
}

/* 
   Compare 256 bit integers X and Y.

   uint32_t *x : 256 bit integer x
   uint32_t *y : 256 bit integer y

   returns 
      true          : x < y
      false         : x >= y
*/
bool ltu256_h(const uint32_t *x, const uint32_t *y)
{
  return (mpCompare(x, y, NWORDS_256BIT) < 0);
}

/*
   Generates N 256 bit samples with incremements of inc starting at start. If sample reached value of mod,
   value goes back to 0

   uint32_t *samples : Vector containing output samples. Vector is of length nsamples
   uint32_t nsamples : Number of samples to generate
   uint32_t *start   : First sample value 
   uint32_t inc      : sample increment 
   uint32_t *mod     : if different from NULL, it is maximum sample value. If generation reaches this value, it will go back to 0.  
*/
void rangeu256_h(uint32_t *samples, uint32_t nsamples, const uint32_t  *start, uint32_t inc, const uint32_t *mod)
{
   uint32_t i;
   uint32_t _inc[] = {inc,0,0,0,0,0,0,0};

   memcpy(samples,start,sizeof(uint32_t)*NWORDS_256BIT);

   for (i=1; i < nsamples; i++){
     mpAdd(&samples[i*NWORDS_256BIT], &samples[(i-1)*NWORDS_256BIT], _inc, NWORDS_256BIT);
     if ((mod != NULL) && (compu256_h(&samples[i*NWORDS_256BIT], mod) >= 0)){
         do{
           subu256_h(&samples[i*NWORDS_256BIT], mod);
         }while(compu256_h(&samples[i*NWORDS_256BIT],mod) >=0);
     }
   }
}

/* 
   Convert 256 bit number to montgomery representation of one of the two prime 
      p1 = 21888242871839275222246405745257275088696311157297823662689037894645226208583L
      p2 = 21888242871839275222246405745257275088548364400416034343698204186575808495617L

   uint32_t *z   : montgomery represention of input sample x. Z is 256 bits
   uint32_t *x   : input 256 bit sample
   uint32_t pidx : prime select. if 0, use p1. If 1, use p2
*/   
void to_montgomery_h(uint32_t *z, const uint32_t *x, uint32_t pidx)
{
  const uint32_t *R2 = CusnarksR2Get((mod_t)pidx);
  montmult_h(z,x,R2, pidx);
}

/* 
   Convert 256 bit number from montgomery representation of one of the two prime 
      p1 = 21888242871839275222246405745257275088696311157297823662689037894645226208583L
      p2 = 21888242871839275222246405745257275088548364400416034343698204186575808495617L

   uint32_t *z   : normal represention of input sample x. Z is 256 bits
   uint32_t *x   : input 256 bit sample in montgomery format
   uint32_t pidx : prime select. if 0, use p1. If 1, use p2
*/   
void from_montgomery_h(uint32_t *z, const uint32_t *x, uint32_t pidx)
{
  const uint32_t *one = CusnarksOneGet();
  montmult_h(z,x,one, pidx);
}


/*
    Removes higher order coefficient equal to 0

    uint32_t *pin    : poly vector
    uint32_t n_coeff : number of polynomial coefficients

    returns number of remaining coefficients
*/
uint32_t zpoly_norm_h(uint32_t *pin, uint32_t n_coeff)
{
  const uint32_t *Zero = CusnarksZeroGet();
  for (int i=n_coeff-1; i>=0; i--){ 
    if (mpCompare(&pin[i*NWORDS_256BIT],Zero,NWORDS_256BIT)){
       return (uint32_t) i+1;    
    }
  }
  return 0;
}

/*
   Sort 256-bit samples in ascending order.  Input samples indexes are actually sorted. Samples are
    left unsorted. 

   uint32_t *idx  : output vector containing sorted indexed. Size of idx is len.
   uint32_t *v    : input vector of size len 256 bit samples
   uint32_t len   : number of samples to sort 
  
*/
void sortu256_idx_h(uint32_t *idx, const uint32_t *v, uint32_t len)
{
  uint32_t i;

  for (i=0;i < len; i++){  
    idx[i] = i;
  }

   //std::sort(idx, idx+len, [&v](uint32_t i1, uint32_t i2){ return (v[i1*NWORDS_256BIT] < v[i2*NWORDS_256BIT]);});
   std::sort(idx, idx+len, 
       [&v](uint32_t i1, uint32_t i2){ 
         return (ltu256_h((const uint32_t*)&v[i1*NWORDS_256BIT],(const uint32_t *)&v[i2*NWORDS_256BIT]));});
}

/*
   Substract two 256 bit samples. x -= y
  
   uint32_t *x : vector x
   uint32_t *y : vector y
   
*/

void subu256_h(uint32_t *x, const uint32_t *y)
{
   uint32_t z[NWORDS_256BIT];

   mpSubtract(z, x, y, NWORDS_256BIT);
   memcpy(x,z,sizeof(uint32_t)*NWORDS_256BIT);
}   
/****************************************************************************/
/** [1]
*
* This function implements the Montgomery Modular Multiplication (MMM)
* Finely Integrated Operand Scanning (FIOS) algorithm. The FIOS method
* interleaves multiplication and reduction operations. Requires NDigits+3
* words of temporary storage.
*
* U = MontMult(A,B,N)
*
* Reference:
* Analyzing and Comparing Montgomery Multiplication Algorithms
* IEEE Micro, 16(3):26-33,June 1996
* By: Cetin Koc, Tolga Acar, and Burton Kaliski
*
* @param U is the MMM result
* @param A is the n-residue input, A' = A*R mod N
* @param B is the n-residue input, B' = B*R mod N
* @param N is the modulus
* @param NPrime is a pre-computed constant, NPrime = (1-R*Rbar)/N
* @param NDigits is the integer precision of the arguments (C,A,B,N,NPrime)
*
* @return None.
*
* @note  None.
*****************************************************************************/
void montmult_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx)
{
  int i, j;
  uint32_t S, C, C1, C2, M[2], X[2];
  uint32_t T[NWORDS_256BIT_FIOS];
  const uint32_t *N = CusnarksPGet((mod_t)pidx);
  const uint32_t *NPrime = CusnarksNPGet((mod_t)pidx);

  memset(T, 0, sizeof(uint32_t)*(NWORDS_256BIT_FIOS));

  for(i=0; i<NWORDS_256BIT; i++) {
    // (C,S) = t[0] + a[0]*b[i], worst case 2 words
    spMultiply(X, A[0], B[i]); // X[Upper,Lower] = a[0]*b[i]
    C = mpAdd(&S, T+0, X+0, 1); // [C,S] = t[0] + X[Lower]
    mpAdd(&C, &C, X+1, 1);// [~,C] = C + X[Upper], No carry

    //printf("0 - C : %u, S: %u\n",C,S);
    //printf("0 - A[0] : %u, B[i]: %u T[0] : %u\n",A[0],B[i], T[0]);
    // ADD(t[1],C)
    mpAddWithCarryProp(T, C, 1, NWORDS_256BIT_FIOS);
    //printf("T\n");
    //printU256Number(T);

    // m = S*n'[0] mod W, where W=2^32
    // Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
    spMultiply(M, S, NPrime[0]);
    //printf("M[0]:%u, M[1]: %u\n",M[0], M[1]);

    // (C,S) = S + m*n[0], worst case 2 words
    spMultiply(X, M[0], N[0]); // X[Upper,Lower] = m*n[0]
    C = mpAdd(&S, &S, X+0, 1); // [C,S] = S + X[Lower]
    mpAdd(&C, &C, X+1, 1);  // [~,C] = C + X[Upper]
    //printf("1 - C : %u, S: %u\n",C,S);

    for(j=1; j<NWORDS_256BIT; j++) {
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      spMultiply(X, A[j], B[i]);   // X[Upper,Lower] = a[j]*b[i], double precision
      C1 = mpAdd(&S, T+j, &C, 1);  // (C1,S) = t[j] + C
      //printf("2 - C1 : %u, S: %u\n",C1,S);
      C2 = mpAdd(&S, &S, X+0, 1);  // (C2,S) = S + X[Lower]
      //printf("3 - C2 : %u, S: %u\n",C1,S);
      //printf("X[0] : %u, X[1]: %u\n",X[0],X[1]);
      mpAdd(&C, &C1, X+1, 1);   // (~,C)  = C1 + X[Upper], doesn't produce carry
      //printf("4 - C : %u\n",C);
      mpAdd(&C, &C, &C2, 1);    // (~,C)  = C + C2, doesn't produce carry
      //printf("5 - C : %u\n",C);

      // ADD(t[j+1],C)
      mpAddWithCarryProp(T, C, j+1, NWORDS_256BIT_FIOS);
      //printf("T\n");
      //printU256Number(T);

      // (C,S) = S + m*n[j]
      spMultiply(X, M[0], N[j]); // X[Upper,Lower] = m*n[j]
      C = mpAdd(&S, &S, X+0, 1); // [C,S] = S + X[Lower]
      mpAdd(&C, &C, X+1, 1);  // [~,C] = C + X[Upper]
      //printf("6 - C : %u, S: %u\n",C,S);

      // t[j-1] = S
      T[j-1] = S;
      //printf("T\n");
      //printU256Number(T);
    }

    // (C,S) = t[s] + C
    C = mpAdd(&S, T+NWORDS_256BIT, &C, 1);
    //printf("6 - C : %u, S: %u\n",C,S);
    // t[s-1] = S
    T[NWORDS_256BIT-1] = S;
    // t[s] = t[s+1] + C
    mpAdd(T+NWORDS_256BIT, T+NWORDS_256BIT+1, &C, 1);
    // t[s+1] = 0
    T[NWORDS_256BIT+1] = 0;
  }

  /* Step 3: if(u>=n) return u-n else return u */
  if(mpCompare(T, N, NWORDS_256BIT) >= 0) {
    mpSubtract(T, T, N, NWORDS_256BIT);
  }

  memcpy(U, T, sizeof(uint32_t)*NWORDS_256BIT);
}

// I am leaving this as a separate function to test both implementations are equal
void montsquare_h(uint32_t *U, const uint32_t *A, uint32_t pidx)
{
  int i, j;
  uint32_t S, C, C1, C2, M[2], X[2], X1[2], carry;
  uint32_t T[NWORDS_256BIT_FIOS];
  const uint32_t *N = CusnarksPGet((mod_t)pidx);
  const uint32_t *NPrime = CusnarksNPGet((mod_t)pidx);

  memset(T, 0, sizeof(uint32_t)*(NWORDS_256BIT_FIOS));

  for(i=0; i<NWORDS_256BIT; i++) {
    // (C,S) = t[0] + a[0]*b[i], worst case 2 words
    spMultiply(X, A[i], A[i]); // X[Upper,Lower] = a[0]*b[i]
    C = mpAdd(&S, T+0, X+0, 1); // [C,S] = t[0] + X[Lower]
    mpAdd(&C, &C, X+1, 1);  // [~,C] = C + X[Upper], No carry

    // ADD(t[1],C)
    mpAddWithCarryProp(T, C, 1, NWORDS_256BIT_FIOS);
    //carry = mpAdd(&T[1], &T[1], &C, 1); 

    // m = S*n'[0] mod W, where W=2^32
    // Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
    spMultiply(M, S, NPrime[0]);

    // (C,S) = S + m*n[0], worst case 2 words
    spMultiply(X, M[0], N[0]); // X[Upper,Lower] = m*n[0]
    C = mpAdd(&S, &S, X+0, 1); // [C,S] = S + X[Lower]
    mpAdd(&C, &C, X+1, 1);  // [~,C] = C + X[Upper]
    
    
    // (C,S) = t[j] + a[i+1]*a[i+1] + C, worst case 2 words
    spMultiply(X1, A[i+1], A[i]);
    C1 = mpAdd(&S, &T[i+1], &C, 1);  // (C1,S) = t[i+1] + C
    C2 = mpAdd(&S, &S, X1+0, 1);  // (C2,S) = S + X[Lower]
    mpAdd(&C, &C1, X1+1, 1);   // (~,C)  = C1 + X[Upper], doesn't produce carry
    mpAdd(&C, &C, &C2, 1);    // (~,C)  = C + C2, doesn't produce carry
   
    // ADD(t[i+2],C)
    mpAddWithCarryProp(T, C, i+2, NWORDS_256BIT_FIOS);
   
    // (C,S) = S + m*n[j]
    spMultiply(X, M[0], N[1]); // X[Upper,Lower] = m*n[j]
    C = mpAdd(&S, &S, X+0, 1); // [C,S] = S + X[Lower]
    C1 = mpAdd(&S, &S, X1+0, 1); // [C,S] = S + X[Lower]
    C+=C1;
    C1=mpAdd(&S, &X1[1], X+1, 1);  // [~,C] = C + X[Upper]
    C+=C1;
    mpAdd(&C, &C, &S, 1);  // [~,C] = C + X[Upper]
   
    // t[j-1] = S
    T[0] = S;

    C2=0;
    for(j=2; j<NWORDS_256BIT; j++) {
      // (C,S) = t[j] + 2*a[j]*a[i] + C, worst case 2 words
      spMultiply(X, A[j], A[i]);   // X[Upper,Lower] = a[j]*b[i], double precision
      C1 = (X[0] >> 31)+C2;
      X[0] <<= 1;
      C2 = X[1] >> 31;
      X[1] = (X[1] << 1) + C1;
      C1 = mpAdd(&X[0], &X[0], &C, 1);
      C = mpAdd(&S, &T[j], &X[0], 1);
      C += C1;
      mpAdd(&C, &C, X+1, 1);  

      // ADD(t[j+1],C)
      mpAddWithCarryProp(T, C, j+1, NWORDS_256BIT_FIOS);
   
      // (C,S) = S + m*n[j]
      spMultiply(X, M[0], N[j]); // X[Upper,Lower] = m*n[j]
      C = mpAdd(&S, &S, X+0, 1); // [C,S] = S + X[Lower]
      mpAdd(&C, &C, X+1, 1);  // [~,C] = C + X[Upper]
   
      T[j-1] = S;
    }

    // (C,S) = t[s] + C
    C = mpAdd(&S, T+NWORDS_256BIT, &C, 1);
    //printf("6 - C : %u, S: %u\n",C,S);
    // t[s-1] = S
    T[NWORDS_256BIT-1] = S;
    // t[s] = t[s+1] + C
    mpAdd(T+NWORDS_256BIT, T+NWORDS_256BIT+1, &C, 1);
    // t[s+1] = 0
    T[NWORDS_256BIT+1] = 0;
  }

  /* Step 3: if(u>=n) return u-n else return u */
  if(compu256_h(T, N) >= 0) {
    mpSubtract(T, T, N, NWORDS_256BIT);
  }

  memcpy(U, T, sizeof(uint32_t)*NWORDS_256BIT);
}


// Improved speed (in Cuda at least) by substituting mpAddWithCarryProp by mpAdd
// I am leaving this as a separate function to test both implementations are equal
void montmult_h2(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx)
{
  int i, j;
  uint32_t S, C, C1, C2, M[2], X[2], carry;
  uint32_t T[NWORDS_256BIT_FIOS];
  const uint32_t *NPrime = CusnarksNPGet((mod_t)pidx);
  const uint32_t *N = CusnarksPGet((mod_t)pidx);

  memset(T, 0, sizeof(uint32_t)*(NWORDS_256BIT_FIOS));

  printf("A\n");
  printU256Number(A);
  printf("B\n");
  printU256Number(B);
  for(i=0; i<NWORDS_256BIT; i++) {
    // (C,S) = t[0] + a[0]*b[i], worst case 2 words
    spMultiply(X, A[0], B[i]); // X[Upper,Lower] = a[0]*b[i]
    C = mpAdd(&S, T+0, X+0, 1); // [C,S] = t[0] + X[Lower]
    mpAdd(&C, &C, X+1, 1);  // [~,C] = C + X[Upper], No carry

    printf("0 - C : %u, S: %u\n",C,S);
    //printf("0 - A[0] : %u, B[i]: %u T[0] : %u\n",A[0],B[i], T[0]);
    // ADD(t[1],C)
    //mpAddWithCarryProp(T, C, 1);
    carry = mpAdd(&T[1], &T[1], &C, 1); 
    printf("C3: %d\n",carry);
    printf("T\n");
    printU256Number(T);

    // m = S*n'[0] mod W, where W=2^32
    // Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
    spMultiply(M, S, NPrime[0]);
    printf("M[0]:%u, M[1]: %u\n",M[0], M[1]);

    // (C,S) = S + m*n[0], worst case 2 words
    spMultiply(X, M[0], N[0]); // X[Upper,Lower] = m*n[0]
    C = mpAdd(&S, &S, X+0, 1); // [C,S] = S + X[Lower]
    mpAdd(&C, &C, X+1, 1);  // [~,C] = C + X[Upper]
    printf("1 - C : %u, S: %u\n",C,S);

    for(j=1; j<NWORDS_256BIT; j++) {
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      spMultiply(X, A[j], B[i]);   // X[Upper,Lower] = a[j]*b[i], double precision
      C1 = mpAdd(&S, T+j, &C, 1);  // (C1,S) = t[j] + C
      printf("2 - C1 : %u, S: %u\n",C1,S);
      C2 = mpAdd(&S, &S, X+0, 1);  // (C2,S) = S + X[Lower]
      printf("3 - C2 : %u, S: %u\n",C2,S);
      printf("X[0] : %u, X[1]: %u\n",X[0],X[1]);
      mpAdd(&C, &C1, X+1, 1);   // (~,C)  = C1 + X[Upper], doesn't produce carry
      printf("4 - C : %u\n",C);
      mpAdd(&C, &C, &C2, 1);    // (~,C)  = C + C2, doesn't produce carry
      printf("5 - C : %u\n",C);
   
      // ADD(t[j+1],C)
      C += carry;
      carry = mpAdd(&T[j+1], &T[j+1], &C, 1); 
      //mpAddWithCarryProp(T, C, j+1);
      printf("T\n");
      printU256Number(T);
   
      // (C,S) = S + m*n[j]
      spMultiply(X, M[0], N[j]); // X[Upper,Lower] = m*n[j]
      C = mpAdd(&S, &S, X+0, 1); // [C,S] = S + X[Lower]
      mpAdd(&C, &C, X+1, 1);  // [~,C] = C + X[Upper]
      //printf("6 - C : %u, S: %u\n",C,S);
   
      // t[j-1] = S
      T[j-1] = S;
      printf("T\n");
      printU256Number(T);
    }

    mpAddWithCarryProp(T, carry, NWORDS_256BIT, NWORDS_256BIT_FIOS);
    // (C,S) = t[s] + C
    C = mpAdd(&S, T+NWORDS_256BIT, &C, 1);
    printf("6 - C : %u, S: %u\n",C,S);
    // t[s-1] = S
    T[NWORDS_256BIT-1] = S;
    // t[s] = t[s+1] + C
    mpAdd(T+NWORDS_256BIT, T+NWORDS_256BIT+1, &C, 1);
    // t[s+1] = 0
    T[NWORDS_256BIT+1] = 0;
  }

  /* Step 3: if(u>=n) return u-n else return u */
  if(compu256_h(T, N) >= 0) {
    mpSubtract(T, T, N, NWORDS_256BIT);
  }

  memcpy(U, T, sizeof(uint32_t)*NWORDS_256BIT);
}

/****************************************************************************/
/** [1]
*
* This function implements the Montgomery Modular Multiplication (MMM)
*  (SOS) algorithm.
*
* U = MontMult(A,B,N)
*
* Reference:
* Analyzing and Comparing Montgomery Multiplication Algorithms
* IEEE Micro, 16(3):26-33,June 1996
* By: Cetin Koc, Tolga Acar, and Burton Kaliski
*
* @param U is the MMM result
* @param A is the n-residue input, A' = A*R mod N
* @param B is the n-residue input, B' = B*R mod N
* @param N is the modulus
* @param NPrime is a pre-computed constant, NPrime = (1-R*Rbar)/N
* @param NDigits is the integer precision of the arguments (C,A,B,N,NPrime)
*
* @return None.
*
* @note  None.
*****************************************************************************/
void montmult_sos_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx)
{
 int i, j;
 uint32_t S, C, C1, C2, M[]={0,0,0}, X[]={0,0};
 uint32_t T[NWORDS_256BIT_SOS];
 const uint32_t *NPrime = CusnarksNPGet((mod_t)pidx);
 const uint32_t *N = CusnarksPGet((mod_t)pidx);
 uint32_t C3=0,C4;
 C2 = 0;
 memset(T, 0, sizeof(uint32_t)*(NWORDS_256BIT_SOS));

 if (memcmp(A,B,NWORDS_256BIT*sizeof(uint32_t))){
    for(i=0; i<NWORDS_256BIT; i++) {
       C = 0;
       for (j=0; j<NWORDS_256BIT; j++){
          //(C,S) := t[i+j] + a[j]*b[i] + C
          spMultiply(X, A[j], B[i]);   
          C1 = mpAdd(&X[0], &X[0], &C, 1);
          C = mpAdd(&S, &T[i+j], &X[0], 1);
          C +=C1;
          mpAdd(&C, &C, X+1, 1);  
          T[i+j] = S;
       } 
       T[i+NWORDS_256BIT] = C;
    }
  } else {
     // squaring bit
     for(i=0; i<NWORDS_256BIT; i++) {
       //(C,S) := t[i+i] + a[i]*a[i]
       spMultiply(X, A[i], A[i]);
       C = mpAdd(&S, &T[i+i], &X[0], 1);
       mpAdd(&C, &C, X+1, 1);  
       T[i+i] = S;
       for (j=i+1; j<NWORDS_256BIT; j++){
         //(C,S) := t[i+j] + 2*a[j]*a[i] + C
         spMultiply(X, A[j], A[i]);
         C1 = (X[0] >> 31)+C2;
         X[0] <<= 1;
         C2 = X[1] >> 31;
         X[1] = (X[1] << 1) + C1;
         C1 = mpAdd(&X[0], &X[0], &C, 1);
         C = mpAdd(&S, &T[i+j], &X[0], 1);
         C += C1;
         mpAdd(&C, &C, X+1, 1);  
         T[i+j] = S;
       } 
       T[i+NWORDS_256BIT] += C;
     }
  }

  for (i=0; i<NWORDS_256BIT;i++){
    C = 0;
    //m := t[i]*n'[0] mod W
    spMultiply(M, T[i], NPrime[0]);
    for (j=0; j< NWORDS_256BIT; j++){
         //(C,S) := t[i+j] + m*n[j] + C
         spMultiply(X, M[0], N[j]);
         C1 = mpAdd(&X[0], &X[0], &C, 1);
         C = mpAdd(&S, &T[i+j], &X[0], 1);
         C +=C1;
	 //C += C3[i+j+1];
	 //C3[i+j+1] = 0;
	 C += ((C3 >> (i+j+1)) & 1);
	 C3 &= (0xFFFFFFFF ^ (1 << (1+j+i))); 
         ////printf("2 - %x, %d\n",C3, i+j+1);
         mpAdd(&C, &C, X+1, 1);  
	 T[i+j] = S;
    }
    //ADD (t[i+s],C)
    C4 = mpAdd(&T[i+NWORDS_256BIT], &T[i+NWORDS_256BIT], &C, 1);
    C3 |= (C4 << (i+NWORDS_256BIT+1));  
    //printf("%x, %d\n",C3, i+NWORDS_256BIT+1);
  }
  //printU256Number(&C3[NWORDS_256BIT]);
  memcpy(U,&T[NWORDS_256BIT],(NWORDS_256BIT)*sizeof(uint32_t));

 /* Step 3: if(u>=n) return u-n else return u */
 if(compu256_h(U, N) >= 0) {
    mpSubtract(U, U, N, NWORDS_256BIT);
 }

}





/*
  Recursive 4 step N 256 bit sample FFT. Read https://www.davidhbailey.com/dhbpapers/fftq.pdf for more info
   1) Get input samples in a N=N1xN2 matrix, filling the matrix by columns. Compute N1 N2 point FFT (FFT of every row)
   2) Multiply resulting N1xN2 Ajk matrix by root[j*k]
   3) Transpose resulting matrix to N2xN1 matrix
   4) Perform N2 N1 point FFT

   Retrieve data columnwise from the resulting N2xN1 matrix
   Function is 2D because when computing FFT of rows/columns, the 4-step procedure is repeated

   uint32_t *A     : Input vector containing ordered samples in Montgomery format. Input vector has 1<<(Nrows+Ncols) samples. Resulting FFT is returned in vector A
                        Output samples are ordered
   uint32_t *roots : input roots (first root is 1) in Montgomery. If roots are inverse, IFFT is computed
   uint32_t Nrows  : Number of rows in starting matrix (N1)
   uint32_t fft_Nyx   : Number of columns N12 in secondary matrix (N1=N11xN12)
   uint32_t Ncols  : Number of columns in starting matrix (N2)
   uint32_t fft_Nxx : Number of columns N22 in secondary matrix (N2=N21xN22)
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
   uint32_t mode    : Debug mode. If 0, run normal FFT operation. 
                                  If 1, stop after first step
                                  If 2, stop after second step
                                  If 3, stop after third step
*/
void ntt_parallel2D_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, uint32_t pidx, uint32_t mode)
{
  uint32_t Anrows = (1<<Nrows);
  uint32_t Ancols = (1<<Ncols);
  uint32_t Mnrows = Ancols;
  uint32_t Mncols = Anrows;
  uint32_t *M = (uint32_t *) malloc (Anrows * Ancols * NWORDS_256BIT * sizeof(uint32_t));
  uint32_t *reducedR = (uint32_t *) malloc (MAX(Mncols,Mnrows) * NWORDS_256BIT * sizeof(uint32_t));
  uint32_t i,j;
  uint32_t tmp_mode = mode;
  

  transpose_h(M,A,Anrows, Ancols);

  for(i=0;i<Mncols;i++){
    memcpy(&reducedR[i*NWORDS_256BIT], &roots[i*NWORDS_256BIT*Mnrows],sizeof(uint32_t)*NWORDS_256BIT);
  }

  if (mode == 3) { tmp_mode = 0;}

  for (i=0;i < Mnrows; i++){
    ntt_parallel_h(&M[i*NWORDS_256BIT*Mncols], reducedR, Nrows - fft_Nyx, fft_Nyx, pidx, tmp_mode);
  }
  

  if (mode == 1) { 
     memcpy(A,M,Ancols * Anrows * NWORDS_256BIT * sizeof(uint32_t));
     return; 
  }
 
  for (i=0;i < Mnrows; i++){
    for (j=0;j < Mncols; j++){   
        montmult_h(&M[i*NWORDS_256BIT*Mncols+j*NWORDS_256BIT], &M[i*NWORDS_256BIT*Mncols+j*NWORDS_256BIT], &roots[i*j*NWORDS_256BIT], pidx);
    }
  }
  transpose_h(A,M,Mnrows, Mncols);

  if (mode == 2){
     return;
  }

  for(i=0;i<Mnrows;i++){
    memcpy(&reducedR[i*NWORDS_256BIT], &roots[i*NWORDS_256BIT*Mncols],sizeof(uint32_t)*NWORDS_256BIT);
  }

  for (i=0;i < Anrows; i++){
    ntt_parallel_h(&A[i*NWORDS_256BIT*Ancols], reducedR, Ncols - fft_Nxx, fft_Nxx, pidx, mode);
  }

  transpose_h(M,A,Anrows, Ancols);
  memcpy(A,M,Ancols * Anrows * NWORDS_256BIT * sizeof(uint32_t));

  //for (i=0;i < Anrows; i++){
    //for (j=0;j < Ancols; j++){   
        //printf("OUT(:%d/%d)\n",i,j); 
        //printU256Number(&A[i*NWORDS_256BIT*Ancols+j*NWORDS_256BIT]);
    //}
  //}

  free(M);
  free(reducedR);
}

/*
  Recursive 4 step N 256 bit sample IFFT. Read https://www.davidhbailey.com/dhbpapers/fftq.pdf for more info
   1) Get input samples in a N=N1xN2 matrix, filling the matrix by columns. Compute N1 N2 point FFT (FFT of every row)
   2) Multiply resulting N1xN2 Ajk matrix by inv_root[j*k]
   3) Transpose resulting matrix to N2xN1 matrix
   4) Perform N2 N1 point FFT
   5) Divide each coefficient by number of samples

   Retrieve data columnwise from the resulting N2xN1 matrix
   Function is 2D because when computing FFT of rows/columns, the 4-step procedure is repeated

   uint32_t *A     : Input vector containing ordered samples in Montgomery format. Input vector has 1<<(Nrows+Ncols) samples. Resulting FFT is returned in vector A
                        Output samples are ordered
   uint32_t *roots : input roots (first root is 1) in Montgomery. If roots are inverse, IFFT is computed
   uint32_t *format : if 0, output is in normal format. If 1, outout is montgomery
   uint32_t Nrows  : Number of rows in starting matrix (N1)
   uint32_t fft_Nyx   : Number of columns N12 in secondary matrix (N1=N11xN12)
   uint32_t Ncols  : Number of columns in starting matrix (N2)
   uint32_t fft_Nxx : Number of columns N22 in secondary matrix (N2=N21xN22)
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
   uint32_t mode    : Debug mode. If 0, run normal FFT operation. 
                                  If 1, stop after first step
                                  If 2, stop after second step
                                  If 3, stop after third step
*/

void intt_parallel2D_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, uint32_t pidx, uint32_t mode)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);

  ntt_parallel2D_h(A, roots, Nrows, fft_Nyx,  Ncols, fft_Nxx, pidx, mode);

  for (i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&A[i*NWORDS_256BIT], &A[i*NWORDS_256BIT], &scaler[(Nrows + Ncols)*NWORDS_256BIT], pidx);
  }
}

/*
  4 step N 256 bit sample FFT. Read https://www.davidhbailey.com/dhbpapers/fftq.pdf for more info
   1) Get input samples in a N=N1xN2 matrix, filling the matrix by columns. Compute N1 N2 point FFT (FFT of every row)
   2) Multiply resulting N1xN2 Ajk matrix by root[j*k]
   3) Transpose resulting matrix to N2xN1 matrix
   4) Perform N2 N1 point FFT

   Retrieve data columnwise from the resulting N2xN1 matrix

   uint32_t *A     : Input vector containing ordered samples in Montgomery format. Input vector has 1<<(Nrows+Ncols) samples. Resulting FFT is returned in vector A
                        Output samples are ordered
   uint32_t *roots : input roots (first root is 1) in Montgomery. If roots are inverse, IFFT is computed
   uint32_t Nrows  : Number of rows in starting matrix (N1)
   uint32_t Ncols  : Number of columns in starting matrix (N2)
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
   uint32_t mode    : Debug mode. If 0, run normal FFT operation. 
                                  If 1, stop after first step
                                  If 2, stop after second step
                                  If 3, stop after third step
*/
void ntt_parallel_h(uint32_t *A, const uint32_t *roots, uint32_t Ncols, uint32_t Nrows, uint32_t pidx, uint32_t mode)
{
  uint32_t Anrows = (1<<Nrows);
  uint32_t Ancols = (1<<Ncols);
  uint32_t Mnrows = Ancols;
  uint32_t Mncols = Anrows;
  uint32_t *M = (uint32_t *) malloc (Anrows * Ancols * NWORDS_256BIT * sizeof(uint32_t));
  uint32_t *reducedR = (uint32_t *) malloc (MAX(Mncols/2,Mnrows/2) * NWORDS_256BIT * sizeof(uint32_t));
  uint32_t i,j;
  

  transpose_h(M,A,Anrows, Ancols);

  for(i=0;i<Mncols/2;i++){
    memcpy(&reducedR[i*NWORDS_256BIT], &roots[i*NWORDS_256BIT*Mnrows],sizeof(uint32_t)*NWORDS_256BIT);
  }


  for (i=0;i < Mnrows; i++){
    ntt_h(&M[i*NWORDS_256BIT*Mncols], reducedR, Nrows, pidx);
    for (j=0;j < Mncols; j++){  
      montmult_h(&M[i*NWORDS_256BIT*Mncols+j*NWORDS_256BIT], &M[i*NWORDS_256BIT*Mncols+j*NWORDS_256BIT], &roots[i*j*NWORDS_256BIT], pidx);
    }
  }

  transpose_h(A,M,Mnrows, Mncols);

  if ( (mode == 1) || (mode == 3) ){
    return;
  }

  for(i=0;i<Mnrows/2;i++){
    memcpy(&reducedR[i*NWORDS_256BIT], &roots[i*NWORDS_256BIT*Mncols],sizeof(uint32_t)*NWORDS_256BIT);
  }

  for (i=0;i < Anrows; i++){
    ntt_h(&A[i*NWORDS_256BIT*Ancols], reducedR, Ncols, pidx);
  }

  transpose_h(M,A,Anrows, Ancols);
  memcpy(A,M,Ancols * Anrows * NWORDS_256BIT * sizeof(uint32_t));

  free(M);
  free(reducedR);
}

/*
  4 step N 256 bit sample IFFT. Read https://www.davidhbailey.com/dhbpapers/fftq.pdf for more info
   1) Get input samples in a N=N1xN2 matrix, filling the matrix by columns. Compute N1 N2 point FFT (FFT of every row)
   2) Multiply resulting N1xN2 Ajk matrix by inv_root[j*k]
   3) Transpose resulting matrix to N2xN1 matrix
   4) Perform N2 N1 point FFT
   5) Divide each coefficient by number of samples

   Retrieve data columnwise from the resulting N2xN1 matrix

   uint32_t *A     : Input vector containing ordered samples in Montgomery format. Input vector has 1<<(Nrows+Ncols) samples. Resulting FFT is returned in vector A
                        Output samples are ordered
   uint32_t *roots : input roots (first root is 1) in Montgomery. If roots are inverse, IFFT is computed
   uint32_t *format : if 0, output is in normal format. If 1, outout is montgomery
   uint32_t Nrows  : Number of rows in starting matrix (N1)
   uint32_t Ncols  : Number of columns in starting matrix (N2)
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
   uint32_t mode    : Debug mode. If 0, run normal FFT operation. 
                                  If 1, stop after first step
                                  If 2, stop after second step
                                  If 3, stop after third step
*/
void intt_parallel_h(uint32_t *A, const uint32_t *roots,uint32_t format, uint32_t Nrows, uint32_t Ncols, uint32_t pidx, uint32_t mode)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);

  ntt_parallel_h(A, roots, Ncols, Nrows, pidx, mode);

  for (i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&A[i*NWORDS_256BIT], &A[i*NWORDS_256BIT], &scaler[(Nrows + Ncols)*NWORDS_256BIT], pidx);
  }
}


/*
   Computes the forward number-theoretic transform of the given vector in place,
   with respect to the given primitive nth root of unity under the given modulus.
   The length of the vector must be a power of 2.

   NOTE https://www.nayuki.io/page/number-theoretic-transform-integer-dft

   uint32_t *A     : ordered input vector of length 1<<levels in montgomery format. Ordered result is 
                  is stored in A as well.
   uint32_t *roots : input roots (first root is 1) in Montgomery. If roots are inverse, IFFT is computed. 
   uint32_t levels : 1<<levels is the number of samples in the FFT
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void ntt_h(uint32_t *A, const uint32_t *roots, uint32_t levels, uint32_t pidx)
{
   uint32_t *vector = A;
   uint32_t n = 1 << levels;
   uint32_t i,j,k,l,size, halfsize, tablestep;
   uint32_t left[NWORDS_256BIT], right[NWORDS_256BIT];

   for (i=0; i < n ; i++){
      j = reverse(i, levels);
      if (j > i){
         swap(&vector[i*NWORDS_256BIT],&vector[j*NWORDS_256BIT]);
      }
   }

   size = 2;
   while (size <= n){
     halfsize = size >> 1; 
     tablestep = n/size;
     for (i=0; i<n; i+=size){
        k = 0;
        for (j=i; j<i+halfsize; j++){
           l = j + halfsize;
           memcpy(left, &vector[j*NWORDS_256BIT], sizeof(uint32_t)*NWORDS_256BIT);
           montmult_h(right,&vector[l*NWORDS_256BIT], &roots[k*NWORDS_256BIT], pidx);
           addm_h(&vector[j*NWORDS_256BIT], left, right, pidx);
           subm_h(&vector[l*NWORDS_256BIT], left, right, pidx);
           k += tablestep;
        }
     }
     size *= 2;
  }
}

/*
   Computes the inverse number-theoretic transform of the given vector in place,
   with respect to the given primitive nth root of unity under the given modulus.
   The length of the vector must be a power of 2.

   NOTE https://www.nayuki.io/page/number-theoretic-transform-integer-dft

   uint32_t *A     : ordered input vector of length 1<<levels in montgomery format. Ordered result is 
                  is stored in A as well.
   uint32_t *roots : input inverse roots (first root is 1) in Montgomery. 
   uint32_t *format : if 0, output is in normal format. If 1, outout is montgomery
   uint32_t levels : 1<<levels is the number of samples in the FFT
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void intt_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, uint32_t pidx)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);
  
  ntt_h(A, roots, levels, pidx);

  for (i=0; i< (1<<levels); i++){
     montmult_h(&A[i*NWORDS_256BIT], &A[i*NWORDS_256BIT], &scaler[levels * NWORDS_256BIT], pidx);
  }
  
}

/*
  Generate format of FFT from number of samples. Parameters include 1D FFT/2D FFT/ 3D FFT/ 4D FFT, size of 
   matrix for multi D FFT,...

  fft_params_t *ntt_params : pointer to structure containing resulting FFT format
  uint32_t nsamples : number of samples of FFT
  
*/
void ntt_build_h(fft_params_t *ntt_params, uint32_t nsamples)
{
  uint32_t levels = (uint32_t) ceil(log2(nsamples));
  memset(ntt_params,0,sizeof(fft_params_t));
  ntt_params->padding = (1 << levels) - nsamples;
  ntt_params->levels = levels;
  
  if (nsamples <= 32){
    ntt_params->fft_type =  FFT_T_1D;
    ntt_params->fft_N[0] = levels;

  } else if (nsamples <= 1024) {
    ntt_params->fft_type =  FFT_T_2D;
    ntt_params->fft_N[(1<<FFT_T_2D)-1] = levels/2;
    ntt_params->fft_N[(1<<FFT_T_2D)-2] = levels - levels/2;

  } else if (nsamples <= (1<<20) ) {
    ntt_params->fft_type =  FFT_T_3D;
    ntt_params->fft_N[(1<<FFT_T_3D)-1] = levels/2;
    ntt_params->fft_N[(1<<FFT_T_3D)-2] = levels - levels/2;
    ntt_params->fft_N[(1<<FFT_T_3D)-3] = levels/4;
    ntt_params->fft_N[(1<<FFT_T_3D)-4] = (levels - levels/2)/2;

  } else {
    ntt_params->fft_type =  FFT_T_4D;
    ntt_params->fft_N[(1<<FFT_T_4D)-1] = levels/2;
    ntt_params->fft_N[(1<<FFT_T_4D)-2] = levels - levels/2;
    ntt_params->fft_N[(1<<FFT_T_4D)-3] = levels/4;
    ntt_params->fft_N[(1<<FFT_T_4D)-4] = (levels - levels/2)/2;
    
    levels = ntt_params->fft_N[(1<<FFT_T_4D)-1];
    ntt_params->fft_N[(1<<FFT_T_4D)-5] = levels/2;
    ntt_params->fft_N[(1<<FFT_T_4D)-6] = levels - levels/2;
    ntt_params->fft_N[(1<<FFT_T_4D)-7] = levels/4;
    ntt_params->fft_N[(1<<FFT_T_4D)-8] = (levels - levels/2)/2;
  }
}

/*
  modular addition of 256 bit numbers : Z = X + Y mod P

  uint32_t *z : Output 256 bit number
  uint32_t *x : Input 256 bit number 1
  uint32_t *y : Input 256 bit number 2
  uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void addm_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
   uint32_t tmp[NWORDS_256BIT];
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   mpAdd(tmp, x, y, NWORDS_256BIT);
   if(mpCompare(tmp, N, NWORDS_256BIT) >= 0) {
      mpSubtract(tmp, tmp, N, NWORDS_256BIT);
   }

   memcpy(z, tmp, sizeof(uint32_t)*NWORDS_256BIT);
}

/*
  modular substraction of 256 bit numbers : Z = X - Y mod P

  uint32_t *z : Output 256 bit number
  uint32_t *x : Input 256 bit number 1
  uint32_t *y : Input 256 bit number 2
  uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void subm_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
   uint32_t tmp[NWORDS_256BIT];
   const uint32_t *N = CusnarksPGet((mod_t)pidx);

   mpSubtract(tmp, x, y, NWORDS_256BIT);
   if(mpCompare(tmp, N, NWORDS_256BIT) >= 0) {
       mpAdd(tmp, tmp, N, NWORDS_256BIT);
   }

   memcpy(z, tmp, sizeof(uint32_t)*NWORDS_256BIT);
}

/*
  Computes N roots of unity from a given primitive root. Roots are in montgomery format

  uint32_t *roots : Output vector containing computed roots. Size of vector is nroots
  uint32_t *primitive_root : Primitive root 
  uint32_t nroots : Number of roots
  uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void find_roots_h(uint32_t *roots, const uint32_t *primitive_root, uint32_t nroots, uint32_t pidx)
{
  uint32_t i;
  const uint32_t *_1 = CusnarksOnMonteGet((mod_t)pidx);
  
  memcpy(roots,_1,sizeof(uint32_t)*NWORDS_256BIT);
  memcpy(&roots[NWORDS_256BIT],primitive_root,sizeof(uint32_t)*NWORDS_256BIT);
  for (i=2;i<nroots; i++){
    montmult_h(&roots[i*NWORDS_256BIT], &roots[(i-1)*NWORDS_256BIT], primitive_root, pidx);
  }

  return;
}

