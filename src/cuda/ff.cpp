
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
//   Util functions for host. Functions mostly implement BI bit arithmetic and polynomial of finite field elementss
//
//   BI bit samples are represented by N 32-bit words (NWORDS_FP/NWORDS_FR), where word 0 is the least significant word
//
//   A polynomial of degree N is represented by N+1 BI bit samples ((N+1) * NWORDS_FR/NWORDS_FP), where degree 0 coefficient
//   is stored in the first NWORDS_FR/NWORDS_FP words
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
#include <omp.h>

#include "types.h"
#include "constants.h"
#include "rng.h"
#include "log.h"
#include "bigint.h"
#include "ff.h"

#ifdef _CASM
#include "fr.h"
#include "fp.h"
#endif

#ifdef PARALLEL_EN
static  uint32_t parallelism_enabled =  1;
#else
static  uint32_t parallelism_enabled =  0;
#endif

static void almmontinv_h(uint32_t *r, uint32_t *k, uint32_t *a, uint32_t pidx);

t_subm getcb_subm_h( uint32_t pidx)
{
  t_subm subm_cb;

#ifdef _CASM
  if (pidx == MOD_FP){
     subm_cb = &Fp_rawSub;
  } else {
     subm_cb = &Fr_rawSub;
  }
#else
  if (pidx == MOD_FP){
     subm_cb = &submp_h;
  } else {
     subm_cb = &submr_h;
  }
#endif
  return subm_cb;
}

t_addm getcb_addm_h( uint32_t pidx)
{
  t_addm addm_cb;

#ifdef _CASM
  if (pidx == MOD_FP){
     addm_cb = &Fp_rawAdd;
  } else {
     addm_cb = &Fr_rawAdd;
  }
#else
  if (pidx == MOD_FP){
     addm_cb = &addmp_h;
  } else {
     addm_cb = &addmr_h;
  }
#endif
  return addm_cb;
}

t_mulm getcb_mulm_h( uint32_t pidx)
{
  t_mulm mulm_cb;
  
#ifdef _CASM
  if (pidx == MOD_FP){
     mulm_cb = &Fp_rawMMul;
  } else {
     mulm_cb = &Fr_rawMMul;
  }
#else

  if (pidx == MOD_FP){
     mulm_cb = &montmultp_h;
  } else {
     mulm_cb = &montmultr_h;
  }
#endif
  return mulm_cb;
}

t_sqm getcb_sqm_h( uint32_t pidx)
{
  t_sqm sqm_cb;
  if (pidx == MOD_FP){
     sqm_cb = &Fp_rawMSquare;
  } else {
     sqm_cb = &Fr_rawMSquare;
  }
  return sqm_cb;
}

t_tomont getcb_tomont_h( uint32_t pidx)
{
  t_tomont tom_cb;
  if (pidx == MOD_FP){
     tom_cb = &Fp_toMont;
  } else {
     tom_cb = &Fr_toMont;
  }
  return tom_cb;
}

t_frommont getcb_frommont_h( uint32_t pidx)
{
  t_frommont fromm_cb;
  if (pidx == MOD_FP){
     fromm_cb = &Fp_fromMont;
  } else {
     fromm_cb = &Fr_fromMont;
  }
  return fromm_cb;
}

void Fp_toMont(uint32_t *z, const uint32_t *x)
{
  const uint32_t *R2 = CusnarksR2Get((mod_t)MOD_FP);
  Fp_rawMMul(z,x,R2);
}

void Fp_fromMont(uint32_t *z, const uint32_t *x)
{
  const uint32_t *one = CusnarksOneGet((mod_t)MOD_FP);
  Fp_rawMMul(z,x,one);
}

void Fr_toMont(uint32_t *z, const uint32_t *x)
{
  const uint32_t *R2 = CusnarksR2Get((mod_t)MOD_FR);
  Fr_rawMMul(z,x,R2);
}

void Fr_fromMont(uint32_t *z, const uint32_t *x)
{
  const uint32_t *one = CusnarksOneGet((mod_t)MOD_FR);
  Fr_rawMMul(z,x,one);
}


/****************************************************************************/
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
void montmultp_h(uint32_t *z, const uint32_t *x, const uint32_t *y)
{
 montmult_h(z,x, y, MOD_FP);
}
void montmultr_h(uint32_t *z, const uint32_t *x, const uint32_t *y)
{
 montmult_h(z,x, y, MOD_FR);
}
void montmult_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx)
{
  #ifndef _CASM
  int i, j;
  t_uint64 S, C, C1, C2, C3=0, M[2], X[2], carry;
  uint32_t T[NWORDS_FP+4];
  const uint32_t *NPrime = CusnarksNPGet((mod_t)pidx);
  const uint32_t *N = CusnarksPGet((mod_t)pidx);

  const t_uint64 *dA = (t_uint64 *)A;
  const t_uint64 *dB = (t_uint64 *)B;
  t_uint64 *dU = (t_uint64 *)U;
  const t_uint64 *dNP = (t_uint64 *)NPrime;
  const t_uint64 *dN = (t_uint64 *)N;
  t_uint64 *dT = (t_uint64 *)T;
  const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

  memset(T, 0, sizeof(uint32_t)*(NWORDS_FP+4));

  for(i=0; i<PSize/2; i++) {
    // (C,S) = t[0] + a[0]*b[i], worst case 2 words
    mulu64_h(X, &dA[0], &dB[i]); // X[Upper,Lower] = a[0]*b[i]
    C = addu64_h(&S, dT+0, X+0); // [C,S] = t[0] + X[Lower]
    addu64_h(&C, &C, X+1);  // [~,C] = C + X[Upper], No carry
    //printf("1[%d]: C: %llx S: %llx\n",i,(uint64_t)C, (uint64_t)S); 

    // ADD(t[1],C)
    carry = addu64_h(&dT[1], &dT[1], &C); 
    // m = S*n'[0] mod W, where W=2^32
    // Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
    mulu64_h(M, &S, dNP);

    // (C,S) = S + m*n[0], worst case 2 words
    mulu64_h(X, &M[0], dN); // X[Upper,Lower] = m*n[0]
    C = addu64_h(&S, &S, X+0); // [C,S] = S + X[Lower]
    addu64_h(&C, &C, X+1);  // [~,C] = C + X[Upper]

    for(j=1; j<PSize/2; j++) {
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      mulu64_h(X,&dA[j], &dB[i]);   // X[Upper,Lower] = a[j]*b[i], double precision
      C1 = addu64_h(&S, dT+j, &C);  // (C1,S) = t[j] + C
      C2 = addu64_h(&S, &S, X+0);  // (C2,S) = S + X[Lower]
      addu64_h(&C, &C1, X+1);   // (~,C)  = C1 + X[Upper], doesn't produce carry
      C3 = addu64_h(&C, &C, &C2);    // (~,C)  = C + C2, it DOES produce carry
       
      C3 += addu64_h(&C, &C, &carry);    // (~,C)  = C + C2, It DOES produce carry

      carry = addu64_h(&dT[j+1], &dT[j+1], &C) + C3; 
   
      // (C,S) = S + m*n[j]
      mulu64_h(X, M, &dN[j]); // X[Upper,Lower] = m*n[j]
      C = addu64_h(&dT[j-1], &S, X+0); // [C,S] = S + X[Lower]
      addu64_h(&C, &C, X+1);  // [~,C] = C + X[Upper]
   
    }

    // (C,S) = t[s] + C
    C = addu64_h(&dT[PSize/2-1], dT+PSize/2, &C);
    addu64_h(dT+PSize/2, dT+PSize/2+1, &C);
    dT[PSize/2+1] = 0;
  }

  /* Step 3: if(u>=n) return u-n else return u */
  if(compuBI_h(T, N, PSize) >= 0) {
    subuBI_h(T, (const uint32_t *)T, N, PSize);
  }

  memcpy(U, T, sizeof(uint32_t)*PSize);

 #else
    if (pidx == MOD_FP ){
       Fp_rawMMul(U, A, B);
    } else {
       Fr_rawMMul(U, A, B);
    }
 #endif
}

void montmult_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
  uint32_t t0[NWORDS_FP], t1[NWORDS_FP];
  uint32_t t2[NWORDS_FP], t3[NWORDS_FP];
  const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

 #ifndef _CASM
  montmult_h(t0,x,y,pidx);
  montmult_h(t1,&x[PSize],&y[PSize],pidx);

  addm_h(t2,x,&x[PSize],pidx);
  addm_h(t3,y,&y[PSize],pidx);
  montmult_h(t2,t2,t3,pidx);
  subm_h(z,t0,t1,pidx);
  addm_h(&z[PSize],t0,t1,pidx);
  subm_h(&z[PSize],t2,&z[PSize],pidx);
 #else
  t_subm subm_cb = getcb_subm_h(pidx);
  t_addm addm_cb = getcb_addm_h(pidx);
  t_mulm mulm_cb = getcb_mulm_h(pidx);
 
  mulm_cb(t0,x,y);
  mulm_cb(t1,&x[PSize],&y[PSize]);

  addm_cb(t2,x,&x[PSize]);
  addm_cb(t3,y,&y[PSize]);
  mulm_cb(t2,t2,t3);
  subm_cb(z,t0,t1);
  addm_cb(&z[PSize],t0,t1);
  subm_cb(&z[PSize],t2,&z[PSize]);
 #endif
  
}

// I am leaving this as a separate function to test both implementations are equal
void montsquare_h(uint32_t *U, const uint32_t *A, uint32_t pidx)
{
  #ifndef _CASM
    montmult_h(U,A,A,pidx);
  #else
    if (pidx == MOD_FP){
      Fp_rawMSquare(U,A);
    } else {
      Fr_rawMSquare(U,A);
    }
  #endif
}

void montsquare_ext_h(uint32_t *U, const uint32_t *A, uint32_t pidx)
{
  const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);
  #ifndef _CASM
    montmult_ext_h(U,A,A,pidx);
  #else
    uint32_t t0[NWORDS_FP], t1[NWORDS_FP];
    uint32_t t2[NWORDS_FP], t3[NWORDS_FP];
    void (*subm_cb)(uint32_t *, const uint32_t *, const uint32_t *) = &Fr_rawSub;
    void (*addm_cb)(uint32_t *, const uint32_t *, const uint32_t *) = &Fr_rawAdd;
    void (*sqm_cb)(uint32_t *, const uint32_t *) = &Fr_rawMSquare;

    if (pidx == MOD_FP){
     subm_cb = &Fp_rawSub;
     addm_cb = &Fp_rawAdd;
     sqm_cb = &Fp_rawMSquare;
    } 
    sqm_cb(t0,A);
    sqm_cb(t1,&A[PSize]);

    addm_cb(t2,A,&A[PSize]);
    sqm_cb(t2,t2);
    subm_cb(U,t0,t1);
    addm_cb(&U[PSize],t0,t1);
    subm_cb(&U[PSize],t2,&U[PSize]);
    
  #endif
}

void montmultN_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t n, uint32_t pidx)
{
  uint32_t i;
  const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i<n; i++){
     montmult_h(&U[i*PSize], &A[i*PSize], &B[i*PSize], pidx);
  }
}
void montmultN_ext_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t n, uint32_t pidx)
{
  uint32_t i;
  const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i<n; i++){
     montmult_ext_h(&U[2*i*PSize], &A[2*i*PSize], &B[2*i*PSize], pidx);
  }
}

/* I
   Convert BI bit number from montgomery representation of one of the two prime 

   uint32_t *z   : normal represention of input sample x. 
   uint32_t *x   : input BI bit sample in montgomery format
   uint32_t pidx : prime select. if 0, use p1. If 1, use p2
*/   
void from_montgomery_h(uint32_t *z, const uint32_t *x, uint32_t pidx)
{
  const uint32_t *one = CusnarksOneGet((mod_t)pidx);
  montmult_h(z,x,one, pidx);
}


void from_montgomeryN_h(uint32_t *z, const uint32_t *x, uint32_t n, uint32_t pidx, uint32_t strip_last)
{
  uint32_t i;
  const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

  if (!strip_last){
    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for(i=0; i<n;i++){
      from_montgomery_h(&z[i*PSize], &x[i*PSize], pidx);
    }
  } else if (strip_last == 1) {
    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for(i=0; i<n;i++){
      int rem = i%3;
      if (rem != 2){
         from_montgomery_h(&z[(2*(i/3)+rem)*PSize], &x[i*PSize], pidx);
      }
      
    }
  } else if (strip_last == 2){
    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for(i=0; i<n;i++){
      int rem = i%6;
      if (rem < 4){
        from_montgomery_h(&z[(4*(i/6)+rem)*PSize], &x[i*PSize], pidx);
      }
    }
  }
}

/*
   Generate N BI bit random samples

   uint32_t *x       : output vector containing BI bit samples. Vector is of length nsamples
   uint32_t nsamples : Number of BI bit samples to generate
   uint32_t *p       : If different from null, samples will be less than p (p is a BI bit number)
   
*/
void setRandomBI(uint32_t *x, const uint32_t nsamples, const uint32_t *p, const uint32_t biSize)
{
  int j;
  _RNG* rng = _RNG::get_instance(x[0]);

  memset(x,0,biSize*sizeof(uint32_t)*nsamples);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (j=0; j < nsamples; j++){
    uint32_t nwords;
    uint32_t nbits;
    rng->randu32(&nwords,1);
    rng->randu32(&nbits,1);

    nwords %= biSize;
    nbits %= NBITS_WORD; 

    rng->randu32(&x[j*biSize],nwords+1); 

    x[j*biSize+nwords] &= ((1 << nbits)-1);
    if ((p!= NULL) && (nwords==biSize-1) && (compuBI_h(&x[j*biSize], p, biSize) >= 0)){
         do{
           subuBI_h(&x[j*biSize], p, biSize);
         }while(compuBI_h(&x[j*biSize],p, biSize) >=0);
    }
  }
}

void setRandomBI(uint32_t *x, const uint32_t nsamples, int32_t min_nwords, int32_t max_nwords, const uint32_t *p, const uint32_t biSize)
{
  int j;
  _RNG* rng = _RNG::get_instance(x[0]);

  memset(x,0,biSize*sizeof(uint32_t)*nsamples);
  if (min_nwords == -1){
	  min_nwords = 0;
  }
  if (max_nwords == -1){
	  max_nwords =biSize - 1;
  }

  /*
  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  */
  for (j=0; j < nsamples; j++){
    uint32_t nwords;
    uint32_t nbits;
    do {
      rng->randu32(&nwords,1);
      nwords %= biSize;

    }while(nwords < min_nwords || nwords > max_nwords);

    rng->randu32(&nbits,1);

    nbits %= NBITS_WORD;

    rng->randu32(&x[j*biSize],nwords+1); 
    x[j*biSize+nwords] &= ((1 << nbits)-1);
    if ((p!= NULL) && (nwords==biSize-1) && (compuBI_h(&x[j*biSize], p, biSize) >= 0)){
         do{
           subuBI_h(&x[j*biSize], p, biSize);
         }while(compuBI_h(&x[j*biSize],p, biSize) >=0);
    }
  }
}

/*
   Generates N BI bit samples with incremements of inc starting at start. If sample reached value of mod,
   value goes back to 0

   uint32_t *samples : Vector containing output samples. Vector is of length nsamples
   uint32_t nsamples : Number of samples to generate
   uint32_t *start   : First sample value 
   uint32_t inc      : sample increment 
   uint32_t *mod     : if different from NULL, it is maximum sample value. If generation reaches this value, it will go back to 0.  
*/
void rangeuBI_h(uint32_t *samples, uint32_t nsamples, const uint32_t  *start, uint32_t inc, const uint32_t *mod, const uint32_t biSize)
{
   uint32_t i;
   uint32_t _inc[] = {inc,0,0,0,0,0,0,0};

   memcpy(samples,start,sizeof(uint32_t)*biSize);

   for (i=1; i < nsamples; i++){
     adduBI_h(&samples[i*biSize], &samples[(i-1)*biSize], _inc, biSize);
     if ((mod != NULL) && (compuBI_h(&samples[i*biSize], mod, biSize) >= 0)){
         do{
           subuBI_h(&samples[i*biSize], mod, biSize);
         }while(compuBI_h(&samples[i*biSize],mod, biSize) >=0);
     }
   }
}

/* 
   Convert BI bit number to montgomery representation of one of the two prime 

   uint32_t *z   : montgomery represention of input sample x. 
   uint32_t *x   : input BI bit sample
   uint32_t pidx : prime select. if 0, use p1. If 1, use p2
*/   
void to_montgomery_h(uint32_t *z, const uint32_t *x, uint32_t pidx)
{
  const uint32_t *R2 = CusnarksR2Get((mod_t)pidx);
  montmult_h(z,x,R2, pidx);
}

void to_montgomeryN_h(uint32_t *z, const uint32_t *x, uint32_t n, uint32_t pidx)
{
  uint32_t i;
  const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for(i=0; i<n;i++){
    to_montgomery_h(&z[i*PSize], &x[i*PSize], pidx);
  }
}


/*
  modular addition of BI bit numbers : Z = X + Y mod P

  uint32_t *z : Output BI bit number
  uint32_t *x : Input BI bit number 1
  uint32_t *y : Input BI bit number 2
  uint32_t pidx    : index of BI modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void addmp_h(uint32_t *z, const uint32_t *x, const uint32_t *y)
{
 addm_h(z,x, y, MOD_FP);
}
void addmr_h(uint32_t *z, const uint32_t *x, const uint32_t *y)
{
 addm_h(z,x, y, MOD_FR);
}
void addm_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
   #ifndef _CASM
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);
   adduBI_h(z, x, y, PSize);
   if(compuBI_h(z, N, PSize) >= 0) {
      subuBI_h(z, z, N, PSize);
   }
   #else
    if (pidx == MOD_FP ){
       Fp_rawAdd(z, x, y);
    } else {
       Fr_rawAdd(z, x, y);
    }
	
   #endif

}

void addm_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
   const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

   addm_h(z,x,y,pidx);
   addm_h(&z[PSize],&x[PSize],&y[PSize],pidx);
}
/*
  modular substraction of BI bit numbers : Z = X - Y mod P

  uint32_t *z : Output BI bit number
  uint32_t *x : Input BI bit number 1
  uint32_t *y : Input BI bit number 2
  uint32_t pidx    : index of BI modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void submp_h(uint32_t *z, const uint32_t *x, const uint32_t *y)
{
 subm_h(z,x, y, MOD_FP);
}
void submr_h(uint32_t *z, const uint32_t *x, const uint32_t *y)
{
 subm_h(z,x, y, MOD_FR);
}
void subm_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
  #ifndef _CASM
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

   subuBI_h(z, x, y, PSize);
   if(z[PSize-1] > N[PSize-1]){
       adduBI_h(z, z, N, PSize);
   }

  #else
    if (pidx == MOD_FP ){
       Fp_rawSub(z, x, y);
    } else {
       Fr_rawSub(z, x, y);
    }
  #endif
}
void subm_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
   const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);
   subm_h(z,x,y,pidx);
   subm_h(&z[PSize],&x[PSize],&y[PSize],pidx);
}


/*
  Montgomery Modular Inverse - Revisited
  E. Savas, C.K.Koc
  IEEE trasactions on Computers Vol49, No 7. July 2000
*/
void montinv_h(uint32_t *y, uint32_t *x,  uint32_t pidx)
{
   uint32_t k;
   FP_INIT_ARRONE(t);
   const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

   almmontinv_h(y,&k, x, pidx);
   if ( k <= PSize*NBITS_WORD){
      to_montgomery_h(y,y,pidx);
      k+=PSize*NBITS_WORD;
   }
   shlluBI_h(t,t,2 * PSize * NBITS_WORD - k, PSize);
   to_montgomery_h(t,t,pidx);
   montmult_h(y, y,t,pidx);
}

void almmontinv_h(uint32_t *r, uint32_t *k, uint32_t *a, uint32_t pidx)
{
  const uint32_t *P = CusnarksPGet((mod_t)pidx);
  const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

  uint32_t u[NWORDS_FP], v[NWORDS_FP];
  FP_INIT_ARRONE(s);
  FP_INIT_ARRZERO(r1);
  FP_INIT_ARRZERO(zero);
  //uint32_t s[] = {1,0,0,0,0,0,0,0};
  //uint32_t r1[] = {0,0,0,0,0,0,0,0};
  //uint32_t zero[] = {0,0,0,0,0,0,0,0};
  uint32_t i = 0;
  uint32_t t0,t1,t2,t3;

  memcpy(u,P,PSize*sizeof(uint32_t));
  memcpy(v,a,PSize*sizeof(uint32_t));
  *k = 0;

  //Phase 1 - ALmost inverse r = a^(-1) * 2 ^k, n<=k<=2n
  // u is  < BIbits
  // v is < BI bits, < u
  // s is  1     
  // r1 is 0

  while(compuBI_h(v,zero, PSize) != 0){
     if (getbituBI_h(u,0) == 0){
        shlruBI_h(u,u,1, PSize);
        shlluBI_h(s,s,1, PSize);
     } else if (getbituBI_h(v,0) == 0){
        shlruBI_h(v,v,1, PSize);
        shlluBI_h(r1,r1,1, PSize);
     } else if (compuBI_h(u,v, PSize) > 0) {
        subuBI_h(u,v, PSize);
        shlruBI_h(u,u,1, PSize);
        adduBI_h(r1,s, PSize);
        shlluBI_h(s,s,1, PSize);
     } else {
        subuBI_h(v,u, PSize);
        shlruBI_h(v,v,1, PSize);
        adduBI_h(s,r1, PSize);
        shlluBI_h(r1,r1,1, PSize);
     }
     (*k)++;
  }
  
  if (compuBI_h(r1,P, PSize) >= 0){
      subuBI_h(r1,P, PSize);
  }
  subuBI_h(r, (uint32_t *)P,r1, PSize);
  uint32_t  tmp_msb = msbuBI_h(a, PSize); 
}

void montinv_ext_h(uint32_t *y, uint32_t *x,  uint32_t pidx)
{
  uint32_t t0[NWORDS_FP], t1[NWORDS_FP];
  const uint32_t *Zero = CusnarksZeroGet((mod_t)pidx);
  const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

  montsquare_h(t0,x,pidx);
  montsquare_h(t1,&x[PSize], pidx);
  addm_h(t0,t0,t1,pidx);
  montinv_h(t0,t0,pidx);
  
  montmult_h(y,x,t0,pidx);
  montmult_h(&y[PSize],&x[PSize],t0,pidx);
  subm_h(&y[PSize],Zero,&y[PSize],pidx);
}




