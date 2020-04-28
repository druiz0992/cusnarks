
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
#include <omp.h>

#include "types.h"
#include "constants.h"
#include "rng.h"
#include "log.h"
#include "bigint.h"
#include "ff.h"

#ifdef _CASM
#include "fr.h"
#include "fq.h"
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

  if (pidx == MOD_GROUP){
     subm_cb = &Fr_rawSub;
  } else {
     subm_cb = &Fq_rawSub;
  }
  return subm_cb;
}

t_addm getcb_addm_h( uint32_t pidx)
{
  t_addm addm_cb;
  if (pidx == MOD_GROUP){
     addm_cb = &Fr_rawAdd;
  } else {
     addm_cb = &Fq_rawAdd;
  }
  return addm_cb;
}

t_mulm getcb_mulm_h( uint32_t pidx)
{
  t_mulm mulm_cb;
  if (pidx == MOD_GROUP){
     mulm_cb = &Fr_rawMMul;
  } else {
     mulm_cb = &Fq_rawMMul;
  }
  return mulm_cb;
}

t_sqm getcb_sqm_h( uint32_t pidx)
{
  t_sqm sqm_cb;
  if (pidx == MOD_GROUP){
     sqm_cb = &Fr_rawMSquare;
  } else {
     sqm_cb = &Fq_rawMSquare;
  }
  return sqm_cb;
}

t_tomont getcb_tomont_h( uint32_t pidx)
{
  t_tomont tom_cb;
  if (pidx == MOD_GROUP){
     tom_cb = &Fr_toMont;
  } else {
     tom_cb = &Fq_toMont;
  }
  return tom_cb;
}

t_frommont getcb_frommont_h( uint32_t pidx)
{
  t_frommont fromm_cb;
  if (pidx == MOD_GROUP){
     fromm_cb = &Fr_fromMont;
  } else {
     fromm_cb = &Fq_fromMont;
  }
  return fromm_cb;
}

void Fr_toMont(uint32_t *z, const uint32_t *x)
{
  const uint32_t *R2 = CusnarksR2Get((mod_t)MOD_GROUP);
  Fr_rawMMul(z,x,R2);
}

void Fr_fromMont(uint32_t *z, const uint32_t *x)
{
  const uint32_t *one = CusnarksOneGet();
  Fr_rawMMul(z,x,one);
}

void Fq_toMont(uint32_t *z, const uint32_t *x)
{
  const uint32_t *R2 = CusnarksR2Get((mod_t)MOD_FIELD);
  Fq_rawMMul(z,x,R2);
}

void Fq_fromMont(uint32_t *z, const uint32_t *x)
{
  const uint32_t *one = CusnarksOneGet();
  Fq_rawMMul(z,x,one);
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
void montmult_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx)
{
  #ifndef _CASM
  int i, j;
  t_uint64 S, C, C1, C2, C3=0, M[2], X[2], carry;
  uint32_t T[NWORDS_256BIT_FIOS+1];
  const uint32_t *NPrime = CusnarksNPGet((mod_t)pidx);
  const uint32_t *N = CusnarksPGet((mod_t)pidx);

  const t_uint64 *dA = (t_uint64 *)A;
  const t_uint64 *dB = (t_uint64 *)B;
  t_uint64 *dU = (t_uint64 *)U;
  const t_uint64 *dNP = (t_uint64 *)NPrime;
  const t_uint64 *dN = (t_uint64 *)N;
  t_uint64 *dT = (t_uint64 *)T;

  memset(T, 0, sizeof(uint32_t)*(NWORDS_256BIT_FIOS+1));

  /*
  printf("A\n");
  printU256Number(A);
  printf("B\n");
  printU256Number(B);

  printf("N\n");
  printU256Number(N);

  printf("NPrime[0] : %u\n",NPrime[0]);
  */

  for(i=0; i<NWORDS_256BIT/2; i++) {
    // (C,S) = t[0] + a[0]*b[i], worst case 2 words
    mulu64_h(X, &dA[0], &dB[i]); // X[Upper,Lower] = a[0]*b[i]
    C = addu64_h(&S, dT+0, X+0); // [C,S] = t[0] + X[Lower]
    addu64_h(&C, &C, X+1);  // [~,C] = C + X[Upper], No carry
    //printf("1[%d]: C: %llx S: %llx\n",i,(uint64_t)C, (uint64_t)S); 

    /*
    printf("0 - C : %u, S: %u\n",C,S);
    printf("0 - A[0] : %u, B[i]: %u T[0] : %u\n",A[0],B[i], T[0]);
    */
    // ADD(t[1],C)
    //mpAddWithCarryProp(T, C, 1);
    carry = addu64_h(&dT[1], &dT[1], &C); 
    //printf("a[%d]: C: %llx T[1]: %llx\n",i,(uint64_t)carry, (uint64_t)dT[1]); 
    /*
    printf("C3: %u\n",carry);
    printf("T\n");
    printU256Number(T);
    */

    // m = S*n'[0] mod W, where W=2^32
    // Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
    mulu64_h(M, &S, dNP);
    //printf("b[%d]: M: %llx, N: %llx\n",i,(uint64_t)(M[0]),(uint64_t)dN[0]);
    /*
    printf("M[0]:%u, M[1]: %u\n",M[0], M[1]);
    */

    // (C,S) = S + m*n[0], worst case 2 words
    mulu64_h(X, &M[0], dN); // X[Upper,Lower] = m*n[0]
    /*
    printf("1 - X[1] %u, X[0] : %u\n",X[1], X[0]);
    */
    C = addu64_h(&S, &S, X+0); // [C,S] = S + X[Lower]
    addu64_h(&C, &C, X+1);  // [~,C] = C + X[Upper]
    /*
    printf("1 - C : %u, S: %u, X[1] %u, X[0] : %u\n\n",C,S, X[1], X[0]);
    */
    //printf("2[%d]: C: %llx S: %llx, carry: %llx\n",i,(uint64_t)C, (uint64_t)S, (uint64_t)carry); 

    for(j=1; j<NWORDS_256BIT/2; j++) {
      // (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
      mulu64_h(X,&dA[j], &dB[i]);   // X[Upper,Lower] = a[j]*b[i], double precision
      C1 = addu64_h(&S, dT+j, &C);  // (C1,S) = t[j] + C
      /*
      printf("2 - C1 : %u, S: %u\n",C1,S);
      */
      C2 = addu64_h(&S, &S, X+0);  // (C2,S) = S + X[Lower]
      /*
      printf("3 - C2 : %u, S: %u\n",C2,S);
      printf("X[0] : %u, X[1]: %u\n",X[0],X[1]);
      */
      addu64_h(&C, &C1, X+1);   // (~,C)  = C1 + X[Upper], doesn't produce carry
      /*
      printf("4 - C : %u\n",C);
      */
      C3 = addu64_h(&C, &C, &C2);    // (~,C)  = C + C2, it DOES produce carry
      /*
      printf("5 - C : %u, C3 : %u\n",C, C3);
      */
       
      /*
      // Fix this!!!! TODO
      if (C3 > 0){
        printf("Te pille\n");
      }
      */
      // ADD(t[j+1],C)
      //C += carry;
      //printf("3[%d-%d]: C1: %llx C: %llx S: %llx\n",i,j,(uint64_t)C3,(uint64_t) C, (uint64_t)S); 
      C3 += addu64_h(&C, &C, &carry);    // (~,C)  = C + C2, It DOES produce carry
      /*
      if (C3 > 0){
        printf("Te pille v2\n");
      }
      */

      //printf("c[%d-%d]: C1: %llu C: %llx T[j+1]: %llx\n",i,j,(uint64_t) C3,(uint64_t)C, (uint64_t)dT[j+1]); 
      carry = addu64_h(&dT[j+1], &dT[j+1], &C) + C3; 
      //printf("4[%d-%d]: C1: %llx C: %llx S: %llx, carry: %llx\n",i,j,(uint64_t) C3,(uint64_t)C, (uint64_t)dT[j+1],(uint64_t)carry); 
      //mpAddWithCarryProp(T, C, j+1);
      /*
      printf("T(%u)\n", carry);
      printU256Number(T);
     */
   
      // (C,S) = S + m*n[j]
      mulu64_h(X, M, &dN[j]); // X[Upper,Lower] = m*n[j]
      C = addu64_h(&dT[j-1], &S, X+0); // [C,S] = S + X[Lower]
      addu64_h(&C, &C, X+1);  // [~,C] = C + X[Upper]
   
      // t[j-1] = S
      //dT[j-1] = S;
      /*
      printf("T[%d]\n", j-1);
      printU256Number(T);
      */
      //printU256Number("T1 : \n",T);
    }

    //mpAddWithCarryProp(T, carry, NWORDS_256BIT, NWORDS_256BIT_FIOS);
    // (C,S) = t[s] + C
    C = addu64_h(&dT[NWORDS_256BIT/2-1], dT+NWORDS_256BIT/2, &C);
    /*
    printf("6 - C : %u, S: %u\n",C,S);
    */
    // t[s-1] = S
    //dT[NWORDS_256BIT/2-1] = S;
    // t[s] = t[s+1] + C
    addu64_h(dT+NWORDS_256BIT/2, dT+NWORDS_256BIT/2+1, &C);
    // t[s+1] = 0
    dT[NWORDS_256BIT/2+1] = 0;
    //printU256Number("T2 : \n",T);
  }

  //printU256Number("T : \n",T);
  /* Step 3: if(u>=n) return u-n else return u */
  if(compu256_h(T, N) >= 0) {
    subu256_h(T, (const uint32_t *)T, N);
  }

  memcpy(U, T, sizeof(uint32_t)*NWORDS_256BIT);
  //printU256Number("U : \n",U);

 #else
    if (pidx == MOD_GROUP ){
       Fr_rawMMul(U, A, B);
    } else {
       Fq_rawMMul(U, A, B);
    }
 #endif
}

void montmult_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
  uint32_t t0[NWORDS_256BIT], t1[NWORDS_256BIT];
  uint32_t t2[NWORDS_256BIT], t3[NWORDS_256BIT];

 #ifndef _CASM
  montmult_h(t0,x,y,pidx);
  montmult_h(t1,&x[NWORDS_256BIT],&y[NWORDS_256BIT],pidx);

  addm_h(t2,x,&x[NWORDS_256BIT],pidx);
  addm_h(t3,y,&y[NWORDS_256BIT],pidx);
  montmult_h(t2,t2,t3,pidx);
  subm_h(z,t0,t1,pidx);
  addm_h(&z[NWORDS_256BIT],t0,t1,pidx);
  subm_h(&z[NWORDS_256BIT],t2,&z[NWORDS_256BIT],pidx);
 #else
  t_subm subm_cb = getcb_subm_h(pidx);
  t_addm addm_cb = getcb_addm_h(pidx);
  t_mulm mulm_cb = getcb_mulm_h(pidx);
 
  mulm_cb(t0,x,y);
  mulm_cb(t1,&x[NWORDS_256BIT],&y[NWORDS_256BIT]);

  addm_cb(t2,x,&x[NWORDS_256BIT]);
  addm_cb(t3,y,&y[NWORDS_256BIT]);
  mulm_cb(t2,t2,t3);
  subm_cb(z,t0,t1);
  addm_cb(&z[NWORDS_256BIT],t0,t1);
  subm_cb(&z[NWORDS_256BIT],t2,&z[NWORDS_256BIT]);
 #endif
  
}

// I am leaving this as a separate function to test both implementations are equal
void montsquare_h(uint32_t *U, const uint32_t *A, uint32_t pidx)
{
  #ifndef _CASM
    montmult_h(U,A,A,pidx);
  #else
    if (pidx == MOD_GROUP){
      Fr_rawMSquare(U,A);
    } else {
      Fq_rawMSquare(U,A);
    }
  #endif
}

void montsquare_ext_h(uint32_t *U, const uint32_t *A, uint32_t pidx)
{
  #ifndef _CASM
    montmult_ext_h(U,A,A,pidx);
  #else
    uint32_t t0[NWORDS_256BIT], t1[NWORDS_256BIT];
    uint32_t t2[NWORDS_256BIT], t3[NWORDS_256BIT];
    void (*subm_cb)(uint32_t *, const uint32_t *, const uint32_t *) = &Fq_rawSub;
    void (*addm_cb)(uint32_t *, const uint32_t *, const uint32_t *) = &Fq_rawAdd;
    void (*sqm_cb)(uint32_t *, const uint32_t *) = &Fq_rawMSquare;

    if (pidx == MOD_GROUP){
     subm_cb = &Fr_rawSub;
     addm_cb = &Fr_rawAdd;
     sqm_cb = &Fr_rawMSquare;
    } 
    sqm_cb(t0,A);
    sqm_cb(t1,&A[NWORDS_256BIT]);

    addm_cb(t2,A,&A[NWORDS_256BIT]);
    sqm_cb(t2,t2);
    subm_cb(U,t0,t1);
    addm_cb(&U[NWORDS_256BIT],t0,t1);
    subm_cb(&U[NWORDS_256BIT],t2,&U[NWORDS_256BIT]);
    
  #endif
}

void montmultN_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t n, uint32_t pidx)
{
  uint32_t i;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i<n; i++){
     montmult_h(&U[i*NWORDS_256BIT], &A[i*NWORDS_256BIT], &B[i*NWORDS_256BIT], pidx);
  }
}
void montmultN_ext_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t n, uint32_t pidx)
{
  uint32_t i;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i<n; i++){
     montmult_ext_h(&U[2*i*NWORDS_256BIT], &A[2*i*NWORDS_256BIT], &B[2*i*NWORDS_256BIT], pidx);
  }
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


void from_montgomeryN_h(uint32_t *z, const uint32_t *x, uint32_t n, uint32_t pidx, uint32_t strip_last)
{
  uint32_t i;

  if (!strip_last){
    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for(i=0; i<n;i++){
      from_montgomery_h(&z[i*NWORDS_256BIT], &x[i*NWORDS_256BIT], pidx);
    }
  } else if (strip_last == 1) {
    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for(i=0; i<n;i++){
      int rem = i%3;
      if (rem != 2){
         from_montgomery_h(&z[(2*(i/3)+rem)*NWORDS_256BIT], &x[i*NWORDS_256BIT], pidx);
      }
      
    }
  } else if (strip_last == 2){
    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for(i=0; i<n;i++){
      int rem = i%6;
      if (rem < 4){
        from_montgomery_h(&z[(4*(i/6)+rem)*NWORDS_256BIT], &x[i*NWORDS_256BIT], pidx);
      }
    }
  }
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

  memset(x,0,NWORDS_256BIT*sizeof(uint32_t)*nsamples);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (j=0; j < nsamples; j++){
    uint32_t nwords;
    uint32_t nbits;
    rng->randu32(&nwords,1);
    rng->randu32(&nbits,1);

    nwords %= NWORDS_256BIT;
    nbits %= 32;

    rng->randu32(&x[j*NWORDS_256BIT],nwords+1); 

    x[j*NWORDS_256BIT+nwords] &= ((1 << nbits)-1);
    if ((p!= NULL) && (nwords==NWORDS_256BIT-1) && (compu256_h(&x[j*NWORDS_256BIT], p) >= 0)){
         do{
           subu256_h(&x[j*NWORDS_256BIT], p);
         }while(compu256_h(&x[j*NWORDS_256BIT],p) >=0);
    }
  }
}
void setRandom256(uint32_t *x, const uint32_t nsamples, int32_t min_nwords, int32_t max_nwords, const uint32_t *p)
{
  int j;
  _RNG* rng = _RNG::get_instance(x[0]);

  memset(x,0,NWORDS_256BIT*sizeof(uint32_t)*nsamples);
  if (min_nwords == -1){
	  min_nwords = 0;
  }
  if (max_nwords == -1){
	  max_nwords = NWORDS_256BIT - 1;
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
      nwords %= NWORDS_256BIT;

    }while(nwords < min_nwords || nwords > max_nwords);

    rng->randu32(&nbits,1);

    nbits %= 32;

    rng->randu32(&x[j*NWORDS_256BIT],nwords+1); 
    x[j*NWORDS_256BIT+nwords] &= ((1 << nbits)-1);
    if ((p!= NULL) && (nwords==NWORDS_256BIT-1) && (compu256_h(&x[j*NWORDS_256BIT], p) >= 0)){
         do{
           subu256_h(&x[j*NWORDS_256BIT], p);
         }while(compu256_h(&x[j*NWORDS_256BIT],p) >=0);
    }
  }
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
     addu256_h(&samples[i*NWORDS_256BIT], &samples[(i-1)*NWORDS_256BIT], _inc);
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

void to_montgomeryN_h(uint32_t *z, const uint32_t *x, uint32_t n, uint32_t pidx)
{
  uint32_t i;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for(i=0; i<n;i++){
    to_montgomery_h(&z[i*NWORDS_256BIT], &x[i*NWORDS_256BIT], pidx);
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
   #ifndef _CASM
   //uint32_t tmp[NWORDS_256BIT];
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   addu256_h(z, x, y);
   if(compu256_h(z, N) >= 0) {
      subu256_h(z, z, N);
   }
   #else
    if (pidx == MOD_GROUP ){
       Fr_rawAdd(z, x, y);
    } else {
       Fq_rawAdd(z, x, y);
    }
	
   #endif

   //memcpy(z, tmp, sizeof(uint32_t)*NWORDS_256BIT);
}

void addm_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
   addm_h(z,x,y,pidx);
   addm_h(&z[NWORDS_256BIT],&x[NWORDS_256BIT],&y[NWORDS_256BIT],pidx);
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
  #ifndef _CASM
   const uint32_t *N = CusnarksPGet((mod_t)pidx);

   subu256_h(z, x, y);
   //if(compu256_h(z, N) >= 0) {
   if(z[NWORDS_256BIT-1] > N[NWORDS_256BIT-1]){
       addu256_h(z, z, N);
   }

   //memcpy(z, tmp, sizeof(uint32_t)*NWORDS_256BIT);
  #else
    if (pidx == MOD_GROUP ){
       Fr_rawSub(z, x, y);
    } else {
       Fq_rawSub(z, x, y);
    }
  #endif
}
void subm_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
   subm_h(z,x,y,pidx);
   subm_h(&z[NWORDS_256BIT],&x[NWORDS_256BIT],&y[NWORDS_256BIT],pidx);
}


/*
  Montgomery Modular Inverse - Revisited
  E. Savas, C.K.Koc
  IEEE trasactions on Computers Vol49, No 7. July 2000
*/
void montinv_h(uint32_t *y, uint32_t *x,  uint32_t pidx)
{
#if 1
   uint32_t k;
   uint32_t t[] = {1,0,0,0,0,0,0,0};

   almmontinv_h(y,&k, x, pidx);
   if ( k <= NWORDS_256BIT*NBITS_WORD){
      to_montgomery_h(y,y,pidx);
      k+=NWORDS_256BIT*NBITS_WORD;
   }
   shllu256_h(t,t,2 * NWORDS_256BIT * NBITS_WORD - k);
   to_montgomery_h(t,t,pidx);
   montmult_h(y, y,t,pidx);
#else
   uint32_t k;
   uint32_t t[] = {1,0,0,0,0,0,0,0};
   uint32_t t_idx;

   const uint32_t *R[2];
   R[0] = CusnarksR2Get((mod_t)pidx);
   R[1] = CusnarksR3Get((mod_t)pidx);
   uint32_t shift[2];

   almmontinv_h(y,&k, x, pidx);

   t_idx = 2*NWORDS_256BIT*NBITS_WORD/k-1;
   shift[0] = 2*NWORDS_256BIT * NBITS_WORD - k;
   shift[1] = NWORDS_256BIT * NBITS_WORD - k;

   shllu256_h(t,t,shift[t_idx]);
   montmult_h(y, y, R[t_idx],pidx);
   montmult_h(y, y, t,pidx);

#endif
}
void almmontinv_h(uint32_t *r, uint32_t *k, uint32_t *a, uint32_t pidx)
{
  const uint32_t *P = CusnarksPGet((mod_t)pidx);

  uint32_t u[NWORDS_256BIT], v[NWORDS_256BIT];
  uint32_t s[] = {1,0,0,0,0,0,0,0};
  uint32_t r1[] = {0,0,0,0,0,0,0,0};
  uint32_t i = 0;
  uint32_t t0,t1,t2,t3;
  uint32_t tmp[NWORDS_256BIT];
  uint32_t zero[] = {0,0,0,0,0,0,0,0};

  memcpy(u,P,NWORDS_256BIT*sizeof(uint32_t));
  memcpy(v,a,NWORDS_256BIT*sizeof(uint32_t));
  *k = 0;

  //Phase 1 - ALmost inverse r = a^(-1) * 2 ^k, n<=k<=2n
  // u is  < 256bits
  // v is < 256 bits, < u
  // s is  1     
  // r1 is 0

  while(compu256_h(v,zero) != 0){
     if (getbitu256_h(u,0) == 0){
        shlru256_h(u,u,1);
        shllu256_h(s,s,1);
     } else if (getbitu256_h(v,0) == 0){
        shlru256_h(v,v,1);
        shllu256_h(r1,r1,1);
     } else if (compu256_h(u,v) > 0) {
        subu256_h(u,v);
        shlru256_h(u,u,1);
        addu256_h(r1,s);
        shllu256_h(s,s,1);
     } else {
        subu256_h(v,u);
        shlru256_h(v,v,1);
        addu256_h(s,r1);
        shllu256_h(r1,r1,1);
     }
     (*k)++;
  }
  
  if (compu256_h(r1,P) >= 0){
      subu256_h(r1,P);
  }
  subu256_h(r, (uint32_t *)P,r1);
  uint32_t  tmp_msb = msbu256_h(a); 
}

void montinv_ext_h(uint32_t *y, uint32_t *x,  uint32_t pidx)
{
  uint32_t t0[NWORDS_256BIT], t1[NWORDS_256BIT];
  const uint32_t *Zero = CusnarksZeroGet();

  montsquare_h(t0,x,pidx);
  montsquare_h(t1,&x[NWORDS_256BIT], pidx);
  addm_h(t0,t0,t1,pidx);
  montinv_h(t0,t0,pidx);
  
  montmult_h(y,x,t0,pidx);
  montmult_h(&y[NWORDS_256BIT],&x[NWORDS_256BIT],t0,pidx);
  subm_h(&y[NWORDS_256BIT],Zero,&y[NWORDS_256BIT],pidx);
}




