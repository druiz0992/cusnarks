
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
#include <sys/sysinfo.h>
#include <pthread.h>
#include <unistd.h> 
#include <omp.h>
#include <x86intrin.h>

#include "types.h"
#include "constants.h"
#include "rng.h"
#include "log.h"
#include "utils_host.h"


#define MAX_DIGIT 0xFFFFFFFFUL
#define MAX(X,Y)  ((X)>=(Y) ? (X) : (Y))
#define MIN(X,Y)  ((X)<(Y) ? (X) : (Y))

//MPROC VARS
static  pthread_mutex_t utils_lock;
static  uint32_t utils_nprocs = 1;
static  uint32_t utils_mproc_init = 0;

static uint32_t utils_N[MAX_NCORES_OMP * NWORDS_256BIT * ECP2_JAC_OUTDIMS];
static uint32_t utils_zinv[2 * MAX_NCORES_OMP * NWORDS_256BIT];
static uint32_t utils_zinv_sq[2 * MAX_NCORES_OMP * NWORDS_256BIT];


static uint32_t *M_transpose;
static uint32_t *M_mul;

#ifdef PARALLEL_EN
static  uint32_t parallelism_enabled =  1;
#else
static  uint32_t parallelism_enabled =  0;
#endif


// Internal functions
void almmontinv_h(uint32_t *r, uint32_t *k, uint32_t *a, uint32_t pidx);

// Mproc
void mproc_init_h(void);

inline void mulu64_h(t_uint64 p[2], const t_uint64 *x, const t_uint64 *y)
{
 p[0] = _mulx_u64(x[0],y[0],&p[1]);
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

static void _ntt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx);
static void _ntt_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx);
static void _intt_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx);
static void _intt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx);
static void ntt_reorder_h(uint32_t *A, uint32_t levels, uint32_t astride);
static void ntt_parallel_T_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, int32_t direction, fft_mode_t fft_mode, uint32_t pidx);
static void _ntt_parallel_T_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, int32_t direction, fft_mode_t fft_mode, uint32_t pidx);
static void montmult_reorder_h(uint32_t *A, const uint32_t *roots, uint32_t levels, uint32_t pidx);
static void montmult_parallel_reorder_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, uint32_t rstride, uint32_t direction, uint32_t pidx);

//////

/*
  Bit reverse 32 bit number

  uint32_t x : input number
  uint32_t bits : number of bits
  
  returns bit reversed input number
*/
inline uint32_t reverse32(uint32_t x, uint32_t bits)
{
  // from http://graphics.stanford.edu/~seander/bithacks.html
  // swap odd and even bits
  x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
  // swap consecutixe pairs
  x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
  // swap nibbles ... 
  x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4);
  // swap bytes
  x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8);
  // swap 2-byte long pairs
  x = ( x >> 16             ) | ( x               << 16);

  x = ( x >> (32 - bits));
  return x;
}
inline uint32_t reverse16(uint32_t x, uint32_t bits)
{
  // from http://graphics.stanford.edu/~seander/bithacks.html
  // swap odd and even bits
  x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
  // swap consecutixe pairs
  x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
  // swap nibbles ... 
  x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4);
  // swap bytes
  x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8);

  x = ( x >> (16 - bits));
  return x;
}

inline uint32_t reverse8(uint32_t x, uint32_t bits)
{
  // from http://graphics.stanford.edu/~seander/bithacks.html
  // swap odd and even bits
  x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
  // swap consecutixe pairs
  x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
  // swap nibbles ... 
  x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4);

  x = ( x >> (8 - bits));
  return x;
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
  
  /*
  SWAP(dX[0],dY[0]);
  SWAP(dX[1],dY[1]);
  SWAP(dX[2],dY[2]);
  SWAP(dX[3],dY[3]);
  */
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
// input matrix mxn
void transpose_h(uint32_t *min, uint32_t in_nrows, uint32_t in_ncols)
{
   uint32_t m = sizeof(uint32_t) * NBITS_BYTE - __builtin_clz(in_nrows) - 1;
   uint32_t n = sizeof(uint32_t) * NBITS_BYTE - __builtin_clz(in_ncols) - 1;

   if (in_nrows == in_ncols){
     transpose_square_h(min, in_nrows);
     return;
   }
   const uint32_t *tt = CusnarksTidxGet();

   uint32_t idx = tt[m];
   const uint32_t N_1 = (1 << n) - 1;

   uint32_t val[NWORDS_256BIT];
   uint32_t cur_pos = tt[idx++], trans_pos, step=tt[idx++];
   uint32_t max_count = tt[idx++], max_el = tt[idx++];

   memcpy(val, &min[cur_pos * NWORDS_256BIT], sizeof(uint32_t)*NWORDS_256BIT);
   for (uint32_t ccount=0; ccount < max_count; ccount++){
     for (uint32_t elcount=0; elcount < max_el; elcount+=2){
        trans_pos = (cur_pos >> n) + ((cur_pos & N_1) << m);
        swapu256_h(&min[trans_pos * NWORDS_256BIT], val);
        cur_pos = (trans_pos >> n ) + ((trans_pos & N_1) << m);
        swapu256_h(&min[cur_pos * NWORDS_256BIT], val);
     }
     cur_pos = trans_pos + step;
   }
   cur_pos = tt[idx++];
   step = tt[idx++];
   max_count = tt[idx++];
   max_el = tt[idx++];
}

void transpose_square_h(uint32_t *min, uint32_t in_nrows)
{
  uint32_t m = sizeof(uint32_t) * NBITS_BYTE - __builtin_clz(in_nrows) - 1;
  for(uint32_t i=0; i<in_nrows; i++){
    for(uint32_t j=i+1; j <in_nrows; j++){
        uint32_t cur_pos = (i << m) + j;
        uint32_t trans_pos = (j << m) + i;
        swapu256_h(&min[trans_pos << NWORDS_256BIT_SHIFT], &min[cur_pos << NWORDS_256BIT_SHIFT]);
    } 
  }
}
/*
   Initalize multiprocessing components
    - mutex : utils_lock
    - number of processors
*/
void mproc_init_h()
{
  if (utils_mproc_init) {
    return;
  }

  utils_nprocs = get_nprocs_conf() > MAX_NCORES_OMP ? MAX_NCORES_OMP : get_nprocs_conf();
  omp_set_num_threads(utils_nprocs);
  utils_mproc_init = 1;

  if (pthread_mutex_init(&utils_lock, NULL) != 0){
     exit(1);
  }

  //logInfo("N Procs available : %d\n", utils_nprocs);
}

/*
   Launch server to evaluate Mpolys. Several threads interact
    to evaluate mpoly. Result is protected with mutex.
    
    mpoly_eval_t struct:
      uint32_t *pout         : output poly data
      const uint32_t *scalar : multiplying scalar
      uint32_t *pin          : input mpoly
      uint32_t reduce_coeff  : apply montgomery reduction to
                                 scalar
      uint32_t start_idx     : starting mpoly idx
      uint32_t last_idx      : last mpoly idx
      uint32_t max_threads   : number of threads. If 0, single thread is run
      uint32_t pidx          : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void mpoly_eval_server_h(mpoly_eval_t *args)
{
  if ((!args->max_threads) || (!utils_mproc_init)) {
    mpoly_eval_h((void *)args);
    return;
  }

  #ifndef PARALLEL_EN
    mpoly_eval_h((void *)args);
    return;
  #endif
  int nthreads = args->max_threads > utils_nprocs ? utils_nprocs : args->max_threads;
  uint32_t nvars = args->last_idx - args->start_idx;

  //printf("N threads : %d\n", nthreads);
  //printf("N vars    : %d\n", nvars);

  uint32_t vars_per_thread = nvars/nthreads;
  uint32_t i;
  uint32_t start_idx, last_idx;
   
  pthread_t *workers = (pthread_t *) malloc(nthreads * sizeof(pthread_t));
  mpoly_eval_t *w_args  = (mpoly_eval_t *)malloc(nthreads * sizeof(mpoly_eval_t));

  
  //printf ("Creating  %d threads, with %d vars per thread. Start idx: %d, Last idx %d\n",
   //       nthreads, vars_per_thread,args->start_idx, args->last_idx);

  for(i=0; i< nthreads; i++){
     start_idx = i * vars_per_thread;
     last_idx = (i+1) * vars_per_thread;
     if ( (i == nthreads - 1) && (last_idx != nvars) ){
         last_idx = nvars;
     }
     memcpy(&w_args[i], args, sizeof(mpoly_eval_t ));

     w_args[i].start_idx = start_idx;
     w_args[i].last_idx = last_idx;
     w_args[i].thread_id = i;

     //printf("Thread %d : start_idx : %d, last_idx : %d\n", i, w_args[i].start_idx,w_args[i].last_idx);
     if ( pthread_create(&workers[i], NULL, &mpoly_eval_h, (void *) &w_args[i]) ){
       free(workers);
       free(w_args);
       exit(1);
     }
  }

  for (i=0; i < nthreads; i++){
    pthread_join(workers[i], NULL);
  }

  free(workers);
  free(w_args);
}


void *mpoly_eval_h(void *vargs)
{
  mpoly_eval_t *args = (mpoly_eval_t *) vargs;
  uint32_t n_zpoly = args->pin[0];
  uint32_t zcoeff_d_offset = 1 + n_zpoly;
  uint32_t zcoeff_v_offset;
  uint32_t n_zcoeff;
  uint32_t scl[NWORDS_256BIT];
  uint32_t i,j;
  uint32_t zcoeff_v_in[NWORDS_256BIT], *zcoeff_v_out, zcoeff_d;
  uint32_t accum_n_zcoeff=0;

  /*
  printf("N zpoly: %d\n",n_zpoly);
  printf("Zcoeff D Offset : %d\n",zcoeff_d_offset);
  */
   //printf("Thread id: %d, Start idx : %d, Last idx : %d\n", args->thread_id, args->start_idx, args->last_idx);
  //TODO Change : If coeffs are accumulated, I don't need to do the accumulation
  //accum_n_zcoeff = args->pin[args->start_idx];
  
  for (i=0; i<args->start_idx; i++){
    accum_n_zcoeff += args->pin[i+1];
  }
 
  zcoeff_d_offset = accum_n_zcoeff*(NWORDS_256BIT+1) +1 + n_zpoly;

  for (i=args->start_idx; i<args->last_idx; i++){
    to_montgomery_h(scl, &args->scalar[i*NWORDS_256BIT], args->pidx);
    /*
    printf("In Scalar : \n");
    printU256Number(&args->scalar[i*NWORDS_256BIT]);
    printf("Out Scalar : \n");
    printU256Number(scl);
    */
    n_zcoeff = args->pin[1+i];
    accum_n_zcoeff += n_zcoeff;   
    //prev_n_zcoeff = n_zcoeff;
 
    //accum_n_zcoeff = args->pin[1+i];
    //n_zcoeff = accum_n_zcoeff - prev_n_zcoeff;
    //prev_n_zcoeff = accum_n_zcoeff;
    zcoeff_v_offset = zcoeff_d_offset + n_zcoeff;

    /*
    if ((i< 5) || (i > args->last_idx-5)){
      printf("N Zcoeff[%d] : %d\n", i, n_zcoeff);
      printf("Accum N Zcoeff[%d] : %d\n", i, accum_n_zcoeff);
      printf("Zcoeff D Offset : %d\n",zcoeff_d_offset);
      printf("ZCoeff_v_offset[%d] : %d\n", i , zcoeff_v_offset);
    }   
    */
    //printf("Thread id: %d Idx : %d. N Coeff : %d, Accum coeff : %d\n", args->thread_id, i, n_zcoeff, accum_n_zcoeff);

    for (j=0; j< n_zcoeff; j++){
       zcoeff_d = args->pin[zcoeff_d_offset+j];
       //memcpy(zcoeff_v_in , &args->pin[zcoeff_v_offset+j*NWORDS_256BIT], sizeof(uint32_t)*NWORDS_256BIT);
       zcoeff_v_out = &args->pout[zcoeff_d*NWORDS_256BIT];
       /*
       if ( ((i<5) || (i > args->last_idx-5)) && ((j<5) || (j>n_zcoeff-5))){
         printf("V[%d] in \n", zcoeff_d);
         printU256Number(zcoeff_v_in);
       }
       */
       //printf("%u, %u, %u, %u, %u, %u\n",i,j,zcoeff_d, n_zcoeff, zcoeff_v_offset, zcoeff_d_offset);
       montmult_h(zcoeff_v_in, &args->pin[zcoeff_v_offset+j*NWORDS_256BIT], scl, args->pidx);
       if(args->reduce_coeff){
         to_montgomery_h(zcoeff_v_in, zcoeff_v_in, args->pidx);
       }
       /*
       if ( ((i<5) || (i > args->last_idx-5)) && ((j<5) || (j>n_zcoeff-5))){
         printf("V[%d] in after mult \n", zcoeff_d);
         printU256Number(zcoeff_v_in);
         printf("V[%d] out before add \n", zcoeff_d);
         printU256Number(zcoeff_v_out);
       }
       */
       if (args->max_threads > 1){
         pthread_mutex_lock(&utils_lock);
         //printf("Mutex locked(%d)\n",args->thread_id);
         //fflush(stdin);
       }

       addm_h(zcoeff_v_out, zcoeff_v_out, zcoeff_v_in, args->pidx);

       if (args->max_threads > 1){
         //printf("Mutex unlocked(%d)\n", args->thread_id);
         //fflush(stdin);
         pthread_mutex_unlock(&utils_lock);
       }
       /*
       if ( ((i<5) || (i > args->last_idx-5)) && ((j<5) || (j>n_zcoeff-5))){
         printf("V[%d] out after add \n", zcoeff_d);
         printU256Number(zcoeff_v_out);
       }
       */
    }
    zcoeff_d_offset = accum_n_zcoeff*(NWORDS_256BIT+1) +1 + n_zpoly;
  }
}

void r1cs_to_mpoly_len_h(uint32_t *coeff_len, uint32_t *cin, cirbin_hfile_t *header, uint32_t extend)
{
  uint32_t n_coeff,i,j, poly_idx, prev_n_coeff, const_offset;

  const_offset = cin[0]+1;
  prev_n_coeff = 0;

  for (i=0; i < header->nConstraints; i++){
     n_coeff = cin[1+i];
     for (j=0; j < n_coeff - prev_n_coeff ;j++){
       poly_idx = cin[const_offset+j];
       coeff_len[poly_idx]++;
     }
     const_offset += ((n_coeff - prev_n_coeff) * (NWORDS_256BIT+1));
     prev_n_coeff = n_coeff;
  }

  if (extend){
    for (i=0; i < header->nPubInputs + header->nOutputs + 1; i++){
       coeff_len[i]++;
    }
  }
}

/*
  pout : 
   [0] ........... N Polys = Nvars
   [1 .. NVars] .. N coeff Poly[0..NVars-1] 
   [NVars + 1 .. NcoeffPoly[0]]
*/
void r1cs_to_mpoly_h(uint32_t *pout, uint32_t *cin, cirbin_hfile_t *header, uint32_t to_mont, uint32_t pidx, uint32_t extend)
{
  uint32_t *tmp_poly, *cum_c_poly, *cum_v_poly;
  uint32_t i,j;
  uint32_t poly_idx, const_offset, n_coeff,prev_n_coeff, coeff_offset, coeff_idx;
  uint32_t c_offset, v_offset;
  const uint32_t *One;
  
  One = CusnarksOneMontGet((mod_t)pidx);

  tmp_poly = (uint32_t *) calloc(header->nVars,sizeof(uint32_t *));
  cum_c_poly = (uint32_t *) calloc(header->nVars+1,sizeof(uint32_t *));
  cum_v_poly = (uint32_t *) calloc(header->nVars+1,sizeof(uint32_t *));

  cum_c_poly[0] = pout[0];
  cum_v_poly[0] = pout[0] + pout[1];

  for (i=1; i < header->nVars+1;i++){
    cum_c_poly[i] = pout[i] * (NWORDS_256BIT+1) + cum_c_poly[i-1];
    //cum_v_poly[i] = pout[i] * (NWORDS_256BIT+1) + cum_v_poly[i-1];
    cum_v_poly[i] = cum_c_poly[i] + pout[i+1];
  }

  const_offset = cin[0]+1;
  prev_n_coeff = 0;

  for (i=0; i < header->nConstraints; i++){
     n_coeff = cin[1+i];
     coeff_offset = const_offset + n_coeff - prev_n_coeff;
     for (j=0; j < n_coeff - prev_n_coeff ;j++){
       poly_idx = cin[const_offset+j];
       coeff_idx = tmp_poly[poly_idx]++;
       pout[cum_c_poly[poly_idx]+coeff_idx+1]=i;
       if (to_mont){
         to_montgomery_h(&pout[cum_v_poly[poly_idx]+1+coeff_idx*NWORDS_256BIT],
                         &cin[coeff_offset], pidx);
       } else {
         memcpy(&pout[cum_v_poly[poly_idx]+1+coeff_idx*NWORDS_256BIT], &cin[coeff_offset] ,NWORDS_256BIT * sizeof(uint32_t));
       }
       coeff_offset += NWORDS_256BIT;
     }
     const_offset += ((n_coeff - prev_n_coeff) * (NWORDS_256BIT+1));
     prev_n_coeff = n_coeff;
  }

  if (extend){
    for (i=0; i < header->nPubInputs + header->nOutputs + 1; i++){
       coeff_idx = tmp_poly[i]++;
       pout[cum_c_poly[i]+1+coeff_idx]=i + header->nConstraints;
       memcpy(&pout[cum_v_poly[i]+1+coeff_idx*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);
    }
  }
  //TODO
  /*
  for (i=1; i < header->nVars;i++){
    pout[i+1] += pout[i];
  }
  */

  free(tmp_poly);
  free(cum_c_poly);
  free(cum_v_poly);
}

/*
  Read header circuit binary file

  char * filename      : location of file to be written

  circuit bin header file format:
*/
void readU256CircuitFileHeader_h(cirbin_hfile_t *hfile, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");
  fread(&hfile->nWords, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->nPubInputs, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->nOutputs, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->nVars, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->nConstraints, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->cirformat, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->R1CSA_nWords, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->R1CSB_nWords, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->R1CSC_nWords, sizeof(unsigned long long), 1, ifp); 
  fclose(ifp);

}
/*
  Read circuit binary file
       
*/
void readU256CircuitFile_h(uint32_t *samples, const char *filename, unsigned long long nwords=0)
{
  FILE *ifp = fopen(filename,"rb");
  unsigned long long i=0;
  if (!nwords){
    while (!feof(ifp)){
      fread(&samples[i++], sizeof(uint32_t), 1, ifp); 
    }
  } else {
      fread(samples, sizeof(uint32_t), nwords, ifp); 
  }
  fclose(ifp);

}

void readR1CSFileHeader_h(r1csv1_t *r1cs_hdr, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");
  uint32_t k=0,i;
  uint32_t tmp_word, n_coeff;

  r1cs_hdr->R1CSA_nCoeff=0;
  r1cs_hdr->R1CSB_nCoeff=0;
  r1cs_hdr->R1CSC_nCoeff=0;

  fread(&r1cs_hdr->magic_number, sizeof(uint32_t), 1, ifp); 
  if (r1cs_hdr->magic_number != R1CS_HDR_MAGIC_NUMBER){
    printf("Unexpected R1CS header format\n");
    fclose(ifp);
    exit(1);
  }

  fread(&r1cs_hdr->version, sizeof(uint32_t), 1, ifp); 
  if (r1cs_hdr->version != R1CS_HDR_V01){
    printf("Unexpected R1CS version\n");
    fclose(ifp);
    exit(1);
  }

  fread(&r1cs_hdr->word_width_bytes, sizeof(uint32_t), 1, ifp); 
  if (r1cs_hdr->word_width_bytes != 4){
    printf("Unexpected R1CS word width\n");
    fclose(ifp);
    exit(1);
  }

  fread(&r1cs_hdr->nVars, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPubOutputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPubInputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPrivInputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nConstraints, sizeof(uint32_t), 1, ifp); 

  while (!feof(ifp)){
    fread(&n_coeff, sizeof(uint32_t), 1, ifp); 
    if (k%3 == R1CSA_IDX){
      r1cs_hdr->R1CSA_nCoeff+= (n_coeff);
    } else if (k%3 == R1CSB_IDX){
      r1cs_hdr->R1CSB_nCoeff+= (n_coeff);
    } else {
      r1cs_hdr->R1CSC_nCoeff+= (n_coeff);
    }
    for (i=0; i< n_coeff; i++){
      fseek(ifp, 4, SEEK_CUR);
      fread(&tmp_word, 1, 1, ifp); 
      fseek(ifp, tmp_word, SEEK_CUR);
    }
    k++;
  }
  fclose(ifp);
  
  return;
}
  

void readR1CSFile_h(uint32_t *samples, const char *filename, r1csv1_t *r1cs, r1cs_idx_t r1cs_idx )
{
  FILE *ifp = fopen(filename,"rb");
  uint32_t tmp_word, n_coeff;
  uint32_t r1cs_offset=0, r1cs_coeff_offset=1+r1cs->nConstraints, r1cs_val_offset = 1+r1cs->nConstraints;
  uint32_t k=0, accum_coeffs=0, i,j;

  samples[r1cs_offset++] = r1cs->nConstraints;
  
  fseek(ifp, R1CS_HDR_START_OFFSET_NWORDS * sizeof(uint32_t), SEEK_SET);

  while (!feof(ifp)){
    fread(&n_coeff, sizeof(uint32_t), 1, ifp); 
    if (k%3 == r1cs_idx) {
      accum_coeffs+= ((uint32_t) n_coeff);
      samples[r1cs_offset++] = accum_coeffs;
      r1cs_val_offset += n_coeff;
      for (i=0; i< n_coeff; i++){
        fread(&samples[r1cs_coeff_offset++], sizeof(uint32_t), 1, ifp); 
        fread(&tmp_word, 1,1, ifp);
        for(j=0; j <tmp_word; j++){
           fread(&samples[r1cs_val_offset+j], 1, 1, ifp); 
        }
        r1cs_val_offset += NWORDS_256BIT;
      }
      r1cs_coeff_offset = r1cs_val_offset;

    }  else {
      for (i=0; i< n_coeff; i++){
        fseek(ifp, 4, SEEK_CUR);
        fread(&tmp_word, 1, 1, ifp); 
        fseek(ifp, tmp_word, SEEK_CUR);
      }
    }
    
    k++;
  }

  fclose(ifp);
}


/*
  Read header PK binary file

  char * filename      : location of file to be read

  circuit bin header file format:
*/
void readU256PKFileHeader_h(pkbin_hfile_t *hfile, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");
  fread(&hfile->nWords, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->ftype, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->protocol, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->Rbitlen, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->k_binformat, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->k_ecformat, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->nVars, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->nPublic, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->domainSize, sizeof(uint32_t), 1, ifp); 
  fclose(ifp);

}
/*
  Read PK binary file
       
*/
void readU256PKFile_h(uint32_t *samples, const char *filename, unsigned long long nwords=0)
{
  readU256CircuitFile_h(samples, filename, nwords);
}



/*
  Write binary file

  t_uint32_t * samples : input vector containing samples. Vector is of length nwords 
  char * filename      : location of file to be written
  uint32_t nwords      : Number of samples to write.
*/
void writeU256DataFile_h(uint32_t *samples, const char *filename, unsigned long long nwords)
{
  FILE *ifp = fopen(filename,"wb");
  fwrite(samples, sizeof(uint32_t), nwords, ifp); 
  fclose(ifp);

}

void appendU256DataFile_h(uint32_t *samples, const char *filename, unsigned long long nwords)
{
  FILE *ifp = fopen(filename,"ab");
  fseek(ifp, 0, SEEK_END);
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

void readWitnessFile_h(uint32_t *samples, const char *filename, uint32_t fmt,  const unsigned long long inlen)
{
  unsigned long long i;
  unsigned long long nwords;
  uint32_t wsize;
  uint32_t wmore;
  uint32_t nwords32;
  uint32_t r[NWORDS_256BIT];
  FILE *ifp = fopen(filename,"rb");
  

  fread(&nwords,sizeof(uint32_t),WITNESS_HEADER_N_LEN_NWORDS,ifp);
  fread(&wsize,sizeof(uint32_t),WITNESS_HEADER_SIZE_LEN_NWORDS,ifp);
  fread(&wmore,sizeof(uint32_t),WITNESS_HEADER_OTHER_LEN_NWORDS,ifp); 
  if (!fmt){
    fseek(ifp, 32, SEEK_SET);
  }

  for (i=0;i<inlen; i++){
    fread(&samples[i*NWORDS_256BIT],sizeof(uint32_t),NWORDS_256BIT,ifp);
  }
  
  fclose(ifp);
}
void writeWitnessFile_h(uint32_t *samples, const char *filename, const unsigned long long nwords)
{
  uint32_t wsize = NWORDS_256BIT;
  uint32_t wmore = 0;
  FILE *ifp = fopen(filename,"wb");

  fwrite(&nwords, sizeof(uint64_t), 1, ifp); 
  fwrite(&wsize, sizeof(uint32_t), 1, ifp); 
  fwrite(&wmore, sizeof(uint32_t), 1, ifp); 

  fseek(ifp, WITNESS_HEADER_LEN_NWORDS * sizeof(uint32_t), SEEK_SET);

  fwrite(samples, sizeof(uint32_t), nwords*NWORDS_256BIT, ifp); 

}
/*
  Display u256 samples
 
  TODO - substitute by log function
*/
void printU256Number(const uint32_t *x)
{
  for (uint32_t i=0; i < NWORDS_256BIT; i++){
    printf("%x ",x[i]);
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

void montmultN_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t n, uint32_t pidx)
{
  uint32_t i;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i<n; i++){
     montmult_h(&z[i*NWORDS_256BIT], &x[i*NWORDS_256BIT], &y[i*NWORDS_256BIT], pidx);
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


void from_montgomeryN_h(uint32_t *z, const uint32_t *x, uint32_t n, uint32_t pidx, uint32_t strip_last=0)
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

void ec_stripc_h(uint32_t *z, uint32_t *x, uint32_t n)
{
  uint32_t i;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i< n; i++){
     memmove(&z[i*2*NWORDS_256BIT],&x[i*3*NWORDS_256BIT], 2 * NWORDS_256BIT * sizeof(uint32_t));
  }
}
void ec2_stripc_h(uint32_t *z, uint32_t *x, uint32_t n)
{
  uint32_t i;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i< n; i++){
     memmove(&z[i*4*NWORDS_256BIT],&x[i*6*NWORDS_256BIT], 4 * NWORDS_256BIT * sizeof(uint32_t));
  }
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
    if (compu256_h(&pin[i*NWORDS_256BIT],Zero)){
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
         return (ltu256_h((const uint32_t*)&v[i1*NWORDS_256BIT],(const uint32_t *)&v[i2*NWORDS_256BIT]));});
  }
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
}

void montmult_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
  uint32_t t0[NWORDS_256BIT], t1[NWORDS_256BIT];
  uint32_t t2[NWORDS_256BIT], t3[NWORDS_256BIT];

  montmult_h(t0,x,y,pidx);
  montmult_h(t1,&x[NWORDS_256BIT],&y[NWORDS_256BIT],pidx);

  addm_h(t2,x,&x[NWORDS_256BIT],pidx);
  addm_h(t3,y,&y[NWORDS_256BIT],pidx);
  montmult_h(t2,t2,t3,pidx);
  subm_h(z,t0,t1,pidx);
  addm_h(&z[NWORDS_256BIT],t0,t1,pidx);
  subm_h(&z[NWORDS_256BIT],t2,&z[NWORDS_256BIT],pidx);
  
}

// I am leaving this as a separate function to test both implementations are equal
void montsquare_h(uint32_t *U, const uint32_t *A, uint32_t pidx)
{
  montmult_h(U,A,A,pidx);
}

void montsquare_ext_h(uint32_t *U, const uint32_t *A, uint32_t pidx)
{
  montmult_ext_h(U,A,A,pidx);
}

#if 0
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
    subu256_h(U, U, N);
 }

}
#endif


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
*/

void intt_parallel2D_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, t_uint64 rstride, uint32_t pidx)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);

  ntt_parallel2D_h(A, roots, Nrows, fft_Nyx,  Ncols, fft_Nxx, rstride,1,pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&A[i*NWORDS_256BIT], &A[i*NWORDS_256BIT], &scaler[(Nrows + Ncols)*NWORDS_256BIT], pidx);
  }
}


void intt_parallel3D_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t N_fftx, uint32_t N_ffty, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, uint32_t pidx)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);

  ntt_parallel3D_h(A, roots, N_fftx, N_ffty, Nrows, fft_Nyx,  Ncols, fft_Nxx, 1,pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0;i < 1 << (N_fftx + N_ffty); i++){
      montmult_h(&A[i*NWORDS_256BIT], &A[i*NWORDS_256BIT], &scaler[(N_fftx + N_ffty)*NWORDS_256BIT], pidx);
  }
}

/*
  4 step N 256 bit sample FFT. Read https://www.davidhbailey.com/dhbpapers/fftq.pdf for more info
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
   uint32_t rstride : Root stride
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void ntt_parallel2D_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, t_uint64 rstride, int32_t direction, uint32_t pidx)
{
  uint32_t Anrows = (1<<Nrows);
  uint32_t Ancols = (1<<Ncols);
  uint32_t Mnrows = Ancols;
  uint32_t Mncols = Anrows;
  uint32_t *M = (uint32_t *) malloc (Anrows * Ancols * NWORDS_256BIT * sizeof(uint32_t));

  transpose_h(M,A,Anrows, Ancols);

  for (uint32_t i=0;i < Mnrows; i++){
    ntt_parallel_h(&M[i*NWORDS_256BIT*Mncols], roots, fft_Nyx, Nrows - fft_Nyx, rstride*Mnrows, direction, FFT_T_DIT,pidx);
    for (uint32_t j=0;j < Mncols; j++){   
        montmult_h(&M[i*NWORDS_256BIT*Mncols+j*NWORDS_256BIT], &M[i*NWORDS_256BIT*Mncols+j*NWORDS_256BIT], &roots[rstride*i*j*NWORDS_256BIT], pidx);
    }
  }

  transpose_h(A,M,Mnrows, Mncols);

  for (uint32_t i=0;i < Anrows; i++){
    ntt_parallel_h(&A[i*NWORDS_256BIT*Ancols], roots, fft_Nxx,Ncols - fft_Nxx, rstride*Mncols, direction, FFT_T_DIT, pidx);
  }

  transpose_h(M,A,Anrows, Ancols);
  memcpy(A,M,Ancols * Anrows * NWORDS_256BIT * sizeof(uint32_t));

  free(M);
}

void ntt_parallel3D_h(uint32_t *A, const uint32_t *roots, uint32_t Nfft_x, uint32_t Nfft_y, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, int32_t direction, uint32_t pidx)
{
  uint32_t Anrows = (1<<Nfft_y);
  uint32_t Ancols = (1<<Nfft_x);
  uint32_t Mnrows = Ancols;
  uint32_t Mncols = Anrows;
  uint32_t *M = (uint32_t *) malloc (Anrows * Ancols * NWORDS_256BIT * sizeof(uint32_t));
  uint32_t i,j;

  transpose_h(M,A,Anrows, Ancols);

  for (i=0;i < Mnrows; i++){
    ntt_parallel2D_h(&M[i*NWORDS_256BIT*Mncols], roots, Nrows, fft_Nyx, Nfft_y - Nrows, Nrows - fft_Nyx, Mnrows, direction,pidx);
    for (j=0;j < Mncols; j++){   
        montmult_h(&M[i*NWORDS_256BIT*Mncols+j*NWORDS_256BIT], &M[i*NWORDS_256BIT*Mncols+j*NWORDS_256BIT], &roots[i*j*NWORDS_256BIT], pidx);
    }
  }
  
  transpose_h(A,M,Mnrows, Mncols);

  for (i=0;i < Anrows; i++){
    ntt_parallel2D_h(&A[i*NWORDS_256BIT*Ancols], roots, Ncols,fft_Nxx,  Nfft_x- Ncols, Ncols - fft_Nxx, Mncols, direction, pidx);
  }

  transpose_h(M,A,Anrows, Ancols);
  memcpy(A,M,Ancols * Anrows * NWORDS_256BIT * sizeof(uint32_t));

  free(M);
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
   uint32_t rstride : roots stride
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void ntt_parallel_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, int32_t direction, fft_mode_t fft_mode,  uint32_t pidx)
{
  uint32_t *M = (uint32_t *) malloc ( (1 << (Nrows + Ncols)) * NWORDS_256BIT * sizeof(uint32_t));

  ntt_parallel_T_h(A, roots, Nrows, Ncols, rstride, direction, fft_mode, pidx);

  transpose_h(M,A,1<<Nrows, 1<<Ncols);
  memcpy(A,M, (1 << (Nrows + Ncols)) * NWORDS_256BIT * sizeof(uint32_t));

  free(M);
}

static void ntt_parallel_T_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, int32_t direction, fft_mode_t fft_mode, uint32_t pidx)
{
  uint32_t Anrows = (1<<Nrows);
  uint32_t Ancols = (1<<Ncols);
  int64_t ridx;
  void (*fft_ptr)(uint32_t *, const uint32_t *, uint32_t , t_uint64 , t_uint64 , int32_t , uint32_t ) = ntt_h;
  if (fft_mode == FFT_T_DIF){
      fft_ptr = ntt_dif_h;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < Ancols; i++){
    fft_ptr(&A[i<<NWORDS_256BIT_SHIFT], roots, Nrows,Ancols, rstride<<Ncols, direction,  pidx);
  }


  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << (Nrows + Ncols); i++){
    ridx = (rstride * (i >> Ncols) * (i & (Ancols - 1)) * direction ) & (rstride * Anrows * Ancols - 1);
    montmult_h(&A[i<<NWORDS_256BIT_SHIFT],
               &A[i<<NWORDS_256BIT_SHIFT],
               &roots[ridx << NWORDS_256BIT_SHIFT],
               pidx);
  }
  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < Anrows; i++){
    fft_ptr(&A[i<<Ncols+NWORDS_256BIT_SHIFT], roots, Ncols,1, rstride<<Nrows, direction, pidx);
  }
}

static void _ntt_parallel_T_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, int32_t direction, fft_mode_t fft_mode, uint32_t pidx)
{
  void (*fft_ptr)(uint32_t *, const uint32_t *, uint32_t , t_uint64 , t_uint64 , int32_t , uint32_t ) = _ntt_h;
  if (fft_mode == FFT_T_DIF){
      fft_ptr = _ntt_dif_h;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << Ncols; i++){
    fft_ptr(&A[i<<NWORDS_256BIT_SHIFT], roots, Nrows,1 << Ncols, rstride<<Ncols, direction,  pidx);
  }

  montmult_parallel_reorder_h(A, roots, Nrows, Ncols, rstride, direction, pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << Nrows; i++){
    fft_ptr(&A[i<<(Ncols+NWORDS_256BIT_SHIFT)], roots, Ncols,1, rstride<<Nrows, direction, pidx);
  }
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
*/
void intt_parallel_h(uint32_t *A, const uint32_t *roots,uint32_t format, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, fft_mode_t fft_mode, uint32_t pidx)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);

  ntt_parallel_h(A, roots, Nrows, Ncols, rstride, -1, fft_mode, pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&A[i<<NWORDS_256BIT_SHIFT], &A[i<<NWORDS_256BIT_SHIFT], &scaler[(Nrows + Ncols)<<NWORDS_256BIT_SHIFT], pidx);
  }
}


/*
   Computes the forward number-theoretic transform of the given vector in place,
   with respect to the given primitive nth root of unity under the given modulus.
   The length of the vector must be a power of 2.
   NOTE https://www.nayuki.io/page/number-theoretic-transform-integer-dft

   NOTE https://www.nayuki.io/page/number-theoretic-transform-integer-dft

   uint32_t *A     : ordered input vector of length 1<<levels in montgomery format. Ordered result is 
                  is stored in A as well.
   uint32_t *roots : input roots (first root is 1) in Montgomery. If roots are inverse, IFFT is computed. 
   uint32_t levels : 1<<levels is the number of samples in the FFT
   uint32_t rstride : roots stride
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
static void _ntt_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx)
{
   int32_t n = 1 << levels;
   uint32_t i,j,l, halfsize;
   int32_t k, tablestep, size, ridx;
   uint32_t right[NWORDS_256BIT];

   for(size=2; size <= n; size *=2){
     halfsize = size >> 1; 
     tablestep = (1 * direction)*n/size;
     for (i=0; i<n; i+=size){
        k = 0;
        for (j=i; j<i+halfsize; j++){
           l = j + halfsize;
           ridx = (rstride*k) & (rstride * n-1);
           montmult_h(right,&A[(astride*l)<<NWORDS_256BIT_SHIFT], &roots[ridx<<NWORDS_256BIT_SHIFT], pidx);
           subm_h(&A[(astride*l)<<NWORDS_256BIT_SHIFT], &A[(astride*j)<<NWORDS_256BIT_SHIFT], right, pidx);
           addm_h(&A[(astride*j)<<NWORDS_256BIT_SHIFT], &A[(astride*j)<<NWORDS_256BIT_SHIFT], right, pidx);
           k += tablestep;
        }
     }
  }
}
void ntt_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx)
{
   ntt_reorder_h(A, levels, astride);
   _ntt_h(A, roots, levels, astride, rstride, direction, pidx);
}

static void _ntt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx)
{
   uint32_t i,j,k ,s, t;
   uint32_t right[NWORDS_256BIT];

   for (i=0; i<levels; i++){
     for (j=0; j < 1<< i; j++) {
       for (k=0; k < 1 << (levels - i - 1); k++){
          s = j * (1 << (levels - i)) + k;
          t = s + (1 << (levels - i - 1));
          subm_h(right,
                 &A[astride*s<<NWORDS_256BIT_SHIFT],
                 &A[astride*t<<NWORDS_256BIT_SHIFT],
                 pidx);
          addm_h(&A[astride*s<<NWORDS_256BIT_SHIFT],
                 &A[astride*s<<NWORDS_256BIT_SHIFT],
                 &A[astride*t<<NWORDS_256BIT_SHIFT], pidx);
          montmult_h(&A[astride*t<<NWORDS_256BIT_SHIFT],
                     right,
                     &roots[((rstride*(1<<i)*k*direction) & (rstride*(1 << levels)-1))<<NWORDS_256BIT_SHIFT], pidx);
       }
     }
   }
}

static void ntt_reorder_h(uint32_t *A, uint32_t levels, uint32_t astride)
{
  uint32_t (*reverse_ptr) (uint32_t, uint32_t);
  
   if (levels <= 8){
      reverse_ptr = reverse8;
   } else if (levels <= 16) {
      reverse_ptr = reverse16;
   } else {
      reverse_ptr = reverse32;
   }

   #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
   #endif
   for (uint32_t i=0; i < 1 << levels ; i++){
      uint32_t j = reverse_ptr(i, levels);
      if (j > i){
         swapu256_h(&A[astride*i<<NWORDS_256BIT_SHIFT],&A[astride*j<<NWORDS_256BIT_SHIFT]);
      }
   }
}

static void montmult_reorder_h(uint32_t *A, const uint32_t *roots, uint32_t levels, uint32_t pidx)
{
  uint32_t (*reverse_ptr) (uint32_t, uint32_t);

  if (levels <= 8){
      reverse_ptr = reverse8;
   } else if (levels <= 16) {
      reverse_ptr = reverse16;
   } else {
      reverse_ptr = reverse32;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i<1 << levels; i++){
    uint32_t j = reverse_ptr(i, levels);
    if (j > i){
      montmult_h(&A[i<<NWORDS_256BIT_SHIFT], &A[i<<NWORDS_256BIT_SHIFT], &roots[j<<NWORDS_256BIT_SHIFT], pidx);
      montmult_h(&A[j<<NWORDS_256BIT_SHIFT], &A[j<<NWORDS_256BIT_SHIFT], &roots[i<<NWORDS_256BIT_SHIFT], pidx);
    } else if (j == i) {
      montmult_h(&A[i<<NWORDS_256BIT_SHIFT], &A[i<<NWORDS_256BIT_SHIFT], &roots[i<<NWORDS_256BIT_SHIFT], pidx);
    } 
  }
}

static void montmult_parallel_reorder_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, uint32_t rstride, uint32_t direction, uint32_t pidx)
{
  uint32_t (*reverse_ptr) (uint32_t, uint32_t);
  uint32_t levels = Nrows + Ncols;

  if ((levels+1) <= 8){
      reverse_ptr = reverse8;
   } else if ((levels+1) <= 16) {
      reverse_ptr = reverse16;
   } else {
      reverse_ptr = reverse32;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (t_uint64 i=0;i<1 << levels; i++){
    int32_t ridx = (rstride * (i >> Ncols) * (i & ( (1 << Ncols) - 1)) * direction ) & ( (rstride  << (Nrows + Ncols)) - 1);
    t_uint64 j = reverse_ptr(ridx, levels+1);
    montmult_h(&A[i<<NWORDS_256BIT_SHIFT], &A[i<<NWORDS_256BIT_SHIFT], &roots[j<<NWORDS_256BIT_SHIFT], pidx);
  }
}


void ntt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx)
{
   _ntt_dif_h(A, roots, levels, astride, rstride, direction, pidx);
   ntt_reorder_h(A, levels, astride);
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
void intt_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx)
{
  ntt_reorder_h(A, levels, 1);
  _intt_h(A, roots, format, levels, rstride, pidx);
}

static void _intt_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);
  
  _ntt_h(A, roots, levels, 1,rstride, -1, pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i< (1<<levels); i++){
     montmult_h(&A[i<<NWORDS_256BIT_SHIFT], &A[i<<NWORDS_256BIT_SHIFT], &scaler[levels << NWORDS_256BIT_SHIFT], pidx);
  }
  
}
void intt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx)
{
  _intt_dif_h(A, roots, format, levels, rstride, pidx);
  ntt_reorder_h(A, levels, 1);
}

static void _intt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);

  _ntt_dif_h(A, roots, levels, 1,rstride,-1, pidx);
  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i< (1<<levels); i++){
     montmult_h(&A[i<<NWORDS_256BIT_SHIFT], &A[i<<NWORDS_256BIT_SHIFT], &scaler[levels << NWORDS_256BIT_SHIFT], pidx);
  }
}

void interpol_odd_h(uint32_t *A, const uint32_t *roots, uint32_t levels,t_uint64 rstride, uint32_t pidx)
{ 
  _intt_dif_h(A, roots, 1, levels,rstride, pidx);
  montmult_reorder_h(A,roots,levels, pidx);
  _ntt_h(A, roots, levels,1, rstride,1, pidx);
}

void M_init_h(uint32_t nroots)
{
  M_transpose = (uint32_t *) malloc ( (t_uint64) nroots * NWORDS_256BIT * sizeof(uint32_t));
  M_mul = (uint32_t *) malloc ( (t_uint64)(nroots+1) * NWORDS_256BIT * sizeof(uint32_t));
}

void M_free_h(void)
{
  free (M_transpose);
  free (M_mul);
}

void interpol_parallel_odd_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, uint32_t pidx)
{
  intt_parallel_h(A, roots, 1, Nrows, Ncols, rstride, FFT_T_DIF, pidx);
  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&A[i<<NWORDS_256BIT_SHIFT], &A[i<<NWORDS_256BIT_SHIFT], &roots[i<<NWORDS_256BIT_SHIFT], pidx);
  }
  ntt_parallel_h(A, roots, Nrows, Ncols, rstride, 1, FFT_T_DIT,  pidx);
}

uint32_t *nttmul_parallel_h(uint32_t *A, uint32_t *B, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, uint32_t pidx)
{
  t_uint64 size = (t_uint64) (1 << (Nrows + Ncols + 1)) * NWORDS_256BIT;
  uint32_t mNrows = Nrows, mNcols = Ncols;
  if (Nrows < Ncols){
    mNrows++;
  } else {
    mNcols++;
  }
  memcpy(M_mul, A, size/2 * sizeof(uint32_t));
  memcpy(&M_mul[size/2], B, size/2 * sizeof(uint32_t));

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&M_mul[(2*i)<<NWORDS_256BIT_SHIFT], &A[i<<NWORDS_256BIT_SHIFT], &B[i<<NWORDS_256BIT_SHIFT], pidx);
  }

  interpol_parallel_odd_h(A, roots, Nrows, Ncols, rstride, pidx);
  interpol_parallel_odd_h(B, roots, Nrows, Ncols, rstride, pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&M_mul[(2*i+1)<<NWORDS_256BIT_SHIFT], &A[i<<NWORDS_256BIT_SHIFT], &B[i<<NWORDS_256BIT_SHIFT], pidx);
  }
  return M_mul;

  intt_parallel_h(M_mul, roots, 0, mNrows, mNcols, 1, FFT_T_DIF, pidx);
  return M_mul;
}

/*

  Generate format of FFT from number of samples. Parameters include 1D FFT/2D FFT/ 3D FFT/ 4D FFT, size of 
   matrix for multi D FFT,...

  fft_params_t *ntt_params : pointer to structure containing resulting FFT format
  uint32_t nsamples : number of samples of FFT
  
*/
void ntt_build_h(fft_params_t *ntt_params, uint32_t nsamples)
{
  uint32_t min_samples = 1 << 6;
  uint32_t levels = (uint32_t) ceil(log2(MAX(nsamples, min_samples)));
  memset(ntt_params,0,sizeof(fft_params_t));
  ntt_params->padding = (1 << levels) - nsamples;
  ntt_params->levels = levels;
  
  if (MAX(min_samples, nsamples) <= 32){
    ntt_params->fft_type =  FFT_T_1D;
    ntt_params->fft_N[0] = levels;

// } else if (nsamples <= 1024) {
//    ntt_params->fft_type =  FFT_T_2D;
//    ntt_params->fft_N[(1<<FFT_T_2D)-1] = levels/2;
//    ntt_params->fft_N[(1<<FFT_T_2D)-2] = levels - levels/2;

  } else if (MAX(min_samples, nsamples) <= (1<<20)){
  //} else if (MAX(min_samples, nsamples) <= (1<<12)){
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
   //uint32_t tmp[NWORDS_256BIT];
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   addu256_h(z, x, y);
   if(compu256_h(z, N) >= 0) {
      subu256_h(z, z, N);
   }

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
   uint32_t tmp[NWORDS_256BIT];
   const uint32_t *N = CusnarksPGet((mod_t)pidx);

   subu256_h(z, x, y);
   if(compu256_h(z, N) >= 0) {
       addu256_h(z, z, N);
   }

   //memcpy(z, tmp, sizeof(uint32_t)*NWORDS_256BIT);
}
void subm_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx)
{
   subm_h(z,x,y,pidx);
   subm_h(&z[NWORDS_256BIT],&x[NWORDS_256BIT],&y[NWORDS_256BIT],pidx);
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
  const uint32_t *_1 = CusnarksOneMontGet((mod_t)pidx);
  
  memcpy(roots,_1,sizeof(uint32_t)*NWORDS_256BIT);
  memcpy(&roots[NWORDS_256BIT],primitive_root,sizeof(uint32_t)*NWORDS_256BIT);
  for (i=2;i<nroots; i++){
    montmult_h(&roots[i*NWORDS_256BIT], &roots[(i-1)*NWORDS_256BIT], primitive_root, pidx);
  }

  return;
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
uint32_t getbitu256g_h(uint32_t *x, uint32_t n, uint32_t group_size)
{
  uint32_t w, b,i, val=0;
  
  w = n >> NBITS_WORD_LOG2;
  b = n & NBITS_WORD_MOD;

  for (i = 0; i < group_size; i++){
    val |= (( (x[w+NWORDS_256BIT*i] >> b) & 0x1) << i);
  }

  return val;
}
/*
  Montgomery Modular Inverse - Revisited
  E. Savas, C.K.Koc
  IEEE trasactions on Computers Vol49, No 7. July 2000
*/
void montinv_h(uint32_t *y, uint32_t *x,  uint32_t pidx)
{
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
}
void almmontinv_h(uint32_t *r, uint32_t *k, uint32_t *a, uint32_t pidx)
{
  const uint32_t *P = CusnarksPGet((mod_t)pidx);

  uint32_t u[NWORDS_256BIT], v[NWORDS_256BIT];
  uint32_t s[] = {1,0,0,0,0,0,0,0};
  uint32_t r1[] = {0,0,0,0,0,0,0,0};
  uint32_t i = 0;

  const uint32_t *zero = CusnarksZeroGet();
  
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

void ec_jacadd_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t x_cpy[NWORDS_256BIT * ECP_JAC_OUTDIMS];
  uint32_t y_cpy[NWORDS_256BIT * ECP_JAC_OUTDIMS];

  uint32_t Z1sq[NWORDS_256BIT], Z1cube[NWORDS_256BIT];
  uint32_t Z2sq[NWORDS_256BIT], Z2cube[NWORDS_256BIT];
  uint32_t U1[NWORDS_256BIT], U2[NWORDS_256BIT];
  uint32_t S1[NWORDS_256BIT], S2[NWORDS_256BIT];

  uint32_t *X1, *Y1, *Z1;
  uint32_t *X2, *Y2, *Z2;
  uint32_t *X3, *Y3, *Z3;
  uint32_t *H, *R, *Hsq, *Hcube;
  
  X1 = x_cpy; Y1 = &x_cpy[NWORDS_256BIT]; Z1 = &x_cpy[2*NWORDS_256BIT];
  X2 = y_cpy; Y2 = &y_cpy[NWORDS_256BIT]; Z2 = &y_cpy[2*NWORDS_256BIT];
  X3 = z; Y3 = &z[NWORDS_256BIT]; Z3 = &z[2*NWORDS_256BIT];
  H = U2;  R = S2; Hsq = Y3; Hcube = Z3;

  if (ec_iseq_h( x,
                &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT]) ) {

          memmove( z, y, sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_OUTDIMS);
          return;

  } else if (ec_iseq_h( y,
                &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT]) ) {

          memmove( z, x, sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_OUTDIMS);
          return;
  }

  memcpy(x_cpy, x, sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_OUTDIMS);
  memcpy(y_cpy, y, sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_OUTDIMS);

  montsquare_h(Z1sq, Z1, pidx);
  montmult_h(Z1cube, Z1sq, Z1, pidx);
  montsquare_h(Z2sq, Z2, pidx);
  montmult_h(Z2cube, Z2sq, Z2, pidx);

  montmult_h(U1, X1, Z2sq, pidx);
  montmult_h(U2, X2, Z1sq, pidx);
  montmult_h(S1, Y1, Z2cube, pidx);
  montmult_h(S2, Y2, Z1cube, pidx);

  if (equ256_h(U1, U2)){
     if (!equ256_h(S1, S2)) {
              memmove(
                  z,
                  &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT],
                  sizeof(uint32_t)*ECP_JAC_OUTDIMS * NWORDS_256BIT);
               return;
     } else {
          ec_jacdouble_h(z, x, pidx);
          return;
     }
  }

 subm_h(H,U2,U1,pidx);  // H is U2
 subm_h(R, S2, S1, pidx);  // R is S2

 montsquare_h(Hsq, H, pidx);  // Hsq is Y3
 montmult_h(Hcube, H, Hsq, pidx);  // Hcube is Z3

 montsquare_h(X3, R, pidx);
 subm_h(X3, X3, Hcube, pidx);
 montmult_h(U1,U1, Hsq, pidx);
 subm_h(X3, X3, U1, pidx);
 subm_h(X3, X3, U1, pidx);

 montmult_h(S1,S1,Hcube, pidx);
 subm_h(U1, U1, X3, pidx);
 montmult_h(Y3,R, U1, pidx);
 subm_h(Y3, Y3, S1, pidx);

 montmult_h(Z3, Z1, Z2, pidx);
 montmult_h(Z3, Z3, H, pidx);

}

/*
 * EC_P JAC = EC_P AFF + EC_P JAC
*/
void ec_jacaddmixed_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t x_cpy[NWORDS_256BIT * ECP_JAC_INDIMS];
  uint32_t y_cpy[NWORDS_256BIT * ECP_JAC_OUTDIMS];
  const uint32_t *One = CusnarksOneMontGet(pidx);

  uint32_t Z2sq[NWORDS_256BIT], Z2cube[NWORDS_256BIT];
  uint32_t U1[NWORDS_256BIT];
  uint32_t S1[NWORDS_256BIT];

  uint32_t *X1, *Y1;
  uint32_t *X2, *Y2, *Z2;
  uint32_t *X3, *Y3, *Z3;
  uint32_t *H, *R, *Hsq, *Hcube;
  
  X1 = x_cpy; Y1 = &x_cpy[NWORDS_256BIT]; 
  X2 = y_cpy; Y2 = &y_cpy[NWORDS_256BIT]; Z2 = &y_cpy[2*NWORDS_256BIT];
  X3 = z; Y3 = &z[NWORDS_256BIT]; Z3 = &z[2*NWORDS_256BIT];
  H = X2;  R = Y2; Hsq = Y3; Hcube = Z3;

  if (ec_iseq_h( x,
                &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT]) ) {

          memmove( z, y, sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_OUTDIMS);
          return;

  } else if (ec_iseq_h( y,
                &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT]) ) {

          memmove( z, x, sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_INDIMS);
          memmove(&z[2*NWORDS_256BIT], One, sizeof(uint32_t) * NWORDS_256BIT);
          return;
  }

  memcpy(x_cpy, x, sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_INDIMS);
  memcpy(y_cpy, y, sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_OUTDIMS);

  montsquare_h(Z2sq, Z2, pidx);
  montmult_h(Z2cube, Z2sq, Z2, pidx);

  montmult_h(U1, X1, Z2sq, pidx);
  montmult_h(S1, Y1, Z2cube, pidx);

  if (equ256_h(U1, X2)){
     if (!equ256_h(S1, Y2)) {
              memmove(
                  z,
                  &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT],
                  sizeof(uint32_t)*ECP_JAC_OUTDIMS * NWORDS_256BIT);
               return;
     } else {
          ec_jacdouble_h(z, y, pidx);
          return;
     }
  }

 subm_h(H,X2,U1,pidx);  // H = U2 - U1
 subm_h(R, Y2, S1, pidx);  // R = S2 - S1

 montsquare_h(Hsq, H, pidx);  // Hsq = H * H
 montmult_h(Hcube, H, Hsq, pidx);  // Hcube  = H * H * H

 montsquare_h(X3, R, pidx); // X3 = R * R
 subm_h(X3, X3, Hcube, pidx); // R^2 - H^3
 montmult_h(U1,U1, Hsq, pidx); // U1 = U1 * H^2
 subm_h(X3, X3, U1, pidx);     // X3 = R^2 - H^3 - U1 * H^2
 subm_h(X3, X3, U1, pidx);     // X3 = R^2 - H^3 - 2*(U1 * H^2)

 montmult_h(S1,S1,Hcube, pidx);  // S1 = S1 * H^3
 subm_h(U1, U1, X3, pidx);       // U1 = U1 * H^2  - X3
 montmult_h(Y3,R, U1, pidx);     // Y3 = R * (U1 * H^2 - X3)
 subm_h(Y3, Y3, S1, pidx);       // Y3 = R * (U1 * H^2 - X3) - S1*H^3

 montmult_h(Z3, Z2, H, pidx);

}


void ec2_jacadd_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t x_cpy[NWORDS_256BIT * ECP2_JAC_OUTDIMS];
  uint32_t y_cpy[NWORDS_256BIT * ECP2_JAC_OUTDIMS];

  uint32_t Z1sq[NWORDS_256BIT*2], Z1cube[NWORDS_256BIT*2];
  uint32_t Z2sq[NWORDS_256BIT*2], Z2cube[NWORDS_256BIT*2];
  uint32_t U1[NWORDS_256BIT*2], U2[NWORDS_256BIT*2];
  uint32_t S1[NWORDS_256BIT*2], S2[NWORDS_256BIT*2];

  uint32_t *X1, *Y1, *Z1;
  uint32_t *X2, *Y2, *Z2;
  uint32_t *X3, *Y3, *Z3;
  uint32_t *H, *R, *Hsq, *Hcube;
  
  X1 = x_cpy; Y1 = &x_cpy[2*NWORDS_256BIT]; Z1 = &x_cpy[4*NWORDS_256BIT];
  X2 = y_cpy; Y2 = &y_cpy[2*NWORDS_256BIT]; Z2 = &y_cpy[4*NWORDS_256BIT];
  X3 = z;     Y3 = &z[2*NWORDS_256BIT];    Z3 = &z[4*NWORDS_256BIT];
  H = U2;  R = S2; Hsq = Y3; Hcube = Z3;

  if (ec2_iseq_h( x,
                &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT])){

          memmove( z, y, sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_OUTDIMS);
          return;

  } else if (ec2_iseq_h( y,
                &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT])){

          memmove( z, x, sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_OUTDIMS);
          return;
  }

  memcpy(x_cpy, x, sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_OUTDIMS);
  memcpy(y_cpy, y, sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_OUTDIMS);

  montsquare_ext_h(Z1sq, Z1, pidx);
  montmult_ext_h(Z1cube, Z1sq, Z1, pidx);
  montsquare_ext_h(Z2sq, Z2, pidx);
  montmult_ext_h(Z2cube, Z2sq, Z2, pidx);

  montmult_ext_h(U1, X1, Z2sq, pidx);
  montmult_ext_h(U2, X2, Z1sq, pidx);
  montmult_ext_h(S1, Y1, Z2cube, pidx);
  montmult_ext_h(S2, Y2, Z1cube, pidx);

  if (equ256_h(U1, U2) && equ256_h(&U1[NWORDS_256BIT], &U2[NWORDS_256BIT])){
     if (!equ256_h(S1, S2) || !equ256_h(&S1[NWORDS_256BIT], &S2[NWORDS_256BIT])){
              memmove(
                  z,
                  &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT],
                  sizeof(uint32_t)*ECP2_JAC_OUTDIMS * NWORDS_256BIT);
               return;
     } else {
          ec2_jacdouble_h(z, x, pidx);
          return;
     }
  }

 subm_ext_h(H,U2,U1,pidx);  // H is U2
 subm_ext_h(R, S2, S1, pidx);  // R is S2

 montsquare_ext_h(Hsq, H, pidx);  // Hsq is Y3
 montmult_ext_h(Hcube, H, Hsq, pidx);  // Hcube is Z3

 montsquare_ext_h(X3, R, pidx);
 subm_ext_h(X3, X3, Hcube, pidx);
 montmult_ext_h(U1,U1, Hsq, pidx);
 subm_ext_h(X3, X3, U1, pidx);
 subm_ext_h(X3, X3, U1, pidx);

 montmult_ext_h(S1,S1,Hcube, pidx);
 subm_ext_h(U1, U1, X3, pidx);
 montmult_ext_h(Y3,R, U1, pidx);
 subm_ext_h(Y3, Y3, S1, pidx);

 montmult_ext_h(Z3, Z1, Z2, pidx);
 montmult_ext_h(Z3, Z3, H, pidx);

}

/*
 * EC_P JAC = EC_P AFF + EC_P JAC
*/
void ec2_jacaddmixed_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t x_cpy[NWORDS_256BIT * ECP2_JAC_INDIMS];
  uint32_t y_cpy[NWORDS_256BIT * ECP2_JAC_OUTDIMS];
  const uint32_t *One = CusnarksOneMontGet(pidx);

  uint32_t Z2sq[NWORDS_256BIT*2], Z2cube[NWORDS_256BIT*2];
  uint32_t U1[NWORDS_256BIT*2];
  uint32_t S1[NWORDS_256BIT*2];

  uint32_t *X1, *Y1;
  uint32_t *X2, *Y2, *Z2;
  uint32_t *X3, *Y3, *Z3;
  uint32_t *H, *R, *Hsq, *Hcube;
  
  X1 = x_cpy; Y1 = &x_cpy[2*NWORDS_256BIT]; 
  X2 = y_cpy; Y2 = &y_cpy[2*NWORDS_256BIT]; Z2 = &y_cpy[4*NWORDS_256BIT];
  X3 = z; Y3 = &z[2*NWORDS_256BIT]; Z3 = &z[4*NWORDS_256BIT];
  H = X2;  R = Y2; Hsq = Y3; Hcube = Z3;

  if (ec2_iseq_h( x,
                &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT]) ) {

          memmove( z, y, sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_OUTDIMS);
          return;

  } else if (ec2_iseq_h( y,
                &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT]) ) {

          memmove( z, x, sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_INDIMS);
          memmove(&z[4*NWORDS_256BIT], One, sizeof(uint32_t) * NWORDS_256BIT);
          memset(&z[5*NWORDS_256BIT], 0, sizeof(uint32_t)*NWORDS_256BIT);
          return;
  }

  memcpy(x_cpy, x, sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_INDIMS);
  memcpy(y_cpy, y, sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_OUTDIMS);

  montsquare_ext_h(Z2sq, Z2, pidx);
  montmult_ext_h(Z2cube, Z2sq, Z2, pidx);

  montmult_ext_h(U1, X1, Z2sq, pidx);
  montmult_ext_h(S1, Y1, Z2cube, pidx);

  if (equ256_h(U1, X2) && equ256_h(&U1[NWORDS_256BIT], &X2[NWORDS_256BIT])){
     if (!equ256_h(S1, Y2) || !equ256_h(&S1[NWORDS_256BIT], &Y2[NWORDS_256BIT])) {
              memmove(
                  z,
                  &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT],
                  sizeof(uint32_t)*ECP2_JAC_OUTDIMS * NWORDS_256BIT);
               return;
     } else {
          ec2_jacdouble_h(z, y, pidx);
          return;
     }
  }

 subm_ext_h(H,X2,U1,pidx);  // H = U2 - U1
 subm_ext_h(R, Y2, S1, pidx);  // R = S2 - S1

 montsquare_ext_h(Hsq, H, pidx);  // Hsq = H * H
 montmult_ext_h(Hcube, H, Hsq, pidx);  // Hcube  = H * H * H

 montsquare_ext_h(X3, R, pidx); // X3 = R * R
 subm_ext_h(X3, X3, Hcube, pidx); // R^2 - H^3
 montmult_ext_h(U1,U1, Hsq, pidx); // U1 = U1 * H^2
 subm_ext_h(X3, X3, U1, pidx);     // X3 = R^2 - H^3 - U1 * H^2
 subm_ext_h(X3, X3, U1, pidx);     // X3 = R^2 - H^3 - 2*(U1 * H^2)

 montmult_ext_h(S1,S1,Hcube, pidx);  // S1 = S1 * H^3
 subm_ext_h(U1, U1, X3, pidx);       // U1 = U1 * H^2  - X3
 montmult_ext_h(Y3,R, U1, pidx);     // Y3 = R * (U1 * H^2 - X3)
 subm_ext_h(Y3, Y3, S1, pidx);       // Y3 = R * (U1 * H^2 - X3) - S1*H^3

 montmult_ext_h(Z3, Z2, H, pidx);

}


void ec_jacdouble_h(uint32_t *z, uint32_t *x, uint32_t pidx)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t x_cpy[NWORDS_256BIT*ECP_JAC_OUTDIMS];
  uint32_t *X1 = x_cpy;
  uint32_t *Y1 = &x_cpy[NWORDS_256BIT];
  uint32_t *Z1 = &x_cpy[2*NWORDS_256BIT];
  uint32_t *X2 = z;
  uint32_t *Y2 = &z[NWORDS_256BIT];
  uint32_t *Z2 = &z[2*NWORDS_256BIT];
  uint32_t Ysq[NWORDS_256BIT], Ysqsq[NWORDS_256BIT];
  uint32_t Zsq[NWORDS_256BIT], S[NWORDS_256BIT], M[NWORDS_256BIT];


  if (ec_iseq_h(x,
              &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT])){

          memcpy( z, x, sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_OUTDIMS);
          return;
  }
  memcpy(x_cpy, x, sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_OUTDIMS);
  montsquare_h(Ysq, Y1, pidx);
  montsquare_h(Ysqsq,  Ysq, pidx);
  montsquare_h(Zsq, Z1, pidx);
 
  montmult_h(S, X1, Ysq, pidx);
  addm_h(S, S, S, pidx);
  addm_h(S, S, S, pidx);
  
  montsquare_h(M, X1, pidx);

  addm_h(X2, M, M, pidx);
  addm_h(M, X2, M, pidx);

  montsquare_h(X2, M, pidx);

  montmult_h(Z2, Y1, Z1, pidx);
  subm_h(X2, X2, S, pidx);
  subm_h(X2, X2, S, pidx);

  addm_h(Y2, Ysqsq, Ysqsq, pidx);
 
  addm_h(Y2, Y2, Y2, pidx);
  addm_h(Y2, Y2, Y2, pidx);

  //addm_h(Y2, Y2, Y2, pidx);
  
  subm_h(S, S, X2, pidx);
  montmult_h(S, M, S, pidx);
  subm_h(Y2, S, Y2, pidx);
   
  addm_h(Z2, Z2, Z2, pidx);
}

void ec2_jacdouble_h(uint32_t *z, uint32_t *x, uint32_t pidx)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t x_cpy[NWORDS_256BIT*ECP2_JAC_OUTDIMS];
  uint32_t *X1 = x_cpy;
  uint32_t *Y1 = &x_cpy[2*NWORDS_256BIT];
  uint32_t *Z1 = &x_cpy[4*NWORDS_256BIT];
  uint32_t *X2 = z;
  uint32_t *Y2 = &z[2*NWORDS_256BIT];
  uint32_t *Z2 = &z[4*NWORDS_256BIT];
  uint32_t Ysq[2*NWORDS_256BIT], Ysqsq[2*NWORDS_256BIT];
  uint32_t Zsq[2*NWORDS_256BIT], S[2*NWORDS_256BIT], M[2*NWORDS_256BIT];

  if (ec2_iseq_h(x,
                &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT])){

          memcpy( z, x, sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_OUTDIMS);
          return;
  }
  memcpy(x_cpy, x, sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_OUTDIMS);
  montsquare_ext_h(Ysq, Y1, pidx);
  montsquare_ext_h(Ysqsq,  Ysq, pidx);
  montsquare_ext_h(Zsq, Z1, pidx);
 
  montmult_ext_h(S, X1, Ysq, pidx);
  addm_ext_h(S, S, S, pidx);
  addm_ext_h(S, S, S, pidx);
  
  montsquare_ext_h(M, X1, pidx);

  addm_ext_h(X2, M, M, pidx);
  addm_ext_h(M, X2, M, pidx);

  montsquare_ext_h(X2, M, pidx);
  montmult_ext_h(Z2, Y1, Z1, pidx);
  subm_ext_h(X2, X2, S, pidx);
  subm_ext_h(X2, X2, S, pidx);
  
  addm_ext_h(Y2, Ysqsq, Ysqsq, pidx);
 
  addm_ext_h(Y2, Y2, Y2, pidx);
  addm_ext_h(Y2, Y2, Y2, pidx);

  
  subm_ext_h(S, S, X2, pidx);
  montmult_ext_h(S, M, S, pidx);
  subm_ext_h(Y2, S, Y2, pidx);
   
  addm_ext_h(Z2, Z2, Z2, pidx);
}


uint32_t * ec_inittable(uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t table_order, uint32_t pidx, uint32_t add_last=0)
{
   uint32_t n_tables = (table_order + n - 1)/table_order;
   uint32_t i;
   uint32_t table_size = 1<< table_order;
   const uint32_t *ECInf = CusnarksMiscKGet();
   const uint32_t *One = CusnarksOneMontGet(pidx);
   uint32_t ndims = ECP_JAC_OUTDIMS;
   if (add_last){
      ndims = ECP_JAC_INDIMS;
   }

   uint32_t *ec_table = ectable;
   if (ectable == NULL){
     ec_table = (uint32_t *) malloc(n_tables * table_size * NWORDS_256BIT * ECP_JAC_OUTDIMS * sizeof(uint32_t));
   }

   #ifndef TEST_MODE
     #pragma omp parallel for if(parallelism_enabled)
   #endif
   for (i=0; i< n_tables; i++){
      // init element 0 of table
      memcpy(&ec_table[(i*table_size)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
            &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT],
            sizeof(uint32_t) * ECP_JAC_OUTDIMS * NWORDS_256BIT);
      uint32_t k=0, last_pow2, n_els = 0;
      for (uint32_t j=1; j< table_size; j++){
         // if power of 2    
         if  ((j & (j-1)) == 0){
             last_pow2 = j;
             if (n_els < n){
                memcpy(&ec_table[(i*table_size+j)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
                   &x[(i*table_order+k)*NWORDS_256BIT*ndims],
                   sizeof(uint32_t) * ndims * NWORDS_256BIT);

                if (add_last){
                   memcpy(&ec_table[(i*table_size+j)*NWORDS_256BIT*ECP_JAC_OUTDIMS+ECP_JAC_INDIMS*NWORDS_256BIT],
                      One,
                      sizeof(uint32_t) * NWORDS_256BIT);
                }
             } else {
                 memcpy(&ec_table[(i*table_size+j)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
                        &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT],
                        sizeof(uint32_t) * ECP_JAC_OUTDIMS * NWORDS_256BIT);
             }
             k++;
             n_els++;
         } else {
             ec_jacadd_h( &ec_table[(i*table_size+j)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
                          &ec_table[(i*table_size+last_pow2)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
                          &ec_table[(i*table_size+j-last_pow2)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
                          pidx);
                           
         }      
      } 
   } 

   return ec_table;
}

uint32_t * ec2_inittable(uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t table_order, uint32_t pidx, uint32_t add_last=0)
{
   uint32_t n_tables = (table_order + n - 1)/table_order;
   uint32_t i;
   uint32_t table_size = 1<< table_order;
   const uint32_t *ECInf = CusnarksMiscKGet();
   const uint32_t *One = CusnarksOneMontGet(pidx);
   uint32_t ndims = ECP2_JAC_OUTDIMS;
   if (add_last){
      ndims = ECP2_JAC_INDIMS;
   }
 
   uint32_t *ec_table = ectable;
   if (ectable == NULL){
     ec_table = (uint32_t *) malloc(n_tables * table_size * NWORDS_256BIT * ECP2_JAC_OUTDIMS * sizeof(uint32_t));
   }

   #ifndef TEST_MODE
     #pragma omp parallel for if(parallelism_enabled)
   #endif
   for (i=0; i< n_tables; i++){
      // init element 0 of table
      memcpy(&ec_table[(i*table_size)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
            &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT],
            sizeof(uint32_t) * ECP2_JAC_OUTDIMS * NWORDS_256BIT);
      uint32_t k=0, last_pow2, n_els = 0;
      for (uint32_t j=1; j< table_size; j++){
         // if power of 2    
         if  ((j & (j-1)) == 0){
             last_pow2 = j;
             if (n_els < n){
                memcpy(&ec_table[(i*table_size+j)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
                   &x[(i*table_order+k)*NWORDS_256BIT*ndims],
                   sizeof(uint32_t) * ndims * NWORDS_256BIT);

                if (add_last){
                   memcpy(&ec_table[(i*table_size+j)*NWORDS_256BIT*ECP2_JAC_OUTDIMS+4*NWORDS_256BIT],
                      One,
                      sizeof(uint32_t) * NWORDS_256BIT);
                   memset(&ec_table[(i*table_size+j)*NWORDS_256BIT*ECP2_JAC_OUTDIMS+5*NWORDS_256BIT],
	     	          0, sizeof(uint32_t) * NWORDS_256BIT);
                }
             } else {
                 memcpy(&ec_table[(i*table_size+j)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
                        &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT],
                        sizeof(uint32_t) * ECP2_JAC_OUTDIMS * NWORDS_256BIT);
             }
             k++;
             n_els++;
         } else {
             ec2_jacadd_h( &ec_table[(i*table_size+j)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
                          &ec_table[(i*table_size+last_pow2)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
                          &ec_table[(i*table_size+j-last_pow2)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
                          pidx);
                           
         }      
      } 
   } 

   return ec_table;
}


void ec_jacscmul_opt_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t order, uint32_t pidx, uint32_t add_last=0)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t i;
  int debug_tid = 5;
  uint32_t ndims = ECP_JAC_OUTDIMS;
  if (add_last){
      ndims = ECP_JAC_INDIMS;
  }
  uint32_t n_tables = (order + n - 1)/order;
  uint32_t table_size = 1 << order; 

  uint32_t * ec_table = ec_inittable(x, ectable, n, order, pidx, add_last);


  /*
  for(i=debug_tid * table_size; i< (debug_tid+1)*table_size; i++){
        printf("T[%d] :\n",i-debug_tid*table_size);
        printU256Number(&ec_table[i * NWORDS_256BIT * ECP_JAC_OUTDIMS]);
        printU256Number(&ec_table[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+NWORDS_256BIT]);
        printU256Number(&ec_table[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+2*NWORDS_256BIT]);
  }
  */

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i<n_tables ; i++){
     uint32_t tid = omp_get_thread_num();

     // Q=0
     memcpy(
            &z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
            &ECInf[(pidx * MISC_K_N + MISC_K_INF) * NWORDS_256BIT],
            sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_OUTDIMS
          );

     uint32_t msb = 255;
     uint32_t tmp_msb;

     for(uint32_t j=0; j< order; j++){
       if (order*i + j < n){
          tmp_msb = msbu256_h(&scl[i*order*NWORDS_256BIT+j*NWORDS_256BIT]); 
          if (tmp_msb < msb){
             msb = tmp_msb;
          }
       }

     }
     /*
     if (i == debug_tid){
         printf("msb : %d\n",msb);
     }
      */
     msb = 255 - msb;
     for (int j=msb; j>=0 ; j--){
        uint32_t b = getbitu256g_h(&scl[i*order * NWORDS_256BIT], j, order);


        ec_jacdouble_h(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                       &z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                       pidx);
        if (b) {
           ec_jacadd_h(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                       &ec_table[(i * table_size + b) * NWORDS_256BIT *ECP_JAC_OUTDIMS],
                       &z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                       pidx);
        }
        /*
        if (i==debug_tid){ 
          printf("offset : %d, b : %d\n",j, b);
          printf("Q :\n");
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS]);
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+NWORDS_256BIT]);
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+2*NWORDS_256BIT]);
        }
        */

     }
  }

  if (ectable == NULL){
    free(ec_table);
  }
}

void ec2_jacscmul_opt_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t order, uint32_t pidx, uint32_t add_last=0)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t i;
  int debug_tid = 8191;
  uint32_t ndims = ECP2_JAC_OUTDIMS;
  if (add_last){
      ndims = ECP2_JAC_INDIMS;
  }
  uint32_t n_tables = (order + n - 1)/order;
  uint32_t table_size = 1 << order; 

  uint32_t * ec_table = ec2_inittable(x, ectable, n, order, pidx, add_last);

 /*
  for(i=debug_tid * table_size; i< (debug_tid+1)*table_size; i++){
        printf("T[%d] :\n",i-debug_tid*table_size);
        printU256Number(&ec_table[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS]);
        printU256Number(&ec_table[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS+NWORDS_256BIT]);
        printU256Number(&ec_table[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS+2*NWORDS_256BIT]);
        printU256Number(&ec_table[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS+3*NWORDS_256BIT]);
        printU256Number(&ec_table[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS+4*NWORDS_256BIT]);
        printU256Number(&ec_table[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS+5*NWORDS_256BIT]);
  }
  */

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i<n_tables ; i++){
     uint32_t tid = omp_get_thread_num();

     // Q=0
     memcpy(
            &z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
            &ECInf[(pidx * MISC_K_N + MISC_K_INF2) * NWORDS_256BIT],
            sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_OUTDIMS
          );

     uint32_t msb = 255;
     uint32_t tmp_msb;

     for(uint32_t j=0; j< order; j++){
       if (order*i + j < n){
          tmp_msb = msbu256_h(&scl[i*order*NWORDS_256BIT+j*NWORDS_256BIT]); 
          if (tmp_msb < msb){
             msb = tmp_msb;
          }
       }

     }
     /*
     if (i == debug_tid){
         printf("msb : %d\n",msb);
     }
      */
     msb = 255 - msb;
     for (int j=msb; j>=0 ; j--){
        uint32_t b = getbitu256g_h(&scl[i*order * NWORDS_256BIT], j, order);


        ec2_jacdouble_h(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                       &z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                       pidx);
        if (b) {
           ec2_jacadd_h(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                       &ec_table[(i * table_size + b) * NWORDS_256BIT *ECP2_JAC_OUTDIMS],
                       &z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                       pidx);
        }
        /*
        if (i==debug_tid){ 
          printf("offset : %d, b : %d\n",j, b);
          printf("Q :\n");
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS]);
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+NWORDS_256BIT]);
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+2*NWORDS_256BIT]);
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+3*NWORDS_256BIT]);
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+4*NWORDS_256BIT]);
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+5*NWORDS_256BIT]);
        }
        */

     }
  }

  if (ectable == NULL){
    free(ec_table);
  }
}


void ec_jacscmul_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last=0)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  const uint32_t *zero = CusnarksZeroGet();
  const uint32_t *One = CusnarksOneMontGet(pidx);
  uint32_t i;
  uint32_t ndims = ECP_JAC_OUTDIMS;
  void (*add_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t) = &ec_jacadd_h;
  if (add_last){
      ndims = ECP_JAC_INDIMS;
      add_cb = &ec_jacaddmixed_h;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i<n ; i++){
     uint32_t tid = omp_get_thread_num();
     uint32_t *N = &utils_N[tid * NWORDS_256BIT * ECP_JAC_OUTDIMS];
     memcpy(
            &z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
            &ECInf[(pidx * MISC_K_N + MISC_K_INF) * NWORDS_256BIT],
            sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_OUTDIMS
          );
      // if x == inf || scl == 0 => y = inf
     if ( equ256_h( &scl[i * NWORDS_256BIT], zero) ||
          ec_iseq_h(&x[i * ndims * NWORDS_256BIT],
                   &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT]) ){
              continue;
     }

     if (!add_last){
       memcpy(
            &utils_N[tid * NWORDS_256BIT * ECP_JAC_OUTDIMS],
            &x[i * NWORDS_256BIT * ECP_JAC_INDIMS],
            sizeof(uint32_t) * NWORDS_256BIT * ndims
          );
     } else{
        N = &x[i * NWORDS_256BIT * ECP_JAC_INDIMS];
     }
     uint32_t msb = 255 - msbu256_h(&scl[i*NWORDS_256BIT]);

     for (int32_t j=msb; j >=0 ; j--){
        uint32_t b0 = getbitu256_h(&scl[i * NWORDS_256BIT], j);
        ec_jacdouble_h(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                       &z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                       pidx);
        if (b0) {
           add_cb(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                  N,
                  &z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                  pidx);
        }
     }
  }
}

void ec2_jacscmul_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last=0)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  const uint32_t *zero = CusnarksZeroGet();
  const uint32_t *One = CusnarksOneMontGet(pidx);
  uint32_t i;
  uint32_t ndims = ECP2_JAC_OUTDIMS;
  void (*add_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t) = &ec2_jacadd_h;
  if (add_last){
      ndims = ECP2_JAC_INDIMS;
      add_cb = &ec2_jacaddmixed_h;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i<n ; i++){
     uint32_t tid = omp_get_thread_num();
     uint32_t *N = &utils_N[tid * NWORDS_256BIT * ECP2_JAC_OUTDIMS];
     memcpy(
            &z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
            &ECInf[(pidx * MISC_K_N + MISC_K_INF2) * NWORDS_256BIT],
            sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_OUTDIMS
          );
      // if x == inf || scl == 0 => y = inf
     if (equ256_h( &scl[i * NWORDS_256BIT], zero) ||
         ec2_iseq_h(&x[i * ndims * NWORDS_256BIT],
                   &ECInf[(pidx * MISC_K_N+MISC_K_INF2)*NWORDS_256BIT])) {
         
              continue;
     }

     if (!add_last){
       memcpy(
            &utils_N[tid * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
            &x[i * NWORDS_256BIT * ndims],
            sizeof(uint32_t) * NWORDS_256BIT * ndims
          );
     } else {
        N = &x[i * NWORDS_256BIT * ECP2_JAC_INDIMS];
     }

     uint32_t msb = 255 - msbu256_h(&scl[i*NWORDS_256BIT]); 

     for (int32_t j=msb; j >= 0 ; j--){
        uint32_t b0 = getbitu256_h(&scl[i * NWORDS_256BIT], j);
        ec2_jacdouble_h(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                       &z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                       pidx);
        if (b0) {
           add_cb(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                     N,
                     &z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                     pidx);
        }
     }
  }
}


uint32_t msbu256_h(uint32_t *x)
{
  int i,j;
  uint32_t count=0; 

  for(i=NWORDS_256BIT-1; i >= 0; i--){
     for(j=31; j >= 0; j--){
        if (x[i] & (1 << j)) { 
           return count;
        } else {
          count++;
        }
     }
  }
}

void ec_jac2aff_h(uint32_t *y, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t strip_last=0)
{
  const uint32_t *One = CusnarksOneMontGet(pidx);
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t ndims = ECP_JAC_OUTDIMS;
  if (strip_last == 1){
     ndims = ECP_JAC_INDIMS;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0; i< n; i++){
     uint32_t tid = omp_get_thread_num();
     if (ec_iseq_h(
               &x[i*ECP_JAC_OUTDIMS*NWORDS_256BIT],
               &ECInf[(pidx*MISC_K_N+MISC_K_INF)*NWORDS_256BIT])){
           memmove(
             &y[i*ndims*NWORDS_256BIT],
             &x[i*ECP_JAC_OUTDIMS*NWORDS_256BIT],
            sizeof(uint32_t)*ndims*NWORDS_256BIT
                 );
            continue;
     }
     //zinv = x[Z].inv()
     montinv_h(&utils_zinv[tid * NWORDS_256BIT],
               &x[2*NWORDS_256BIT+i*ECP_JAC_OUTDIMS*NWORDS_256BIT],
               pidx);
     //zinv_sq = zinv * zinv
     montsquare_h(&utils_zinv_sq[tid * NWORDS_256BIT],
                  &utils_zinv[tid * NWORDS_256BIT],
                  pidx);
     // zinv = zinv_sq * zinv
     montmult_h(&utils_zinv[tid * NWORDS_256BIT],
                &utils_zinv_sq[tid * NWORDS_256BIT],
                &utils_zinv[tid * NWORDS_256BIT], 
                pidx);
     // y[X] = x[X] * zinv_sq
     montmult_h(&y[i*ndims*NWORDS_256BIT],
                &x[i*ECP_JAC_OUTDIMS*NWORDS_256BIT],
                &utils_zinv_sq[tid * NWORDS_256BIT],
                pidx);
     // y[Y] = x[Y] * zinv
     montmult_h(&y[NWORDS_256BIT + i*ndims*NWORDS_256BIT],
                &x[NWORDS_256BIT + i*ECP_JAC_OUTDIMS*NWORDS_256BIT],
                &utils_zinv[tid * NWORDS_256BIT], pidx);

     if (!strip_last){
     // y[Z] = 1
        memcpy(&y[2*NWORDS_256BIT+i*ECP_JAC_OUTDIMS*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);
     }
  }
}

void ec2_jac2aff_h(uint32_t *y, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t strip_last=0)
{
  const uint32_t *One = CusnarksOneMontGet(pidx);
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t ndims = ECP2_JAC_OUTDIMS;
  if (strip_last == 1){
     ndims = ECP2_JAC_INDIMS;
  }


  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0; i< n; i++){
     uint32_t tid = omp_get_thread_num();
     if (ec2_iseq_h(&x[i*ECP2_JAC_OUTDIMS*NWORDS_256BIT],
                 &ECInf[(pidx*MISC_K_N+MISC_K_INF2)*NWORDS_256BIT])){
        memmove(&y[i*ndims*NWORDS_256BIT],
                &x[i*ECP2_JAC_OUTDIMS*NWORDS_256BIT],
                sizeof(uint32_t)*ndims*NWORDS_256BIT);
        continue;
     }
     montinv_ext_h(&utils_zinv[tid * 2*NWORDS_256BIT],
                   &x[4*NWORDS_256BIT+i*ECP2_JAC_OUTDIMS*NWORDS_256BIT],
                   pidx);

     montmult_ext_h(&utils_zinv_sq[2 * tid * NWORDS_256BIT],
                    &utils_zinv[2 * tid * NWORDS_256BIT],
                    &utils_zinv[2 * tid * NWORDS_256BIT],
                    pidx);
     montmult_ext_h(&utils_zinv[2*tid*NWORDS_256BIT],
                    &utils_zinv_sq[2*tid*NWORDS_256BIT],
                    &utils_zinv[2*tid*NWORDS_256BIT], 
                    pidx);

     montmult_ext_h(&y[i*ndims*NWORDS_256BIT],
                    &x[i*ECP2_JAC_OUTDIMS*NWORDS_256BIT],
                    &utils_zinv_sq[2*tid*NWORDS_256BIT], pidx);
     montmult_ext_h(&y[2*NWORDS_256BIT + i*ndims*NWORDS_256BIT],
                    &x[2*NWORDS_256BIT + i*ECP2_JAC_OUTDIMS*NWORDS_256BIT], 
                    &utils_zinv[2*tid*NWORDS_256BIT],
                    pidx);
     if (!strip_last){
        // y[Z] = 1
        memcpy(&y[4*NWORDS_256BIT+i*ECP2_JAC_OUTDIMS*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);
        memset(&y[5*NWORDS_256BIT+i*ECP2_JAC_OUTDIMS*NWORDS_256BIT], 0, sizeof(uint32_t)*NWORDS_256BIT);
     }
  }
}

uint32_t ec_isoncurve_h(uint32_t *x, uint32_t is_affine, uint32_t pidx)
{
  // TODO : Check ec_jac2aff_h and copy parallel omp
  const uint32_t *ECInf = CusnarksMiscKGet();
  const uint32_t *ecbn_params = CusnarksEcbn128ParamsGet();
  uint32_t tmp_p [ECP_JAC_INDIMS * NWORDS_256BIT];
  uint32_t y1[NWORDS_256BIT], y2[NWORDS_256BIT];

  if (ec_iseq_h(x,
               &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT])){
     return 2;

  } else if (is_affine){
      memcpy(tmp_p,x,2*NWORDS_256BIT*sizeof(uint32_t));

  } else {
      ec_jac2aff_h(tmp_p, x, 1, pidx, 1);
  }
  
  montsquare_h(y1, &tmp_p[NWORDS_256BIT], pidx);
  
  montsquare_h(y2, tmp_p, pidx);
  montmult_h(y2, y2, tmp_p, pidx);

  addm_h(y2,y2, &ecbn_params[pidx*ECBN128_PARAM_N + ECBN128_PARAM_B] , pidx);

  if (equ256_h(y1,y2) ){
    return 1;
  } else {
    return 0;
  }
}

uint32_t ec2_isoncurve_h(uint32_t *x, uint32_t is_affine, uint32_t pidx)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  const uint32_t *ecbn_params = CusnarksEcbn128ParamsGet();
  uint32_t tmp_p [ECP2_JAC_INDIMS * NWORDS_256BIT];
  uint32_t y1[2*NWORDS_256BIT], y2[2*NWORDS_256BIT];

  if (ec2_iseq_h(x,
               &ECInf[(pidx*MISC_K_N+MISC_K_INF2)*NWORDS_256BIT])){
     return 2;

  } else if (is_affine){
      memcpy(tmp_p,x,4*NWORDS_256BIT*sizeof(uint32_t));

  } else {
      ec2_jac2aff_h(tmp_p, x, 1, pidx, 1);
  }
  
  montsquare_ext_h(y1, &tmp_p[2*NWORDS_256BIT], pidx);

  montsquare_ext_h(y2, tmp_p, pidx);
  montmult_ext_h(y2, y2,tmp_p, pidx);

  addm_h(y2,y2,
        &ecbn_params[pidx*ECBN128_PARAM_N + ECBN128_PARAM_B2X] , pidx);
  addm_h(&y2[NWORDS_256BIT],&y2[NWORDS_256BIT],
        &ecbn_params[pidx*ECBN128_PARAM_N + ECBN128_PARAM_B2Y] , pidx);

  if ( equ256_h(y1, y2) && equ256_h(&y1[NWORDS_256BIT], &y2[NWORDS_256BIT])){
    return 1;
  } else {
    return 0;
  }
}

uint32_t ec_isinf(const uint32_t *x, const uint32_t pidx)
{
  const uint32_t *ECInf = CusnarksMiscKGet();

  if (ec_iseq_h( x,
                &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT]) )
    return 1;
  else 
    return 0;
}

uint32_t ec2_isinf(const uint32_t *x, const uint32_t pidx)
{
  const uint32_t *ECInf = CusnarksMiscKGet();

  if (ec_iseq_h( x,
                &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT]) )
    return 1;
  else 
    return 0;
}
void ec_isinf(uint32_t *z, const uint32_t *x, const uint32_t n, const uint32_t pidx)
{
  uint32_t j;
  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (j=0; j < n; j++){
    z[j] = ec_isinf(&x[j*ECP_JAC_INDIMS],pidx);
  } 
}

void ec2_isinf(uint32_t *z, const uint32_t *x, const uint32_t n, const uint32_t pidx)
{
  uint32_t j;
  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (j=0; j < n; j++){
    z[j] = ec2_isinf(&x[ECP2_JAC_INDIMS],pidx);
  } 
}
void ec_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last)
{
  uint32_t i;
  uint32_t zout[ECP_JAC_OUTDIMS*NWORDS_256BIT];
  uint32_t x1[ECP_JAC_OUTDIMS*NWORDS_256BIT], x2[ECP_JAC_OUTDIMS*NWORDS_256BIT];
  uint32_t *x1_ptr = x1, *x2_ptr = x2;
  uint32_t outdims = ECP_JAC_OUTDIMS;
  const uint32_t *One;
  
  One = CusnarksOneMontGet((mod_t)pidx);

  if (strip_last){
    outdims = ECP_JAC_INDIMS;
  }
  
  if (add_in){
    if (n > 1) {
      memcpy(x1,x,NWORDS_256BIT*ECP_JAC_INDIMS*sizeof(uint32_t));
      memcpy(&x1[2*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);

      memcpy(x2,&x[ECP_JAC_INDIMS*NWORDS_256BIT],NWORDS_256BIT*ECP_JAC_INDIMS*sizeof(uint32_t));
      memcpy(&x2[2*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);
    } else {
      memcpy(zout,x,NWORDS_256BIT*ECP_JAC_INDIMS*sizeof(uint32_t));
      memcpy(&zout[2*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);
    }

  } else {
    if (n > 1){
      x1_ptr = x;
      x2_ptr = &x[ECP_JAC_OUTDIMS*NWORDS_256BIT];
    } else {
      memcpy(zout,x,NWORDS_256BIT*ECP_JAC_OUTDIMS*sizeof(uint32_t));
    }
    
  }

  if (n > 1) {
    ec_jacadd_h(zout,x1_ptr,x2_ptr,pidx);

    for (i=2; i<n; i++){
      if (add_in){
        memcpy(x1,&x[i*ECP_JAC_INDIMS*NWORDS_256BIT],NWORDS_256BIT*ECP_JAC_INDIMS*sizeof(uint32_t));
        memcpy(&x1[2*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);
      } else {
        x1_ptr = &x[i*ECP_JAC_OUTDIMS*NWORDS_256BIT];
      }
      ec_jacadd_h(zout,zout,x1_ptr,pidx);
    }
   }

   if (to_aff){
     ec_jac2aff_h(z,zout,1,pidx, strip_last);
   } else {
     memcpy(z,zout,outdims*NWORDS_256BIT*sizeof(uint32_t));
   }

}

void ec2_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last)
{
  uint32_t i;
  uint32_t zout[ECP2_JAC_OUTDIMS*NWORDS_256BIT];
  uint32_t x1[ECP2_JAC_OUTDIMS*NWORDS_256BIT], x2[ECP2_JAC_OUTDIMS*NWORDS_256BIT];
  uint32_t *x1_ptr = x1, *x2_ptr = x2;
  uint32_t outdims = ECP2_JAC_OUTDIMS;
  const uint32_t *One;
  
  One = CusnarksOneMontGet((mod_t)pidx);

  if (strip_last){
    outdims = ECP2_JAC_INDIMS;
  }
  
  if (add_in){
    if (n > 1){
      memcpy(x1,x,NWORDS_256BIT*ECP2_JAC_INDIMS*sizeof(uint32_t));
      memcpy(&x1[4*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);
      memset(&x1[5*NWORDS_256BIT], 0, sizeof(uint32_t)*NWORDS_256BIT);

      memcpy(x2,&x[ECP2_JAC_INDIMS*NWORDS_256BIT],NWORDS_256BIT*ECP2_JAC_INDIMS*sizeof(uint32_t));
      memcpy(&x2[4*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);
      memset(&x2[5*NWORDS_256BIT], 0, sizeof(uint32_t)*NWORDS_256BIT);
    } else {
      memcpy(zout,x,NWORDS_256BIT*ECP2_JAC_INDIMS*sizeof(uint32_t));
      memcpy(&zout[4*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);
      memset(&zout[5*NWORDS_256BIT], 0, sizeof(uint32_t)*NWORDS_256BIT);
    }
  } else {
    if (n > 1){
      x1_ptr = x;
      x2_ptr = &x[ECP2_JAC_OUTDIMS*NWORDS_256BIT];
    } else {
      memcpy(zout,x,NWORDS_256BIT*ECP2_JAC_OUTDIMS*sizeof(uint32_t));
    }
  }

  if (n > 1){
    ec2_jacadd_h(zout,x1_ptr,x2_ptr,pidx);

    for (i=2; i<n; i++){
      if (add_in){
        memcpy(x1,&x[i*ECP2_JAC_INDIMS*NWORDS_256BIT],NWORDS_256BIT*ECP2_JAC_INDIMS*sizeof(uint32_t));
        memcpy(&x1[4*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT);
        memset(&x1[5*NWORDS_256BIT], 0, sizeof(uint32_t)*NWORDS_256BIT);
  
      } else {
        x1_ptr = &x[i*ECP2_JAC_OUTDIMS*NWORDS_256BIT];
      }
      ec2_jacadd_h(zout,zout,x1_ptr,pidx);
    }
  }

   if (to_aff){
     ec2_jac2aff_h(z,zout,1,pidx, strip_last);
   } else {
     memcpy(z,zout,outdims*NWORDS_256BIT*sizeof(uint32_t));
   }

}

uint32_t ec_jacreduce_init_h(uint32_t **ectable, uint32_t **scmul, uint32_t n, uint32_t order)
{
  uint32_t ntables =  (order + n - 1) / order;

  //initialize tables and scmul values
  *ectable = (uint32_t *)malloc((1<<order) * ntables * ECP_JAC_OUTDIMS * NWORDS_256BIT * sizeof(uint32_t));
  *scmul = (uint32_t *)malloc(ntables * ECP_JAC_OUTDIMS * NWORDS_256BIT * sizeof(uint32_t));

  return ntables;
}

uint32_t ec2_jacreduce_init_h(uint32_t **ectable, uint32_t **scmul, uint32_t n, uint32_t order)
{
  uint32_t ntables =  (order + n - 1) / order;

  //initialize tables and scmul values
  *ectable = (uint32_t *)malloc((1<<order) * ntables * ECP2_JAC_OUTDIMS * NWORDS_256BIT * sizeof(uint32_t));
  *scmul = (uint32_t *)malloc(ntables * ECP2_JAC_OUTDIMS * NWORDS_256BIT * sizeof(uint32_t));

  return ntables;
}

void ec_jacreduce_del_h(uint32_t *ectable, uint32_t *scmul)
{
  free(ectable);
  free(scmul);
}

void ec_jacreduce_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last)
{
  uint32_t *ectable, *scmul;
  uint32_t ntables;

  ntables = ec_jacreduce_init_h(&ectable, &scmul, n, U256_BSELM);
  ec_jacscmul_opt_h(scmul, scl, x, ectable, n, U256_BSELM, pidx, add_in);
  ec_jacaddreduce_h(z, scmul, ntables, pidx, to_aff, 0, strip_last);

  ec_jacreduce_del_h(ectable, scmul);
}

void ec2_jacreduce_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last)
{
  uint32_t *ectable, *scmul;
  uint32_t ntables;

  ntables = ec2_jacreduce_init_h(&ectable, &scmul, n, U256_BSELM);

  ec2_jacscmul_opt_h(scmul, scl, x, ectable, n, U256_BSELM, pidx, add_in);
  ec2_jacaddreduce_h(z, scmul, ntables, pidx, to_aff, 0, strip_last);

  ec_jacreduce_del_h(ectable, scmul);
}
void field_roots_compute_h(uint32_t *roots, uint32_t nbits)
{
  uint32_t i, pidx = MOD_FIELD;
  const  uint32_t *One = CusnarksOneMontGet((mod_t)pidx);
  const uint32_t *proots = CusnarksPrimitiveRootsFieldGet(nbits);
  
  memcpy(roots, One, NWORDS_256BIT*sizeof(uint32_t));
  if (nbits > 1){
    memcpy(&roots[NWORDS_256BIT], proots, NWORDS_256BIT*sizeof(uint32_t));
  }

  for(i=2; i < (1 << nbits); i++){
     montmult_h(&roots[i*NWORDS_256BIT],&roots[(i-1)*NWORDS_256BIT], proots, pidx);
  }
}

void mpoly_from_montgomery_h(uint32_t *x, uint32_t pidx)
{
  uint32_t i;
  uint32_t offset = 1 + x[0];

  for (i=0; i < x[0];i++){
    offset += x[i+1];
    from_montgomeryN_h(&x[offset], &x[offset], x[i+1], pidx);
    offset += (x[i+1]*NWORDS_256BIT);
  }
}

void mpoly_to_montgomery_h(uint32_t *x, uint32_t pidx)
{
  uint32_t i;
  uint32_t offset = 1 + x[0];

  for (i=0; i < x[0];i++){
    offset += x[i+1];
    to_montgomeryN_h(&x[offset], &x[offset], x[i+1], pidx);
    offset += (x[i+1]*NWORDS_256BIT);
  }
}

void computeIRoots_h(uint32_t *iroots, uint32_t *roots, uint32_t nroots)
{
  uint32_t i;

  if (roots == iroots){
    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for(i=1; i<nroots/2; i++){
      swapu256_h(&iroots[i*NWORDS_256BIT], &roots[(nroots-i)*NWORDS_256BIT]);
    }
  } else {
    memcpy(iroots, roots,NWORDS_256BIT*sizeof(uint32_t));
    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for (i=1; i<nroots; i++){
      memcpy(&iroots[i*NWORDS_256BIT], &roots[(nroots-i)*NWORDS_256BIT],NWORDS_256BIT*sizeof(uint32_t));
    }
  }
}

void init_h(void)
{
  #ifdef PARALLEL_EN
  if (!utils_mproc_init) {
    mproc_init_h();
  }
  #endif
}

void release_h(void)
{
  #ifdef PARALLEL_EN
  if (utils_mproc_init) {
     pthread_mutex_destroy(&utils_lock); 
     utils_nprocs = 1;
     utils_mproc_init=0;
   }
  #endif
}
