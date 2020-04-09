
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
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "types.h"
#include "constants.h"
#include "rng.h"
#include "log.h"
#include "utils_host.h"


#define MAX_DIGIT 0xFFFFFFFFUL
#define MAX(X,Y)  ((X)>=(Y) ? (X) : (Y))
#define MIN(X,Y)  ((X)<(Y) ? (X) : (Y))

#define MAX_NCORES_OMP        (32)
#define _CASM
//MPROC VARS
static  pthread_mutex_t utils_lock;
static  pthread_barrier_t utils_barrier;
static  pthread_cond_t utils_cond;
static  uint32_t utils_nprocs = 1;
static  uint32_t utils_mproc_init = 0;
static  uint32_t utils_ectable_ready = 1;
static  uint32_t utils_done = 0;

static uint32_t utils_N[MAX_NCORES_OMP * NWORDS_256BIT * ECP2_JAC_OUTDIMS];
static uint32_t utils_zinv[2 * MAX_NCORES_OMP * NWORDS_256BIT];
static uint32_t utils_zinv_sq[2 * MAX_NCORES_OMP * NWORDS_256BIT];
static uint32_t utils_EPout[MAX_NCORES_OMP * ECP2_JAC_OUTDIMS * NWORDS_256BIT];
static uint32_t utils_EPin[EC_JACREDUCE_TABLE_LEN * MAX_NCORES_OMP * ECP2_JAC_OUTDIMS * NWORDS_256BIT];
static uint32_t utils_ectable[(U256_BSELM << EC_JACREDUCE_BATCH_SIZE) * MAX_NCORES_OMP * ECP2_JAC_OUTDIMS * NWORDS_256BIT <<U256_BSELM];


static uint32_t *M_transpose = NULL;
static uint32_t *M_mul = NULL;
static uint32_t *Scl_idx = NULL;

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
 //p[0] = _mulx_u64(x[0],y[0],&p[1]);
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

inline void init_invtable(inv_t *data_table, uint32_t *u, uint32_t *v, uint32_t *s, uint32_t *r1, uint32_t *zero)
{
  // x0 = x0 - x1;
  // x0 = x0 >> 1
  // x4 = x2 + x3;
  // x3 = x3 << 1
  data_table[0].x0 = u;
  data_table[0].x1 = zero;
  data_table[0].x2 = zero;
  data_table[0].x3 = s;
  data_table[0].x4 = s;

  data_table[1].x0 = v;
  data_table[1].x1 = zero;
  data_table[1].x2 = zero;
  data_table[1].x3 = r1;
  data_table[1].x4 = r1;

  data_table[2].x0 = u;
  data_table[2].x1 = v;
  data_table[2].x2 = r1;
  data_table[2].x3 = s;
  data_table[2].x4 = r1;

  data_table[3].x0 = v;
  data_table[3].x1 = u;
  data_table[3].x2 = s;
  data_table[3].x3 = r1;
  data_table[3].x4 = s;
}

inline void almmontinv_step_h(inv_t *table)
{
  // x0 = x0 - x1;
  // x0 = x0 >> 1
  // x4 = x2 + x3;
  // x3 = x3 << 1
  subu256_h(table->x0, table->x0, table->x1);
  shlru256_h(table->x0, table->x0,1);
  addu256_h(table->x4, table->x2,table->x3);
  shllu256_h(table->x3, table->x3,1);
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
static void ntt_interpolandmul_init_h(uint32_t *A, uint32_t *B, uint32_t *mNrows, uint32_t *mNcols, uint32_t nRows, uint32_t nCols);
static uint32_t launch_client_h( void * (*f_ptr) (void* ), pthread_t *workers, void *w_args, uint32_t size, uint32_t max_threads, uint32_t detach);
static void ec_jacdouble_finish_h(void *args);
static void ec_inittable_ready_h(void *args);
static void ec_print_EPin(void *args);
void *interpol_and_mul_h(void *args);
void *ec_jacreduce_batch_h(void *args);
void *ec_jacreduce_batch_precomputed_h(void *args);
void *ec_read_table_h(void *args);
void ec_jacaddreduce_finish_h(void *args);
void ec2_jacaddreduce_finish_h(void *args);

#ifdef _CASM
//extern "C" void rawAddLL_R(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fr_rawAdd(uint32_t *r, const uint32_t *, const uint32_t *b);
//extern "C" void rawSubLL_R(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fr_rawSub(uint32_t *r, const uint32_t *, const uint32_t *b);
//extern "C" void rawMontgomeryMul_R(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fr_rawMMul(uint32_t *r, const uint32_t *, const uint32_t *b);
//extern "C" void rawMontgomerySquare_R(uint32_t *r, const uint32_t *x);
extern "C" void Fr_rawMSquare(uint32_t *r, const uint32_t *x);
//extern "C" void rawAddLL_Q(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fq_rawAdd(uint32_t *r, const uint32_t *, const uint32_t *b);
//extern "C" void rawSubLL_Q(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fq_rawSub(uint32_t *r, const uint32_t *, const uint32_t *b);
//extern "C" void rawMontgomeryMul_Q(uint32_t *r, const uint32_t *, const uint32_t *b);
extern "C" void Fq_rawMMul(uint32_t *r, const uint32_t *, const uint32_t *b);
//extern "C" void rawMontgomerySquare_Q(uint32_t *r, const uint32_t *x);
extern "C" void Fq_rawMSquare(uint32_t *r, const uint32_t *x);
#endif
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

inline void util_wait_h(uint32_t thread_id, void (*f_ptr) (void *), void * args)
{
   pthread_barrier_wait(&utils_barrier);
  if (f_ptr){
     if (thread_id == 0){
       f_ptr(args);
     }
     pthread_barrier_wait(&utils_barrier);
  }
}

static uint32_t launch_client_h( void * (*f_ptr) (void* ), pthread_t *workers, void *w_args, uint32_t size, uint32_t max_threads, uint32_t detach=0)
{
  uint32_t i;

  for (i=0; i < max_threads; i++)
  {
     //printf("Thread %d : start_idx : %d, last_idx : %d . ptr : %x\n", i, w_args[i].start_idx,w_args[i].last_idx, f_ptr);
     if ( pthread_create(&workers[i], NULL, f_ptr, (void *) w_args+i*size) ){
       //printf("error\n");
       return 0;
     }
    if (detach){
      pthread_detach(workers[i]);
    }
  }

  if (detach == 0){
    //printf("Max threads : %d\n",w_args[0].max_threads);
    for (i=0; i < max_threads; i++){
      pthread_join(workers[i], NULL);
    }
  }
  return 1;
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
void transpose_h(uint32_t *mout, const uint32_t *min,  uint32_t start_row, uint32_t last_row, uint32_t in_nrows, uint32_t in_ncols)
{
  uint32_t i,j,k;

  for (i=start_row; i<last_row; i++){
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
   uint32_t nelems = in_nrows*in_ncols-2;

   if (in_nrows == in_ncols){
     transpose_square_h(min, in_nrows);
     return;
   }
   const uint32_t *tt = CusnarksTidxGet();

   uint32_t idx = tt[m];
   const uint32_t N_1 = (1 << n) - 1;
   const uint32_t NM_1 = nelems+1;

   uint32_t val[NWORDS_256BIT];
   uint32_t cur_pos = tt[idx++], trans_pos, step=tt[idx++];
   uint32_t max_count = tt[idx++], max_el = tt[idx++];

   while (nelems > 0){
     memcpy(val, &min[cur_pos * NWORDS_256BIT], sizeof(uint32_t)*NWORDS_256BIT);
     for (uint32_t ccount=0; ccount < max_count; ccount++){
       for (uint32_t elcount=0; elcount < max_el; elcount++){
          trans_pos = (cur_pos >> n) + ((cur_pos & N_1) << m);
          swapu256_h(&min[trans_pos * NWORDS_256BIT], val);
          nelems--;
          //cur_pos = (trans_pos >> n ) + ((trans_pos & N_1) << m);
          //swapu256_h(&min[cur_pos * NWORDS_256BIT], val);
          cur_pos = trans_pos;
       }
       cur_pos = (trans_pos + step) & NM_1;
       memcpy(val, &min[cur_pos * NWORDS_256BIT], sizeof(uint32_t)*NWORDS_256BIT);
     }
     cur_pos = tt[idx++];
     step = tt[idx++];
     max_count = tt[idx++];
     max_el = tt[idx++];
   }
}

void transpose_square_h(uint32_t *min, uint32_t in_nrows)
{
  uint32_t m = sizeof(uint32_t) * NBITS_BYTE - __builtin_clz(in_nrows) - 1;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for(uint32_t i=0; i<in_nrows; i++){
    for(uint32_t j=i+1; j <in_nrows; j++){
        uint32_t cur_pos = (i << m) + j;
        uint32_t trans_pos = (j << m) + i;
        swapu256_h(&min[trans_pos << NWORDS_256BIT_SHIFT], &min[cur_pos << NWORDS_256BIT_SHIFT]);
    } 
  }
}

void transposeBlock_h(uint32_t *mout, uint32_t *min, uint32_t in_nrows, uint32_t in_ncols, uint32_t block_size)
{
    for (uint32_t i = 0; i < in_nrows; i += block_size) {
        for(uint32_t j = 0; j < in_ncols; ++j) {
            for(uint32_t b = 0; b < block_size && i + b < in_nrows; ++b) {
               for (uint32_t k=0; k< NWORDS_256BIT; k++){
                  mout[(j*in_nrows + i + b)*NWORDS_256BIT+k] = min[((i + b)*in_ncols + j)*NWORDS_256BIT + k];
               }
            }
        }
    }
}
void transposeBlock_h(uint32_t *mout, uint32_t *min, uint32_t start_row, uint32_t last_row, uint32_t in_nrows, uint32_t in_ncols, uint32_t block_size)
{
    for (uint32_t i = start_row; i < last_row; i += block_size) {
        for(uint32_t j = 0; j < in_ncols; ++j) {
            for(uint32_t b = 0; b < block_size && i + b < last_row; ++b) {
               for (uint32_t k=0; k< NWORDS_256BIT; k++){
                  mout[(j*in_nrows + i + b)*NWORDS_256BIT+k] = min[((i + b)*in_ncols + j)*NWORDS_256BIT + k];
               }
            }
        }
    }
}
/*
   Initalize multiprocessing components
    - mutex : utils_lock
    - number of processors
*/
uint32_t get_nprocs_h()
{
  uint32_t max_cores = get_nprocs_conf();

  if (max_cores > MAX_NCORES_OMP){
	  return MAX_NCORES_OMP;
  } else {
	  return max_cores;
  }
}

void mproc_init_h()
{
  if (utils_mproc_init) {
    return;
  }

  utils_nprocs = get_nprocs_h(); 
  omp_set_num_threads(utils_nprocs);
  utils_mproc_init = 1;

  if (pthread_mutex_init(&utils_lock, NULL) != 0){
     exit(1);
  }

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
  t_uint64 n_zpoly = (t_uint64) args->pin[0];
  t_uint64 zcoeff_d_offset = 1 + n_zpoly;
  t_uint64 zcoeff_v_offset;
  t_uint64 n_zcoeff;
  uint32_t scl[NWORDS_256BIT];
  t_uint64 i,j;
  uint32_t zcoeff_v_in[NWORDS_256BIT], *zcoeff_v_out;
  t_uint64 zcoeff_d;
  t_uint64 accum_n_zcoeff=0;

  /*
  printf("N zpoly: %d\n",n_zpoly);
  printf("Zcoeff D Offset : %d\n",zcoeff_d_offset);
  */
   //printf("Thread id: %d, Start idx : %d, Last idx : %d\n", args->thread_id, args->start_idx, args->last_idx);
  //TODO Change : If coeffs are accumulated, I don't need to do the accumulation
  //accum_n_zcoeff = args->pin[args->start_idx];
  
  for (i=0; i<args->start_idx; i++){
    accum_n_zcoeff += (t_uint64) args->pin[i+1];
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
    n_zcoeff = (t_uint64) args->pin[1+i];
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
       zcoeff_d = (t_uint64) args->pin[zcoeff_d_offset+j];
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
       /*
       if(args->reduce_coeff){
         to_montgomery_h(zcoeff_v_in, zcoeff_v_in, args->pidx);
       }
       */
       /*
       if ( ((i<5) || (i > args->last_idx-5)) && ((j<5) || (j>n_zcoeff-5))){
         printf("V[%d] in after mult \n", zcoeff_d);
         printU256Number(zcoeff_v_in);
         printf("V[%d] out before add \n", zcoeff_d);
         printU256Number(zcoeff_v_out);
       }
       */
       //if (args->max_threads > 1){
         //pthread_mutex_lock(&utils_lock);
         //printf("Mutex locked(%d)\n",args->thread_id);
         //fflush(stdin);
       //}

       addm_h(zcoeff_v_out, zcoeff_v_out, zcoeff_v_in, args->pidx);

       //if (args->max_threads > 1){
         //printf("Mutex unlocked(%d)\n", args->thread_id);
         //fflush(stdin);
         //pthread_mutex_unlock(&utils_lock);
       //}
       /*
       if ( ((i<5) || (i > args->last_idx-5)) && ((j<5) || (j>n_zcoeff-5))){
         printf("V[%d] out after add \n", zcoeff_d);
         printU256Number(zcoeff_v_out);
       }
       */
    }
    zcoeff_d_offset = accum_n_zcoeff*(NWORDS_256BIT+1) +1 + n_zpoly;
  }

  return NULL;
}

void r1cs_to_mpoly_len_h(uint32_t *coeff_len, uint32_t *cin, cirbin_hfile_t *header, uint32_t extend)
{
  uint32_t i,j, poly_idx;
  t_uint64 n_coeff, prev_n_coeff,const_offset;

  const_offset = (t_uint64) cin[0]+1;
  prev_n_coeff = 0;
  //printf("N constraints : %d\n",header->nConstraints);
  for (i=0; i < header->nConstraints; i++){
     n_coeff = (t_uint64)cin[1+i];
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
  t_uint64 poly_idx, const_offset, n_coeff,prev_n_coeff, coeff_offset, coeff_idx, c_offset, v_offset;
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

  const_offset = (t_uint64) cin[0]+1;
  prev_n_coeff = 0;

  for (i=0; i < header->nConstraints; i++){
     n_coeff = (t_uint64) cin[1+i];
     coeff_offset = const_offset + n_coeff - prev_n_coeff;
     for (j=0; j < n_coeff - prev_n_coeff ;j++){
       poly_idx = (t_uint64) cin[const_offset+j];
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

#ifdef ROLLUP
void readR1CSFileHeader_h(r1csv1_t *r1cs_hdr, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");
  uint32_t k=0,i;
  uint32_t tmp_word, n_coeff;
  t_int64 offset;

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

  fseek(ifp, R1CS_HDR_FIELDDEFSIZE_OFFSET_NBYTES * sizeof(char), SEEK_SET);
  fread(&offset, sizeof(uint32_t), 1, ifp); 
  offset &= 0xFFFF;
  //printf("offset : %d\n", offset);
  fseek(ifp, (R1CS_HDR_FIELDDEFSIZE_OFFSET_NBYTES + 8 + offset) * sizeof(char), SEEK_SET);

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
  fread(&r1cs_hdr->nLabels, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nConstraints, sizeof(uint32_t), 1, ifp); 

  fread(&offset, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->constraintLen, sizeof(uint32_t), 2, ifp); 

  /*
  printf("word_width_bytes : %d\n", r1cs_hdr->word_width_bytes);
  printf("nVars : %d\n", r1cs_hdr->nVars);
  printf("nPubOutputs : %d\n", r1cs_hdr->nPubOutputs);
  printf("nPubInputs : %d\n", r1cs_hdr->nPubInputs);
  printf("nPrivInputs : %d\n", r1cs_hdr->nPrivInputs);
  printf("nLabels : %d\n", r1cs_hdr->nLabels);
  printf("nConstraints : %d\n", r1cs_hdr->nConstraints);
  printf("Const section len : %lld\n",r1cs_hdr->constraintLen);
  */

  r1cs_hdr->constraintOffset = ftell(ifp);

  r1cs_hdr->R1CSA_nCoeff = 0;
  r1cs_hdr->R1CSB_nCoeff = 0;
  r1cs_hdr->R1CSC_nCoeff = 0;

  offset = r1cs_hdr->constraintLen;

  while (offset > 0){
    fread(&n_coeff, sizeof(uint32_t), 1, ifp); 
    offset-=sizeof(uint32_t);
    if (k%3 == R1CSA_IDX){
      r1cs_hdr->R1CSA_nCoeff+= (n_coeff);
    } else if (k%3 == R1CSB_IDX){
      r1cs_hdr->R1CSB_nCoeff+= (n_coeff);
    } else {
      r1cs_hdr->R1CSC_nCoeff+= (n_coeff);
    }
    for (i=0; i< n_coeff; i++){
      fseek(ifp, 4, SEEK_CUR);
      fread(&tmp_word, sizeof(char), 1, ifp); 
      tmp_word &= 0xFF;
      fseek(ifp, tmp_word, SEEK_CUR);
      offset-=(tmp_word + sizeof(char) + sizeof(uint32_t));
    }
    k++;
  }

  /*
  printf("N coeff R1CSA : %d\n", r1cs_hdr->R1CSA_nCoeff);
  printf("N coeff R1CSB : %d\n", r1cs_hdr->R1CSB_nCoeff);
  printf("N coeff R1CSC : %d\n", r1cs_hdr->R1CSC_nCoeff);
  
  printf("end of constraints : %lld\n",ftell(ifp));
  fread(&offset, sizeof(uint32_t), 1, ifp); 
  printf("Lable section len : %lld\n",offset);
  */

  fclose(ifp);

  return;
}
  

void readR1CSFile_h(uint32_t *samples, const char *filename, r1csv1_t *r1cs, r1cs_idx_t r1cs_idx )
{
  FILE *ifp = fopen(filename,"rb");
  uint32_t tmp_word, n_coeff;
  uint32_t r1cs_offset=0, r1cs_coeff_offset=1+r1cs->nConstraints, r1cs_val_offset = 1+r1cs->nConstraints;
  uint32_t k=0, accum_coeffs=0, i,j;
  t_int64 offset = r1cs->constraintLen;

  samples[r1cs_offset++] = r1cs->nConstraints;
  
  //printf("constraint len : %lld\n",offset);
  fseek(ifp, r1cs->constraintOffset, SEEK_SET);

  while (!offset){
    fread(&n_coeff, sizeof(uint32_t), 1, ifp); 
    offset-=sizeof(uint32_t);
    if (k%3 == r1cs_idx) {
      accum_coeffs+= ((uint32_t) n_coeff);
      samples[r1cs_offset++] = accum_coeffs;
      r1cs_val_offset += n_coeff;
      for (i=0; i< n_coeff; i++){
        fread(&samples[r1cs_coeff_offset++], sizeof(uint32_t), 1, ifp); 
        fread(&tmp_word, 1,1, ifp);
	tmp_word &= tmp_word & 0xFF;
        for(j=0; j <tmp_word; j++){
           fread(&samples[r1cs_val_offset+j], 1, 1, ifp); 
        }
        offset-=(tmp_word + sizeof(char) + sizeof(uint32_t));
        r1cs_val_offset += NWORDS_256BIT;
      }
      r1cs_coeff_offset = r1cs_val_offset;

    }  else {
      for (i=0; i< n_coeff; i++){
        fseek(ifp, 4, SEEK_CUR);
        fread(&tmp_word, 1, 1, ifp); 
	tmp_word &=0xFF;
        fseek(ifp, tmp_word, SEEK_CUR);
        offset-=(tmp_word + sizeof(char) + sizeof(uint32_t));
      }
    }
    
    k++;
  }

  fclose(ifp);
}

#else
void readR1CSFileHeader_h(r1csv1_t *r1cs_hdr, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");
  uint32_t k=0,i;
  uint32_t tmp_word, n_coeff;
  t_int64 offset;
  uint32_t section_type;

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
    printf("Unexpected R1CS version %d\n", r1cs_hdr->version);
    fclose(ifp);
    exit(1);
  }

  fread(&r1cs_hdr->nsections, sizeof(uint32_t), 1, ifp); 
  //printf("N sections : %d\n",r1cs_hdr->nsections);
  fread(&section_type, sizeof(uint32_t), 1, ifp); 
  //printf("Section type : %d\n",section_type);
  if (section_type != R1CS_HDR_SECTION_TYPE){
    printf("Unexpected section : %d\n",section_type);
    fclose(ifp);
    exit(1);
  }

  fread(&offset, sizeof(t_uint64), 1, ifp); 
  //printf("HEADER Section Length : %lld\n", offset);
  //fseek(ifp, R1CS_HDR_FIELDDEFSIZE_OFFSET_NBYTES * sizeof(char), SEEK_SET);
  fread(&offset, sizeof(uint32_t), 1, ifp); 
  //printf("Field Size : %lld\n", offset);
  fseek(ifp, offset , SEEK_CUR);

  //fread(&r1cs_hdr->word_width_bytes, sizeof(uint32_t), 1, ifp); 
  //if (r1cs_hdr->word_width_bytes != 4){
     //printf("Unexpected R1CS word width %d\n",r1cs_hdr->word_width_bytes);
     //fclose(ifp);
     //exit(1);
  //}

  fread(&r1cs_hdr->nVars, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPubOutputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPubInputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPrivInputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nLabels, sizeof(t_uint64), 1, ifp); 
  fread(&r1cs_hdr->nConstraints, sizeof(uint32_t), 1, ifp); 

  fread(&offset, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->constraintLen, sizeof(uint32_t), 2, ifp); 

  /*
  printf("word_width_bytes : %d\n", r1cs_hdr->word_width_bytes);
  printf("nVars : %d\n", r1cs_hdr->nVars);
  printf("nPubOutputs : %d\n", r1cs_hdr->nPubOutputs);
  printf("nPubInputs : %d\n", r1cs_hdr->nPubInputs);
  printf("nPrivInputs : %d\n", r1cs_hdr->nPrivInputs);
  printf("nLabels : %d\n", r1cs_hdr->nLabels);
  printf("nConstraints : %d\n", r1cs_hdr->nConstraints);
  printf("Const section len : %lld\n",r1cs_hdr->constraintLen);
  */
  

  r1cs_hdr->constraintOffset = ftell(ifp);

  r1cs_hdr->R1CSA_nCoeff = 0;
  r1cs_hdr->R1CSB_nCoeff = 0;
  r1cs_hdr->R1CSC_nCoeff = 0;

  offset = r1cs_hdr->constraintLen;

  while (offset > 0){
    fread(&n_coeff, sizeof(uint32_t), 1, ifp); 
    //printf("N coeff : %d\n",n_coeff);
    offset-=sizeof(uint32_t);
    if (k%3 == R1CSA_IDX){
      r1cs_hdr->R1CSA_nCoeff+= (n_coeff);
    } else if (k%3 == R1CSB_IDX){
      r1cs_hdr->R1CSB_nCoeff+= (n_coeff);
    } else {
      r1cs_hdr->R1CSC_nCoeff+= (n_coeff);
    }
    fseek(ifp, n_coeff*(32+4), SEEK_CUR);
    offset -= (n_coeff*36);
    /*for (i=0; i< n_coeff; i++){
      fseek(ifp, 4, SEEK_CUR);
      fread(&tmp_word, sizeof(char), 1, ifp); 
      tmp_word &= 0xFF;
      fseek(ifp, tmp_word, SEEK_CUR);
      offset-=(tmp_word + sizeof(char) + sizeof(uint32_t));
    }
    */
    k++;
  }

  /*
  printf("N coeff R1CSA : %d\n", r1cs_hdr->R1CSA_nCoeff);
  printf("N coeff R1CSB : %d\n", r1cs_hdr->R1CSB_nCoeff);
  printf("N coeff R1CSC : %d\n", r1cs_hdr->R1CSC_nCoeff);
  
  printf("end of constraints : %lld\n",ftell(ifp));
  */
  fread(&offset, sizeof(uint32_t), 1, ifp); 
  offset &=0xFFFF;
  //printf("Lable section len : %lld\n",offset);
  

  fclose(ifp);

  return;
}
  

void readR1CSFile_h(uint32_t *samples, const char *filename, r1csv1_t *r1cs, r1cs_idx_t r1cs_idx )
{
  FILE *ifp = fopen(filename,"rb");
  uint32_t tmp_word, n_coeff;
  uint32_t r1cs_offset=0, r1cs_coeff_offset=1+r1cs->nConstraints, r1cs_val_offset = 1+r1cs->nConstraints;
  uint32_t k=0, accum_coeffs=0, i,j;
  t_int64 offset = r1cs->constraintLen;

  samples[r1cs_offset++] = r1cs->nConstraints;
  
  /* printf("constraint LEN : %lld\n",offset);
  printf("constraint Offset : %lld\n",r1cs->constraintOffset);
  */
  fseek(ifp, r1cs->constraintOffset, SEEK_SET);
  //printf("Start\n");

  while (offset){
    fread(&n_coeff, sizeof(uint32_t), 1, ifp); 
    //printf("N COEFF : %d\n",n_coeff);
    offset-=sizeof(uint32_t);
    if (k%3 == r1cs_idx) {
      accum_coeffs+= ((uint32_t) n_coeff);
      samples[r1cs_offset++] = accum_coeffs;
      r1cs_val_offset += n_coeff;
      for (i=0; i< n_coeff; i++){
        fread(&samples[r1cs_coeff_offset++], sizeof(uint32_t), 1, ifp); 
        //fread(&tmp_word, 1,1, ifp);
	//tmp_word &= tmp_word & 0xFF;
	tmp_word = 8;
        //for(j=0; j <tmp_word; j++){
        fread(&samples[r1cs_val_offset], sizeof(uint32_t), tmp_word, ifp); 
        //}
        offset-=36;
        r1cs_val_offset += NWORDS_256BIT;
      }
      r1cs_coeff_offset = r1cs_val_offset;

    }  else {
      /*for (i=0; i< n_coeff; i++){
        fseek(ifp, 4, SEEK_CUR);
        fread(&tmp_word, 1, 1, ifp); 
	tmp_word &=0xFF;
        fseek(ifp, tmp_word, SEEK_CUR);
        offset-=(tmp_word + sizeof(char) + sizeof(uint32_t));
      } */
      fseek(ifp, n_coeff*(32+4), SEEK_CUR);
      offset -= (n_coeff*36);
    }
    
    k++;
  }

  fclose(ifp);

}
#endif

void readECTablesNElementsFile_h(ec_table_offset_t *table_offset, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");

  fread(&table_offset->table_order,     sizeof(uint32_t), 1, ifp); 
  fread(&table_offset->woffset_A,      sizeof(t_uint64), 1, ifp); 
  fread(&table_offset->woffset_B2,     sizeof(t_uint64), 1, ifp); 
  fread(&table_offset->woffset_B1,     sizeof(t_uint64), 1, ifp); 
  fread(&table_offset->woffset_C,      sizeof(t_uint64), 1, ifp); 
  fread(&table_offset->woffset_hExps,  sizeof(t_uint64), 1, ifp); 
  fread(&table_offset->nwords_tdata,   sizeof(t_uint64), 1, ifp); 

  /*
  printf("Order : %d\n",table_offset->table_order);
  printf("Woffset1 A : %ld\n", table_offset->woffset1_A);
  printf("Woffset1 B2 : %ld\n", table_offset->woffset1_B2);
  printf("Woffset1 B1 : %ld\n", table_offset->woffset1_B1);
  printf("Woffset C : %ld\n", table_offset->woffset_C);
  printf("Woffset hExps : %ld\n", table_offset->woffset_hExps);
  printf("N Words : %d\n", table_offset->nwords_tdata);
  */
  

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
  if (count != 1)
  {
    for (i=0;i<insize; i++){
      fread(r,sizeof(uint32_t),NWORDS_256BIT,ifp);
      if (j % count == 0){
        memcpy(&samples[k*NWORDS_256BIT], r, sizeof(uint32_t)*NWORDS_256BIT);
        k++;
      }
      j++;
    }
  } 
  else  {
    fread(samples, sizeof(uint32_t)*outsize, NWORDS_256BIT, ifp);
  }
 
  
  fclose(ifp);
}

void readU256DataFileFromOffset_h(uint32_t *samples, const char *filename, t_uint64 woffset, t_uint64 nwords)
{
  FILE *ifp = fopen(filename,"rb");

  fseek(ifp, woffset * sizeof(uint32_t), SEEK_SET);
  fread(samples,sizeof(uint32_t),nwords,ifp);

  /*
  printf("Offset : %ld, Nwords : %d\n",woffset, nwords);
  printU256Number(&samples[0]);
  printU256Number(&samples[NWORDS_256BIT]);
  printU256Number(&samples[2*NWORDS_256BIT]);
  printU256Number(&samples[3*NWORDS_256BIT]);
  */

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

#if 0
  for (i=0;i<inlen; i++){
    fread(&samples[i*NWORDS_256BIT],sizeof(uint32_t),NWORDS_256BIT,ifp);
  }
#else
    fread(samples,sizeof(uint32_t)*NWORDS_256BIT,inlen,ifp);
#endif
  
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

  //if (Scl_idx != NULL){
     //idx = Scl_idx;
  //} 
  //else {
    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for (i=0;i < len; i++){  
      idx[i] = i;
    }
  //}

  if (sort_en){
     //std::sort(idx, idx+len, [&v](uint32_t i1, uint32_t i2){ return (v[i1*NWORDS_256BIT] < v[i2*NWORDS_256BIT]);});
     std::sort(idx, idx+len, 
       [&v](uint32_t i1, uint32_t i2){ 
         //return (ltu256_h((const uint32_t*)&v[i1*NWORDS_256BIT],(const uint32_t *)&v[i2*NWORDS_256BIT]));});
         return (ltu32_h((const uint32_t*)&v[i1*NWORDS_256BIT+NWORDS_256BIT-1],(const uint32_t *)&v[i2*NWORDS_256BIT+NWORDS_256BIT-1]));});
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
  void (*subm_cb)(uint32_t *, const uint32_t *, const uint32_t *) = &Fq_rawSub;
  void (*addm_cb)(uint32_t *, const uint32_t *, const uint32_t *) = &Fq_rawAdd;
  void (*mulm_cb)(uint32_t *, const uint32_t *, const uint32_t *) = &Fq_rawMMul;

  if (pidx == MOD_GROUP){
     subm_cb = &Fr_rawSub;
     addm_cb = &Fr_rawAdd;
     mulm_cb = &Fr_rawMMul;
  } 
  
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
  ntt_parallel_T_h(A, roots, Nrows, Ncols, rstride, direction, fft_mode, pidx);

  transpose_h(M_transpose,A,1<<Nrows, 1<<Ncols);
  memcpy(A,M_transpose, (1ull << (Nrows + Ncols)) * NWORDS_256BIT * sizeof(uint32_t));
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

void ntt2_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx)
{
  if (levels == 0){
     return;
  }

  ntt2_h(A, roots, levels-1, 2*astride, rstride, direction, pidx);
  ntt2_h(A, roots, levels-1, 2*astride, rstride, direction, pidx);
  
  uint32_t k;

/*
  for (k=0; k < levels-1; k++){
     A[1<< (k + NWORDS_256BIT_SHIFT)] = 
  }
*/
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
                     &roots[((rstride*(1ull<<i)*k*direction) & (rstride*(1ull << levels)-1))<<NWORDS_256BIT_SHIFT], pidx);
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
  if (M_transpose == NULL){
    M_transpose = (uint32_t *) malloc ( (t_uint64) (nroots) * NWORDS_256BIT * sizeof(uint32_t));
  }
  if (M_mul == NULL){
    M_mul = (uint32_t *) malloc ( (t_uint64)(nroots+1) * NWORDS_256BIT * sizeof(uint32_t));
  }
  if (Scl_idx == NULL){
    Scl_idx = (uint32_t *) malloc(nroots* sizeof(uint32_t)); 
    for (uint32_t i=0; i < nroots; i++){
       Scl_idx[i] = i;
    }
  }
}

void M_free_h(void)
{
  free (M_transpose);
  free (M_mul);
  free (Scl_idx);
  M_transpose = NULL;
  M_mul = NULL;
  Scl_idx = NULL;
}

uint32_t *get_Mmul_h()
{
  return M_mul;
}

uint32_t *get_Mtranspose_h()
{
  return M_transpose;
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

uint32_t * ntt_interpolandmul_server_h(ntt_interpolandmul_t *args)
{
  if ((!args->max_threads) || (!utils_mproc_init)) {
    return ntt_interpolandmul_parallel_h(args->A, args->B, args->roots, args->Nrows, args->Ncols, args->rstride, args->pidx);
  }
  #ifndef PARALLEL_EN
    return ntt_interpolandmul_parallel_h(args->A, args->B, args->roots, args->Nrows, args->Ncols, args->rstride, args->pidx);
  #endif

  args->max_threads = MIN(args->max_threads, MIN(utils_nprocs, 1<<MIN(args->Nrows, args->Ncols)));

  uint32_t nvars = 1<<(args->Nrows+args->Ncols);
  uint32_t start_idx, last_idx;
  uint32_t vars_per_thread = nvars/args->max_threads;

  if  ((vars_per_thread & (vars_per_thread-1)) != 0){
    vars_per_thread = sizeof(uint32_t) * NBITS_BYTE - __builtin_clz(args->max_threads/nvars) - 1;
    vars_per_thread = 1 << (vars_per_thread-1);
  }
   
  pthread_t *workers = (pthread_t *) malloc(args->max_threads * sizeof(pthread_t));
  ntt_interpolandmul_t *w_args  = (ntt_interpolandmul_t *)malloc(args->max_threads * sizeof(ntt_interpolandmul_t));
  if (pthread_barrier_init(&utils_barrier, NULL, args->max_threads) != 0){
     exit(1);
  }

  /*
  printf("N threads : %d\n", args->max_threads);
  printf("N vars    : %d\n", nvars);
  printf("Vars per thread : %d\n", vars_per_thread);
  printf("Nrows : %d\n", args->Nrows);
  printf("Ncols : %d\n", args->Ncols);
  printf("nroots : %d\n", args->nroots);
  printf("rstride : %d\n", args->rstride);
  printf("pidx : %d\n", args->pidx);
  */
  
  
  ntt_interpolandmul_init_h(args->A, args->B, &args->mNrows,&args->mNcols, args->Nrows, args->Ncols);

  for(uint32_t i=0; i< args->max_threads; i++){
     start_idx = i * vars_per_thread;
     last_idx = (i+1) * vars_per_thread;
     if ( (i == args->max_threads - 1) && (last_idx != nvars) ){
         last_idx = nvars;
     }
     memcpy(&w_args[i], args, sizeof(ntt_interpolandmul_t));

     w_args[i].start_idx = start_idx;
     w_args[i].last_idx = last_idx;
     w_args[i].thread_id = i;
     
     /*
     printf("Thread : %d, start_idx : %d, end_idx : %d\n",
             w_args[i].thread_id, 
             w_args[i].start_idx,
             w_args[i].last_idx);   
     */
  }

  /*
  printf("mNrows : %d\n", args->mNrows);
  printf("mNcols : %d\n", args->mNcols);
  */
 
  /*
  for (uint32_t i=0; i < 1 << (args->Nrows +args->Ncols); i++){
    printf("[%d] : ",i);
    printU256Number(&args->A[i*NWORDS_256BIT]);
    printU256Number(&args->B[i*NWORDS_256BIT]);
  }
  */

  launch_client_h(interpol_and_mul_h, workers, (void *)w_args, sizeof(ntt_interpolandmul_t), args->max_threads);

  /*
  for (uint32_t i=0; i < 1 << (args->Nrows +args->Ncols); i++){
    printf("[%d] : ",i);
    printU256Number(&args->A[i*NWORDS_256BIT]);
    printU256Number(&args->B[i*NWORDS_256BIT]);
  }
  for (uint32_t i=0; i < 1 << (args->Nrows +args->Ncols+1); i++){
    printf("[%d] : ",i);
    printU256Number(&M_mul[i*NWORDS_256BIT]);
  }
  */

  pthread_barrier_destroy(&utils_barrier);
  free(workers);
  free(w_args);

  //return M_mul;
  return M_transpose;
}

static void ntt_interpolandmul_init_h(uint32_t *A, uint32_t *B, uint32_t *mNrows, uint32_t *mNcols, uint32_t Nrows, uint32_t Ncols)
{
  if (Nrows < Ncols){
    *mNcols = Ncols;
    *mNrows = Nrows+1;
  } else {
    *mNrows = Nrows;
    *mNcols = Ncols+1;
  }
}

void *interpol_and_mul_h(void *args)
{
  ntt_interpolandmul_t *wargs = (ntt_interpolandmul_t *)args;
  uint32_t start_col = wargs->start_idx>>wargs->Nrows, last_col = wargs->last_idx>>wargs->Nrows;
  uint32_t start_row = wargs->start_idx>>wargs->Ncols, last_row = wargs->last_idx>>wargs->Ncols;
  uint32_t Ancols = 1 << wargs->Ncols;
  uint32_t Anrows = 1 << wargs->Nrows;
  uint32_t Amncols = 1 << wargs->mNcols;
  uint32_t Amnrows = 1 << wargs->mNrows;
  uint32_t *save_A, *save_B;
  const uint32_t *scaler_mont = CusnarksIScalerGet((fmt_t)1);
  const uint32_t *scaler_ext = CusnarksIScalerGet((fmt_t)0);
  int64_t ridx;
  uint32_t i;

  // multiply M_mul[2*i] = A * B
  //printf("multiply-even [%d]\n", wargs->thread_id);
  //__builtin_prefetch(&M_mul[(2*wargs->start_idx)<<NWORDS_256BIT_SHIFT]);
  //__builtin_prefetch(&wargs->A[wargs->start_idx<<NWORDS_256BIT_SHIFT]);
  //__builtin_prefetch(&wargs->B[wargs->start_idx<<NWORDS_256BIT_SHIFT]);
  for (i=wargs->start_idx; i < wargs->last_idx; i++){
    //if (i < wargs->last_idx -1){
      //__builtin_prefetch(&M_mul[(2*(i+1))<<NWORDS_256BIT_SHIFT]);
      //__builtin_prefetch(&wargs->A[(i+1)<<NWORDS_256BIT_SHIFT]);
      //__builtin_prefetch(&wargs->B[(i+1)<<NWORDS_256BIT_SHIFT]);
    //}
    montmult_h(&M_mul[(2*i)<<NWORDS_256BIT_SHIFT],
               &wargs->A[i<<NWORDS_256BIT_SHIFT],
               &wargs->B[i<<NWORDS_256BIT_SHIFT],
               wargs->pidx);
  }
  util_wait_h(wargs->thread_id, NULL, NULL);

  //printf("intt0 [%d]\n", wargs->thread_id);
  // A = IFFT_N/2(A); B = IFFT_N/2(B)
  for (i=start_col; i<last_col; i++){
    ntt_h(&wargs->A[i<<NWORDS_256BIT_SHIFT], wargs->roots, wargs->Nrows,1 << wargs->Ncols, wargs->rstride<<wargs->Ncols, -1,  wargs->pidx);
    ntt_h(&wargs->B[i<<NWORDS_256BIT_SHIFT], wargs->roots, wargs->Nrows,1 << wargs->Ncols, wargs->rstride<<wargs->Ncols, -1,  wargs->pidx);
  }
  util_wait_h(wargs->thread_id, NULL, NULL);

  //printf("intt0 il2mul [%d]\n", wargs->thread_id);
  // A[i] = A[i] * l2_IW[i]
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    ridx = (wargs->rstride * (i >> wargs->Ncols) * (i & (Ancols - 1)) * -1) & (wargs->rstride * Anrows * Ancols - 1);
    montmult_h(&wargs->A[i<<NWORDS_256BIT_SHIFT],
               &wargs->A[i<<NWORDS_256BIT_SHIFT],
               &wargs->roots[ridx << NWORDS_256BIT_SHIFT],
               wargs->pidx);
    montmult_h(&wargs->B[i<<NWORDS_256BIT_SHIFT],
               &wargs->B[i<<NWORDS_256BIT_SHIFT],
               &wargs->roots[ridx << NWORDS_256BIT_SHIFT],
               wargs->pidx);
  }
  util_wait_h(wargs->thread_id, NULL, NULL);
  
  //printf("intt1 [%d]\n", wargs->thread_id);
  // A[i] = IFFT_N/2(A).T; B[i] = IFFT_N/2(B).T
  for (i=start_row;i < last_row; i++){
    ntt_h(&wargs->A[(i<<wargs->Ncols+NWORDS_256BIT_SHIFT)], wargs->roots, wargs->Ncols,1, wargs->rstride<<wargs->Nrows, -1, wargs->pidx);
    ntt_h(&wargs->B[(i<<wargs->Ncols+NWORDS_256BIT_SHIFT)], wargs->roots, wargs->Ncols,1, wargs->rstride<<wargs->Nrows, -1, wargs->pidx);
  }

  transposeBlock_h(M_transpose, wargs->A,
              start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(&M_transpose[1ull<<(wargs->Nrows+wargs->Ncols + NWORDS_256BIT_SHIFT)],
              wargs->B, start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  util_wait_h(wargs->thread_id, NULL, NULL);

  save_A = wargs->A;
  save_B = wargs->B;
  wargs->A = M_transpose;
  wargs->B = &M_transpose[1ull<<(wargs->Nrows+wargs->Ncols + NWORDS_256BIT_SHIFT)];

  // A = A * scaler * l3W; B = B * scaler * l3W
  //printf("intt1-scaler-l3mul [%d]\n", wargs->thread_id);
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
      montmult_h(&wargs->A[i<<NWORDS_256BIT_SHIFT],
                 &wargs->A[i<<NWORDS_256BIT_SHIFT],
                 &scaler_mont[(wargs->Nrows + wargs->Ncols)<<NWORDS_256BIT_SHIFT], wargs->pidx);
      montmult_h(&wargs->B[i<<NWORDS_256BIT_SHIFT],
                 &wargs->B[i<<NWORDS_256BIT_SHIFT],
                 &scaler_mont[(wargs->Nrows + wargs->Ncols)<<NWORDS_256BIT_SHIFT], wargs->pidx);
      montmult_h(&wargs->A[i<<NWORDS_256BIT_SHIFT],
                 &wargs->A[i<<NWORDS_256BIT_SHIFT],
                 &wargs->roots[i<<NWORDS_256BIT_SHIFT],
                 wargs->pidx);
      montmult_h(&wargs->B[i<<NWORDS_256BIT_SHIFT],
                 &wargs->B[i<<NWORDS_256BIT_SHIFT],
                 &wargs->roots[i<<NWORDS_256BIT_SHIFT],
                 wargs->pidx);
  }
  util_wait_h(wargs->thread_id, NULL, NULL);

  // A = FFT_N/2(A); B = FFT_N/2(B)
  //printf("ntt2 [%d]\n", wargs->thread_id);
  for (i=start_col;i < last_col; i++){
    ntt_h(&wargs->A[i<<NWORDS_256BIT_SHIFT], wargs->roots, wargs->Nrows,1 << wargs->Ncols, wargs->rstride<<wargs->Ncols, 1,  wargs->pidx);
    ntt_h(&wargs->B[i<<NWORDS_256BIT_SHIFT], wargs->roots, wargs->Nrows,1 << wargs->Ncols, wargs->rstride<<wargs->Ncols, 1,  wargs->pidx);
  }
  util_wait_h(wargs->thread_id, NULL, NULL);

  //printf("ntt2-l2mul [%d]\n", wargs->thread_id);
  // A[i] = A[i] * l2_W[i]
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    ridx = (wargs->rstride * (i >> wargs->Ncols) * (i & (Ancols - 1))) & (wargs->rstride * Anrows * Ancols - 1);
    montmult_h(&wargs->A[i<<NWORDS_256BIT_SHIFT],
               &wargs->A[i<<NWORDS_256BIT_SHIFT],
               &wargs->roots[ridx << NWORDS_256BIT_SHIFT],
               wargs->pidx);
    montmult_h(&wargs->B[i<<NWORDS_256BIT_SHIFT],
               &wargs->B[i<<NWORDS_256BIT_SHIFT],
               &wargs->roots[ridx << NWORDS_256BIT_SHIFT],
               wargs->pidx);
  }
  util_wait_h(wargs->thread_id, NULL, NULL);

  //printf("ntt3 [%d]\n", wargs->thread_id);
  // A = FFT_N/2(A).T; B = FFT_N/2(B).T
  for (i=start_row;i < last_row; i++){
    ntt_h(&wargs->A[i<<wargs->Ncols+NWORDS_256BIT_SHIFT], wargs->roots, wargs->Ncols,1, wargs->rstride<<wargs->Nrows, 1, wargs->pidx);
    ntt_h(&wargs->B[i<<wargs->Ncols+NWORDS_256BIT_SHIFT], wargs->roots, wargs->Ncols,1, wargs->rstride<<wargs->Nrows, 1, wargs->pidx);
  }
  wargs->A=save_A;
  wargs->B=save_B;

  transposeBlock_h(wargs->A,M_transpose, start_row, last_row,1<<wargs->Nrows, 1<<wargs->Ncols, TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(wargs->B,&M_transpose[1ull<<(wargs->Nrows+wargs->Ncols + NWORDS_256BIT_SHIFT)],
                start_row, last_row,1<<wargs->Nrows, 1<<wargs->Ncols, TRANSPOSE_BLOCK_SIZE);
  util_wait_h(wargs->thread_id, NULL, NULL);


  //printf("multiply-odd [%d]\n", wargs->thread_id);
  // M_mul[2*i+1] = A * B
  for (i=wargs->start_idx; i < wargs->last_idx; i++){
    montmult_h(&M_mul[(2*i+1)<<NWORDS_256BIT_SHIFT],
               &wargs->A[i<<NWORDS_256BIT_SHIFT],
               &wargs->B[i<<NWORDS_256BIT_SHIFT],
               wargs->pidx);
  }
  util_wait_h(wargs->thread_id, NULL, NULL);

  start_col = wargs->start_idx>>(wargs->mNrows-1), last_col = wargs->last_idx>>(wargs->mNrows-1);
  start_row = wargs->start_idx>>(wargs->mNcols-1), last_row = wargs->last_idx>>(wargs->mNcols-1);

  //printf("intt4 [%d]\n", wargs->thread_id);
  // A = IFFT_N(A); B = IFFT_N(B)
  for (i=start_col;i < last_col; i++){
    ntt_h(&M_mul[i<<NWORDS_256BIT_SHIFT], wargs->roots, wargs->mNrows,1 << wargs->mNcols, wargs->rstride<<(wargs->mNcols-1), -1,  wargs->pidx);
  }
  util_wait_h(wargs->thread_id, NULL, NULL);

  //printf("intt4-il2mul [%d]\n", wargs->thread_id);
  // A[i] = A[i] * l2_IW[i]
  for (i=wargs->start_idx*2;i < wargs->last_idx*2; i++){
    ridx = ( (wargs->rstride>>1) * (i >> wargs->mNcols) * (i & (Amncols - 1)) * -1) & ((wargs->rstride>>1) * Amnrows * Amncols - 1);
    montmult_h(&M_mul[i<<NWORDS_256BIT_SHIFT],
               &M_mul[i<<NWORDS_256BIT_SHIFT],
               &wargs->roots[ridx << NWORDS_256BIT_SHIFT],
               wargs->pidx);
  }
  util_wait_h(wargs->thread_id, NULL, NULL);

  //printf("intt5 [%d]\n", wargs->thread_id);
  for (i=start_row;i < last_row; i++){
    ntt_h(&M_mul[i<<wargs->mNcols+NWORDS_256BIT_SHIFT], wargs->roots, wargs->mNcols,1, wargs->rstride<<(wargs->mNrows-1), -1, wargs->pidx);
  }
  util_wait_h(wargs->thread_id, NULL, NULL);

  //printf("intt5-scaler-l3mul [%d]\n", wargs->thread_id);
  for (i=wargs->start_idx*2;i < wargs->last_idx*2; i++){
      montmult_h(&M_mul[i<<NWORDS_256BIT_SHIFT],
                 &M_mul[i<<NWORDS_256BIT_SHIFT],
                 &scaler_ext[(wargs->mNrows + wargs->mNcols)<<NWORDS_256BIT_SHIFT], wargs->pidx);
  }

  transposeBlock_h(M_transpose,M_mul,start_row, last_row, 1<<wargs->mNrows, 1<<wargs->mNcols, TRANSPOSE_BLOCK_SIZE);
  util_wait_h(wargs->thread_id, NULL, NULL);

  return NULL;
}

uint32_t *ntt_interpolandmul_parallel_h(uint32_t *A, uint32_t *B, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, uint32_t pidx)
{
  uint32_t mNrows, mNcols;
  ntt_interpolandmul_init_h(A, B, &mNrows, &mNcols, Nrows, Ncols);

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

  intt_parallel_h(M_mul, roots, 0, mNrows, mNcols, 1, FFT_T_DIF, pidx);
  
  /*
  for(uint32_t i=0; i < 1<<(mNcols + mNrows); i++){
    printf("[%d] : ",i);
    printU256Number(&M_mul[i*NWORDS_256BIT]);
  }
  */
 
 
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

#if 0
  inv_t data_table[4];
  uint32_t data_table_r[] = {0,1,0,3,0,1,0,2};

  init_invtable(data_table, u, v, s, r1, zero);

  while(compu256_h(v,zero) != 0){
     t0 = u[0] & 0x1; 
     t1 = (v[0] & 0x1) << 1;
     subu256_h(tmp,v,u);
     t2 = (tmp[NWORDS_256BIT-1] & 0x80000000) >> 29;
     t3 = t0 + t1 + t2;
     almmontinv_step_h(&data_table[data_table_r[t3]]);	  
     (*k)++;
  }
#else
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
#endif
  
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
          //memmove(&z[2*NWORDS_256BIT], One, sizeof(uint32_t) * NWORDS_256BIT);
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

void ec_loadtable_h(uint32_t *x, t_uint64 len, t_uint64 *offset, ecp_t ecp, FILE *ifp)
{
  fseek(ifp, offset[ecp], SEEK_SET);
  fread(x, sizeof(uint32_t), len, ifp); 
  offset[ecp] += len;
}

void ec_inittable_h(uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t table_order, uint32_t pidx, uint32_t add_last=0)
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

   #ifndef TEST_MODE
     #pragma omp parallel for if(parallelism_enabled)
   #endif
   for (i=0; i< n_tables; i++){
      // init element 0 of table
      memcpy(&ectable[(i*table_size)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
            &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT],
            sizeof(uint32_t) * ECP_JAC_OUTDIMS * NWORDS_256BIT);
      uint32_t k=0, last_pow2, n_els=0;
      for (uint32_t j=1; j< table_size; j++){
         if (j < table_size - 1){
              __builtin_prefetch(&ectable[(i*table_size+(j+1))*NWORDS_256BIT*ECP_JAC_OUTDIMS]);
              __builtin_prefetch(&ectable[(i*table_size+last_pow2)*NWORDS_256BIT*ECP_JAC_OUTDIMS]);
              __builtin_prefetch(&ectable[(i*table_size+(j+1)-last_pow2)*NWORDS_256BIT*ECP_JAC_OUTDIMS]);
         }
         // if power of 2    
         if  ((j & (j-1)) == 0){
             //printf("elems : %d,%d, %d, %d\n",n_els+i*table_order,n,j,i);
             last_pow2 = j;
             if (n_els + i*table_order < n){
                memcpy(&ectable[(i*table_size+j)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
                   &x[(i*table_order+k)*NWORDS_256BIT*ndims],
                   sizeof(uint32_t) * ndims * NWORDS_256BIT);

                if (add_last){
                   memcpy(&ectable[(i*table_size+j)*NWORDS_256BIT*ECP_JAC_OUTDIMS+ECP_JAC_INDIMS*NWORDS_256BIT],
                      One,
                      sizeof(uint32_t) * NWORDS_256BIT);
                }
             } else {
		 //printf("Table Overflow : %d, %d, %d, %d\n", n_els+i*table_order, n,i, i*table_size);
                 memcpy(&ectable[(i*table_size+j)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
                        &ECInf[(pidx * MISC_K_N+MISC_K_INF) * NWORDS_256BIT],
                        sizeof(uint32_t) * ECP_JAC_OUTDIMS * NWORDS_256BIT);
             }
	     //printf("Table idx : %d\n",i*table_size+j);
             k++;
             n_els++;
         } else {
             ec_jacadd_h( &ectable[(i*table_size+j)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
                          &ectable[(i*table_size+last_pow2)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
                          &ectable[(i*table_size+j-last_pow2)*NWORDS_256BIT*ECP_JAC_OUTDIMS],
                          pidx);
                           
         }      
      } 
   } 

}

void ec2_inittable_h(uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t table_order, uint32_t pidx, uint32_t add_last=0)
{
   uint32_t n_tables = (table_order + n - 1)/table_order;
   uint32_t i;
   uint32_t table_size = 1<< table_order;
   const uint32_t *ECInf = CusnarksMiscKGet();
   const uint32_t *One = CusnarksOneMont2Get(pidx);
   uint32_t ndims = ECP2_JAC_OUTDIMS;
   if (add_last){
      ndims = ECP2_JAC_INDIMS;
   }
 

   #ifndef TEST_MODE
     #pragma omp parallel for if(parallelism_enabled)
   #endif
   for (i=0; i< n_tables; i++){
      // init element 0 of table
      memcpy(&ectable[(i*table_size)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
            &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT],
            sizeof(uint32_t) * ECP2_JAC_OUTDIMS * NWORDS_256BIT);
      uint32_t k=0, last_pow2=1, n_els = 0;
      for (uint32_t j=1; j< table_size; j++){
         
         if (j < table_size - 1){
              __builtin_prefetch(&ectable[(i*table_size+(j+1))*NWORDS_256BIT*ECP2_JAC_OUTDIMS]);
              __builtin_prefetch(&ectable[(i*table_size+last_pow2)*NWORDS_256BIT*ECP2_JAC_OUTDIMS]);
              __builtin_prefetch(&ectable[(i*table_size+(j+1)-last_pow2)*NWORDS_256BIT*ECP2_JAC_OUTDIMS]);
         }
        
         // if power of 2    
         if  ((j & (j-1)) == 0){
             last_pow2 = j;
             if (n_els +i*table_order < n){
                memcpy(&ectable[(i*table_size+j)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
                   &x[(i*table_order+k)*NWORDS_256BIT*ndims],
                   sizeof(uint32_t) * ndims * NWORDS_256BIT);

                if (add_last){
                   memcpy(&ectable[(i*table_size+j)*NWORDS_256BIT*ECP2_JAC_OUTDIMS+4*NWORDS_256BIT],
                      One,
                      sizeof(uint32_t) * NWORDS_256BIT * 2);
                }
             } else {
                 memcpy(&ectable[(i*table_size+j)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
                        &ECInf[(pidx * MISC_K_N+MISC_K_INF2) * NWORDS_256BIT],
                        sizeof(uint32_t) * ECP2_JAC_OUTDIMS * NWORDS_256BIT);
             }
             k++;
             n_els++;
         } else {
            
             ec2_jacadd_h( &ectable[(i*table_size+j)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
                          &ectable[(i*table_size+last_pow2)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
                          &ectable[(i*table_size+j-last_pow2)*NWORDS_256BIT*ECP2_JAC_OUTDIMS],
                          pidx);
                           
         }      
      } 
   } 
}


void ec_jacscmul_opt_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t order, uint32_t pidx, uint32_t add_last=0, uint32_t compute_table=1 )
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t i;
  int debug_tid = 0;

  uint32_t n_tables = (order + n - 1)/order;
  uint32_t table_size = 1 << order; 
  void (*add_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t) = &ec_jacadd_h;
  uint32_t table_dim = ECP_JAC_OUTDIMS;
  uint32_t *ectable_ptr = ectable;
  uint32_t torder1 = 0;

  if (compute_table){
    if (order > 1){
      ec_inittable_h(x, ectable, n, order, pidx, add_last);
    } else {
      ectable_ptr = x;
      torder1 = 1;
      table_size = 1;
      table_dim = ECP_JAC_INDIMS;
      add_cb = &ec_jacaddmixed_h;
    }
   
  } else {
    add_cb = &ec_jacaddmixed_h;
    table_dim = ECP_JAC_INDIMS;
  }

  /*
  for(i=debug_tid * table_size; i< (debug_tid+1)*table_size; i++){
        printf("T[%d] :\n",i-debug_tid*table_size);
        printU256Number(&ectable[i * NWORDS_256BIT * ECP_JAC_OUTDIMS]);
        printU256Number(&ectable[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+NWORDS_256BIT]);
        printU256Number(&ectable[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+2*NWORDS_256BIT]);
  }
  for(i=debug_tid * order ; i< (debug_tid+1)*order; i++){
     printf("SCL :");
     printU256Number(&scl[i*NWORDS_256BIT]);
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

       else{
                 break;
       }


     }

     msb = 255 - msb;
     for (int j=msb; j>=0 ; j--){
        uint32_t b = getbitu256g_h(&scl[i*order * NWORDS_256BIT], j, MIN(order, n -order*i ));

        ec_jacdouble_h(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                       &z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                       pidx);
        if (b) {
           add_cb(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                       &ectable_ptr[(i * table_size + b - torder1) * NWORDS_256BIT * table_dim],
                       &z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
                       pidx);
        }
       
        /*
        if (i==debug_tid){ 
          printf("offset : %d, b : %d\n",255-j, b);
          printf("Q :\n");
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS]);
          printf("Q :\n");
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+NWORDS_256BIT]);
          printf("Q :\n");
          printU256Number(&z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS+2*NWORDS_256BIT]);
        }
        */
     }
  }
}

void ec2_jacscmul_opt_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t order, uint32_t pidx, uint32_t add_last=0, uint32_t compute_table=1)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t i;
  int debug_tid = 8191;
  uint32_t n_tables = (order + n - 1)/order;
  uint32_t table_size = 1 << order; 
  void (*add_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t) = &ec2_jacadd_h;
  uint32_t table_dim = ECP2_JAC_OUTDIMS;
  uint32_t *ectable_ptr = ectable;
  uint32_t torder1 = 0;

  
  if (compute_table){
    if (order > 1){
      ec2_inittable_h(x, ectable, n, order, pidx, add_last);
    } else {
      ectable_ptr = x;
      torder1 = 1;
      table_size = 1;
      table_dim = ECP2_JAC_INDIMS;
      add_cb = &ec2_jacaddmixed_h;
    }
  } else {
    add_cb = &ec2_jacaddmixed_h;
    table_dim = ECP2_JAC_INDIMS;
  }
  

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
       else {
          break;
       }

     }
     msb = 255 - msb;
     for (int j=msb; j>=0 ; j--){
        uint32_t b = getbitu256g_h(&scl[i*order * NWORDS_256BIT], j, MIN(order, n - order*i) );


        ec2_jacdouble_h(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                       &z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                       pidx);
        if (b) {
           add_cb(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                       &ectable_ptr[(i * table_size + b - torder1) * NWORDS_256BIT * table_dim],
                       &z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
                       pidx);
        }
        
        /*
         if (i==debug_tid){ 
          printf("offset : %d, b : %d\n",j, b);
          printf("Q2[%d] :\n",i);
          printU256Number(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS]);
          printU256Number(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS+NWORDS_256BIT]);
          printU256Number(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS+2*NWORDS_256BIT]);
          printU256Number(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS+3*NWORDS_256BIT]);
          printU256Number(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS+4*NWORDS_256BIT]);
          printU256Number(&z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS+5*NWORDS_256BIT]);
        }
        */
       
     }
  }

}

void ec_jacsc1mul_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last=0)
{
  uint32_t outdims = ECP_JAC_OUTDIMS;
  uint32_t indims = ECP_JAC_OUTDIMS;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif

  for (uint32_t i=0; i<n; i++){
     ec_jacscmul_h(&z[i*NWORDS_256BIT*outdims], &x[i*NWORDS_256BIT], &x[n*NWORDS_256BIT], 1, pidx, add_last);
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

void ec2_jacsc1mul_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last=0)
{
  uint32_t outdims = ECP2_JAC_OUTDIMS;
  uint32_t indims = ECP2_JAC_OUTDIMS;

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0; i<n; i++){
     ec2_jacscmul_h(&z[i*NWORDS_256BIT*outdims], &x[i*NWORDS_256BIT], &x[n*NWORDS_256BIT], 1, pidx, add_last);
  }
}

uint32_t msbu256_h(uint32_t *x)
{
  int i,j;
  uint32_t count=0, n; 
  for(i=NWORDS_256BIT-1; i >= 0; i--){
     #if 0
     for(j=31; j >= 0; j--){
        if (x[i] & (1 << j)) { 
           return count;
        } else {
          count++;
        }
     }
    #else
    n = __builtin_clz(x[i]);
    count += n;
    if (n != 32) return count;
    #endif
  }
}

void ec_jac2aff_h(uint32_t *y, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t strip_last=0)
{
  const uint32_t *One = CusnarksOneMontGet(pidx);
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t ndims = ECP_JAC_OUTDIMS;
  uint32_t zero[] = {0,0,0,0,0,0,0,0};
  if (strip_last == 1){
     ndims = ECP_JAC_INDIMS;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0; i< n; i++){
     uint32_t tid = omp_get_thread_num();
     if (equ256_h(&x[i*ECP_JAC_OUTDIMS*NWORDS_256BIT+2*NWORDS_256BIT], zero)){
           memmove(&y[i*ndims*NWORDS_256BIT],
                &ECInf[(pidx*MISC_K_N+MISC_K_INF)*NWORDS_256BIT],
                sizeof(uint32_t)*ndims*NWORDS_256BIT);
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
  const uint32_t *One = CusnarksOneMont2Get(pidx);
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t ndims = ECP2_JAC_OUTDIMS;
  uint32_t zero[] = {0,0,0,0,0,0,0,0};
  if (strip_last == 1){
     ndims = ECP2_JAC_INDIMS;
  }


  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0; i< n; i++){
     uint32_t tid = omp_get_thread_num();
     if (equ256_h(&x[i*ECP2_JAC_OUTDIMS*NWORDS_256BIT+4*NWORDS_256BIT], zero) &&
        equ256_h(&x[i*ECP2_JAC_OUTDIMS*NWORDS_256BIT+5*NWORDS_256BIT], zero)){
        memmove(&y[i*ndims*NWORDS_256BIT],
                &ECInf[(pidx*MISC_K_N+MISC_K_INF2)*NWORDS_256BIT],
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
        memcpy(&y[4*NWORDS_256BIT+i*ECP2_JAC_OUTDIMS*NWORDS_256BIT], One, sizeof(uint32_t)*NWORDS_256BIT*2);
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

void ec_jacaddreduce_finish_h(void *args)
{
  jacadd_reduced_t *wargs = (jacadd_reduced_t *)args;
  ec_jacaddreduce_h(
          wargs->out_ep,
          &utils_EPout[wargs->thread_id*ECP_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
          wargs->max_threads,
          wargs->pidx, 1, 0, 1, EC_JACREDUCE_FLAGS_REDUCTION | EC_JACREDUCE_FLAGS_FINISH);

  /*
  printU256Number(wargs->out_ep);
  printU256Number(&wargs->out_ep[NWORDS_256BIT]);
  */

}

void ec2_jacaddreduce_finish_h(void *args)
{
  jacadd_reduced_t *wargs = (jacadd_reduced_t *)args;
  ec2_jacaddreduce_h(
          wargs->out_ep,
          &utils_EPout[wargs->thread_id*ECP2_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
          wargs->max_threads,
          wargs->pidx, 1, 0, 1, EC_JACREDUCE_FLAGS_REDUCTION | EC_JACREDUCE_FLAGS_FINISH);

 /*
  printU256Number(wargs->out_ep);
  printU256Number(&wargs->out_ep[NWORDS_256BIT]);
  printU256Number(&wargs->out_ep[2*NWORDS_256BIT]);
  printU256Number(&wargs->out_ep[3*NWORDS_256BIT]);
 */
}

void ec_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last, uint32_t flags)
{
  uint32_t i;
  uint32_t outdims = ECP_JAC_OUTDIMS, indims = ECP_JAC_OUTDIMS;
  const uint32_t *One;
  uint32_t start_idx=1;

  One = CusnarksOneMontGet((mod_t)pidx);

  if (strip_last){
    outdims = ECP_JAC_INDIMS;
  }
  if (add_in){
    indims = ECP_JAC_INDIMS;
  }
  if ((flags & EC_JACREDUCE_FLAGS_REDUCTION) && !(flags & EC_JACREDUCE_FLAGS_FINISH)){
    start_idx = 0;
  } 

  if (n == 1) {
     if (flags & EC_JACREDUCE_FLAGS_FINISH) {
         memmove(utils_EPout,x,indims*NWORDS_256BIT*sizeof(uint32_t));
     } else if (flags & EC_JACREDUCE_FLAGS_REDUCTION && !(flags & EC_JACREDUCE_FLAGS_INIT)){
         ec_jacadd_h(z, z, x,pidx);
     } else if (flags & EC_JACREDUCE_FLAGS_REDUCTION){
         memmove(z,x,indims*NWORDS_256BIT*sizeof(uint32_t));
     }
  }

  if ( (n > 1) && (flags & EC_JACREDUCE_FLAGS_INIT) ){
    start_idx = 2;
    ec_jacadd_h(z, x, &x[ECP_JAC_OUTDIMS << NWORDS_256BIT_SHIFT], pidx);

  }

  if ((n > 1) && (flags & EC_JACREDUCE_FLAGS_REDUCTION))  {

      for (uint32_t j=start_idx; j<n; j++){
        if (flags & EC_JACREDUCE_FLAGS_FINISH){
           ec_jacadd_h(utils_EPout,
                       utils_EPout,
                       &x[(j*ECP_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],pidx);
        } else {
          ec_jacadd_h(z, z, &x[(j*ECP_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],pidx);
        }
      }
   }

   if (flags & EC_JACREDUCE_FLAGS_FINISH){
     if (to_aff){
       ec_jac2aff_h(z,utils_EPout,1,pidx, strip_last);
     } else {
       memmove(z,utils_EPout,outdims*NWORDS_256BIT*sizeof(uint32_t));
     }
   }

}

void ec_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last)
{
  uint32_t i;
  uint32_t outdims = ECP_JAC_OUTDIMS, indims = ECP_JAC_OUTDIMS;
  const uint32_t *One;
  uint32_t vars_per_thread = n, vars_last_thread=n;
  uint32_t n_threads = MIN(n, get_nprocs_h());

  // set number of threads and vars per thread depending on nvars
  if (n >= n_threads*2){
    vars_per_thread = n/n_threads;
    vars_last_thread = n - (n_threads -1)*vars_per_thread;
  } else {
    n_threads = 1;
  }
  
  omp_set_num_threads(n_threads);
  One = CusnarksOneMontGet((mod_t)pidx);

  if (strip_last){
    outdims = ECP_JAC_INDIMS;
  }
  if (add_in){
    indims = ECP_JAC_INDIMS;
  }
  
  if (n > 1) {
    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for (i =0; i < n_threads; i++){
       memcpy(&utils_EPin[(i*ECP_JAC_OUTDIMS<<NWORDS_256BIT_SHIFT) +  (ECP_JAC_INDIMS<<NWORDS_256BIT_SHIFT)],
              One, sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);
       memcpy(&utils_EPin[i*ECP_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
              &x[i*vars_per_thread * indims << NWORDS_256BIT_SHIFT],
              indims*sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);

       memcpy(&utils_EPout[(i*ECP_JAC_OUTDIMS<<NWORDS_256BIT_SHIFT) + (ECP_JAC_INDIMS << NWORDS_256BIT_SHIFT)],
              One, sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);
       memcpy(&utils_EPout[i*ECP_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
              &x[(i*vars_per_thread * indims << NWORDS_256BIT_SHIFT) + (indims<<NWORDS_256BIT_SHIFT)],
              indims*sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);

       ec_jacadd_h(&utils_EPout[i*ECP_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
                   &utils_EPin[i*ECP_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
                   &utils_EPout[i*ECP_JAC_OUTDIMS << NWORDS_256BIT_SHIFT], pidx);
    }

  } else {
    memcpy(utils_EPout,x,ECP_JAC_INDIMS*sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);
    memcpy(&utils_EPout[ECP_JAC_OUTDIMS<<NWORDS_256BIT_SHIFT], One, sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);
  }

  if (n > 1) {

    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for(i=0; i<n_threads;i++){
      uint32_t tid = omp_get_thread_num();
      for (uint32_t j=2; j<vars_last_thread; j++){
        if (j == vars_last_thread && tid < n_threads-1){
          break;
        }
        memcpy(&utils_EPin[(tid*ECP_JAC_OUTDIMS<<NWORDS_256BIT_SHIFT) +  (ECP_JAC_INDIMS<<NWORDS_256BIT_SHIFT)],
                One, sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);
        memcpy(&utils_EPin[(tid*ECP_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],
                 &x[(tid*vars_per_thread*indims<<NWORDS_256BIT_SHIFT) + (j*indims<<NWORDS_256BIT_SHIFT)],
                 indims*sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);

        ec_jacadd_h(&utils_EPout[(tid*ECP_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],
                    &utils_EPout[(tid*ECP_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],
                    &utils_EPin[(tid*ECP_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],pidx);
      }
    }
    /*
    for(i=0; i<4;i++){
       printf("X2[%d]\n",i);
       printU256Number(&utils_EPout[i*ECP2_JAC_OUTDIMS*NWORDS_256BIT]);
       printU256Number(&utils_EPout[i*ECP2_JAC_OUTDIMS*NWORDS_256BIT+2*NWORDS_256BIT]);
       printU256Number(&utils_EPout[i*ECP2_JAC_OUTDIMS*NWORDS_256BIT+4*NWORDS_256BIT]);
    }
    */

    for (i =1; i < n_threads; i++){
       ec_jacadd_h(utils_EPout,
                   utils_EPout,
                   &utils_EPout[i*ECP_JAC_OUTDIMS << NWORDS_256BIT_SHIFT], pidx);
    }
  }

   /*
       printf("XF2\n");
       printU256Number(&utils_EPout[0]);
       printU256Number(&utils_EPout[2*NWORDS_256BIT]);
       printU256Number(&utils_EPout[4*NWORDS_256BIT]);
   */
   if (to_aff){
     ec_jac2aff_h(z,utils_EPout,1,pidx, strip_last);
   } else {
     memcpy(z,utils_EPout,outdims*NWORDS_256BIT*sizeof(uint32_t));
   }
}

void ec2_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last, uint32_t flags)
{
  uint32_t i;
  uint32_t outdims = ECP2_JAC_OUTDIMS, indims = ECP2_JAC_OUTDIMS;
  const uint32_t *One;
  uint32_t start_idx=1;

  One = CusnarksOneMontGet((mod_t)pidx);

  if (strip_last){
    outdims = ECP2_JAC_INDIMS;
  }
  if (add_in){
    indims = ECP2_JAC_INDIMS;
  }
  
  if ((flags & EC_JACREDUCE_FLAGS_REDUCTION) && !(flags & EC_JACREDUCE_FLAGS_FINISH)){
    start_idx = 0;
  } 
  if ( (n > 1) && (flags & EC_JACREDUCE_FLAGS_INIT) ){
    start_idx = 2;
    ec2_jacadd_h(z, x, &x[ECP2_JAC_OUTDIMS << NWORDS_256BIT_SHIFT], pidx);

  } 

  if (n == 1){
     if (flags & EC_JACREDUCE_FLAGS_FINISH) {
       memmove(utils_EPout,x,indims*NWORDS_256BIT*sizeof(uint32_t));
     } else if (flags & EC_JACREDUCE_FLAGS_REDUCTION){
          ec2_jacadd_h(z, z, x,pidx);
     }
  }

  if ((n > 1) && (flags & EC_JACREDUCE_FLAGS_REDUCTION))  {

      for (uint32_t j=start_idx; j<n; j++){
        if (flags & EC_JACREDUCE_FLAGS_FINISH){
           ec2_jacadd_h(utils_EPout,
                       utils_EPout,
                       &x[(j*ECP2_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],pidx);
        } else {
          ec2_jacadd_h(z, z, &x[(j*ECP2_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],pidx);
        }
      }
   }

   if (flags & EC_JACREDUCE_FLAGS_FINISH){
     if (to_aff){
       ec2_jac2aff_h(z,utils_EPout,1,pidx, strip_last);
     } else {
       memmove(z,utils_EPout,outdims*NWORDS_256BIT*sizeof(uint32_t));
     }
   }

}


void ec2_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last)
{
  uint32_t i;
  uint32_t outdims = ECP2_JAC_OUTDIMS, indims = ECP2_JAC_OUTDIMS;
  const uint32_t *One;
  uint32_t vars_per_thread = n, vars_last_thread=n;
  uint32_t n_threads = MIN(n, get_nprocs_h());

  // set number of threads and vars per thread depending on nvars
  if (n >= n_threads*2){
    vars_per_thread = n/n_threads;
    vars_last_thread = n - (n_threads -1)*vars_per_thread;
  } else {
    n_threads = 1;
  }
  
  omp_set_num_threads(n_threads);
  One = CusnarksOneMont2Get((mod_t)pidx);

  if (strip_last){
    outdims = ECP2_JAC_INDIMS;
  }
  if (add_in){
    indims = ECP2_JAC_INDIMS;
  }
  
  if (n > 1) {
    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for (i =0; i < n_threads; i++){
       memcpy(&utils_EPin[(i*ECP2_JAC_OUTDIMS<<NWORDS_256BIT_SHIFT) +  (ECP2_JAC_INDIMS<<NWORDS_256BIT_SHIFT)],
              One, sizeof(uint32_t)*2<<NWORDS_256BIT_SHIFT);
       memcpy(&utils_EPin[i*ECP2_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
              &x[i*vars_per_thread * indims << NWORDS_256BIT_SHIFT],
              indims*sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);

       memcpy(&utils_EPout[(i*ECP2_JAC_OUTDIMS<<NWORDS_256BIT_SHIFT) + (ECP2_JAC_INDIMS << NWORDS_256BIT_SHIFT)],
              One, sizeof(uint32_t)*2<<NWORDS_256BIT_SHIFT);
       memcpy(&utils_EPout[i*ECP2_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
              &x[(i*vars_per_thread * indims << NWORDS_256BIT_SHIFT) + (indims<<NWORDS_256BIT_SHIFT)],
              indims*sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);

       ec2_jacadd_h(&utils_EPout[i*ECP2_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
                   &utils_EPin[i*ECP2_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
                   &utils_EPout[i*ECP2_JAC_OUTDIMS << NWORDS_256BIT_SHIFT], pidx);
    }

  } else {
    memcpy(utils_EPout,x,ECP2_JAC_INDIMS*sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);
    memcpy(&utils_EPout[ECP2_JAC_OUTDIMS<<NWORDS_256BIT_SHIFT], One, sizeof(uint32_t)*2<<NWORDS_256BIT_SHIFT);
  }

  if (n > 1) {

    #ifndef TEST_MODE
      #pragma omp parallel for if(parallelism_enabled)
    #endif
    for(i=0; i<n_threads;i++){
      uint32_t tid = omp_get_thread_num();
      for (uint32_t j=2; j<vars_last_thread; j++){
        if (j == vars_last_thread && tid < n_threads-1){
          break;
        }
        memcpy(&utils_EPin[(tid*ECP2_JAC_OUTDIMS<<NWORDS_256BIT_SHIFT) +  (ECP2_JAC_INDIMS<<NWORDS_256BIT_SHIFT)],
                One, sizeof(uint32_t)*2<<NWORDS_256BIT_SHIFT);
        memcpy(&utils_EPin[(tid*ECP2_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],
                 &x[(tid*vars_per_thread*indims<<NWORDS_256BIT_SHIFT) + (j*indims<<NWORDS_256BIT_SHIFT)],
                 indims*sizeof(uint32_t)<<NWORDS_256BIT_SHIFT);

        ec2_jacadd_h(&utils_EPout[(tid*ECP2_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],
                    &utils_EPout[(tid*ECP2_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],
                    &utils_EPin[(tid*ECP2_JAC_OUTDIMS)<<NWORDS_256BIT_SHIFT],pidx);
      }
    }

    for (i =1; i < n_threads; i++){
       ec2_jacadd_h(utils_EPout,
                   utils_EPout,
                   &utils_EPout[i*ECP2_JAC_OUTDIMS << NWORDS_256BIT_SHIFT], pidx);
    }
  }

   if (to_aff){
     ec2_jac2aff_h(z,utils_EPout,1,pidx, strip_last);
   } else {
     memcpy(z,utils_EPout,outdims*NWORDS_256BIT*sizeof(uint32_t));
   }
}

uint32_t ec2_jacreduce_init_h(uint32_t **ectable, uint32_t **scmul, uint32_t n, uint32_t order)
{
  uint32_t ntables =  (order + n - 1) / order;

  //initialize tables and scmul values
  *ectable = (uint32_t *)malloc((1ull<<order) * ntables * ECP2_JAC_OUTDIMS * NWORDS_256BIT * sizeof(uint32_t));
  *scmul = (uint32_t *)malloc(ntables * ECP2_JAC_OUTDIMS * NWORDS_256BIT * sizeof(uint32_t));

  return ntables;
}

void ec_jacreduce_del_h(uint32_t *ectable, uint32_t *scmul)
{
  free(ectable);
  free(scmul);
}

void ec2_jacreduce_server_h(jacadd_reduced_t *args)
{
  ec_jacreduce_server_h(args);
}

void ec_jacreduce_server_h(jacadd_reduced_t *args)
{
  uint32_t start_idx, last_idx;
  uint32_t vars_per_thread = args->n;
  uint32_t max_threads = get_nprocs_h();
  uint32_t compute_table = args->ec_table==NULL ? 1 : 0;
  uint32_t outdims = ECP_JAC_OUTDIMS;
  uint32_t indims = ECP_JAC_INDIMS;
  uint32_t order = args->order;

  if (args->ec2){
    outdims = ECP2_JAC_OUTDIMS;
    indims  = ECP2_JAC_INDIMS;
  }

  // configure max threads 
  if (args->max_threads == 0){
    args->max_threads = max_threads;
  } else {
    args->max_threads = MIN(args->max_threads, max_threads);
  }

  // set number of threads and vars per thread depending on nvars
  if (compute_table){
    if (args->n >= args->max_threads*(order << EC_JACREDUCE_BATCH_SIZE)*order){
      vars_per_thread = args->n/args->max_threads;
      vars_per_thread = ((vars_per_thread + (order << EC_JACREDUCE_BATCH_SIZE) * order -1) / 
                          ((order << EC_JACREDUCE_BATCH_SIZE) * order)) * 
                          ((order << EC_JACREDUCE_BATCH_SIZE) * order);
    } else {
      args->max_threads = 1;
    }
  }  else {
      vars_per_thread = EC_JACREDUCE_TABLE_LEN / (args->max_threads * order) * order;
  }

  if (!utils_mproc_init) {
    exit(1);
  }
  #ifndef PARALLEL_EN
    exit(1);
  #endif

  pthread_t *workers = (pthread_t *) malloc(args->max_threads * sizeof(pthread_t));
  jacadd_reduced_t *w_args  = (jacadd_reduced_t *)malloc(args->max_threads * sizeof(jacadd_reduced_t));
  if (pthread_barrier_init(&utils_barrier, NULL, args->max_threads) != 0){
     exit(1);
  }
 
  /*
  printf("N threads : %d\n", args->max_threads);
  printf("N vars    : %d\n", args->n);
  printf("Vars per thread : %d\n", vars_per_thread);
  printf("pidx : %d\n", args->pidx);
  printf("compute table: %d\n", compute_table);
  printf("filename : %d\n",args->filename);
  printf("total words : %ld\n",args->total_words);
  printf("offset : %lld\n",args->offset);
 */
 
  
  
  /*
  for(uint32_t i=0; i<args->n; i++){
    printf("%d\n",i);
    printU256Number(&args->scl[i*NWORDS_256BIT]);
    printU256Number(&args->x[i*NWORDS_256BIT*indims]);
    printU256Number(&args->x[i*NWORDS_256BIT*indims+NWORDS_256BIT]);
  }
  */
  
  for(uint32_t i=0; i< args->max_threads; i++){
     start_idx = i * vars_per_thread;
     last_idx = (i+1) * vars_per_thread;
     if (i == args->max_threads - 1){
       if (compute_table){
          last_idx = args->n;
       } else {
          last_idx = EC_JACREDUCE_TABLE_LEN;
       }
     }
     memcpy(&w_args[i], args, sizeof(jacadd_reduced_t));

     w_args[i].start_idx = start_idx;
     w_args[i].last_idx = last_idx;
     w_args[i].thread_id = i;
    
   /*         
     printf("Thread : %d, start_idx : %d, end_idx : %d\n",
             w_args[i].thread_id, 
             w_args[i].start_idx,
             w_args[i].last_idx);   
  */
    
  
  }

  parallelism_enabled = 0;

  if (compute_table){
    launch_client_h(ec_jacreduce_batch_h, workers,(void *) w_args, sizeof(jacadd_reduced_t), args->max_threads);
  } else {
    if (pthread_cond_init(&utils_cond, NULL) != 0){
     exit(1);
    }

    pthread_t *workers_table;
    ec_table_desc_t *w_table_args;

    if (w_args->filename != NULL){
      workers_table = (pthread_t *) malloc(1 * sizeof(pthread_t));
      w_table_args  = (ec_table_desc_t *)malloc(1 * sizeof(ec_table_desc_t));
      w_table_args->filename = w_args->filename;
      w_table_args->ec_table = w_args->ec_table;
      w_table_args->offset = w_args->offset;
      w_table_args->total_words = w_args->total_words;
      w_table_args->ec2 = w_args->ec2;
      w_table_args->order = w_args->order;
 
      launch_client_h(ec_read_table_h, workers_table,(void *) w_table_args, sizeof(ec_table_desc_t), 1, 1);

      /*
      printf("Table Offset[%d] : %d\n",w_args->thread_id, w_args->offset);
      printf("EC2[%d] : %d\n",w_args->thread_id, w_args->ec2);
      */
      
    } 

    memset(
            utils_EPin, 0, outdims * NWORDS_256BIT * EC_JACREDUCE_TABLE_LEN * sizeof(uint32_t)
          );
    launch_client_h(ec_jacreduce_batch_precomputed_h, workers,(void *) w_args, sizeof(jacadd_reduced_t), args->max_threads);

    pthread_cond_destroy(&utils_cond);

    if (w_args->filename != NULL){
      free(workers_table);
      free(w_table_args);
    }
  }

  pthread_barrier_destroy(&utils_barrier);
  
  free(workers);
  free(w_args);

  parallelism_enabled = 1;

  return; 

}

void *ec_jacreduce_batch_h(void *args)
{
  #define REDUCE_MODE 0
  jacadd_reduced_t *wargs = (jacadd_reduced_t *)args;
  uint32_t order = wargs->order;
  #if REDUCE_MODE
  uint32_t n_batches = (wargs->last_idx - wargs->start_idx + (order << EC_JACREDUCE_BATCH_SIZE)*order - 1)/((order << EC_JACREDUCE_BATCH_SIZE)*order);
  #else
  uint32_t n_batches = (wargs->n - 0 + (order << EC_JACREDUCE_BATCH_SIZE)*order - 1)/((order << EC_JACREDUCE_BATCH_SIZE)*order);
  #endif
  uint32_t flags[] = {EC_JACREDUCE_FLAGS_INIT | EC_JACREDUCE_FLAGS_REDUCTION, 
                      EC_JACREDUCE_FLAGS_REDUCTION};
  uint32_t compute_table = wargs->ec_table==NULL ? 1 : 0;
  uint32_t nsamples_offset=0;
  uint32_t nsamples;
  uint32_t indims = ECP_JAC_INDIMS;
  uint32_t outdims = ECP_JAC_OUTDIMS;
  void (*ec_jacscmul_opt_cb) (uint32_t *, uint32_t *, uint32_t *, uint32_t *,
                uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) = &ec_jacscmul_opt_h;
  void (*ec_jacaddreduce_cb)(uint32_t *, uint32_t *, uint32_t, uint32_t , uint32_t , uint32_t , uint32_t , uint32_t ) = &ec_jacaddreduce_h;
  void (*ec_jacaddreduce_finish_cb)(void *) = &ec_jacaddreduce_finish_h;

  if (wargs->ec2){
    indims = ECP2_JAC_INDIMS;
    outdims = ECP2_JAC_OUTDIMS;
    ec_jacscmul_opt_cb = &ec2_jacscmul_opt_h;
    ec_jacaddreduce_cb = &ec2_jacaddreduce_h;
    ec_jacaddreduce_finish_cb = &ec2_jacaddreduce_finish_h;
  }
  uint32_t *table_ptr = &utils_ectable[wargs->thread_id * (order << EC_JACREDUCE_BATCH_SIZE)*outdims<<(NWORDS_256BIT_SHIFT+order)];
  uint32_t *EPout = &utils_EPout[wargs->thread_id*outdims << NWORDS_256BIT_SHIFT];
  uint32_t *EPin  = &utils_EPin[wargs->thread_id*(order << EC_JACREDUCE_BATCH_SIZE)*outdims << NWORDS_256BIT_SHIFT];

  //printf("[%d] - N batches : %d %d\n",wargs->thread_id, n_batches, wargs->ec2);

  #if REDUCE_MODE
  for (uint32_t i=0; i < n_batches; i++){
    nsamples = MIN((order << EC_JACREDUCE_BATCH_SIZE)*order,(wargs->last_idx - wargs->start_idx) - nsamples_offset);
    if (i < n_batches - 1){
       __builtin_prefetch(&wargs->scl[(wargs->start_idx + nsamples_offset + nsamples)<<NWORDS_256BIT_SHIFT],
       __builtin_prefetch( &wargs->x[(wargs->start_idx + nsamples_offset + nsamples)*indims<<NWORDS_256BIT_SHIFT],
    }
  #else
  for (uint32_t i=0; i < n_batches; i+=wargs->max_threads){
    if (i < n_batches - wargs->max_threads){
       __builtin_prefetch(&wargs->scl[(i + wargs->max_threads + wargs->thread_id)*(order << EC_JACREDUCE_BATCH_SIZE)*order<<NWORDS_256BIT_SHIFT]);
       __builtin_prefetch(&wargs->x[(i + wargs->max_threads + wargs->thread_id)*(order << EC_JACREDUCE_BATCH_SIZE)*order*indims<<NWORDS_256BIT_SHIFT]);
    }
    nsamples = MIN((order << EC_JACREDUCE_BATCH_SIZE)*order*wargs->max_threads,(wargs->n - 0) - nsamples_offset);
    if (nsamples >= wargs->thread_id * (order << EC_JACREDUCE_BATCH_SIZE)*order){  
      nsamples = MIN((order << EC_JACREDUCE_BATCH_SIZE)*order, nsamples - wargs->thread_id * (order << EC_JACREDUCE_BATCH_SIZE)*order);
    } else {
       break;
    }
  #endif
    
   /* printf("%d-%d-%d-%d\n",wargs->thread_id,i,
                              nsamples,
                              MIN((order << EC_JACREDUCE_BATCH_SIZE),(nsamples+order-1)/(order)));
   */
   

    ec_jacscmul_opt_cb(
           EPin,
           #if REDUCE_MODE
           &wargs->scl[(wargs->start_idx + nsamples_offset)<<NWORDS_256BIT_SHIFT],
           &wargs->x[(wargs->start_idx + nsamples_offset)*indims<<NWORDS_256BIT_SHIFT],
           #else
           &wargs->scl[(i + wargs->thread_id)*(order << EC_JACREDUCE_BATCH_SIZE)*order<<NWORDS_256BIT_SHIFT],
           &wargs->x[(i + wargs->thread_id)*(order << EC_JACREDUCE_BATCH_SIZE)*order*indims<<NWORDS_256BIT_SHIFT],
           #endif
           table_ptr,
           nsamples,
           order, wargs->pidx, 1,1);
    
    ec_jacaddreduce_cb(
          EPout,
          EPin,
          MIN((order << EC_JACREDUCE_BATCH_SIZE),(nsamples + order - 1)/order),
          wargs->pidx, 0, 0, 0, flags[i>0]);
    

    #if REDUCE_MODE
    nsamples_offset += (order << EC_JACREDUCE_BATCH_SIZE)*order;
    #else
    nsamples_offset += (order << EC_JACREDUCE_BATCH_SIZE)*order*wargs->max_threads;
    #endif
  }
  util_wait_h(wargs->thread_id, ec_jacaddreduce_finish_cb, wargs);

  return NULL;
}


void *ec_read_table_h(void *args)
{
  ec_table_desc_t *wargs = (ec_table_desc_t *)args;
  t_uint64 total_words_read = 0;
  uint32_t words_read=0;
  uint32_t ec_table_offset[] = {0, (U256_BSELM << EC_JACREDUCE_BATCH_SIZE) * ECP_JAC_INDIMS * NWORDS_256BIT << U256_BSELM};
  uint32_t n_words = (wargs->order<<EC_JACREDUCE_BATCH_SIZE) * ECP_JAC_INDIMS * NWORDS_256BIT << wargs->order;
  uint32_t ec_table_idx=1;

  if (wargs->ec2){
     ec_table_offset[1] = (U256_BSELM<<EC_JACREDUCE_BATCH_SIZE) * ECP2_JAC_INDIMS * NWORDS_256BIT << U256_BSELM;
     n_words = (wargs->order<<EC_JACREDUCE_BATCH_SIZE) * ECP2_JAC_INDIMS * NWORDS_256BIT << wargs->order;
  }
  int err;
  
  // open file
  FILE *ifp = fopen(wargs->filename,"rb");

  // go to initial offset
  err = fseek(ifp, wargs->offset , SEEK_SET);
  //printf("Initial offset (words) : %lld, err : %d, File : %s\n",wargs->offset, err,wargs->filename);

  //while (total_words_read <= wargs->total_words){
  while (total_words_read < wargs->total_words){
     //wait till ready
     pthread_mutex_lock(&utils_lock);
     //printf("Waiting for signal(%d-%d, %d/%d)...!\n",utils_ectable_ready, ec_table_idx, total_words_read, wargs->total_words);
     while(utils_ectable_ready){
       pthread_cond_wait(&utils_cond, &utils_lock);
     }
     words_read = fread(&utils_ectable[ec_table_offset[ec_table_idx]], sizeof(uint32_t), n_words, ifp);
     total_words_read += words_read;
     //printf("Read %d words. New Offset %d/%lld. Words read : %d\n",n_words, total_words_read, wargs->total_words, words_read); 
     ec_table_idx ^= 1;
     utils_ectable_ready = 1;
     pthread_cond_signal(&utils_cond);
     pthread_mutex_unlock(&utils_lock);
     if ( words_read == 0 || utils_done){   
        break;
     }
      
     //printf("Signal received. Continue ...!\n" );
  }
  //printf("File read complete\n");
  fclose(ifp);

  return NULL;
}
//

void *ec_jacreduce_batch_precomputed_h(void *args)
{
  jacadd_reduced_t *wargs = (jacadd_reduced_t *)args;
  utils_done = 0;
  uint32_t i,b, n_batches;
  int32_t j, start_msb, last_msb;
  uint32_t order = wargs->order;
  uint32_t *utils_EPin_ptr = utils_EPin;
  uint32_t ec_table_offset[] = {0, (U256_BSELM << EC_JACREDUCE_BATCH_SIZE) * ECP_JAC_INDIMS * NWORDS_256BIT << U256_BSELM};
  uint32_t ec_table_idx=1;
  uint32_t *utils_ectable_ptr = wargs->ec_table;  
  uint32_t torder1 = (order == 1 && wargs->filename == NULL);

  uint32_t indims = ECP_JAC_INDIMS;
  uint32_t outdims = ECP_JAC_OUTDIMS;
  uint32_t yoffset = ECP_JAC_OUTYOFFSET;
  void (*ec_jacaddmixed_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t) = &ec_jacaddmixed_h;

  if (wargs->ec2){
    indims = ECP2_JAC_INDIMS;
    outdims = ECP2_JAC_OUTDIMS;
    yoffset = ECP2_JAC_OUTYOFFSET;
    ec_table_offset[1] = (U256_BSELM << EC_JACREDUCE_BATCH_SIZE) * ECP2_JAC_INDIMS * NWORDS_256BIT << U256_BSELM;
    ec_jacaddmixed_cb = &ec2_jacaddmixed_h;
  }

  // init table
  last_msb = wargs->start_idx;
  start_msb = wargs->last_idx-1;
  const  uint32_t *One = CusnarksOneMontGet((mod_t)wargs->pidx);
  for (j=start_msb; j>=last_msb; j--){
        memcpy(&utils_EPin_ptr[j*outdims*NWORDS_256BIT+yoffset], One, NWORDS_256BIT * sizeof(uint32_t));
  }
 
  for (i=0; i < wargs->n; i += order){

     if (i % ( (order << EC_JACREDUCE_BATCH_SIZE) * order) == 0 && wargs->filename != NULL){
        ec_table_idx ^= 1;
        if (i == 0){
          utils_ectable_ptr = wargs->ec_table;
        } else {
          utils_ectable_ptr = &utils_ectable[ec_table_offset[ec_table_idx]];
        }
        //printf("Idx : %d-%d,%d/%d\n",wargs->thread_id, ec_table_idx, i,wargs->n);
        util_wait_h(wargs->thread_id, ec_inittable_ready_h, wargs);
        //load next data batch from tables
     }
 
     if (i <= wargs->n - order){
       __builtin_prefetch(&wargs->scl[(i+1)*NWORDS_256BIT]);
       __builtin_prefetch(&utils_ectable_ptr[(i+1)*indims << (NWORDS_256BIT_SHIFT + order - torder1)]); 
     }

     // prefetch : wargs->scl, utils_ectable_ptr[indims << ]
     for (j=start_msb; j>=last_msb ; j--){
        b = getbitu256g_h(&wargs->scl[i * NWORDS_256BIT], j, order);
     
        if (b){
          ec_jacaddmixed_cb(
            &utils_EPin_ptr[j*outdims*NWORDS_256BIT],
            &utils_ectable_ptr[(b-torder1)*indims*NWORDS_256BIT],
            &utils_EPin_ptr[j*outdims*NWORDS_256BIT], wargs->pidx);
        }
      
     }
     utils_ectable_ptr += indims << (NWORDS_256BIT_SHIFT + order - torder1);
  }
  utils_done = 1;

  util_wait_h(wargs->thread_id, ec_jacdouble_finish_h, wargs);
  utils_ectable_ready = 1;

  return NULL;
}

static void ec_print_EPin(void *args)
{
       for (uint32_t j=0; j<256 ; j++){
          printf("EP idx : %d\n",j);
          printU256Number(&utils_EPin[j*ECP_JAC_OUTDIMS*NWORDS_256BIT]);
          printU256Number(&utils_EPin[j*ECP_JAC_OUTDIMS*NWORDS_256BIT+NWORDS_256BIT]);
          printU256Number(&utils_EPin[j*ECP_JAC_OUTDIMS*NWORDS_256BIT+2*NWORDS_256BIT]);
       }
}
static void ec_inittable_ready_h(void *args)
{
  //printf("Switching tables(%d)\n",utils_ectable_ready);
  pthread_mutex_lock(&utils_lock);
  //printf("Table is ready 0\n");
  while(!utils_ectable_ready){
    pthread_cond_wait(&utils_cond, &utils_lock);
  }
  //printf("Table is ready\n");
  utils_ectable_ready = 0;
  pthread_cond_signal(&utils_cond);
  pthread_mutex_unlock(&utils_lock);
}

static void ec_jacdouble_finish_h(void *args)
{
  jacadd_reduced_t *wargs = (jacadd_reduced_t *)args;
  uint32_t i;
  uint32_t outdims = ECP_JAC_OUTDIMS;
  void (*ec_jacdouble_cb)(uint32_t *, uint32_t *, uint32_t) = &ec_jacdouble_h;
  void (*ec_jacadd_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t) = &ec_jacadd_h;
  void (*ec_jac2aff_cb)(uint32_t *, uint32_t *, uint32_t, uint32_t, uint32_t) = &ec_jac2aff_h;

  if (wargs->ec2){
    outdims = ECP2_JAC_OUTDIMS;
    ec_jacdouble_cb = &ec2_jacdouble_h;
    ec_jacadd_cb = &ec2_jacadd_h;
    ec_jac2aff_cb = &ec2_jac2aff_h;
  }

  uint32_t *P = &utils_EPin[(EC_JACREDUCE_TABLE_LEN - 1)*outdims * NWORDS_256BIT];

  /*
  printf("255\n");
  printU256Number(P);
  printU256Number(&P[NWORDS_256BIT]);
  printU256Number(&P[2*NWORDS_256BIT]);
  */

  for (i=EC_JACREDUCE_TABLE_LEN - 1; i>0 ; i--){
     ec_jacdouble_cb( P, P, wargs->pidx );
     ec_jacadd_cb( P, P, &utils_EPin[(i-1)*outdims * NWORDS_256BIT], wargs->pidx);

  /*
     printf("%d\n",i-1);
     printU256Number(&utils_EPin[(i-1)*ECP_JAC_OUTDIMS * NWORDS_256BIT]);
     printU256Number(&utils_EPin[(i-1)*ECP_JAC_OUTDIMS * NWORDS_256BIT+NWORDS_256BIT]);
     printU256Number(&utils_EPin[(i-1)*ECP_JAC_OUTDIMS * NWORDS_256BIT+2*NWORDS_256BIT]);
     printU256Number(P);
     printU256Number(&P[NWORDS_256BIT]);
     printU256Number(&P[2*NWORDS_256BIT]);
  */

  }
  ec_jac2aff_cb(wargs->out_ep, P, 1, wargs->pidx, 1);
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

  M_init_h(1<< CusnarksGetNRoots());
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
  M_free_h();
}

int createSharedMemBuf(void **shmem, unsigned long long size)
{
  int shmid;
  // give your shared memory an id, anything will do
  key_t key = SHMEM_WITNESS_KEY;
 
  // Setup shared memory
  if ((shmid = shmget(key, size, IPC_CREAT | 0666)) < 0)
  {
     return -1;
  }
  // Attached shared memory
  if ((*shmem = shmat(shmid, NULL, 0)) == (char *) -1)
  {
     return -1;
  }

  return shmid;
}

void destroySharedMemBuf(void *shmem, int shmid)
{
   // Detach and remove shared memory
   shmdt(shmem);
   shmctl(shmid, IPC_RMID, NULL);
}
	
void fail_h(void)
{
  assert(NULL);
}

