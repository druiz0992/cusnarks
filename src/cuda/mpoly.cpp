
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
// File name  : ntt.cpp
//
// Date       : 6/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//
// ------------------------------------------------------------------

#include <stdio.h>
#include <omp.h>

#include "types.h"
#include "constants.h"
#include "rng.h"
#include "log.h"
#include "utils_host.h"
#include "bigint.h"
#include "ff.h"
#include "ntt.h"
#include "mpoly.h"

#ifdef PARALLEL_EN
static  uint32_t parallelism_enabled =  1;
#else
static  uint32_t parallelism_enabled =  0;
#endif

void mpoly_addm(void *args);

static  pthread_cond_t utils_cond;        

/*
static uint32_t **utils_mpoly;

void mpoly_addm(void *args)
{
  mpoly_eval_t *wargs = (mpoly_eval_t *)args;
  t_addm addm_cb =  getcb_addm_h(wargs->pidx);
  uint32_t *pout = wargs->pout;

  printf("N coeff : %d, Max threads : %d, Pidx : %d\n", wargs->ncoeff, wargs->max_threads, wargs->pidx);

  for (uint32_t i=0; i <wargs->ncoeff; i++){
     for (uint32_t j=1; j < wargs->max_threads; j++){
       uint32_t *pout2 = utils_mpoly[j];
       addm_cb(&pout[i*NWORDS_FR], &pout[i*NWORDS_FR], &pout2[i*NWORDS_FR]);
     }
  }
}


void mpoly_init_h(uint32_t nroots)
{
  uint32_t nprocs = get_nprocs_h();

  utils_mpoly = (uint32_t **) malloc(nprocs * sizeof(uint32_t *));
  for (uint32_t i=1; i < nprocs; i++){
    utils_mpoly[i] = (uint32_t *) malloc((t_uint64) (nroots/2 ) * NWORDS_FR * sizeof(uint32_t)); 
  }
}

void mpoly_free_h()
{
  uint32_t nprocs = get_nprocs_h();
  for (uint32_t i=1; i < nprocs; i++){
    free(utils_mpoly[i]);
 }
 free(utils_mpoly);
}
*/

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
  if ((!args->max_threads) || (!utils_isinit_h())) {
    if (args->mode==0) {
      mpoly_eval_h((void *)args);
    } else {
      mpolys_eval_h((void *)args);
    }
    return;
  }

  #ifndef PARALLEL_EN
    if (args->mode==0) {
      mpoly_eval_h((void *)args);
    } else {
      mpolys_eval_h((void *)args);
    }
    return;
  #endif
  uint32_t nprocs = get_nprocs_h();
  int nthreads = args->max_threads > nprocs ? nprocs : args->max_threads;
  unsigned long long nvars = args->last_idx - args->start_idx;
  if (args->mode==1){
    nvars = args->ncoeff;
  }

  //printf("N threads : %d\n", nthreads);
  //printf("N vars    : %d\n", nvars);

  unsigned long long vars_per_thread = nvars/nthreads;
  uint32_t i;
  unsigned long long start_idx, last_idx;

  if (args->mode==1){
     vars_per_thread = vars_per_thread * ZKEY_COEFF_NWORDS;
  }
   
  pthread_t *workers = (pthread_t *) malloc(nthreads * sizeof(pthread_t));
  mpoly_eval_t *w_args  = (mpoly_eval_t *)malloc(nthreads * sizeof(mpoly_eval_t));
  init_barrier_h(args->max_threads);

  
  //printf ("Creating  %d threads, with %d vars per thread. Start idx: %d, Last idx %d\n",
   //       nthreads, vars_per_thread,args->start_idx, args->last_idx);

  for(i=0; i< nthreads; i++){
     start_idx = i * vars_per_thread;
     last_idx = (i+1) * vars_per_thread;
     /*
     if (i > 0){
        memset(utils_mpoly[i], 0, sizeof(uint32_t) * (1ull << CusnarksGetNRoots()));
     }
     */
     if ( (i == nthreads - 1) && (last_idx != nvars) ){
         last_idx = nvars;
         if (args->mode == 1){
           last_idx = args->last_idx;
         }
     }
     memcpy(&w_args[i], args, sizeof(mpoly_eval_t ));

     w_args[i].start_idx = start_idx;
     w_args[i].last_idx = last_idx;
     w_args[i].thread_id = i;

     //printf("Thread %d : start_idx : %d, last_idx : %d\n", i, w_args[i].start_idx,w_args[i].last_idx);
     if (args->mode == 0) {
        if ( pthread_create(&workers[i], NULL, &mpoly_eval_h, (void *) &w_args[i]) ){
          free(workers);
          free(w_args);
          exit(1);
        }
     } else {
        if ( pthread_create(&workers[i], NULL, &mpolys_eval_h, (void *) &w_args[i]) ){
          free(workers);
          free(w_args);
          exit(1);
        }
     }
  }

  for (i=0; i < nthreads; i++){
    pthread_join(workers[i], NULL);
  }
  del_barrier_h();

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
  uint32_t scl[NWORDS_FR];
  t_uint64 i,j;
  uint32_t zcoeff_v_in[NWORDS_FR], *zcoeff_v_out;
  t_uint64 zcoeff_d;
  t_uint64 accum_n_zcoeff=0;
  uint32_t *pout= args->pout;
  /*
  if (args->thread_id != 0) {
    pout= utils_mpoly[args->thread_id];
  } 
  */

  t_addm addm_cb =  getcb_addm_h(args->pidx);
  t_mulm mulm_cb =  getcb_mulm_h(args->pidx);
  t_tomont tom_cb = getcb_tomont_h(args->pidx);
  
  //TODO Change : If coeffs are accumulated, I don't need to do the accumulation
  //accum_n_zcoeff = args->pin[args->start_idx];
  
  for (i=0; i<args->start_idx; i++){
    accum_n_zcoeff += (t_uint64) args->pin[i+1];
  }
 
  zcoeff_d_offset = accum_n_zcoeff*(NWORDS_FR+1) +1 + n_zpoly;

  for (i=args->start_idx; i<args->last_idx; i++){
    tom_cb(scl, &args->scalar[i*NWORDS_FR]);
    n_zcoeff = (t_uint64) args->pin[1+i];
    accum_n_zcoeff += n_zcoeff;   
 
    zcoeff_v_offset = zcoeff_d_offset + n_zcoeff;

    for (j=0; j< n_zcoeff; j++){
       zcoeff_d = (t_uint64) args->pin[zcoeff_d_offset+j];
       zcoeff_v_out = &pout[zcoeff_d*NWORDS_FR];
       mulm_cb(zcoeff_v_in, &args->pin[zcoeff_v_offset+j*NWORDS_FR], scl);
       addm_cb(zcoeff_v_out, zcoeff_v_out, zcoeff_v_in);
    }
    zcoeff_d_offset = accum_n_zcoeff*(NWORDS_FR+1) +1 + n_zpoly;
  }

  return NULL;
}


void *mpolys_eval_h(void *vargs)
{
  mpoly_eval_t *args = (mpoly_eval_t *) vargs;
  uint32_t *pin = args->pin;
  uint32_t *pout, *coeff, *scl;
  uint32_t tmp[NWORDS_FR];
  uint32_t w_idx, p_idx;

  t_addm addm_cb =  getcb_addm_h(args->pidx);
  t_mulm mulm_cb =  getcb_mulm_h(args->pidx);
 
  //printf("last_idx : %d\n",args->last_idx); 
  for (uint32_t i=args->start_idx; i<args->last_idx; i+=ZKEY_COEFF_NWORDS){
      coeff = &pin[i];
      w_idx = coeff[ZKEY_COEFF_SIGNAL_OFFSET];
      scl = (uint32_t *)&args->scalar[w_idx*NWORDS_FR];
      if (coeff[ZKEY_COEFF_MATRIX_OFFSET] == 0) {
         p_idx = 0 + coeff[ZKEY_COEFF_CONSTRAINT_OFFSET];
      } else {
         p_idx = args->reduce_coeff + coeff[ZKEY_COEFF_CONSTRAINT_OFFSET];
      }
      pout = &args->pout[p_idx*NWORDS_FR];
      mulm_cb(tmp, &coeff[ZKEY_COEFF_VAL_OFFSET], scl);
      addm_cb(pout, pout, tmp);
      //printf("%d, %d, %d\n", coeff[0], coeff[1], coeff[2]);
  }
/*
  for (uint32_t i=0; i < 64; i++){
     for (uint32_t j=0; j < 4; j++){
       printf("%u ",(args->pout[i] >> (8 * j)) & 0xFF);
     }
  }
*/

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
     const_offset += ((n_coeff - prev_n_coeff) * (NWORDS_FR+1));
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
    cum_c_poly[i] = pout[i] * (NWORDS_FR+1) + cum_c_poly[i-1];
    //cum_v_poly[i] = pout[i] * (NWORDS_FR+1) + cum_v_poly[i-1];
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
         to_montgomery_h(&pout[cum_v_poly[poly_idx]+1+coeff_idx*NWORDS_FR],
                         &cin[coeff_offset], pidx);
       } else {
         memcpy(&pout[cum_v_poly[poly_idx]+1+coeff_idx*NWORDS_FR], &cin[coeff_offset] ,NWORDS_FR * sizeof(uint32_t));
       }
       coeff_offset += NWORDS_FR;
     }
     const_offset += ((n_coeff - prev_n_coeff) * (NWORDS_FR+1));
     prev_n_coeff = n_coeff;
  }

  if (extend){
    for (i=0; i < header->nPubInputs + header->nOutputs + 1; i++){
       coeff_idx = tmp_poly[i]++;
       pout[cum_c_poly[i]+1+coeff_idx]=i + header->nConstraints;
       memcpy(&pout[cum_v_poly[i]+1+coeff_idx*NWORDS_FR], One, sizeof(uint32_t)*NWORDS_FR);
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


void mpoly_from_montgomery_h(uint32_t *x, uint32_t pidx)
{
  uint32_t i;
  uint32_t offset = 1 + x[0];

  for (i=0; i < x[0];i++){
    offset += x[i+1];
    from_montgomeryN_h(&x[offset], &x[offset], x[i+1], pidx,0);
    offset += (x[i+1]*NWORDS_FR);
  }
}

void mpoly_to_montgomery_h(uint32_t *x, uint32_t pidx)
{
  uint32_t i;
  uint32_t offset = 1 + x[0];

  for (i=0; i < x[0];i++){
    offset += x[i+1];
    to_montgomeryN_h(&x[offset], &x[offset], x[i+1], pidx);
    offset += (x[i+1]*NWORDS_FR);
  }
}





