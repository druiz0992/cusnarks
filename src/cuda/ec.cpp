
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
// File name  : ec.cpp
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
#include "ec.h"

#ifdef PARALLEL_EN
static  uint32_t parallelism_enabled =  1;
#else
static  uint32_t parallelism_enabled =  0;
#endif

#define MAX_NCORES_OMP (32)
#define MIN(X,Y)  ((X)<(Y) ? (X) : (Y))
#define MAX(X,Y)  ((X)>(Y) ? (X) : (Y))

static  pthread_mutex_t utils_lock;      
static  pthread_cond_t utils_cond;        
static  uint32_t utils_ectable_ready = 1; 
static  uint32_t utils_done = 0;          

static uint32_t *utils_N=NULL;
static uint32_t *utils_zinv=NULL;
static uint32_t *utils_zinv_sq=NULL;
static uint32_t *utils_EPout=NULL;
static uint32_t *utils_EPin=NULL;
static uint32_t *utils_ectable=NULL;
static uint32_t *utils_R=NULL;
static uint32_t *utils_Ridx=NULL;


static void ec_jacdouble_finish_h(void *args);
static void ec_jacdouble_pippengers_finish_h(void *args);
static void ec_inittable_ready_h(void *args);
static void ec_print_EPin(void *args);
static void ec_initP_h(uint32_t *z, uint32_t n, uint32_t ec2, uint32_t pidx);
static void ec_jacscmul_opt_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t order, uint32_t ec2, uint32_t pidx, uint32_t pippen_conf, uint32_t* Rin);

void *ec_jacreduce_batch_h(void *args);
void *ec_jacreduce_batch_precomputed_h(void *args);
void *ec_read_table_h(void *args);
void *ec_init_table_h(void *args);
void ec_jacaddreduce_finish_h(void *args);
void ec2_jacaddreduce_finish_h(void *args);
void ec_loadtable_h(uint32_t *x, t_uint64 len, t_uint64 *offset, ecp_t ecp, FILE *ifp);


void ec_init_h(void)
{
  uint32_t nprocs = get_nprocs_h();

  utils_N    = (uint32_t *)calloc(nprocs * NWORDS_256BIT * ECP2_JAC_OUTDIMS, sizeof(uint32_t));
  utils_zinv = (uint32_t *)calloc(2 * nprocs * NWORDS_256BIT, sizeof(uint32_t));
  utils_zinv_sq = (uint32_t *) calloc(2 * nprocs * NWORDS_256BIT, sizeof(uint32_t));
  utils_EPout = (uint32_t *) calloc(nprocs * ECP2_JAC_OUTDIMS * NWORDS_256BIT, sizeof(uint32_t));
  utils_EPin = (uint32_t *) calloc(EC_JACREDUCE_TABLE_LEN * nprocs * ECP2_JAC_OUTDIMS * NWORDS_256BIT, sizeof(uint32_t));
  utils_ectable = (uint32_t *)calloc((MAX_U256_BSELM << EC_JACREDUCE_BATCH_SIZE) * nprocs * ECP2_JAC_OUTDIMS * NWORDS_256BIT <<MAX_U256_BSELM, sizeof (uint32_t));
  utils_R = (uint32_t *) calloc((1ull<<MAX_PIPPENGERS_CONF)*nprocs*ECP2_JAC_OUTDIMS*NWORDS_256BIT, sizeof(uint32_t));
  utils_Ridx = (uint32_t *) calloc((1ull<<MAX_PIPPENGERS_CONF), sizeof(uint32_t));
  for (uint32_t i=0; i< (1ull<<MAX_PIPPENGERS_CONF) ; i++){
    utils_Ridx[i]=i;
  }

  if (pthread_mutex_init(&utils_lock, NULL) != 0){
     exit(1);
  }
  if (utils_N == NULL || utils_zinv == NULL || utils_zinv_sq == NULL || 
      utils_EPout == NULL || utils_EPin == NULL || utils_ectable == NULL){
     exit(1);
  }

}

void ec_free_h(void)
{
   pthread_mutex_destroy(&utils_lock); 
   free(utils_N);
   free(utils_zinv);
   free(utils_zinv_sq);
   free(utils_EPout);
   free(utils_EPin);
   free(utils_ectable);
   free(utils_R);
   free(utils_Ridx);
   
   utils_R = NULL;
   utils_Ridx = NULL;
   utils_N=NULL;
   utils_zinv=NULL;
   utils_zinv_sq=NULL;
   utils_EPout=NULL;
   utils_EPin=NULL;
   utils_ectable=NULL;
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


static void ec_jacscmul_opt_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t order, uint32_t ec2, uint32_t pidx, uint32_t pippen_conf, uint32_t* Rin)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  uint32_t i;
  uint32_t n_tables = (order + n - 1)/order;
  uint32_t table_size = 1 << order; 
  void (*add_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t) = &ec_jacadd_h;
  void (*addm_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t) = &ec_jacaddmixed_h;
  uint32_t (*getbit_cb)(uint32_t *, uint32_t, uint32_t) = &getbitu256_h;
  uint32_t table_dim = ECP_JAC_OUTDIMS;
  uint32_t outdim = ECP_JAC_OUTDIMS;
  uint32_t indim = ECP_JAC_INDIMS;
  uint32_t *ectable_ptr = ectable;
  uint32_t torder1 = 0;
  uint32_t add_last = (Rin != NULL);

  if (ec2) {
    table_dim = ECP2_JAC_OUTDIMS;
    outdim = ECP2_JAC_OUTDIMS;
    indim = ECP2_JAC_INDIMS;
    add_cb = &ec2_jacadd_h;
    addm_cb = &ec2_jacaddmixed_h;
    if (order > 1){
       ec2_inittable_h(x, ectable, n, order, pidx, add_last );
    } else {
      ectable_ptr = x;
      torder1 = 1;
      table_size = 1;
      table_dim = ECP2_JAC_INDIMS;
      add_cb = &ec2_jacaddmixed_h;
      if (!Rin){
        table_dim = ECP2_JAC_OUTDIMS;
        add_cb = &ec2_jacadd_h;
      }
    }
  } else {
    if (order > 1){
       ec_inittable_h(x, ectable, n, order, pidx, add_last);
    } else {
      ectable_ptr = x;
      torder1 = 1;
      table_size = 1;
      table_dim = ECP_JAC_INDIMS;
      add_cb = &ec_jacaddmixed_h;
      if (!Rin){
        table_dim = ECP_JAC_OUTDIMS;
        add_cb = &ec_jacadd_h;
      }
    }
  }

  int lsb = pippen_conf;
  uint32_t msb = 255;
  uint32_t stride = NWORDS_256BIT * order;

  if (!Rin){
        msb = pippen_conf - 1;
        lsb = 0;
        getbit_cb = &getbitu32_h;
        stride = order;
  }

  for (i=0; i<n_tables ; i++){
     if (Rin){
       uint32_t tmp_msb;
       msb = 255;
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
     } 

     for (int j=msb; j >= lsb ; j--){
        uint32_t b = getbit_cb(&scl[i*stride], j, MIN(order, n -order*i ));

        if (b) {
           add_cb(&z[j * NWORDS_256BIT * outdim],
                       &ectable_ptr[(i * table_size + b - torder1) * NWORDS_256BIT * table_dim],
                       &z[j * NWORDS_256BIT * outdim],
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
     if (Rin) {
       uint32_t r=0;
       const uint32_t mod_p = (1 << pippen_conf) - 1;
       for (int j=0; j < MIN(order, n-order*i) ; j++){
          uint32_t b = scl[i*order*NWORDS_256BIT + j* NWORDS_256BIT] & mod_p;
          //printf("Table[%d], j:%d, b:%d\n",i,j,b);
  
          if (b) {
                addm_cb(&Rin[b*NWORDS_256BIT*outdim],
                                 &x[(i*order+j)*NWORDS_256BIT*indim],
                                 &Rin[b*NWORDS_256BIT*outdim], 
                                 pidx);
                }
      }
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
  uint32_t toaff = wargs->offset != 0;
  ec_jacaddreduce_h(
          wargs->out_ep,
          &utils_EPout[wargs->thread_id*ECP_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
          wargs->max_threads,
          wargs->pidx, toaff, 0, toaff, EC_JACREDUCE_FLAGS_REDUCTION | EC_JACREDUCE_FLAGS_FINISH);

  /*
  printU256Number(wargs->out_ep);
  printU256Number(&wargs->out_ep[NWORDS_256BIT]);
  */

}

void ec2_jacaddreduce_finish_h(void *args)
{
  jacadd_reduced_t *wargs = (jacadd_reduced_t *)args;
  uint32_t toaff = wargs->offset != 0;
  ec2_jacaddreduce_h(
          wargs->out_ep,
          &utils_EPout[wargs->thread_id*ECP2_JAC_OUTDIMS << NWORDS_256BIT_SHIFT],
          wargs->max_threads,
          wargs->pidx, toaff, 0, toaff, EC_JACREDUCE_FLAGS_REDUCTION | EC_JACREDUCE_FLAGS_FINISH);

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
  uint32_t compute_table = args->ec_table==NULL ? !args->compute_table : 0;
  uint32_t outdims = ECP_JAC_OUTDIMS;
  uint32_t indims = ECP_JAC_INDIMS;
  uint32_t order = args->order;
  utils_done = 0;

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
      vars_per_thread = ((vars_per_thread + ((order << EC_JACREDUCE_BATCH_SIZE) * order) -1) / 
                          (((order << EC_JACREDUCE_BATCH_SIZE) * order))-1) * 
                          ((order << EC_JACREDUCE_BATCH_SIZE) * order);
    } else {
      args->max_threads = 1;
    }
  }  else {
      vars_per_thread = EC_JACREDUCE_TABLE_LEN / (args->max_threads * order) * order;
  }

  if (!utils_isinit_h()) {
    exit(1);
  }
  #ifndef PARALLEL_EN
    exit(1);
  #endif
  
  pthread_t *workers = (pthread_t *) malloc(args->max_threads * sizeof(pthread_t));
  jacadd_reduced_t *w_args  = (jacadd_reduced_t *)malloc(args->max_threads * sizeof(jacadd_reduced_t));
  init_barrier_h(args->max_threads);
 
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
    launch_client_h(ec_jacreduce_batch_h, workers,(void *) w_args, sizeof(jacadd_reduced_t), args->max_threads,0);

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
      utils_ectable_ready = 1;
 
      launch_client_h(ec_read_table_h, workers_table,(void *) w_table_args, sizeof(ec_table_desc_t), 1, 1);

      /*
      printf("Table Offset[%d] : %d\n",w_args->thread_id, w_args->offset);
      printf("EC2[%d] : %d\n",w_args->thread_id, w_args->ec2);
      */
      
    } else if (args->compute_table){ 
      workers_table = (pthread_t *) malloc(1 * sizeof(pthread_t));
      w_table_args  = (ec_table_desc_t *)malloc(1 * sizeof(ec_table_desc_t));
      w_table_args->filename = NULL;
      w_table_args->ec_table = w_args->ec_table;
      w_table_args->offset = 0;
      w_table_args->total_words = w_args->n;
      w_table_args->ec2 = w_args->ec2;
      w_table_args->order = w_args->order;
      w_table_args->indata = w_args->x;
      w_table_args->pidx = w_args->pidx;
 
      utils_ectable_ready = 0;
      launch_client_h(ec_init_table_h, workers_table,(void *) w_table_args, sizeof(ec_table_desc_t), 1, 1);

      /*
      printf("Table Offset[%d] : %d\n",w_args->thread_id, w_args->offset);
      printf("EC2[%d] : %d\n",w_args->thread_id, w_args->ec2);
      */
      
    } 

    memset(
            utils_EPin, 0, outdims * NWORDS_256BIT * EC_JACREDUCE_TABLE_LEN * sizeof(uint32_t)
          );
    launch_client_h(ec_jacreduce_batch_precomputed_h, workers,(void *) w_args, sizeof(jacadd_reduced_t), args->max_threads,0);

    pthread_cond_destroy(&utils_cond);
    del_barrier_h();

    if (w_args->filename != NULL){
      free(workers_table);
      free(w_table_args);
    }
  }

  free(workers);
  free(w_args);

  parallelism_enabled = 1;

  return; 

}

void *ec_jacreduce_batch_h(void *args)
{
  jacadd_reduced_t *wargs = (jacadd_reduced_t *)args;
  uint32_t order = wargs->order;
  uint32_t n_batches = (wargs->last_idx - wargs->start_idx + (order << EC_JACREDUCE_BATCH_SIZE)*order - 1)/((order << EC_JACREDUCE_BATCH_SIZE)*order);
  uint32_t nsamples_offset=0, next_nsamples_offset=0;
  uint32_t nsamples, next_nsamples;
  uint32_t indims = ECP_JAC_INDIMS;
  uint32_t outdims = ECP_JAC_OUTDIMS;

  void (*ec_jacscmul_opt_cb) (uint32_t *, uint32_t *, uint32_t *, uint32_t *,
                uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t *) = &ec_jacscmul_opt_h;
  void (*ec_jacaddreduce_finish_cb)(void *) = &ec_jacaddreduce_finish_h;

  if (wargs->ec2){
    indims = ECP2_JAC_INDIMS;
    outdims = ECP2_JAC_OUTDIMS;
    ec_jacaddreduce_finish_cb = &ec2_jacaddreduce_finish_h;
  }

  uint32_t *table_ptr = &utils_ectable[wargs->thread_id * (order << EC_JACREDUCE_BATCH_SIZE)*outdims<<(NWORDS_256BIT_SHIFT+order)];
  uint32_t *EPin  = &utils_EPin[wargs->thread_id*EC_JACREDUCE_TABLE_LEN * outdims * NWORDS_256BIT];
  uint32_t *Rin   =  EPin;
  if (wargs->pippen){
    Rin   =  &utils_R[wargs->thread_id*(1ull<<MAX_PIPPENGERS_CONF) * outdims * NWORDS_256BIT]; 
    ec_initP_h(Rin, 1ull<<wargs->pippen, wargs->ec2,  wargs->pidx);
  }

  //printf("[%d] - N batches : %d %d, %d\n",wargs->thread_id, n_batches, wargs->ec2, wargs->n);

  ec_initP_h(EPin, EC_JACREDUCE_TABLE_LEN, wargs->ec2,  wargs->pidx);

  for (uint32_t i=0; i < n_batches; i++){
    nsamples = MIN((order << EC_JACREDUCE_BATCH_SIZE)*order,(wargs->last_idx - wargs->start_idx) - nsamples_offset);
    // prefetch
    if (i < n_batches - 1){
       next_nsamples_offset += (order << EC_JACREDUCE_BATCH_SIZE)*order;
       next_nsamples = MIN((order << EC_JACREDUCE_BATCH_SIZE)*order,(wargs->last_idx - wargs->start_idx) - next_nsamples_offset);
       __builtin_prefetch(&wargs->scl[(wargs->start_idx + next_nsamples_offset + next_nsamples)<<NWORDS_256BIT_SHIFT]);
       __builtin_prefetch( &wargs->x[(wargs->start_idx + next_nsamples_offset + next_nsamples)*indims<<NWORDS_256BIT_SHIFT]);
    }
    
    ec_jacscmul_opt_cb(
           EPin,
           &wargs->scl[(wargs->start_idx + nsamples_offset)<<NWORDS_256BIT_SHIFT],
           &wargs->x[(wargs->start_idx + nsamples_offset)*indims<<NWORDS_256BIT_SHIFT],
           table_ptr,
           nsamples,
           order, wargs->ec2, wargs->pidx,
           wargs->pippen,
           Rin);

    nsamples_offset += (order << EC_JACREDUCE_BATCH_SIZE)*order;
  }

  ec_jacdouble_finish_h(wargs);
  wargs->offset=1;
  if (wargs->pippen){
    wargs->offset=0;
  }
  wait_h(wargs->thread_id, ec_jacaddreduce_finish_cb, wargs);

  if (wargs->pippen) {
    // Previous result is stored in utils_EPout[0] in jacobian format
    ec_initP_h(EPin, wargs->pippen, wargs->ec2,  wargs->pidx);

    nsamples_offset=0; next_nsamples_offset=0;
    n_batches = ((1<<wargs->pippen) + (order << EC_JACREDUCE_BATCH_SIZE)*order - 1)/((order << EC_JACREDUCE_BATCH_SIZE)*order);
  
    for (uint32_t i=0; i < n_batches; i++){
      nsamples = MIN((order << EC_JACREDUCE_BATCH_SIZE)*order,(1<<wargs->pippen) - nsamples_offset);
      
      ec_jacscmul_opt_cb(
             EPin,
             &utils_Ridx[nsamples_offset],
             &Rin[nsamples_offset*outdims<<NWORDS_256BIT_SHIFT],
             table_ptr,
             nsamples,
             order, wargs->ec2, wargs->pidx, wargs->pippen, NULL);
  
      nsamples_offset += (order << EC_JACREDUCE_BATCH_SIZE)*order;
    }

    ec_jacdouble_pippengers_finish_h(wargs);
    wargs->offset=1;
    wait_h(wargs->thread_id, ec_jacaddreduce_finish_cb, wargs);
  }

  return NULL;
}

void *ec_init_table_h(void *args)
{
  ec_table_desc_t *wargs = (ec_table_desc_t *)args;
  void (*ec_inittable_cb)(uint32_t *, uint32_t *, uint32_t , uint32_t, uint32_t, uint32_t) = &ec_inittable_h;
  uint32_t total_points_read = 0;
  uint32_t ec_table_offset[] = {0, (MAX_U256_BSELM << EC_JACREDUCE_BATCH_SIZE) * ECP_JAC_OUTDIMS * NWORDS_256BIT << MAX_U256_BSELM};
  uint32_t ec_table_idx=0;
  uint32_t indims = ECP_JAC_INDIMS;
  uint32_t outdims = ECP_JAC_OUTDIMS;
  uint32_t total_elems_read = 0;
  uint32_t nelems = (wargs->order << EC_JACREDUCE_BATCH_SIZE) * wargs->order;

  if (wargs->ec2){
     ec_inittable_cb = &ec2_inittable_h;
     ec_table_offset[1] = (MAX_U256_BSELM<<EC_JACREDUCE_BATCH_SIZE) * ECP2_JAC_OUTDIMS * NWORDS_256BIT << MAX_U256_BSELM;
     indims = ECP2_JAC_INDIMS;
     outdims = ECP2_JAC_OUTDIMS;
  }
  //printf("Nelems : %d\n",nelems);
  while (total_elems_read <= wargs->total_words){
     //wait till ready
     pthread_mutex_lock(&utils_lock);
     while(utils_ectable_ready){
       pthread_cond_wait(&utils_cond, &utils_lock);
     }
     //printf("Start computing tables : %d/%d\n",ec_table_idx,MIN(nelems, wargs->total_words-total_elems_read));
     ec_inittable_cb(&wargs->indata[total_elems_read * indims  * NWORDS_256BIT],
                     &utils_ectable[ec_table_offset[ec_table_idx]],
                     MIN(nelems, wargs->total_words-total_elems_read),
                     wargs->order, wargs->pidx, 1);
     total_elems_read+=nelems;

     ec_table_idx ^= 1;
     utils_ectable_ready = 1;
     pthread_cond_signal(&utils_cond);
     pthread_mutex_unlock(&utils_lock);
  }
  return NULL;
}

void *ec_read_table_h(void *args)
{
  ec_table_desc_t *wargs = (ec_table_desc_t *)args;
  t_uint64 total_words_read = 0;
  uint32_t words_read=0;
  uint32_t ec_table_offset[] = {0, (MAX_U256_BSELM << EC_JACREDUCE_BATCH_SIZE) * ECP_JAC_INDIMS * NWORDS_256BIT << MAX_U256_BSELM};
  uint32_t n_words = (wargs->order<<EC_JACREDUCE_BATCH_SIZE) * ECP_JAC_INDIMS * NWORDS_256BIT << wargs->order;
  uint32_t ec_table_idx=1;
  if (wargs->ec2){
     ec_table_offset[1] = (MAX_U256_BSELM<<EC_JACREDUCE_BATCH_SIZE) * ECP2_JAC_INDIMS * NWORDS_256BIT << MAX_U256_BSELM;
     n_words = (wargs->order<<EC_JACREDUCE_BATCH_SIZE) * ECP2_JAC_INDIMS * NWORDS_256BIT << wargs->order;
  }
  //printf("N words : %d, %d\n",n_words, wargs->total_words);
  int err;
  
  // open file
  FILE *ifp = fopen(wargs->filename,"rb");

  // go to initial offset
  err = fseek(ifp, wargs->offset , SEEK_SET);
  //printf("Initial offset (words) : %lld, err : %d, File : %s\n",wargs->offset, err,wargs->filename);

  while (total_words_read <= wargs->total_words){
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
     //printf("Signal received. Continue ...!\n" );
     if ( words_read < n_words || utils_done){   
        break;
     }
      
  }
  //printf("File read complete\n");
  fclose(ifp);

  return NULL;
}


void *ec_jacreduce_batch_precomputed_h(void *args)
{
  jacadd_reduced_t *wargs = (jacadd_reduced_t *)args;
  utils_done = 0;
  uint32_t i,b, n_batches;
  int32_t j, start_msb, last_msb;
  uint32_t order = wargs->order;
  uint32_t *utils_EPin_ptr = utils_EPin;
  uint32_t ec_table_offset[] = {0, (MAX_U256_BSELM << EC_JACREDUCE_BATCH_SIZE) * ECP_JAC_INDIMS * NWORDS_256BIT << MAX_U256_BSELM};
  uint32_t ec_table_idx=1;
  uint32_t *utils_ectable_ptr = wargs->ec_table;  
  uint32_t torder1 = (order == 1 && wargs->filename == NULL);

  uint32_t indims = ECP_JAC_INDIMS;
  uint32_t outdims = ECP_JAC_OUTDIMS;
  uint32_t yoffset = ECP_JAC_OUTYOFFSET;
  void (*ec_jacadd_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t) = &ec_jacaddmixed_h;

  if (wargs->compute_table){
   ec_jacadd_cb = &ec_jacadd_h;
   indims = ECP_JAC_OUTDIMS;
   ec_table_offset[1] = (MAX_U256_BSELM << EC_JACREDUCE_BATCH_SIZE) * ECP_JAC_OUTDIMS * NWORDS_256BIT << MAX_U256_BSELM;
  }

  if (wargs->ec2){
    outdims = ECP2_JAC_OUTDIMS;
    yoffset = ECP2_JAC_OUTYOFFSET;

    if (wargs->compute_table){
     ec_jacadd_cb = &ec2_jacadd_h;
     indims = ECP2_JAC_OUTDIMS;
     ec_table_offset[1] = (MAX_U256_BSELM << EC_JACREDUCE_BATCH_SIZE) * ECP2_JAC_OUTDIMS * NWORDS_256BIT << MAX_U256_BSELM;

    } else {
      ec_jacadd_cb = &ec2_jacaddmixed_h;
      indims = ECP2_JAC_INDIMS;
      ec_table_offset[1] = (MAX_U256_BSELM << EC_JACREDUCE_BATCH_SIZE) * ECP2_JAC_INDIMS * NWORDS_256BIT << MAX_U256_BSELM;
    }
  }

  // init table
  last_msb = wargs->start_idx;
  start_msb = wargs->last_idx-1;
  const  uint32_t *One = CusnarksOneMontGet((mod_t)wargs->pidx);
  for (j=start_msb; j>=last_msb; j--){
        memcpy(&utils_EPin_ptr[j*outdims*NWORDS_256BIT+yoffset], One, NWORDS_256BIT * sizeof(uint32_t));
  }
  //printf("N elems : %d\n",wargs->n); 
  for (i=0; i < wargs->n; i += order){
     if (i % ( (order << EC_JACREDUCE_BATCH_SIZE) * order) == 0 && (wargs->filename != NULL || wargs->compute_table) ){
        ec_table_idx ^= 1;
        if (i == 0 && !wargs->compute_table){
          utils_ectable_ptr = wargs->ec_table;
        } else {
          utils_ectable_ptr = &utils_ectable[ec_table_offset[ec_table_idx]];
        }
        //printf("Idx : %d-%d,%d/%d\n",wargs->thread_id, ec_table_idx, i,wargs->n);
        wait_h(wargs->thread_id, ec_inittable_ready_h, wargs);
        //printf("Start computing points\n");
        //load next data batch from tables
     }
     if (i <= wargs->n - order){
       __builtin_prefetch(&wargs->scl[(i+1)*NWORDS_256BIT]);
       __builtin_prefetch(&utils_ectable_ptr[(i+1)*indims << (NWORDS_256BIT_SHIFT + order - torder1)]); 
     }

     // prefetch : wargs->scl, utils_ectable_ptr[indims << ]
     for (j=start_msb; j>=last_msb ; j--){
        b = getbitu256_h(&wargs->scl[i * NWORDS_256BIT], j, order);
     
        if (b){
          ec_jacadd_cb(
            &utils_EPin_ptr[j*outdims*NWORDS_256BIT],
            &utils_ectable_ptr[(b-torder1)*indims*NWORDS_256BIT],
            &utils_EPin_ptr[j*outdims*NWORDS_256BIT], wargs->pidx);
        }
     }
     /*
     if (wargs->thread_id == 0){   
       printf("Table : %d\n",i);
       for (uint32_t k=0; k< (1<<wargs->order) * outdims * 1; k++){
         printf("%d\n",k);
         printU256Number(&utils_ectable[ec_table_offset[ec_table_idx] + k*NWORDS_256BIT + (0)*(k<<wargs->order)*outdims*NWORDS_256BIT]);
       }
     }
     */
     utils_ectable_ptr += indims << (NWORDS_256BIT_SHIFT + order - torder1);
  }
  utils_done = 1;

  wait_h(wargs->thread_id, ec_jacdouble_finish_h, wargs);
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
  utils_ectable_ready = 0;
  pthread_cond_signal(&utils_cond);
  pthread_mutex_unlock(&utils_lock);
  //printf("Table is ready\n");
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

  uint32_t *P = &utils_EPin[(wargs->thread_id+1) * EC_JACREDUCE_TABLE_LEN *outdims * NWORDS_256BIT - outdims * NWORDS_256BIT];
  uint32_t *Q = P;

  /*
  printf("255\n");
  printU256Number(P);
  printU256Number(&P[NWORDS_256BIT]);
  printU256Number(&P[2*NWORDS_256BIT]);
  */

  for (i=EC_JACREDUCE_TABLE_LEN - 1; i>0 ; i--){
     ec_jacdouble_cb( P, P, wargs->pidx );
     Q-=outdims*NWORDS_256BIT;
     ec_jacadd_cb( P, P, Q, wargs->pidx);
  
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
  //TODO Maybe add flag to wargs to either do memcpy or convert to affine
  memcpy(&utils_EPout[(wargs->thread_id)*outdims*NWORDS_256BIT] , P, outdims*NWORDS_256BIT*sizeof(uint32_t));
  ec_jac2aff_cb(wargs->out_ep, P, 1, wargs->pidx, 1);
}

static void ec_jacdouble_pippengers_finish_h(void *args)
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

  uint32_t *P = &utils_EPin[(wargs->thread_id+1) *EC_JACREDUCE_TABLE_LEN *outdims * NWORDS_256BIT -(1 + EC_JACREDUCE_TABLE_LEN - wargs->pippen) * outdims * NWORDS_256BIT];
  uint32_t *Q = P;

  /*
  printf("255\n");
  printU256Number(P);
  printU256Number(&P[NWORDS_256BIT]);
  printU256Number(&P[2*NWORDS_256BIT]);
  */

  for (i=wargs->pippen-1; i>0 ; i--){
     ec_jacdouble_cb( P, P, wargs->pidx );
     Q-=outdims*NWORDS_256BIT;
     ec_jacadd_cb( P, P, Q, wargs->pidx);
  
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
  //Add previous result
  if (!wargs->thread_id){
      ec_jacadd_cb(&utils_EPout[(wargs->thread_id)*outdims*NWORDS_256BIT] ,
                 &utils_EPout[(wargs->thread_id)*outdims*NWORDS_256BIT] ,
                 P,  wargs->pidx);
  } else {
     memcpy(&utils_EPout[(wargs->thread_id)*outdims*NWORDS_256BIT] ,
                 P,  outdims*NWORDS_256BIT*sizeof(uint32_t));
  }
}

static void ec_initP_h(uint32_t *z, uint32_t n, uint32_t ec2, uint32_t pidx)
{
  const uint32_t *ECInf = CusnarksMiscKGet();
  for (uint32_t i=0; i < n; i++){
    if (ec2) {
       memcpy(
              &z[i * NWORDS_256BIT * ECP2_JAC_OUTDIMS],
              &ECInf[(pidx * MISC_K_N + MISC_K_INF2) * NWORDS_256BIT],
              sizeof(uint32_t) * NWORDS_256BIT * ECP2_JAC_OUTDIMS
            );
    } else {
       memcpy(
              &z[i * NWORDS_256BIT * ECP_JAC_OUTDIMS],
              &ECInf[(pidx * MISC_K_N + MISC_K_INF) * NWORDS_256BIT],
              sizeof(uint32_t) * NWORDS_256BIT * ECP_JAC_OUTDIMS
            );
    }
  }
}


