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
// File name  : test_utils_host.cpp
//
// Date       : 6/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Util functions for host Test
//
// ------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <omp.h>
#include "types.h"
#include "constants.h"
#include "uint256.h"
#include "bigint.h"
#include "ff.h"
#include "ntt.h"
#include "ec.h"
#include "init.h"
#include "file_utils.h"
#include "transpose.h"
#include "utils_host.h"

#define NROOTS 128
#define MAX_ITER_1M 10
#define MAX_ITER_65K 10
#define MAX_ITER 10
#define NCOLS_1M 10
#define NROWS_1M 10
#define FFT_SIZEXX_1M 5
#define FFT_SIZEYX_1M 5

#define NCOLS_65K 7
#define NROWS_65K 9
#define FFT_SIZEXX_65K 4
#define FFT_SIZEYX_65K 5


#define NCOLS_131K 8
#define NROWS_131K 9

#define NCOLS_131K_3D 5
#define NROWS_131K_3D 4
#define FFT_SIZEXX_131K_3D 3
#define FFT_SIZEYX_131K_3D 2
#define NFFT_X_131K_3D 9
#define NFFT_Y_131K_3D 8

static char ec_table_filename[]="./aux_data/ec_table.tbin";

static char ff_param1_filename[4][100] = {
                                           "./aux_data/ff_param1_0.tbin",
                                           "./aux_data/ff_param1_1.tbin",
                                           "./aux_data/ff_param1_2.tbin",
                                           "./aux_data/ff_param1_3.tbin"
};
                                             
static char ff_param2_filename[4][100] = {
                                           "./aux_data/ff_param2_0.tbin",
                                           "./aux_data/ff_param2_1.tbin",
                                           "./aux_data/ff_param2_2.tbin",
                                           "./aux_data/ff_param2_3.tbin"
};
                                             
static char ff_add_filename[4][100] = {
                                           "./aux_data/ff_add_0.tbin",
                                           "./aux_data/ff_add_1.tbin",
                                           "./aux_data/ff_add_2.tbin",
                                           "./aux_data/ff_add_3.tbin"
};
                                             
static char ff_sub_filename[4][100] = {
                                           "./aux_data/ff_sub_0.tbin",
                                           "./aux_data/ff_sub_1.tbin",
                                           "./aux_data/ff_sub_2.tbin",
                                           "./aux_data/ff_sub_3.tbin"
};

static char ff_tom_filename[4][100] = {
                                           "./aux_data/ff_tom_0.tbin",
                                           "./aux_data/ff_tom_1.tbin",
                                           "./aux_data/ff_tom_2.tbin",
                                           "./aux_data/ff_tom_3.tbin"
};

static char ff_mul_filename[4][100] = {
                                           "./aux_data/ff_mul_0.tbin",
                                           "./aux_data/ff_mul_1.tbin",
                                           "./aux_data/ff_mul_2.tbin",
                                           "./aux_data/ff_mul_3.tbin"
};

static char ff_sq_filename[4][100] = {
                                           "./aux_data/ff_sq_0.tbin",
                                           "./aux_data/ff_sq_1.tbin",
                                           "./aux_data/ff_sq_2.tbin",
                                           "./aux_data/ff_sq_3.tbin"
};

static char ff_inv_filename[4][100] = {
                                           "./aux_data/ff_inv_0.tbin",
                                           "./aux_data/ff_inv_1.tbin",
                                           "./aux_data/ff_inv_2.tbin",
                                           "./aux_data/ff_inv_3.tbin"
};

static char ff_extparam1_filename[4][100] = {
                                           "./aux_data/ff_extparam1_0.tbin",
                                           "./aux_data/ff_extparam1_1.tbin",
                                           "./aux_data/ff_extparam1_2.tbin",
                                           "./aux_data/ff_extparam1_3.tbin"
};
                                             
static char ff_extparam2_filename[4][100] = {
                                           "./aux_data/ff_extparam2_0.tbin",
                                           "./aux_data/ff_extparam2_1.tbin",
                                           "./aux_data/ff_extparam2_2.tbin",
                                           "./aux_data/ff_extparam2_3.tbin"
};
                                             
static char ff_extadd_filename[4][100] = {
                                           "./aux_data/ff_extadd_0.tbin",
                                           "./aux_data/ff_extadd_1.tbin",
                                           "./aux_data/ff_extadd_2.tbin",
                                           "./aux_data/ff_extadd_3.tbin"
};
                                             
static char ff_extsub_filename[4][100] = {
                                           "./aux_data/ff_extsub_0.tbin",
                                           "./aux_data/ff_extsub_1.tbin",
                                           "./aux_data/ff_extsub_2.tbin",
                                           "./aux_data/ff_extsub_3.tbin"
};

static char ff_extmul_filename[4][100] = {
                                           "./aux_data/ff_extmul_0.tbin",
                                           "./aux_data/ff_extmul_1.tbin",
                                           "./aux_data/ff_extmul_2.tbin",
                                           "./aux_data/ff_extmul_3.tbin"
};

static char ff_extsq_filename[4][100] = {
                                           "./aux_data/ff_extsq_0.tbin",
                                           "./aux_data/ff_extsq_1.tbin",
                                           "./aux_data/ff_extsq_2.tbin",
                                           "./aux_data/ff_extsq_3.tbin"
};

static char ff_extinv_filename[4][100] = {
                                           "./aux_data/ff_extinv_0.tbin",
                                           "./aux_data/ff_extinv_1.tbin",
                                           "./aux_data/ff_extinv_2.tbin",
                                           "./aux_data/ff_extinv_3.tbin"
};

static char ecp1_filename[4][100] = {
                                           "./aux_data/ecp1_0.tbin",
                                           "./aux_data/ecp1_2.tbin",
                                           "./aux_data/ecp21_0.tbin",
                                           "./aux_data/ecp21_2.tbin"
};

static char ecp2_filename[4][100] = {
                                           "./aux_data/ecp2_0.tbin",
                                           "./aux_data/ecp2_2.tbin",
                                           "./aux_data/ecp22_0.tbin",
                                           "./aux_data/ecp22_2.tbin"
};

static char scl_filename[4][100] = {
                                           "./aux_data/scl_0.tbin",
                                           "./aux_data/scl_2.tbin",
                                           "./aux_data/scl2_0.tbin",
                                           "./aux_data/scl2_2.tbin"
};

static char ec_add_filename[4][100] = {
                                           "./aux_data/ec_add_0.tbin",
                                           "./aux_data/ec_add_2.tbin",
                                           "./aux_data/ec2_add_0.tbin",
                                           "./aux_data/ec2_add_2.tbin"
};

static char ec_dbl_filename[4][100] = {
                                           "./aux_data/ec_dbl_0.tbin",
                                           "./aux_data/ec_dbl_2.tbin",
                                           "./aux_data/ec2_dbl_0.tbin",
                                           "./aux_data/ec2_dbl_2.tbin"
};

static char ec_aff_filename[4][100] = {
                                           "./aux_data/ec_aff_0.tbin",
                                           "./aux_data/ec_aff_2.tbin",
                                           "./aux_data/ec2_aff_0.tbin",
                                           "./aux_data/ec2_aff_2.tbin"
};

static char ec_mul_filename[4][100] = {
                                           "./aux_data/ec_mul_0.tbin",
                                           "./aux_data/ec_mul_2.tbin",
                                           "./aux_data/ec2_mul_0.tbin",
                                           "./aux_data/ec2_mul_2.tbin"
};

static char ec_rdc_filename[4][100] = {
                                           "./aux_data/ec_rdc_0.tbin",
                                           "./aux_data/ec_rdc_2.tbin",
                                           "./aux_data/ec2_rdc_0.tbin",
                                           "./aux_data/ec2_rdc_2.tbin"
};

uint32_t test_addm(void)
{
 uint32_t c[NWORDS_FP]; 
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx++){
     init_h();
     getDataFileSize(&nwords, ff_param1_filename[pidx+pidx_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

     uint32_t *a = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *b = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords * sizeof(uint32_t));
  
     readDataFile(a, ff_param1_filename[pidx+pidx_offset]);
     readDataFile(b, ff_param2_filename[pidx+pidx_offset]);
     readDataFile(r, ff_add_filename[pidx+pidx_offset]);
  
     for (i=0; i < nwords/ PSize; i++){
       
       addm_h(c, &a[i*PSize], &b[i*PSize], pidx);
  
       if (compuBI_h(&r[i*PSize],c, PSize)){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - N errors(Test_Add) : %d/%d\n",pidx, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(b);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_subm(void)
{
 uint32_t c[NWORDS_FP]; 
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx++){
     init_h();
     getDataFileSize(&nwords, ff_param1_filename[pidx+pidx_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

     uint32_t *a = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *b = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords * sizeof(uint32_t));
  
     readDataFile(a, ff_param1_filename[pidx+pidx_offset]);
     readDataFile(b, ff_param2_filename[pidx+pidx_offset]);
     readDataFile(r, ff_sub_filename[pidx+pidx_offset]);
  
     for (i=0; i < nwords/ PSize; i++){
       
       subm_h(c, &a[i*PSize], &b[i*PSize], pidx);
  
       if (compuBI_h(&r[i*PSize],c, PSize)){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - N errors(Test_Sub) : %d/%d\n",pidx, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(b);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_tomont(void)
{
 uint32_t c[NWORDS_FP]; 
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx++){
     init_h();
     getDataFileSize(&nwords, ff_param1_filename[pidx+pidx_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

     uint32_t *a = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords * sizeof(uint32_t));
  
     readDataFile(a, ff_param1_filename[pidx+pidx_offset]);
     readDataFile(r, ff_tom_filename[pidx+pidx_offset]);
  
     for (i=0; i < nwords/ PSize; i++){
       to_montgomery_h(c, &a[i*PSize], pidx);
  
       if (compuBI_h(&r[i*PSize],c, PSize)){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - N errors(Test_ToMont) : %d/%d\n",pidx, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_mul(void)
{
 uint32_t c[NWORDS_FP]; 
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx++){
     init_h();
     getDataFileSize(&nwords, ff_param1_filename[pidx+pidx_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

     uint32_t *a = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *b = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords * sizeof(uint32_t));
  
     readDataFile(a, ff_param1_filename[pidx+pidx_offset]);
     readDataFile(b, ff_param2_filename[pidx+pidx_offset]);
     readDataFile(r, ff_mul_filename[pidx+pidx_offset]);
  
     for (i=0; i < nwords/ PSize; i++){
       to_montgomery_h(&a[i*PSize], &a[i*PSize], pidx);
       to_montgomery_h(&b[i*PSize], &b[i*PSize], pidx);
       montmult_h(c, &a[i*PSize], &b[i*PSize], pidx);
       from_montgomery_h(c, c, pidx);
  
       if (compuBI_h(&r[i*PSize],c, PSize)){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - N errors(Test_Mul) : %d/%d\n",pidx, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(b);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_square(void)
{
 uint32_t c[NWORDS_FP]; 
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx++){
     init_h();
     getDataFileSize(&nwords, ff_param1_filename[pidx+pidx_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

     uint32_t *a = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords * sizeof(uint32_t));
  
     readDataFile(a, ff_param1_filename[pidx+pidx_offset]);
     readDataFile(r, ff_sq_filename[pidx+pidx_offset]);
  
     for (i=0; i < nwords/ PSize; i++){
       to_montgomery_h(&a[i*PSize], &a[i*PSize], pidx);
       montsquare_h(c, &a[i*PSize], pidx);
       from_montgomery_h(c, c, pidx);
  
       if (compuBI_h(&r[i*PSize],c, PSize)){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - N errors(Test_Square) : %d/%d\n",pidx, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_inv(void)
{
 uint32_t c[NWORDS_FP]; 
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx++){
     init_h();
     getDataFileSize(&nwords, ff_param1_filename[pidx+pidx_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

     uint32_t *a = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords * sizeof(uint32_t));
  
     readDataFile(a, ff_param1_filename[pidx+pidx_offset]);
     readDataFile(r, ff_inv_filename[pidx+pidx_offset]);
  
     for (i=0; i < nwords/ PSize; i++){
       to_montgomery_h(&a[i*PSize], &a[i*PSize], pidx);
       montinv_h (c, &a[i*PSize], pidx);
       from_montgomery_h(c, c, pidx);
  
       if (compuBI_h(&r[i*PSize],c, PSize)){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - N errors(Test_Inverse) : %d/%d\n",pidx, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}


uint32_t test_addm_ext(void)
{
 uint32_t c[NWORDS_FP*2]; 
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx++){
     init_h();
     getDataFileSize(&nwords, ff_extparam1_filename[pidx+pidx_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

     uint32_t *a = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *b = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords * sizeof(uint32_t));
  
     readDataFile(a, ff_extparam1_filename[pidx+pidx_offset]);
     readDataFile(b, ff_extparam2_filename[pidx+pidx_offset]);
     readDataFile(r, ff_extadd_filename[pidx+pidx_offset]);
  
     for (i=0; i < nwords/ (2*PSize); i++){
       
       addm_ext_h(c, &a[i*2*PSize], &b[i*2*PSize], pidx);
  
       if ( compuBI_h(&r[i*2*PSize],c, PSize) || 
            compuBI_h(&r[i*2*PSize+PSize], &c[PSize], PSize) ){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - N errors(Test_AddExt) : %d/%d\n",pidx, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(b);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_subm_ext(void)
{
 uint32_t c[NWORDS_FP*2]; 
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx++){
     init_h();
     getDataFileSize(&nwords, ff_extparam1_filename[pidx+pidx_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

     uint32_t *a = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *b = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords * sizeof(uint32_t));
  
     readDataFile(a, ff_extparam1_filename[pidx+pidx_offset]);
     readDataFile(b, ff_extparam2_filename[pidx+pidx_offset]);
     readDataFile(r, ff_extsub_filename[pidx+pidx_offset]);
  
     for (i=0; i < nwords/ (2*PSize); i++){
       
       subm_ext_h(c, &a[i*2*PSize], &b[i*2*PSize], pidx);
  
       if ( compuBI_h(&r[i*2*PSize],c, PSize) || 
            compuBI_h(&r[i*2*PSize+PSize], &c[PSize], PSize) ){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - N errors(Test_SubExt) : %d/%d\n",pidx, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(b);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_mul_ext(void)
{
 uint32_t c[NWORDS_FP*2]; 
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx++){
     init_h();
     getDataFileSize(&nwords, ff_extparam1_filename[pidx+pidx_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

     uint32_t *a = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *b = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords * sizeof(uint32_t));
  
     readDataFile(a, ff_extparam1_filename[pidx+pidx_offset]);
     readDataFile(b, ff_extparam2_filename[pidx+pidx_offset]);
     readDataFile(r, ff_extmul_filename[pidx+pidx_offset]);
  
     for (i=0; i < nwords/ (2*PSize); i++){
       to_montgomery_h(&a[i*2*PSize], &a[i*2*PSize], pidx);
       to_montgomery_h(&a[i*2*PSize+PSize], &a[i*2*PSize+PSize], pidx);
       to_montgomery_h(&b[i*2*PSize], &b[i*2*PSize], pidx);
       to_montgomery_h(&b[i*2*PSize+PSize], &b[i*2*PSize+PSize], pidx);
       montmult_ext_h(c, &a[i*2*PSize], &b[i*2*PSize], pidx);
       from_montgomery_h(c, c, pidx);
       from_montgomery_h(&c[PSize], &c[PSize], pidx);
  
       if ( compuBI_h(&r[i*2*PSize],c, PSize) || 
            compuBI_h(&r[i*2*PSize+PSize], &c[PSize], PSize) ){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - N errors(Test_MulExt) : %d/%d\n",pidx, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(b);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_square_ext(void)
{
 uint32_t c[NWORDS_FP*2]; 
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx++){
     init_h();
     getDataFileSize(&nwords, ff_extparam1_filename[pidx+pidx_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

     uint32_t *a = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *b = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords * sizeof(uint32_t));
  
     readDataFile(a, ff_extparam1_filename[pidx+pidx_offset]);
     readDataFile(b, ff_extparam2_filename[pidx+pidx_offset]);
     readDataFile(r, ff_extsq_filename[pidx+pidx_offset]);
  
     for (i=0; i < nwords/ (2*PSize); i++){
       to_montgomery_h(&a[i*2*PSize], &a[i*2*PSize], pidx);
       to_montgomery_h(&a[i*2*PSize+PSize], &a[i*2*PSize+PSize], pidx);
       montsquare_ext_h(c, &a[i*2*PSize], pidx);
       from_montgomery_h(c, c, pidx);
       from_montgomery_h(&c[PSize], &c[PSize], pidx);
  
       if ( compuBI_h(&r[i*2*PSize],c, PSize) || 
            compuBI_h(&r[i*2*PSize+PSize], &c[PSize], PSize) ){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - N errors(Test_SquareExt) : %d/%d\n",pidx, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(b);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_inv_ext(void)
{
 uint32_t c[NWORDS_FP*2]; 
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx++){
     init_h();
     getDataFileSize(&nwords, ff_extparam1_filename[pidx+pidx_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

     uint32_t *a = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *b = (uint32_t *) malloc( nwords * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords * sizeof(uint32_t));
  
     readDataFile(a, ff_extparam1_filename[pidx+pidx_offset]);
     readDataFile(b, ff_extparam2_filename[pidx+pidx_offset]);
     readDataFile(r, ff_extinv_filename[pidx+pidx_offset]);
  
     for (i=0; i < nwords/ (2*PSize); i++){
       to_montgomery_h(&a[i*2*PSize], &a[i*2*PSize], pidx);
       to_montgomery_h(&a[i*2*PSize+PSize], &a[i*2*PSize+PSize], pidx);
       montinv_ext_h(c, &a[i*2*PSize], pidx);
       from_montgomery_h(c, c, pidx);
       from_montgomery_h(&c[PSize], &c[PSize], pidx);
  
       if ( compuBI_h(&r[i*2*PSize],c, PSize) || 
            compuBI_h(&r[i*2*PSize+PSize], &c[PSize], PSize) ){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - N errors(Test_InvExt) : %d/%d\n",pidx, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(b);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}








uint32_t test_mul_prof(void)
{
  uint32_t nsamples=1<<25;
  uint32_t retval=0;
  struct timespec start, end;
  double elapsed=0.0;
  uint32_t ncores = get_nprocs_conf();
  
  for (uint32_t pidx=MOD_FP; pidx < MOD_N; pidx++){
    const uint32_t *P =  CusnarksPGet((mod_t)pidx);
    const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

    uint32_t *a = (uint32_t *) malloc( ncores * PSize * sizeof(uint32_t));
    uint32_t *b = (uint32_t *) malloc( ncores * PSize * sizeof(uint32_t));
    uint32_t *r = (uint32_t *) malloc( ncores * PSize * sizeof(uint32_t));
 
    setRandomBI(a, ncores, 0, PSize, P, PSize);
    setRandomBI(b, ncores, 0, PSize, P, PSize);
 
    init_h();
    clock_gettime(CLOCK_MONOTONIC, &start);

    #pragma omp parallel for 
    for (uint32_t i=0; i< nsamples; i++){ 
      uint32_t tid = omp_get_thread_num();
      montmult_h(&r[tid*PSize], (const uint32_t *) &a[tid*PSize],(const uint32_t *) &b[tid*PSize], pidx);
    }
 
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (double) (end.tv_sec - start.tv_sec);
    elapsed += (double) (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Pidx : %d Time(Test_Mul N samples : %u) Time : %f\n",pidx, nsamples, elapsed);
    printf("\033[0m");

    release_h();
    free(r);
    free(a);
    free(b);
  }

  return retval;

}

uint32_t test_addm_prof(void)
{

 uint32_t samples=0;
 int pidx;
 uint32_t retval=0;
 struct timespec start, end;
 double elapsed=0.0;
 uint32_t ncores = get_nprocs_conf();
  
 for (pidx=MOD_FP; pidx < MOD_N; pidx++){   
   init_h();
   samples = 1000000000;
   clock_gettime(CLOCK_MONOTONIC, &start);
   const uint32_t *P =  CusnarksPGet((mod_t)pidx);
   const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

   uint32_t *a = (uint32_t *) malloc( ncores * PSize * sizeof(uint32_t));
   uint32_t *b = (uint32_t *) malloc( ncores * PSize * sizeof(uint32_t));
   uint32_t *r = (uint32_t *) malloc( ncores * PSize * sizeof(uint32_t));
        
   setRandomBI(a, ncores, 0, PSize, P, PSize);
   setRandomBI(b, ncores, 0, PSize, P, PSize);

   #pragma omp parallel for 
   for (uint32_t j=0;j<samples; j++){
  
  
       addm_h(r, a, b, pidx);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (double) (end.tv_sec - start.tv_sec);
    elapsed += (double) (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Pidx : %d - Time(Test_Addm %u) : Time : %f\n",pidx, samples, elapsed);
    printf("\033[0m");
    
    release_h();
    free(a);
    free(b);
    free(r);
  }

  return retval;

}

uint32_t test_subm_prof(void)
{

 uint32_t samples=0;
 int pidx;
 uint32_t retval=0;
 struct timespec start, end;
 double elapsed=0.0;
 uint32_t ncores = get_nprocs_conf();
  
 for (pidx=MOD_FP; pidx < MOD_N; pidx++){   
   init_h();
   samples = 1000000000;
   clock_gettime(CLOCK_MONOTONIC, &start);
   const uint32_t *P =  CusnarksPGet((mod_t)pidx);
   const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

   uint32_t *a = (uint32_t *) malloc( ncores * PSize * sizeof(uint32_t));
   uint32_t *b = (uint32_t *) malloc( ncores * PSize * sizeof(uint32_t));
   uint32_t *r = (uint32_t *) malloc( ncores * PSize * sizeof(uint32_t));
        
   setRandomBI(a, ncores, 0, PSize, P, PSize);
   setRandomBI(b, ncores, 0, PSize, P, PSize);

   #pragma omp parallel for 
   for (uint32_t j=0;j<samples; j++){
       subm_h(r, a, b, pidx);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (double) (end.tv_sec - start.tv_sec);
    elapsed += (double) (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Pidx : %d - Time(Test_Subm %u) : Time : %f\n",pidx, samples, elapsed);
    printf("\033[0m");
    
    release_h();
    free(a);
    free(b);
    free(r);
  }

  return retval;

}


uint32_t test_ntt()
{
  int i,j;
  int pidx=MOD_FR;
  int n_errors=0;
  int levels = 7; 
  int nroots = 1 << levels;
  char roots_f[1000];
  int cusnarks_nroots = 1 << CusnarksGetNRoots();
  const uint32_t *N = CusnarksPGet((mod_t)pidx);
   uint32_t retval=0;

  CusnarksGetFRoots(roots_f, sizeof(roots_f));

  uint32_t *samples = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *result = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *roots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));

  setRandomBI(samples,nroots, N, NWORDS_FR);
  readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);

  memcpy(result, samples, nroots * NWORDS_FR * sizeof(uint32_t));
  ntt_dif_h(samples, roots, levels, 1, 1,1, pidx);
  intt_dif_h(samples, roots, 1, levels, 1, pidx);

  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(FFT DIF-128) : %d/%d\n",pidx, n_errors, j);
  printf("\033[0m");

  ntt_h(samples, roots, levels, 1, 1,1, pidx);
  intt_h(samples, roots, 1, levels, 1,pidx);

  n_errors=0;
  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(FFT DIT-128) : %d/%d\n",pidx, n_errors, j);
  printf("\033[0m");

  free(samples);
  free(result);
  free(roots);
  retval = n_errors;
  return retval;
}

uint32_t test_ntt_parallel(void)
{
  int i,j;
  int pidx=MOD_FR;
  int n_errors=0;
  int cusnarks_nroots = 1 << CusnarksGetNRoots();
  int levels = 7; 
  int nroots = 1 << levels;
  char roots_f[1000];
  int Nrows, Ncols;
  const uint32_t *N = CusnarksPGet((mod_t)pidx);
   uint32_t retval=0;

  Ncols = levels/2;
  Nrows = levels - Ncols;

  CusnarksGetFRoots(roots_f, sizeof(roots_f));

  uint32_t *samples = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *result = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *roots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));

  setRandomBI(samples,nroots, N, NWORDS_FR);
  readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);

  memcpy(result, samples, nroots * NWORDS_FR * sizeof(uint32_t));
  init_h();

  ntt_parallel_h(samples, roots, Nrows, Ncols,1,1, FFT_T_DIT, pidx);
  intt_parallel_h(samples, roots,1, Nrows, Ncols,1, FFT_T_DIT,pidx);

  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(Parallel FFT-DIT) : %d/%d\n",pidx,n_errors, j);
  printf("\033[0m");

  ntt_parallel_h(samples, roots, Nrows, Ncols,1,1, FFT_T_DIF, pidx);
  intt_parallel_h(samples, roots,1, Nrows, Ncols,1, FFT_T_DIF,pidx);

  n_errors = 0;
  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(Parallel FFT-DIF) : %d/%d\n",pidx,n_errors, j);
  printf("\033[0m");

  free(samples);
  free(result);
  free(roots);
  release_h();
  retval = n_errors;
  return retval;
}

uint32_t test_ntt_65K()
{
  int i,j;
  int pidx=MOD_FR;
  int n_errors=0;
  int cusnarks_nroots = 1 << CusnarksGetNRoots();
  char roots_f[1000];
  int levels = 16;
  int nroots = 1 << levels;
  const uint32_t *N = CusnarksPGet((mod_t)pidx);
   uint32_t retval=0;

  CusnarksGetFRoots(roots_f, sizeof(roots_f));

  uint32_t *samples = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *result = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *roots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));

  setRandomBI(samples,nroots, N, NWORDS_FR);
  readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);

  memcpy(result, samples, nroots * NWORDS_FR * sizeof(uint32_t));
  ntt_h(samples, roots, levels,1, 1,1, pidx);
  intt_h(samples, roots, 1, levels,1, pidx);

  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(FFT_DIT-65K) : %d/%d\n",pidx, n_errors, j);
  printf("\033[0m");

  ntt_dif_h(samples, roots, levels,1, 1,1, pidx);
  intt_dif_h(samples, roots, 1, levels,1, pidx);

  n_errors = 0;
  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(FFT-DIF-65K) : %d/%d\n",pidx, n_errors, j);
  printf("\033[0m");

  free(samples);
  free(result);
  free(roots);
  retval = n_errors;
  return retval;
}

uint32_t test_ntt_parallel_65K(void)
{
  int i,j;
  int pidx=MOD_FR;
  int n_errors=0;
  int Nrows = NROWS_65K, Ncols = NCOLS_65K;
  int cusnarks_nroots = 1 << CusnarksGetNRoots();
  int nroots = 1 << (Nrows + Ncols);
  char roots_f[1000];
  int levels=Nrows + Ncols;
  const uint32_t *N = CusnarksPGet((mod_t)pidx);
   uint32_t retval=0;

  CusnarksGetFRoots(roots_f, sizeof(roots_f));

  uint32_t *samples = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *result = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *roots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));

  setRandomBI(samples,nroots, N, NWORDS_FR);
  readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);

  memcpy(result, samples, nroots * NWORDS_FR * sizeof(uint32_t));
  init_h();

  ntt_parallel_h(samples, roots, Nrows, Ncols,1,1, FFT_T_DIT, pidx);
  intt_parallel_h(samples, roots,1, Nrows, Ncols,1, FFT_T_DIT,pidx);

  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
      }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(Parallel FFT-DIT-65K) : %d/%d\n",pidx,n_errors, j);
  printf("\033[0m");

  ntt_parallel_h(samples, roots, Nrows, Ncols,1,1, FFT_T_DIF, pidx);
  intt_parallel_h(samples, roots,1, Nrows, Ncols,1, FFT_T_DIF,pidx);

  n_errors=0;
  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
      }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(Parallel FFT-DIF-65K) : %d/%d\n",pidx,n_errors, j);
  printf("\033[0m");

  free(samples);
  free(result);
  free(roots);
  release_h();
  retval = n_errors;
  return retval;
}

uint32_t test_ntt_1M()
{
  int i,j;
  int pidx=MOD_FR;
  int n_errors=0;
  int cusnarks_nroots = 1 << CusnarksGetNRoots();
  char roots_f[1000];
  int levels = 20;
  int nroots = 1 << levels;
  const uint32_t *N = CusnarksPGet((mod_t)pidx);
  uint32_t retval=0;

  CusnarksGetFRoots(roots_f, sizeof(roots_f));

  uint32_t *samples = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *result = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *roots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));

  setRandomBI(samples,nroots, N, NWORDS_FR);
  readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);

  memcpy(result, samples, nroots * NWORDS_FR * sizeof(uint32_t));
  ntt_h(samples, roots, levels,1, 1,1, pidx);
  intt_h(samples, roots, 1, levels,1, pidx);

  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(FFT-DIT-1M) : %d/%d\n",pidx, n_errors, j);
  printf("\033[0m");

  ntt_dif_h(samples, roots, levels,1, 1,1, pidx);
  intt_dif_h(samples, roots, 1, levels,1, pidx);

  n_errors=0;
  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(FFT-DIF-1M) : %d/%d\n",pidx,n_errors, j);
  printf("\033[0m");

  free(samples);
  free(result);
  free(roots);
  retval = n_errors;
  return retval;
}

uint32_t test_interpol_500K()
{
  int i,j;
  int pidx=MOD_FR;
  int n_errors=0;
  int cusnarks_nroots = 1 << CusnarksGetNRoots();
  char roots_f[1000];
  int levels = 19;
  int nroots = 1 << levels;
  const uint32_t *N = CusnarksPGet((mod_t)pidx);
  uint32_t retval=0;

  CusnarksGetFRoots(roots_f, sizeof(roots_f));

  uint32_t *samples = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *result = (uint32_t *)malloc(2*nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *roots = (uint32_t *)malloc(2*nroots * NWORDS_FR * sizeof(uint32_t));

  setRandomBI(samples,nroots, N, NWORDS_FR);
  readU256DataFile_h(roots,roots_f,cusnarks_nroots,2*nroots);

  memcpy(result, samples, nroots * NWORDS_FR * sizeof(uint32_t));
  memset(&result[nroots * NWORDS_FR],0,nroots*NWORDS_FR*sizeof(uint32_t));
  intt_h(result, roots, 1, levels,2, pidx);
  ntt_h(result, roots, levels+1,1, 1,1, pidx);

  interpol_odd_h(samples, roots,levels, 2, pidx); 

  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[(2*j+1)*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(Interpol-1M) : %d/%d\n",pidx,n_errors, j);
  printf("\033[0m");

  free(samples);
  free(result);
  free(roots);
  retval = n_errors;
  return retval;
}

uint32_t test_interpol_parallel_500K()
{
  int i,j;
  int pidx=MOD_FR;
  int n_errors=0;
  int Nrows = NROWS_1M, Ncols = NCOLS_1M-1;
  int cusnarks_nroots = 1 << CusnarksGetNRoots();
  int nroots = 1 << (Nrows + Ncols);
  char roots_f[1000];
  int levels=Nrows + Ncols;
  const uint32_t *N = CusnarksPGet((mod_t)pidx);
  uint32_t retval=0;

  CusnarksGetFRoots(roots_f, sizeof(roots_f));

  uint32_t *samples = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *result = (uint32_t *)malloc(2*nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *roots = (uint32_t *)malloc(2*nroots * NWORDS_FR * sizeof(uint32_t));

  setRandomBI(samples,nroots, N, NWORDS_FR);
  readU256DataFile_h(roots,roots_f,cusnarks_nroots,2*nroots);

  memcpy(result, samples, nroots * NWORDS_FR * sizeof(uint32_t));
  memset(&result[nroots * NWORDS_FR],0,nroots*NWORDS_FR*sizeof(uint32_t));

  intt_h(result, roots, 1, levels,2, pidx);
  ntt_h(result, roots, levels+1,1, 1,1, pidx);

  init_h();
  interpol_parallel_odd_h(samples, roots,Nrows, Ncols, 2, pidx); 
  release_h();

  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[(2*j+1)*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(Interpol-parallel-1M) : %d/%d\n",pidx,n_errors, j);
  printf("\033[0m");

  free(samples);
  free(result);
  free(roots);
  retval = n_errors;
  return retval;
}


uint32_t test_ntt_parallel_1M(void)
{
  int i,j;
  int pidx=MOD_FR;
  int n_errors=0;
  int Nrows = NROWS_1M, Ncols = NCOLS_1M;
  int cusnarks_nroots = 1 << CusnarksGetNRoots();
  int nroots = 1 << (Nrows + Ncols);
  char roots_f[1000];
  int levels=Nrows + Ncols;
  const uint32_t *N = CusnarksPGet((mod_t)pidx);
  uint32_t retval=0;

  CusnarksGetFRoots(roots_f, sizeof(roots_f));

  uint32_t *samples = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *result = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
  uint32_t *roots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));

  setRandomBI(samples,nroots, N, NWORDS_FR);
  readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);

  memcpy(result, samples, nroots * NWORDS_FR * sizeof(uint32_t));
  init_h();

  ntt_parallel_h(samples, roots, Nrows,Ncols,1,1, FFT_T_DIT,pidx);
  intt_parallel_h(samples, roots,1, Nrows, Ncols,1, FFT_T_DIT,pidx);

  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  } 
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(Parallel FFT-DIT-1M) : %d/%d\n",pidx,n_errors, j);
  printf("\033[0m");

  ntt_parallel_h(samples, roots, Nrows,Ncols,1,1, FFT_T_DIF,pidx);
  intt_parallel_h(samples, roots,1, Nrows,Ncols,1, FFT_T_DIF,pidx);

  n_errors=0;
  for (j=0;j<nroots; j++){
      if (compuBI_h(&samples[j*NWORDS_FR],&result[j*NWORDS_FR], NWORDS_FR)){
          n_errors++;
       }
  } 
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(Parallel FFT-DIF-1M) : %d/%d\n",pidx,n_errors, j);
  printf("\033[0m");

  free(samples);
  free(result);
  free(roots);
  release_h();
  retval = n_errors;
  return retval;
}


uint32_t test_ntt_parallel2D_65K()
{
   int i,j,k;
   int pidx=MOD_FR;
   int n_errors=0;
   int Nrows = NROWS_65K, Ncols = NCOLS_65K;
   int fft_Ny = FFT_SIZEYX_65K, fft_Nx = FFT_SIZEXX_65K;
   int nroots = 1 << (Nrows + Ncols);
   int levels=Nrows + Ncols;
   char roots_f[1000];
   int cusnarks_nroots = 1 << CusnarksGetNRoots();
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   uint32_t retval=0;

   CusnarksGetFRoots(roots_f, sizeof(roots_f));

   uint32_t *samples = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *samples2 = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *roots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *iroots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));

   readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);
   computeIRoots_h(iroots, roots, nroots);
   init_h();

   for (k=0; k < MAX_ITER_65K; k++){
     setRandomBI(samples,nroots, N, NWORDS_FR);
     memcpy(samples2, samples, nroots * NWORDS_FR * sizeof(uint32_t));
 
     intt_parallel2D_h(samples, iroots, 1, Nrows, FFT_SIZEYX_65K, Ncols, FFT_SIZEXX_65K, 1, pidx);
     ntt_parallel2D_h(samples, roots, Nrows, FFT_SIZEYX_65K, Ncols, FFT_SIZEXX_65K,1,1, pidx);

     n_errors = 0;
     for (j=0;j<nroots; j++){
         if (compuBI_h(&samples[j*NWORDS_FR],&samples2[j*NWORDS_FR], NWORDS_FR)){
             n_errors++;
          }
      }
      if (n_errors){
        printf("\033[1;31m");
      }
      printf("Pidx : %d - N errors(FFT 65K) : NTT parallel 2D %d/%d\n",pidx,n_errors, j);
      printf("\033[0m");
      retval += n_errors;
    }
  
    free(samples);
    free(samples2);
    free(roots);
    free(iroots);
    release_h();

    return retval;
}

uint32_t test_nttmul_parallel2D_65K(void)
{
   int i,j,k;
   int pidx=MOD_FR;
   int n_errors=0;
   int Nrows = NROWS_65K, Ncols = NCOLS_65K;
   int fft_Ny = FFT_SIZEYX_65K, fft_Nx = FFT_SIZEXX_65K;
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   int nroots = 1 << (Nrows + Ncols);
   int levels=Nrows + Ncols;
   char roots_f[1000];
   int cusnarks_nroots = 1 << CusnarksGetNRoots();
   uint32_t retval=0;

   CusnarksGetFRoots(roots_f, sizeof(roots_f));

   uint32_t *X1 = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *Y1 = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *X2 = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *Y2 = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *roots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *iroots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));

   readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);
   computeIRoots_h(iroots, roots, nroots);
   init_h();

   for (k=0; k <MAX_ITER_65K; k++){
     memset(X1,0, NWORDS_FR * sizeof(uint32_t)); 
     memset(Y1,0, NWORDS_FR * sizeof(uint32_t)); 
     setRandomBI(X1, nroots/2, N, NWORDS_FR);
     setRandomBI(Y1, nroots/2, N, NWORDS_FR);
     memcpy(X2, X1, nroots * NWORDS_FR * sizeof(uint32_t));
     memcpy(Y2, Y1, nroots * NWORDS_FR * sizeof(uint32_t));
 
     ntt_parallel2D_h(X1, roots, Nrows, FFT_SIZEYX_65K, Ncols, FFT_SIZEXX_65K,1,1, pidx);
     ntt_parallel2D_h(Y1, roots, Nrows, FFT_SIZEYX_65K, Ncols, FFT_SIZEXX_65K,1,1, pidx);

     ntt_h(X2, roots, levels,1, 1,1, pidx);
     ntt_h(Y2, roots, levels,1, 1,1, pidx);
     
     for (j=0; j < nroots; j++){
       montmult_h(&Y1[j*NWORDS_FR], &Y1[j*NWORDS_FR], &X1[j*NWORDS_FR], pidx);
       montmult_h(&Y2[j*NWORDS_FR], &Y2[j*NWORDS_FR], &X2[j*NWORDS_FR], pidx);
     }

     intt_parallel2D_h(Y1, iroots, 1, Nrows, FFT_SIZEYX_65K, Ncols, FFT_SIZEXX_65K, 1, pidx);
     intt_h(Y2, roots, 1,levels, 1, pidx);
     n_errors = 0;
     for (j=0;j<nroots; j++){
         if (compuBI_h(&Y1[j*NWORDS_FR],&Y2[j*NWORDS_FR], NWORDS_FR)){
             n_errors++;
          } 
      }

      if (n_errors){
        printf("\033[1;31m");
      }
      printf("Pidx : %d - N errors(FFTMUL-65K) : NTT parallel 2D File %d/%d\n",pidx,n_errors, j);
      printf("\033[0m");
      retval += n_errors;
    }

    free(X1);
    free(X2);
    free(Y1);
    free(Y2);
    free(roots);
    free(iroots);
    release_h();
    return retval;
}

uint32_t test_ntt_parallel3D_131K()
{
   int i,j,k;
   int pidx=MOD_FR;
   int n_errors=0;
   int Nrows = NROWS_131K, Ncols = NCOLS_131K;
   int nroots = 1 << (Nrows + Ncols);
   int levels=Nrows + Ncols;
   char roots_f[1000];
   int cusnarks_nroots = 1 << CusnarksGetNRoots();
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   uint32_t retval=0;

   CusnarksGetFRoots(roots_f, sizeof(roots_f));
   init_h();

   uint32_t *samples = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *samples2 = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *roots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *iroots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));

   readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);
   computeIRoots_h(iroots, roots, nroots);

   for (k=0; k < MAX_ITER_65K; k++){
     setRandomBI(samples,nroots, N, NWORDS_FR);
     memcpy(samples2, samples, nroots * NWORDS_FR * sizeof(uint32_t));
 
     intt_parallel3D_h(samples, roots, 1, NFFT_X_131K_3D, NFFT_Y_131K_3D, NROWS_131K_3D, FFT_SIZEYX_131K_3D, NCOLS_131K_3D, FFT_SIZEXX_131K_3D, pidx);
     ntt_parallel3D_h(samples, iroots, NFFT_X_131K_3D, NFFT_Y_131K_3D, NROWS_131K_3D, FFT_SIZEYX_131K_3D, NCOLS_131K_3D, FFT_SIZEXX_131K_3D,1, pidx);
     n_errors = 0;

     for (j=0;j<nroots; j++){
         if (compuBI_h(&samples[j*NWORDS_FR],&samples2[j*NWORDS_FR], NWORDS_FR)){
             n_errors++;
          }
      }
      if (n_errors){
        printf("\033[1;31m");
      }
      printf("Pidx : %d - N errors(FFT 131K) : NTT parallel 3D %d/%d\n",pidx,n_errors, j);
      printf("\033[0m");
      retval += n_errors;
    }
  
    free(samples);
    free(samples2);
    free(roots);
    free(iroots);
    release_h();
    return retval;
}


uint32_t test_ntt_parallel2D_1M()
{
   int i,j,k;
   int pidx=MOD_FR;
   int n_errors=0;
   int Nrows = NROWS_1M, Ncols = NCOLS_1M;
   int fft_Ny = FFT_SIZEYX_1M, fft_Nx = FFT_SIZEXX_1M;
   int nroots = 1 << (Nrows + Ncols);
   int levels=Nrows + Ncols;
   char roots_f[1000];
   int cusnarks_nroots = 1 << CusnarksGetNRoots();
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   uint32_t retval=0;

   CusnarksGetFRoots(roots_f, sizeof(roots_f));

   uint32_t *samples = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *samples2 = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *roots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *iroots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));

   readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);
   computeIRoots_h(iroots, roots, nroots);
   init_h();

   for (k=0; k <MAX_ITER_1M; k++){
     setRandomBI(samples, nroots, N, NWORDS_FR);
     memcpy(samples2, samples, nroots * NWORDS_FR * sizeof(uint32_t));
 
     intt_parallel2D_h(samples, roots,1, Nrows, FFT_SIZEYX_1M, Ncols, FFT_SIZEXX_1M,1, pidx);
     ntt_parallel2D_h(samples, iroots, Nrows, FFT_SIZEYX_1M, Ncols, FFT_SIZEXX_1M,1,1, pidx);
 
     n_errors=0; 
     for (j=0;j<nroots; j++){
         if (compuBI_h(&samples[j*NWORDS_FR],&samples2[j*NWORDS_FR], NWORDS_FR)){
             n_errors++;
          } 
      }

      if (n_errors){
        printf("\033[1;31m");
      }
      printf("Pidx : %d - N errors(FFT-1M) : NTT parallel 2D File %d/%d\n",pidx,n_errors, j);
      printf("\033[0m");
      retval += n_errors;
    }

    free(samples);
    free(samples2);
    free(roots);
    free(iroots);
    release_h();
    return retval;
}

uint32_t test_nttmul_parallel2D_1M(void)
{
   int i,j,k;
   int pidx=MOD_FR;
   int n_errors=0;
   int Nrows = NROWS_1M, Ncols = NCOLS_1M;
   int fft_Ny = FFT_SIZEYX_1M, fft_Nx = FFT_SIZEXX_1M;
   int nroots = 1 << (Nrows + Ncols);
   int levels=Nrows + Ncols;
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   char roots_f[1000];
   int cusnarks_nroots = 1 << CusnarksGetNRoots();
   uint32_t retval=0;

   CusnarksGetFRoots(roots_f, sizeof(roots_f));

   uint32_t *X1 = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *Y1 = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *X2 = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *Y2 = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *roots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));
   uint32_t *iroots = (uint32_t *)malloc(nroots * NWORDS_FR * sizeof(uint32_t));

   readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);
   computeIRoots_h(iroots, roots, nroots);
   init_h();

   for (k=0; k <MAX_ITER_1M; k++){
     memset(X1,0, NWORDS_FR * sizeof(uint32_t)); 
     memset(Y1,0, NWORDS_FR * sizeof(uint32_t)); 
     setRandomBI(X1, nroots/2, N, NWORDS_FR);
     setRandomBI(Y1, nroots/2, N, NWORDS_FR);
     memcpy(X2, X1, nroots * NWORDS_FR * sizeof(uint32_t));
     memcpy(Y2, Y1, nroots * NWORDS_FR * sizeof(uint32_t));
 
     ntt_parallel2D_h(X1, roots, Nrows, FFT_SIZEYX_1M, Ncols, FFT_SIZEXX_1M,1,1, pidx);
     ntt_parallel2D_h(Y1, roots, Nrows, FFT_SIZEYX_1M, Ncols, FFT_SIZEXX_1M,1,1, pidx);

     ntt_h(X2, roots, levels,1, 1,1, pidx);
     ntt_h(Y2, roots, levels,1, 1,1, pidx);
     
     for (j=0; j < nroots; j++){
       montmult_h(&Y1[j*NWORDS_FR], &Y1[j*NWORDS_FR], &X1[j*NWORDS_FR], pidx);
       montmult_h(&Y2[j*NWORDS_FR], &Y2[j*NWORDS_FR], &X2[j*NWORDS_FR], pidx);
     }

     intt_parallel2D_h(Y1, iroots,1, Nrows, FFT_SIZEYX_1M, Ncols, FFT_SIZEXX_1M, 1, pidx);
     intt_h(Y2, roots,1,levels,1, pidx);

     n_errors=0;
     for (j=0;j<nroots; j++){
         if (compuBI_h(&Y1[j*NWORDS_FR],&Y2[j*NWORDS_FR], NWORDS_FR)){
             n_errors++;
          } 
      }

      if (n_errors){
        printf("\033[1;31m");
      }
      printf("Pidx : %d - N errors(FFTMUL-1M) : NTT parallel 2D File %d/%d\n",pidx,n_errors, j);
      printf("\033[0m");
      retval += n_errors;
    }

    free(X1);
    free(X2);
    free(Y1);
    free(Y2);
    free(roots);
    free(iroots);
    release_h();
    return retval;
}

uint32_t test_nttmul_randomsize(void)
{
   int i,j,k;
   int pidx=MOD_FR;
   int n_errors=0;
   fft_params_t fft_params;
   int Nrows,Ncols,fft_Nyx,fft_Nxx;
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   uint32_t *X1;
   uint32_t *Y1;
   uint32_t *X2;
   uint32_t *Y2;
   uint32_t *roots, *iroots;
   char roots_f[1000];
   int cusnarks_nroots = 1 << CusnarksGetNRoots();
   uint32_t npoints_raw, npoints, nroots;
   uint32_t retval=0;
   uint32_t min_k=6, max_k = 11;

   CusnarksGetFRoots(roots_f, sizeof(roots_f)); 
   init_h();

   for (k=min_k; k < max_k; k++){
     npoints_raw = (rand() %  ( (1<< k) - (1 << (k - 1))+1)) + (1 << (k-1));
     
     ntt_build_h(&fft_params, npoints_raw);
     npoints = npoints_raw + fft_params.padding;
     nroots = npoints;

     X1 = (uint32_t *)malloc(npoints * NWORDS_FR * sizeof(uint32_t));
     Y1 = (uint32_t *)malloc(npoints * NWORDS_FR * sizeof(uint32_t));
     X2 = (uint32_t *)malloc(npoints * NWORDS_FR * sizeof(uint32_t));
     Y2 = (uint32_t *)malloc(npoints * NWORDS_FR * sizeof(uint32_t));
     roots = (uint32_t *)malloc(npoints* NWORDS_FR * sizeof(uint32_t));
     iroots = (uint32_t *)malloc(npoints* NWORDS_FR * sizeof(uint32_t));

     readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);
     computeIRoots_h(iroots, roots, nroots);

     memset(X1,0, NWORDS_FR * sizeof(uint32_t) * npoints); 
     memset(Y1,0, NWORDS_FR * sizeof(uint32_t) * npoints); 
     setRandomBI(X1, npoints/2, N, NWORDS_FR);
     setRandomBI(Y1, npoints/2, N, NWORDS_FR);
     memcpy(X2, X1, npoints * NWORDS_FR * sizeof(uint32_t));
     memcpy(Y2, Y1, npoints * NWORDS_FR * sizeof(uint32_t));
 
     if (fft_params.fft_type == FFT_T_2D){
        Nrows = fft_params.fft_N[(1<<FFT_T_2D)-1];
        Ncols = fft_params.fft_N[(1<<FFT_T_2D)-2];
        ntt_parallel_h(X1, roots, Nrows, Ncols,1,1, FFT_T_DIT, pidx);
        ntt_parallel_h(Y1, roots, Nrows, Ncols,1,1, FFT_T_DIT, pidx);
     } else if (fft_params.fft_type == FFT_T_3D){
        Nrows = fft_params.fft_N[(1<<FFT_T_3D)-1];
        Ncols = fft_params.fft_N[(1<<FFT_T_3D)-2];
        fft_Nyx = fft_params.fft_N[(1<<FFT_T_3D)-3];
        fft_Nxx = fft_params.fft_N[(1<<FFT_T_3D)-4];
        ntt_parallel2D_h(X1, roots, Nrows, fft_Nyx, Ncols, fft_Nxx,1,1, pidx);
        ntt_parallel2D_h(Y1, roots, Nrows, fft_Nyx, Ncols, fft_Nxx,1,1, pidx);
     } else { 
        printf("\033[1;31m");
        printf("Pidx : %d - FFTMUL-random : Invalid FFT params\n", pidx);
        printf("\033[0m");
        return 1;
     }

     ntt_h(X2, roots, fft_params.levels,1, 1,1, pidx);
     ntt_h(Y2, roots, fft_params.levels,1, 1,1, pidx);
     
     for (j=0; j < npoints; j++){
       montmult_h(&Y1[j*NWORDS_FR], &Y1[j*NWORDS_FR], &X1[j*NWORDS_FR], pidx);
       montmult_h(&Y2[j*NWORDS_FR], &Y2[j*NWORDS_FR], &X2[j*NWORDS_FR], pidx);
     }

     if (fft_params.fft_type == FFT_T_2D){
       intt_parallel_h(Y1, iroots,1, Nrows, Ncols, 1, FFT_T_DIT,pidx);
     } else {
       intt_parallel2D_h(Y1, iroots, 1,Nrows, fft_Nyx, Ncols, fft_Nxx,1, pidx);
     }
     intt_h(Y2, roots,1, fft_params.levels,1, pidx);

     n_errors = 0;
     for (j=0;j<npoints_raw; j++){
         if (compuBI_h(&Y1[j*NWORDS_FR],&Y2[j*NWORDS_FR], NWORDS_FR)){
             n_errors++;
          } 
      }
      retval += n_errors;
      if (n_errors){
        printf("\033[1;31m");
      }
      if (fft_params.fft_type == FFT_T_2D){
         printf("Pidx : %d - N errors(FFTMUL-%d) : 2D[%d/%d] %d/%d\n",pidx,(1<<fft_params.levels),Nrows, Ncols,n_errors, j);
      } else {
         printf("Pidx : %d - N errors(FFTMUL-%d) : 3D[%d/%d/%d/%d] %d/%d\n",pidx,(1<<fft_params.levels),Nrows, fft_Nyx, Ncols, fft_Nxx,n_errors, j);
      }
      printf("\033[0m");

      free(X1);
      free(X2);
      free(Y1);
      free(Y2);
      free(roots);
      free(iroots);
    }
    release_h();
    return retval;

}
uint32_t test_interpol_mul_randomsize(void)
{
   int i,j,k;
   int pidx=1;
   int n_errors=0;
   fft_params_t fft_params;
   int Nrows,Ncols,fft_Nyx,fft_Nxx;
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   uint32_t *X1,*X2,*X3;
   uint32_t *Y1,*Y2,*Y3;
   uint32_t *R;
   uint32_t *roots, *iroots;
   ntt_interpolandmul_t *args;
   char roots_f[1000];
   int cusnarks_nroots = 1 << CusnarksGetNRoots();
   uint32_t npoints_raw, npoints, nroots, nroots2;
   uint32_t retval=0;
   time_t start2, end2, start3, end3;
   //uint32_t min_k=6, max_k = 7;//CusnarksGetNRoots()-1;
   uint32_t min_k=6, max_k = CusnarksGetNRoots()-1;

   CusnarksGetFRoots(roots_f, sizeof(roots_f));
   nroots2 = CusnarksGetNRoots()/2;

   init_h();
   X1 = (uint32_t *)malloc((cusnarks_nroots) * NWORDS_FR * sizeof(uint32_t));
   Y1 = (uint32_t *)malloc((cusnarks_nroots) * NWORDS_FR * sizeof(uint32_t));
   X2 = (uint32_t *)malloc((cusnarks_nroots) * NWORDS_FR * sizeof(uint32_t));
   Y2 = (uint32_t *)malloc((cusnarks_nroots)* NWORDS_FR * sizeof(uint32_t));
   X3 = (uint32_t *)malloc((cusnarks_nroots) * NWORDS_FR * sizeof(uint32_t));
   Y3 = (uint32_t *)malloc((cusnarks_nroots)* NWORDS_FR * sizeof(uint32_t));
   roots = (uint32_t *)malloc((cusnarks_nroots + (1<<nroots2) + (1<<(nroots2+1))) * NWORDS_FR * sizeof(uint32_t));
   args = (ntt_interpolandmul_t *) malloc(sizeof(ntt_interpolandmul_t));

   args->A = X3; args->B = Y3; args->roots = roots; args->pidx=pidx, args->max_threads = get_nprocs_conf();
   args->rstride=2;
   for (k=min_k; k <= max_k; k++){
   //for (k=23; k < 24; k++){
     npoints_raw = (rand() %  ( (1<< k) - (1 << (k - 1))+1)) + (1 << (k-1));
     
     ntt_build_h(&fft_params, npoints_raw);
     npoints = npoints_raw + fft_params.padding;
     nroots = npoints;
     nroots2 = k/2;
     readU256DataFile_h(roots,roots_f,cusnarks_nroots,nroots);
     readU256DataFile_h(&roots[nroots*NWORDS_FR],roots_f, cusnarks_nroots, 1<<nroots2 );
     if (k % 2 == 1){
       readU256DataFile_h(&roots[(nroots+(1<<nroots2))*NWORDS_FR],roots_f, cusnarks_nroots, 1<<(nroots2+1));
     } else {
       readU256DataFile_h(&roots[(nroots+(1<<nroots2))*NWORDS_FR],roots_f, cusnarks_nroots, 1<<(nroots2-1));
     }
     memset(X1,0, NWORDS_FR * sizeof(uint32_t) * npoints); 
     memset(Y1,0, NWORDS_FR * sizeof(uint32_t) * npoints); 
     setRandomBI(X1, npoints/2, N, NWORDS_FR);
     setRandomBI(Y1, npoints/2, N, NWORDS_FR);
     memcpy(X2, X1, npoints * NWORDS_FR * sizeof(uint32_t));
     memcpy(Y2, Y1, npoints * NWORDS_FR * sizeof(uint32_t));
     memcpy(X3, X1, npoints * NWORDS_FR * sizeof(uint32_t));
     memcpy(Y3, Y1, npoints * NWORDS_FR * sizeof(uint32_t));

     Nrows = k/2;
     Ncols = k - Nrows;

     intt_h(X2, roots, 1, fft_params.levels-1,2, pidx);
     intt_h(Y2, roots, 1, fft_params.levels-1,2, pidx);
     ntt_h(X2, roots, fft_params.levels,1, 1,1, pidx);
     ntt_h(Y2, roots, fft_params.levels,1, 1,1, pidx);
     
     for (j=0; j < npoints; j++){
       montmult_h(&Y2[j*NWORDS_FR], &Y2[j*NWORDS_FR], &X2[j*NWORDS_FR], pidx);
     }

     intt_h(Y2, roots,0, fft_params.levels,1, pidx);

     time(&start2);
     R = ntt_interpolandmul_parallel_h(X1,Y1, roots, Nrows, Ncols-1,2, pidx);
     time(&end2);
     n_errors = 0;
     for (j=0;j<npoints_raw; j++){
         if (compuBI_h(&R[j*NWORDS_FR],&Y2[j*NWORDS_FR], NWORDS_FR)){
             n_errors++;
          } 
      }

     if (n_errors){
       printf("\033[1;31m");
     }
     printf("Pidx : %d - N errors(FFTMUL-PARALLEL - %d) : %d/%d\n",pidx,1 << (Nrows+Ncols),n_errors, j);
     printf("\033[0m");
     retval += n_errors;

     args->Nrows = Nrows; args->Ncols=Ncols-1; args->nroots=1<<(Nrows+Ncols); 
     time(&start3);
     R = ntt_interpolandmul_server_h(args);
     time(&end3);
     n_errors = 0;
     for (j=0;j<npoints_raw; j++){
         if (compuBI_h(&R[j*NWORDS_FR],&Y2[j*NWORDS_FR], NWORDS_FR)){
             n_errors++;
          } 
      }
     if (n_errors){
       printf("\033[1;31m");
     }
     printf("Pidx : %d - N errors(FFTMUL-PARALLEL-SERVER - %d) : %d/%d\n",pidx, 1 << (Nrows+Ncols),n_errors, j);
     printf("\033[0m");
     retval += n_errors;
    }

    free(X1);
    free(X2);
    free(X3);
    free(Y1);
    free(Y2);
    free(Y3);
    free(roots);
    free(args);

    release_h();
    return retval;

}

uint32_t test_sort(void)
{
  int n_errors=0;
  uint32_t LEN = 1024;
  uint32_t *samples = (uint32_t *)malloc(LEN*NWORDS_FR*sizeof(uint32_t));
  uint32_t *idx_v = (uint32_t *)malloc(LEN*sizeof(uint32_t));
  uint32_t retval=0;

  setRandomBI(samples, LEN,NULL, NWORDS_FR);

  sortuBI_idx_h(idx_v,samples,LEN,NWORDS_FR, 1);
  for (uint32_t i=0; i< LEN-1; i++){
    if(samples[idx_v[i]*NWORDS_FR+NWORDS_FR-1] > samples[idx_v[i+1]*NWORDS_FR+NWORDS_FR-1]) {
       n_errors++;
    }
  }

   if (n_errors){
    printf("\033[1;31m");
   }
   printf("Pidx : %d, N errors(SORT) : %d/%d\n",MOD_FR, n_errors, LEN);
   printf("\033[0m");
   retval += n_errors;

  free(samples);
  free(idx_v);
  return retval;
}


uint32_t test_interpol_mul_randomsize_prof(void)
{
   int i,j,k;
   int pidx=MOD_FR;
   int n_errors=0;
   fft_params_t fft_params;
   int Nrows,Ncols,fft_Nyx,fft_Nxx;
   const uint32_t *N = CusnarksPGet((mod_t)pidx);
   uint32_t *X1,*X2,*X3;
   uint32_t *Y1,*Y2,*Y3;
   uint32_t *R;
   uint32_t *roots, *iroots;
   ntt_interpolandmul_t *args;
   char roots_f[1000];
   int cusnarks_nroots = 1 << CusnarksGetNRoots();
   uint32_t npoints_raw, npoints, nroots, nroots2;
   uint32_t retval=0;
   //uint32_t min_k=6, max_k = 6;//CusnarksGetNRoots()-1;
   uint32_t min_k=22, max_k = CusnarksGetNRoots()-1;
   struct timespec start, end;
   double elapsed=0.0;
     
   CusnarksGetFRoots(roots_f, sizeof(roots_f));
   nroots2 = CusnarksGetNRoots()/2;

   init_h();
   X1 = (uint32_t *)malloc((cusnarks_nroots) * NWORDS_FR * sizeof(uint32_t));
   Y1 = (uint32_t *)malloc((cusnarks_nroots) * NWORDS_FR * sizeof(uint32_t));
   roots = (uint32_t *)malloc((cusnarks_nroots + (1<<nroots2) + (1<<(nroots2+1))) * NWORDS_FR * sizeof(uint32_t));
   args = (ntt_interpolandmul_t *) malloc(sizeof(ntt_interpolandmul_t));

   args->A = X1; args->B = Y1; args->roots = roots; args->pidx=pidx, args->max_threads = get_nprocs_conf();
   args->rstride=2;
   readU256DataFile_h(roots,roots_f,cusnarks_nroots,cusnarks_nroots);
   memset(X1,0, NWORDS_FR * sizeof(uint32_t) * cusnarks_nroots); 
   memset(Y1,0, NWORDS_FR * sizeof(uint32_t) * cusnarks_nroots); 
   setRandomBI(X1, cusnarks_nroots/2, N, NWORDS_FR);
   setRandomBI(Y1, cusnarks_nroots/2, N, NWORDS_FR);

   for (k=min_k; k <= max_k; k++){
     clock_gettime(CLOCK_MONOTONIC, &start);
     npoints_raw = 1 << k;
     
     ntt_build_h(&fft_params, npoints_raw);
     npoints = npoints_raw + fft_params.padding;
     nroots = npoints;

     Nrows = k/2;
     Ncols = k - Nrows;

     args->Nrows = Nrows; args->Ncols=Ncols-1; args->nroots=1<<(Nrows+Ncols); 
     R = ntt_interpolandmul_server_h(args);

     clock_gettime(CLOCK_MONOTONIC, &end);
     elapsed = (double) (end.tv_sec - start.tv_sec);
     elapsed += (double) (end.tv_nsec - start.tv_nsec) / 1000000000.0;
     //printf("Time(FFTMUL-PARALLEL-SERVER - %d) Time : %f\n", 1 << (Nrows+Ncols), elapsed);
     printf("Pidx : %d - Log2 Constraints : %d, Time : %lld\n", pidx,k,(long long unsigned int) (elapsed*1000));
    }
    
    
    printf("\033[0m");
    free(X1);
    free(Y1);
    free(roots);
    free(args);

    release_h();
    return 0;

}


uint32_t  test_setgetbit()
{
  int pidx = MOD_FR;
  const uint32_t *N = CusnarksPGet((mod_t)pidx);
  uint32_t x[NWORDS_FR];
  uint32_t i,c,b;
  uint32_t n_errors=0;
  uint32_t retval=0;

  for (i=0; i < 10000; i++){
     setRandomBI(x,1, N, NWORDS_FR);  
     c = rand() % (NWORDS_FR * NBITS_WORD);
     setbituBI_h(x,c);
     if (getbituBI_h(x,c) == 0){
       n_errors++;
     }
     
  }
  if (n_errors){
    printf("\033[1;31m");
  }
  printf("Pidx : %d - N errors(SET/GET BIT) : %d/%d\n",pidx, n_errors, 1000);
  printf("\033[0m");
  retval += n_errors;

  return retval;
     
}

uint32_t test_mul_ext_prof(void)
{
  uint32_t nsamples=1<<25;
  uint32_t retval=0;
  struct timespec start, end;
  double elapsed=0.0;
  uint32_t ncores = get_nprocs_conf();
  int pidx;
   
  for (pidx=MOD_FP; pidx < MOD_N; pidx++){
    const uint32_t *P =  CusnarksPGet((mod_t)pidx);
    const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);

    uint32_t *a = (uint32_t *) malloc( 2*ncores * PSize * sizeof(uint32_t));
    uint32_t *b = (uint32_t *) malloc( 2*ncores * PSize * sizeof(uint32_t));
    uint32_t *r = (uint32_t *) malloc( 2*ncores * PSize * sizeof(uint32_t));
      
    setRandomBI(a, 2*ncores, 0, 8, P, PSize);
    setRandomBI(b, 2*ncores, 0, 8, P, PSize);
  
    init_h();
    clock_gettime(CLOCK_MONOTONIC, &start);
  
    #pragma omp parallel for 
    for(uint32_t i=0; i < nsamples; i++){
      uint32_t tid = omp_get_thread_num();
      montmult_ext_h(&r[tid * 2* PSize], 
  		    (const uint32_t *) &a[tid*2*PSize],
  		    (const uint32_t *) &b[tid*2*PSize], pidx);
    }
   
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (double) (end.tv_sec - start.tv_sec);
    elapsed += (double) (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Pidx : %d - Time(Test_Mul Ext. N samples :  %u) Time : %f\n", pidx, nsamples, elapsed);
    printf("\033[0m");
    release_h();
  }

  return retval;

}

uint32_t test_ec2aff(uint32_t ec2)
{
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords_ecp, nwords_res;
 uint32_t necp;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif

 uint32_t ec2_offset = 0;
 uint32_t indims = ECP_JAC_INDIMS;
 uint32_t outdims = ECP_JAC_OUTDIMS;
 void (*jac2aff_cb)(uint32_t *, uint32_t *, t_uint64 , uint32_t, uint32_t) = &ec_jac2aff_h;
 int32_t (*iseq_cb)(const uint32_t *, const uint32_t *) = &ec_iseq_h;

 if (ec2){
   ec2_offset = 2;
   indims = ECP2_JAC_INDIMS;
   outdims = ECP2_JAC_OUTDIMS;
   jac2aff_cb = &ec2_jac2aff_h;
   iseq_cb = &ec2_iseq_h;
   
 }
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx+=2){
     init_h();
     getDataFileSize(&nwords_ecp, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_res, ec_aff_filename[(pidx+pidx_offset)/2+ec2_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);
     necp = nwords_ecp / (PSize * outdims);

     uint32_t *a = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *c = (uint32_t *) malloc( nwords_res * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords_res * sizeof(uint32_t));
  
     readDataFile(a, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(r, ec_aff_filename[(pidx+pidx_offset)/2+ec2_offset]);

     to_montgomeryN_h(a,a,necp*outdims,pidx);
     jac2aff_cb(c, a, necp, pidx, 1);
     from_montgomeryN_h(c,c,necp*indims,pidx,0);

     for (i=0; i < necp; i++){
       if (!iseq_cb(&r[i*PSize*indims],&c[i*PSize*indims])){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - EC2 : %d N errors(Test_ECToAff) : %d/%d\n",pidx, ec2, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(c);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_ec_jacadd(uint32_t ec2)
{
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords_ecp, nwords_res;
 uint32_t necp;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 
 uint32_t ec2_offset = 0;
 uint32_t indims = ECP_JAC_INDIMS;
 uint32_t outdims = ECP_JAC_OUTDIMS;
 void (*jac2aff_cb)(uint32_t *, uint32_t *, t_uint64 , uint32_t, uint32_t) = &ec_jac2aff_h;
 int32_t (*iseq_cb)(const uint32_t *, const uint32_t *) = &ec_iseq_h;
 void (*jacadd_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t ) = &ec_jacadd_h;

 if (ec2){
   ec2_offset = 2;
   indims = ECP2_JAC_INDIMS;
   outdims = ECP2_JAC_OUTDIMS;
   jac2aff_cb = &ec2_jac2aff_h;
   iseq_cb = &ec2_iseq_h;
   jacadd_cb = &ec2_jacadd_h;
   
 }
 uint32_t c[NWORDS_FP * outdims], c1[NWORDS_FP * indims];

 for (pidx = MOD_FP; pidx < MOD_N; pidx+=2){
     init_h();
     getDataFileSize(&nwords_ecp, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_res, ec_add_filename[(pidx+pidx_offset)/2+ec2_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);
     necp = nwords_ecp / (PSize * outdims);

     uint32_t *a = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *b = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords_res * sizeof(uint32_t));
  
     readDataFile(a, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(b, ecp2_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(r, ec_add_filename[(pidx+pidx_offset)/2+ec2_offset]);

     to_montgomeryN_h(a,a,necp*outdims,pidx);
     to_montgomeryN_h(b,b,necp*outdims,pidx);

     for (i=0; i < necp; i++){
       jacadd_cb(c, &a[i*PSize*outdims], &b[i*PSize*outdims], pidx);
       jac2aff_cb(c1, c, 1, pidx, 1);
       from_montgomeryN_h(c1,c1,indims,pidx,0);
       if (!iseq_cb(&r[i*PSize*indims],c1)){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - EC2 : %d,  N errors(Test_ECAdd) : %d/%d\n",pidx, ec2, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(b);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_ec_jacaddmixed(uint32_t ec2)
{
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords_ecp, nwords_res;
 uint32_t necp;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 uint32_t ec2_offset = 0;
 uint32_t indims = ECP_JAC_INDIMS;
 uint32_t outdims = ECP_JAC_OUTDIMS;
 void (*jac2aff_cb)(uint32_t *, uint32_t *, t_uint64 , uint32_t, uint32_t) = &ec_jac2aff_h;
 int32_t (*iseq_cb)(const uint32_t *, const uint32_t *) = &ec_iseq_h;
 void (*jacaddmixed_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t ) = &ec_jacaddmixed_h;

 if (ec2){
   ec2_offset = 2;
   indims = ECP2_JAC_INDIMS;
   outdims = ECP2_JAC_OUTDIMS;
   jac2aff_cb = &ec2_jac2aff_h;
   iseq_cb = &ec2_iseq_h;
   jacaddmixed_cb = &ec2_jacaddmixed_h;
   
 }
 uint32_t c[NWORDS_FP * outdims], c1[NWORDS_FP * indims];
 for (pidx = MOD_FP; pidx < MOD_N; pidx+=2){
     init_h();
     getDataFileSize(&nwords_ecp, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_res, ec_add_filename[(pidx+pidx_offset)/2+ec2_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);
     necp = nwords_ecp / (PSize * outdims);

     uint32_t *a = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *a1 = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *b = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords_res * sizeof(uint32_t));
  
     readDataFile(a, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(b, ecp2_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(r, ec_add_filename[(pidx+pidx_offset)/2+ec2_offset]);

     to_montgomeryN_h(a,a,necp*outdims,pidx);
     to_montgomeryN_h(b,b,necp*outdims,pidx);

     jac2aff_cb(a1, a, necp, pidx, 1);
     for (i=0; i < necp; i++){
       jacaddmixed_cb(c, &a1[i*PSize*indims], &b[i*PSize*outdims], pidx);
       jac2aff_cb(c1, c, 1, pidx, 1);
       from_montgomeryN_h(c1,c1,indims,pidx,0);
       if (!iseq_cb(&r[i*PSize*indims],c1)){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - EC2 : %d,  N errors(Test_ECAddMixed) : %d/%d\n",pidx, ec2,n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(a1);
     free(b);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_ec_jacscmul(uint32_t ec2)
{
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords_ecp, nwords_res, nwords_scl;
 uint32_t necp;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 
 uint32_t ec2_offset = 0;
 uint32_t indims = ECP_JAC_INDIMS;
 uint32_t outdims = ECP_JAC_OUTDIMS;
 void (*jac2aff_cb)(uint32_t *, uint32_t *, t_uint64 , uint32_t, uint32_t) = &ec_jac2aff_h;
 int32_t (*iseq_cb)(const uint32_t *, const uint32_t *) = &ec_iseq_h;
 void (*jacscmul_cb)(uint32_t *, uint32_t *, uint32_t *, uint32_t, uint32_t , uint32_t ) = &ec_jacscmul_h;

 if (ec2){
   ec2_offset = 2;
   indims = ECP2_JAC_INDIMS;
   outdims = ECP2_JAC_OUTDIMS;
   jac2aff_cb = &ec2_jac2aff_h;
   iseq_cb = &ec2_iseq_h;
   jacscmul_cb = &ec2_jacscmul_h;
   
 }

 for (pidx = MOD_FP; pidx < MOD_N; pidx+=2){
     init_h();
     getDataFileSize(&nwords_ecp, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_scl, scl_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_res, ec_mul_filename[(pidx+pidx_offset)/2+ec2_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);
     necp = nwords_ecp / (PSize * outdims);

     uint32_t *a = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *scl = (uint32_t *) malloc( nwords_scl * sizeof(uint32_t));
     uint32_t *c = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *c1 = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords_res * sizeof(uint32_t));
  
     readDataFile(a, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(scl, scl_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(r, ec_mul_filename[(pidx+pidx_offset)/2+ec2_offset]);

     to_montgomeryN_h(a,a,necp*outdims,pidx);

     jacscmul_cb(c, scl, a, necp, pidx, 0);
     jac2aff_cb(c1, c, necp, pidx, 1);
     from_montgomeryN_h(c,c1,necp*indims,pidx,0);

     for (i=0; i < necp; i++){
       if (!ec_iseq_h(&r[i*PSize*indims],&c[i*PSize*indims])){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d -  EC2 : %d, N errors(Test_ECSCMul) : %d/%d\n",pidx,  ec2,n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(scl);
     free(c);
     free(c1);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}

uint32_t test_ec_jacreduce_opt(uint32_t ec2)
{
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;
 struct timespec start, end;
 double elapsed=0.0;
 uint32_t max_order = MAX_U256_BSELM;
 uint32_t min_order = 1;

 t_uint64 nwords_ecp, nwords_res, nwords_scl;
 uint32_t necp;
 
 uint32_t ec2_offset = 0;
 uint32_t indims = ECP_JAC_INDIMS;
 uint32_t outdims = ECP_JAC_OUTDIMS;
 void (*jac2aff_cb)(uint32_t *, uint32_t *, t_uint64 , uint32_t, uint32_t) = &ec_jac2aff_h;
 int32_t (*iseq_cb)(const uint32_t *, const uint32_t *) = &ec_iseq_h;

 if (ec2){
   ec2_offset = 2;
   indims = ECP2_JAC_INDIMS;
   outdims = ECP2_JAC_OUTDIMS;
   jac2aff_cb = &ec2_jac2aff_h;
   iseq_cb = &ec2_iseq_h;
   
 }
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx+=2){
     init_h();
     getDataFileSize(&nwords_ecp, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_scl, scl_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_res, ec_rdc_filename[(pidx+pidx_offset)/2+ec2_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);
     necp = nwords_ecp / (PSize * outdims);

     uint32_t *a = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *a1 = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *scl = (uint32_t *) malloc( nwords_scl * sizeof(uint32_t));
     uint32_t *c = (uint32_t *) malloc( 2*nwords_res * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords_res * sizeof(uint32_t));
     jacadd_reduced_t *args = (jacadd_reduced_t *)malloc(sizeof(jacadd_reduced_t));
  
     readDataFile(a, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(scl, scl_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(r, ec_rdc_filename[(pidx+pidx_offset)/2+ec2_offset]);

     to_montgomeryN_h(a,a,necp*outdims,pidx);
     jac2aff_cb(a1, a, necp, pidx, 1);

     args->out_ep = c;
     args->scl = scl;
     args->x = a1;
     args->n = necp;
     args->ec_table = NULL;
     args->pidx = MOD_FP;
     args->ec2 = ec2;
     args->filename = NULL;
     args->pippen = 0;

     for (uint32_t i=max_order; i >= min_order; i--){
       args->max_threads = get_nprocs_conf();
       // Multiply points
       args->order = i;
       clock_gettime(CLOCK_MONOTONIC, &start);
       ec_jacreduce_server_h(args);
       clock_gettime(CLOCK_MONOTONIC, &end);
       elapsed = (double) (end.tv_sec - start.tv_sec);
       elapsed += (double) (end.tv_nsec - start.tv_nsec) / 1000000000.0;

       from_montgomeryN_h(c,c,indims,pidx,0);

       if (!iseq_cb(r,c)){
          n_errors1++;
       }
    
       if (n_errors1){
         printf("\033[1;31m");
       }

       printf("Pidx %d - N errors(JACREDUCE_OPT-EC2 : %d : N points : %d, Order : %d) : %d - Time : %f\n", 
           pidx,ec2,necp,i, n_errors1, elapsed);
    
       printf("\033[0m");
       n_errors+=n_errors1;
       n_errors1=0;
    }
  

    free(a);
    free(a1);
    free(scl);
    free(c);
    free(r);
    free(args);
    release_h();
  }

  retval = n_errors;
  return retval;
}

uint32_t test_ec_jacreduce_pippen(uint32_t ec2)
{
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;
 struct timespec start, end;
 double elapsed=0.0;

 t_uint64 nwords_ecp, nwords_res, nwords_scl;
 uint32_t necp;
 
 uint32_t ec2_offset = 0;
 uint32_t indims = ECP_JAC_INDIMS;
 uint32_t outdims = ECP_JAC_OUTDIMS;
 void (*jac2aff_cb)(uint32_t *, uint32_t *, t_uint64 , uint32_t, uint32_t) = &ec_jac2aff_h;
 int32_t (*iseq_cb)(const uint32_t *, const uint32_t *) = &ec_iseq_h;

 if (ec2){
   ec2_offset = 2;
   indims = ECP2_JAC_INDIMS;
   outdims = ECP2_JAC_OUTDIMS;
   jac2aff_cb = &ec2_jac2aff_h;
   iseq_cb = &ec2_iseq_h;
   
 }
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx+=2){
     init_h();
     getDataFileSize(&nwords_ecp, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_scl, scl_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_res, ec_rdc_filename[(pidx+pidx_offset)/2+ec2_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);
     necp = nwords_ecp / (PSize * outdims);

     uint32_t *a = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *a1 = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *scl = (uint32_t *) malloc( nwords_scl * sizeof(uint32_t));
     uint32_t *c = (uint32_t *) malloc( 2*nwords_res * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords_res * sizeof(uint32_t));
     jacadd_reduced_t *args = (jacadd_reduced_t *)malloc(sizeof(jacadd_reduced_t));
  
     readDataFile(a, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(scl, scl_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(r, ec_rdc_filename[(pidx+pidx_offset)/2+ec2_offset]);

     to_montgomeryN_h(a,a,necp*outdims,pidx);
     jac2aff_cb(a1, a, necp, pidx, 1);

     args->out_ep = c;
     args->scl = scl;
     args->x = a1;
     args->n = necp;
     args->pidx = MOD_FP;
     args->ec2 = ec2;
     args->pippen = 1;
     args->init = 1;
     args->combine = 1;

     args->max_threads = get_nprocs_conf();
     // Multiply points
     clock_gettime(CLOCK_MONOTONIC, &start);
     ec_jacreduce_server_h(args);
     clock_gettime(CLOCK_MONOTONIC, &end);
     elapsed = (double) (end.tv_sec - start.tv_sec);
     elapsed += (double) (end.tv_nsec - start.tv_nsec) / 1000000000.0;

     from_montgomeryN_h(c,c,indims,pidx,0);

     if (!iseq_cb(r,c)){
        n_errors1++;
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }

     printf("Pidx %d - N errors(JACREDUCE_PIPPEN-EC2 : %d : N points : %d, Order : %d) : %d - Time : %f\n", 
         pidx,ec2,necp,i, n_errors1, elapsed);
    
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;
  

     free(a);
     free(a1);
     free(scl);
     free(c);
     free(r);
     free(args);
     release_h();
  }

  retval = n_errors;
  return retval;
}

uint32_t  test_ec_jacreduce_precompute(uint32_t ec2, uint32_t file, uint32_t compute_table=0)
{
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;
 struct timespec start, end;
 double elapsed=0.0;
 uint32_t max_order = MAX_U256_BSELM;
 uint32_t min_order = 1;
 uint32_t n_tables;
 uint32_t ec2_offset = 0;
 uint32_t indims = ECP_JAC_INDIMS;
 uint32_t outdims = ECP_JAC_OUTDIMS;
 void (*jac2aff_cb)(uint32_t *, uint32_t *, t_uint64, uint32_t, uint32_t) = &ec_jac2aff_h;
 int32_t (*iseq_cb)(const uint32_t *, const uint32_t *) = &ec_iseq_h;
 void (*inittable_cb)(uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t table_order, uint32_t pidx, uint32_t add_last) = &ec_inittable_h;
 if (ec2) {
        indims = ECP2_JAC_INDIMS;
        outdims = ECP2_JAC_OUTDIMS;
        ec2_offset = 2;
        jac2aff_cb = &ec2_jac2aff_h;
        iseq_cb = &ec2_iseq_h;
        inittable_cb = &ec2_inittable_h;
 }

 t_uint64 nwords_ecp, nwords_res, nwords_scl;
 uint32_t necp;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 for (pidx = MOD_FP; pidx < MOD_N; pidx+=2){
     init_h();
     getDataFileSize(&nwords_ecp, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_scl, scl_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_res, ec_rdc_filename[(pidx+pidx_offset)/2+ec2_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);
     necp = nwords_ecp / (PSize * outdims);

     uint32_t *a = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *a1 = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *scl = (uint32_t *) malloc( nwords_scl * sizeof(uint32_t));
     uint32_t *c = (uint32_t *) malloc( 2*nwords_res * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords_res * sizeof(uint32_t));
     jacadd_reduced_t *args = (jacadd_reduced_t *)malloc(sizeof(jacadd_reduced_t));
  
     readDataFile(a, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(scl, scl_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(r, ec_rdc_filename[(pidx+pidx_offset)/2+ec2_offset]);

     to_montgomeryN_h(a,a,necp*outdims,pidx);
     jac2aff_cb(a1, a, necp, pidx, 1);

     args->out_ep = c;
     args->scl = scl;
     args->n = necp;
     args->ec2 = ec2;
     args->pidx = MOD_FP;
     args->filename = NULL;
     args->max_threads = 0;

     if (compute_table) {
       min_order = 2;
     }

     for (uint32_t i=max_order; i >= min_order; i--){
        n_tables = (necp + i - 1) / i;
        uint32_t *ec_table = (uint32_t *) malloc( (1 << i) * (n_tables * NWORDS_FP * outdims) * sizeof(uint32_t) );
        uint32_t *ec_table2 = (uint32_t *) malloc( (1 << i) * (n_tables * NWORDS_FP * outdims) * sizeof(uint32_t) );

        if (!compute_table){
           inittable_cb(a1, ec_table, necp, i, 0, 1);
           jac2aff_cb(ec_table2, ec_table, (1<<i) * n_tables , 0, 1);
        }

       args->ec_table = ec_table2;
       args->order = i;
       args->compute_table=0;

       if (file){
          FILE *ifp = fopen(ec_table_filename,"wb");
          fwrite(ec_table2, sizeof(uint32_t),n_tables * (1<<i) * NWORDS_FP * indims,ifp);
          args->offset = (i << EC_JACREDUCE_BATCH_SIZE) *indims * NWORDS_FP *sizeof(uint32_t)<< i;
          fclose(ifp);
          args->filename = ec_table_filename;
          args->total_words = (1 << i) * (n_tables * NWORDS_FP * indims);
       } else if (compute_table) {
          args->compute_table = 1;
          args->total_words =  n_tables;
          args->x = a1;
       } else if (i==1){
          args->ec_table = a1;
       }


       clock_gettime(CLOCK_MONOTONIC, &start);
       ec_jacreduce_server_h(args);
       clock_gettime(CLOCK_MONOTONIC, &end);
       elapsed = (double) (end.tv_sec - start.tv_sec);
       elapsed += (double) (end.tv_nsec - start.tv_nsec) / 1000000000.0;

       from_montgomeryN_h(c,c,indims,pidx,0);

       if (!iseq_cb(r,c)){
          n_errors1++;
       }
    
       if (n_errors1){
           printf("\033[1;31m");
       }

       printf("Pidx %d - N errors(JACREDUCE_OPT [EC2 : %d, File : %d, N Points : %d, Order : %d]) : %d - Time : %f\n",
             pidx,ec2,file,necp,i,n_errors1, elapsed);
    
       printf("\033[0m");
       n_errors+=n_errors1;
       n_errors1=0;
       free(ec_table);
       free(ec_table2);
    }
  

    free(a);
    free(a1);
    free(scl);
    free(c);
    free(r);
    free(args);
    release_h();
  }

  retval = n_errors;
  return retval;
}


uint32_t test_ec_jacdouble(uint32_t ec2)
{
 int i;
 int pidx;
 uint32_t pidx_offset=0;
 int n_errors=0, n_errors1=0;
 uint32_t retval=0;

 t_uint64 nwords_ecp, nwords_res;
 uint32_t necp;
 
 #ifdef _BLS12381
 pidx_offset = 1 * MOD_N;
 #endif
 

 uint32_t ec2_offset = 0;
 uint32_t indims = ECP_JAC_INDIMS;
 uint32_t outdims = ECP_JAC_OUTDIMS;
 void (*jac2aff_cb)(uint32_t *, uint32_t *, t_uint64 , uint32_t, uint32_t) = &ec_jac2aff_h;
 int32_t (*iseq_cb)(const uint32_t *, const uint32_t *) = &ec_iseq_h;
 void (*jacdouble_cb)(uint32_t *, uint32_t *, uint32_t ) = &ec_jacdouble_h;

 if (ec2){
   ec2_offset = 2;
   indims = ECP2_JAC_INDIMS;
   outdims = ECP2_JAC_OUTDIMS;
   jac2aff_cb = &ec2_jac2aff_h;
   iseq_cb = &ec2_iseq_h;
   jacdouble_cb = &ec2_jacdouble_h;
   
 }
 uint32_t c[NWORDS_FP * outdims], c1[NWORDS_FP * indims];

 for (pidx = MOD_FP; pidx < MOD_N; pidx+=2){
     init_h();
     getDataFileSize(&nwords_ecp, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     getDataFileSize(&nwords_res, ec_dbl_filename[(pidx+pidx_offset)/2+ec2_offset]);
     const uint32_t PSize = CusnarksPSizeGet((mod_t)pidx);
     necp = nwords_ecp / (PSize * outdims);

     uint32_t *a = (uint32_t *) malloc( nwords_ecp * sizeof(uint32_t));
     uint32_t *r = (uint32_t *) malloc( nwords_res * sizeof(uint32_t));
  
     readDataFile(a, ecp1_filename[(pidx+pidx_offset)/2+ec2_offset]);
     readDataFile(r, ec_dbl_filename[(pidx+pidx_offset)/2+ec2_offset]);

     to_montgomeryN_h(a,a,necp*outdims,pidx);

     for (i=0; i < necp; i++){
       jacdouble_cb(c, &a[i*PSize*outdims], pidx);
       jac2aff_cb(c1, c, 1, pidx, 1);
       from_montgomeryN_h(c1,c1,indims,pidx,0);
       if (!iseq_cb(&r[i*PSize*indims],c1)){
          n_errors1++;
       }
     }
  
     if (n_errors1){
       printf("\033[1;31m");
     }
     printf("Pidx %d - EC2 : %d,  N errors(Test_ECDouble) : %d/%d\n",pidx,ec2, n_errors1, i);
     printf("\033[0m");
     n_errors+=n_errors1;
     n_errors1=0;

     free(a);
     free(r);
     release_h();
   }

  retval = n_errors;
  return retval;
}


uint32_t test_transpose(void)
{
  // 2x4 to 64x128
  int min_m=1, max_m=10;
  int nrows, ncols;
  int n_errors=0;
  int i,j;
  uint32_t *samples, *samples2, *result;
  const uint32_t *N = CusnarksPGet((mod_t)1);
  uint32_t retval=0;


  for (i=min_m; i< max_m; i++){
     nrows = 1 << i;
     ncols = 1 << (i+1);
    
     samples = (uint32_t *)malloc(nrows*ncols * NWORDS_FR * sizeof(uint32_t));
     samples2 = (uint32_t *)malloc(nrows*ncols * NWORDS_FR * sizeof(uint32_t));
     result = (uint32_t *)malloc(nrows*ncols * NWORDS_FR * sizeof(uint32_t));

     setRandomBI(samples,nrows*ncols, N, NWORDS_FR); 

     transpose_h(result, samples, nrows, ncols);
     transposeBlock_h(samples2, samples, nrows, ncols, TRANSPOSE_BLOCK_SIZE);
     transpose_h(samples, nrows, ncols);
    
      
     n_errors=0;
     for(j=0; j< nrows*ncols; j++){
       if (!equBI_h(&samples[j*NWORDS_FR], &result[j*NWORDS_FR], NWORDS_FR)){
         n_errors++;
       }
     }
     if (n_errors){
      printf("\033[1;31m");
     }
     printf("Pidx : %d - N errors(Transpose %dx%d) : %d/%d\n",MOD_FR, nrows,ncols,n_errors, nrows*ncols);
     
     n_errors=0;
     for(j=0; j< nrows*ncols; j++){
       if (!equBI_h(&samples2[j*NWORDS_FR], &result[j*NWORDS_FR], NWORDS_FR)){
         n_errors++;
       }
     }
     printf("\033[0m");
     
     /*
     for(j=0; j< nrows*ncols; j++){
       printU256Number(&result[j*NWORDS_256BIT]);
     }
     printf("\n\n");
     for(j=0; j< nrows*ncols; j++){
       printU256Number(&samples[j*NWORDS_256BIT]);
     }
     */
     
     if (n_errors){
      printf("\033[1;31m");
     }
     printf("Pidx : %d - N errors(Transpose2 %dx%d) : %d/%d\n",MOD_FR, nrows,ncols,n_errors, nrows*ncols);
     printf("\033[0m");
     retval += n_errors;

     free(samples);
     free(samples2);
     free(result);
  }
  return retval;
}

uint32_t test_transpose_square(void)
{
  int min_m=3, max_m=13;
  int nrows;
  int n_errors=0;
  int i,j;
  uint32_t *samples, *result;
  const uint32_t *N = CusnarksPGet((mod_t)1);
  double start, end;
  uint32_t retval=0;

  samples = (uint32_t *)malloc((1<<(max_m-1))*(1<<(max_m-1)) * NWORDS_FR * sizeof(uint32_t));
  result = (uint32_t *)malloc((1<<(max_m-1))*(1<<(max_m-1)) * NWORDS_FR * sizeof(uint32_t));

  for (i=min_m=3; i< max_m; i++){
     nrows = 1 << i;
    
     setRandomBI(samples,nrows*nrows, N, NWORDS_FR); 
     
     transpose_h(result, samples, nrows, nrows);
     transpose_h(samples, nrows, nrows);
     
     n_errors=0;
     for(j=0; j< nrows*nrows; j++){
       if (!equBI_h(&samples[i*NWORDS_FR], &result[i*NWORDS_FR], NWORDS_FR)){
         n_errors++;
       }
     }
     if (n_errors){
      printf("\033[1;31m");
     }
     printf("Pidx : %d - N errors(Transpose square %dx%d) : %d/%d\n",MOD_FR,nrows,nrows,n_errors, nrows*nrows);
     printf("\033[0m");
     retval += n_errors;

  }
  free(samples);
  free(result);

  return retval;
}

int main()
{
  uint32_t retval;

  retval+=test_mul_prof();  // Profile montgomery mul 
  retval+=test_mul_ext_prof();  // Profile montgomery mul 
  retval+=test_interpol_mul_randomsize_prof();
  retval+=test_addm_prof();  // Profile addm
  retval+=test_subm_prof();  // Profile subm

  retval+=test_addm();  
  retval+=test_subm();  
  retval+=test_tomont();  
  retval+=test_mul() ;  
  retval+=test_square();
  retval+=test_inv();

  retval+=test_addm_ext();  
  retval+=test_subm_ext();  
  retval+=test_mul_ext() ;  
  retval+=test_square_ext();
  retval+=test_inv_ext();

  retval+=test_setgetbit();

  retval+=test_sort();

  retval+=test_transpose_square();
  retval+=test_transpose();

  retval+=test_ntt();
  retval+=test_ntt_parallel();

  retval+=test_ntt_65K();
  retval+=test_ntt_parallel_65K();
  retval+=test_ntt_parallel2D_65K(); 
  retval+=test_nttmul_parallel2D_65K();

  retval+=test_ntt_parallel3D_131K(); // Forward FFT

  retval+=test_ntt_1M();
  retval+=test_ntt_parallel_1M();
  retval+=test_ntt_parallel2D_1M();
  retval+=test_nttmul_parallel2D_1M();

  retval+=test_nttmul_randomsize();

  retval+=test_interpol_500K();

  retval+=test_interpol_parallel_500K();

  retval+=test_interpol_mul_randomsize();

  //EC1
  retval+=test_ec2aff(0);
  retval+=test_ec_jacdouble(0);
  retval+=test_ec_jacadd(0);
  retval+=test_ec_jacaddmixed(0);
  retval+= test_ec_jacscmul(0);
  retval+=test_ec_jacreduce_opt(0);   
  retval+=test_ec_jacreduce_pippen(0);   
  retval+=test_ec_jacreduce_precompute(0,0,0);   
  retval+=test_ec_jacreduce_precompute(0,0,1);  
  retval+=test_ec_jacreduce_precompute(0,1);   

  //EC2
  retval+=test_ec2aff(1);
  retval+=test_ec_jacdouble(1);
  retval+=test_ec_jacadd(1);
  retval+=test_ec_jacaddmixed(1);
  retval+= test_ec_jacscmul(1);
  retval+=test_ec_jacreduce_opt(1);   
  retval+=test_ec_jacreduce_pippen(1);   
  retval+=test_ec_jacreduce_precompute(1,0,0);   
  retval+=test_ec_jacreduce_precompute(1,0,1);   
  retval+=test_ec_jacreduce_precompute(1,1);   


  if (retval){
    printf("\033[1;31m");
    printf("CPU tests FAILED\n");
  } else {
    printf("All CPU tests PASSED \n");
  }
  printf("\033[0m");

  return 1;
}

