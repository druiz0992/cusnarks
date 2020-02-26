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

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : utils_host.h
//
// Date       : 06/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of small utils functions for host
// ------------------------------------------------------------------

*/
#ifndef _UTILS_HOST_H_
#define _UTILS_HOST_H_

#define NEG(X) (~(X))
#define SWAP(a,b) (((a) ^ (b)) && ((b) ^= (a) ^= (b), (a) ^= (b))) 

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define ADD_FIRSTADD                            \
    "movq    (%[A]), %%rax          \n\t"      \
    "movq    %%rax, (%[C])          \n\t"      \
    "movq    (%[B]), %%rax           \n\t"      \
    "addq    %%rax, (%[C])           \n\t"

#define ADD_NEXTADD(ofs)                                \
    "movq    " STR(ofs) "(%[A]), %%rax          \n\t"   \
    "movq    %%rax, " STR(ofs) "(%[C])          \n\t"   \
    "movq    " STR(ofs) "(%[B]), %%rax          \n\t"   \
    "adcq    %%rax, " STR(ofs) "(%[C])          \n\t"
/* 
   Substract 256 bit integers X and Y.

   uint32_t *x : 256 bit integer x
   uint32_t *y : 256 bit integer y
   returns x - y
*/
inline void subu256_h(uint32_t *c, const uint32_t *a, const uint32_t *b)
{
  uint32_t carry=0;
  const t_uint64 *dA = (t_uint64 *)a;
  const t_uint64 *dB = (t_uint64 *)b;
  t_uint64 *dC = (t_uint64 *)c;
  t_uint64 tmp;


  tmp = NEG(dB[0])+1;
  carry = (tmp < 1);
  dC[0] = dA[0] + tmp;
  carry += (dC[0] < tmp);

  tmp = NEG(dB[1]);
  dC[1] = dA[1] + carry;
  carry = (dC[1] < carry);
  dC[1] += tmp;
  carry += (dC[1] < tmp);

  tmp = NEG(dB[2]);
  dC[2] = dA[2] + carry;
  carry = (dC[2] < carry);
  dC[2] += tmp;
  carry += (dC[2] < tmp);

  tmp = NEG(dB[3]);
  dC[3] = dA[3] + carry;
  carry = (dC[3] < carry);
  dC[3] += tmp;
  carry += (dC[3] < tmp);

}
inline void subu256_h(uint32_t *x, const uint32_t *y)
{
   subu256_h(x, x, y);
} 
  

/* 
   Add 256 bit integers X and Y.

   uint32_t *x : 256 bit integer x
   uint32_t *y : 256 bit integer y
   returns x + y
*/
inline void addu256_h(uint32_t *c, const uint32_t *a, const uint32_t *b)
{
  uint32_t carry=0;
  const t_uint64 *dA = (t_uint64 *)a;
  const t_uint64 *dB = (t_uint64 *)b;
  t_uint64 *dC = (t_uint64 *)c;

#if 0
  uint32_t dC2[NWORDS_256BIT/2];

  dC2[0] = dA[0]; dC2[1] = dA[1]; dC2[2]= dA[2]; dC2[3] = dA[3];

   __asm__
            ("/* perform bignum addition */   \n\t"
             ADD_FIRSTADD
             ADD_NEXTADD(8)
             ADD_NEXTADD(16)
             ADD_NEXTADD(24)
             "done%=:                         \n\t"
             :
             : [A] "r" (dA), [B] "r" (dB), [C] "r" (dC2)
             : "cc", "memory", "%rax");
#else
  t_uint64 tmp = dA[0];

  dC[0] = dA[0] + dB[0];
  carry = (dC[0] < tmp);

  tmp = dB[1];
  dC[1] = dA[1] + carry;
  carry = (dC[1] < carry);
  dC[1] += tmp;
  carry += (dC[1] < tmp);

  tmp = dB[2];
  dC[2] = dA[2] + carry;
  carry = (dC[2] < carry);
  dC[2] += tmp;
  carry += (dC[2] < tmp);

  tmp = dB[3];
  dC[3] = dA[3] + carry;
  carry = (dC[3] < carry);
  dC[3] += tmp;
  carry += (dC[3] < tmp);
 
#endif
}

inline void addu256_h(uint32_t *x, const uint32_t *y)
{
   addu256_h(x, x, y);
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
inline int32_t compu256_h(const uint32_t *a, const uint32_t *b)
{
  uint32_t gt=0, lt=0;
  uint32_t idx = NWORDS_256BIT/2-1;

  const t_uint64 *dA = (const t_uint64 *)a;
  const t_uint64 *dB = (const t_uint64 *)b;
  // idx = 3
  gt = (dA[idx] > dB[idx]);
  lt = (dA[idx] < dB[idx]);
  if (gt) return 1;
  if (lt) return -1;

  // idx = 2
  idx--;
  gt = (dA[idx] > dB[idx]);
  lt = (dA[idx] < dB[idx]);
  if (gt) return 1;
  if (lt) return -1;

  // idx = 1
  idx--;
  gt = (dA[idx] > dB[idx]);
  lt = (dA[idx] < dB[idx]);
  if (gt) return 1;
  if (lt) return -1;

  // idx =0
  idx--;
  gt = (dA[idx] > dB[idx]);
  lt = (dA[idx] < dB[idx]);
  if (gt) return 1;
  if (lt) return -1;

  return 0;

}

/* 
   Compare 256 bit integers X and Y.

   uint32_t *x : 256 bit integer x
   uint32_t *y : 256 bit integer y

   returns 
      1          : x < y
      0         : x >= y
*/
inline int32_t ltu256_h(const uint32_t *x, const uint32_t *y)
{
  return (compu256_h(x, y) < 0);
}

inline int32_t ltu32_h(const uint32_t *x, const uint32_t *y)
{
  return *x < *y;
}

/* 
   Compare 256 bit integers X and Y.

   uint32_t *x : 256 bit integer x
   uint32_t *y : 256 bit integer y

   returns 
      1          : x == y
      0         : x != y
*/
inline int32_t equ256_h(const uint32_t *x, const uint32_t *y)
{
  return (compu256_h(x, y) == 0);
}

/* 
   Compare ECP affine 256 bit integers X and Y. (two coordinates)

   uint32_t *x : 256 bit ECP x
   uint32_t *y : 256 bit ECP y

   returns 
      1          : x == y
      0         : x != y
*/
inline int32_t ec_iseq_h(const uint32_t *x, const uint32_t *y)
{
  return ( (compu256_h(x, y) == 0) &&
	   (compu256_h(&x[NWORDS_256BIT], &y[NWORDS_256BIT])==0) );
}
inline int32_t ec2_iseq_h(const uint32_t *x, const uint32_t *y)
{
  return ( (compu256_h(x, y) == 0) &&
	   (compu256_h(&x[NWORDS_256BIT], &y[NWORDS_256BIT])==0) &&
	   (compu256_h(&x[2*NWORDS_256BIT], &y[2*NWORDS_256BIT])==0) &&
	   (compu256_h(&x[3*NWORDS_256BIT], &y[3*NWORDS_256BIT])==0) );
}


void montmult_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx);
void montmultN_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t n, uint32_t pidx);
void mulN_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t n, uint32_t pidx);
void montmult_ext_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx);
void montmultN_ext_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t n, uint32_t pidx);
void montmult_h2(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx);
//void montmult_sos_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx);
void montsquare_h(uint32_t *U, const uint32_t *A, uint32_t pidx);
void montsquare__ext_h(uint32_t *U, const uint32_t *A, uint32_t pidx);
void ntt_h(uint32_t *A, const uint32_t *roots, uint32_t L, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx);
void intt_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t L, t_uint64 rstride, uint32_t pidx);
void ntt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride,int32_t direction,uint32_t pidx);
void intt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t L, t_uint64 rstride, uint32_t pidx);
void find_roots_h(uint32_t *roots, const uint32_t *primitive_root, uint32_t nroots, uint32_t pidx);
void ntt_parallel_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, int32_t direction, fft_mode_t fft_mode,  uint32_t pidx);
void intt_parallel_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, fft_mode_t fft_mode,  uint32_t pidx);
void ntt_parallel2D_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t fft_Ny,  uint32_t Ncols, uint32_t fft_Nx, t_uint64 rstride, int32_t directinn, uint32_t pidx);
void intt_parallel2D_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t Nrows, uint32_t fft_Ny,  uint32_t Ncols, uint32_t fft_Nx, t_uint64 rstride, uint32_t pidx);
void ntt_parallel3D_h(uint32_t *A, const uint32_t *roots, uint32_t Nfft_x, uint32_t Nfft_y, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, int32_t direction, uint32_t pidx);
void intt_parallel3D_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t N_fftx, uint32_t N_ffty, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, uint32_t pidx);
void interpol_odd_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 rstride, uint32_t pidx);
void interpol_parallel_odd_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, uint32_t pidx);
void M_init_h(uint32_t nroots);
void M_free_h(void);
void ntt_build_h(fft_params_t *ntt_params, uint32_t nsamples);
uint32_t * ntt_interpolandmul_parallel_h(uint32_t *A, uint32_t *B, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, uint32_t pidx);
uint32_t * ntt_interpolandmul_server_h(ntt_interpolandmul_t *args);
void transpose_h(uint32_t *mout, const uint32_t *min, uint32_t in_nrows, uint32_t in_ncols);
void transpose_h(uint32_t *min, uint32_t in_nrows, uint32_t in_ncols);
void transpose_square_h(uint32_t *min, uint32_t in_nrows);
void rangeu256_h(uint32_t *samples, uint32_t nsamples, const uint32_t  *start, uint32_t inc,  const uint32_t *mod);
uint32_t zpoly_norm_h(uint32_t *pin, uint32_t n_coeff);
void sortu256_idx_h(uint32_t *idx, const uint32_t *v, uint32_t len, uint32_t sort_en);
void setRandom(uint32_t *x, const uint32_t);
void setRandom256(uint32_t *x, uint32_t nsamples, const uint32_t *p);
void setRandom256(uint32_t *x, const uint32_t nsamples, int32_t min_nwords, int32_t max_nwords, const uint32_t *p);
void to_montgomery_h(uint32_t *z, const uint32_t *x, uint32_t pidx);
void to_montgomeryN_h(uint32_t *z, const uint32_t *x, uint32_t n, uint32_t pidx);
void ec_stripc_h(uint32_t *z, uint32_t *x, uint32_t n);
void ec2_stripc_h(uint32_t *z, uint32_t *x, uint32_t n);
void ec_jacadd_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx);
void ec_jacaddmixed_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx);
void ec2_jacadd_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx);
void ec2_jacaddmixed_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx);
void ec_jacdouble_h(uint32_t *z, uint32_t *x, uint32_t pidx);
void ec2_jacdouble_h(uint32_t *z, uint32_t *x, uint32_t pidx);
void ec_jacscmul_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last);
void ec2_jacscmul_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last);
void ec_jacscmulx1_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last);
void ec2_jacscmulx1_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last);
void ec_jacscmul_opt_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t order, uint32_t pidx, uint32_t add_last);
void ec2_jacscmul_opt_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t *ectable,
	       	uint32_t n, uint32_t order,  uint32_t pidx, uint32_t add_last);
void ec_inittable_h(uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t table_order, uint32_t pidx, uint32_t add_last);
void ec2_inittable_h(uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t table_order, uint32_t pidx, uint32_t add_last);
void ec_jac2aff_h(uint32_t *y, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t strip_last);
void ec2_jac2aff_h(uint32_t *y, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t strip_last);
void ec_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last, uint32_t flags);
void ec_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last);
void ec2_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last);
uint32_t ec_isoncurve_h(uint32_t *x, uint32_t is_affine, uint32_t pidx);
uint32_t ec2_isoncurve_h(uint32_t *x, uint32_t is_affine, uint32_t pidx);
uint32_t ec_isinf(const uint32_t *x, const uint32_t pidx);
uint32_t ec2_isinf(const uint32_t *x, const uint32_t pidx);
void ec_isinf(uint32_t *z, const uint32_t *x, const uint32_t n, const uint32_t pidx);
void ec2_isinf(uint32_t *z, const uint32_t *x, const uint32_t n, const uint32_t pidx);
uint32_t ec2_jacreduce_init_h(uint32_t **ectable, uint32_t **scmul, uint32_t n, uint32_t order);
void ec_jacreduce_del_h(uint32_t *ectable, uint32_t *scmul);
void ec_jacreduce_server_h(jacadd_reduced_t *args);
void ec2_jacreduce_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last);
void from_montgomery_h(uint32_t *z, const uint32_t *x, uint32_t pidx);
void from_montgomeryN_h(uint32_t *z, const uint32_t *x, uint32_t n, uint32_t pidx, uint32_t strip_last);
void subm_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx);
void subm_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx);
void addm_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx);
void addm_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx);
void printU256Number(const uint32_t *x);
void printU256Number(const char *, const uint32_t *x);
void readU256DataFile_h(uint32_t *samples, const char *filename, uint32_t insize, uint32_t outsize);
void readWitnessFile_h(uint32_t *samples, const char *filename, uint32_t fmt, const unsigned long long insize);
void writeU256DataFile_h(uint32_t *samples, const char *filename, unsigned long long nwords);
void appendU256DataFile_h(uint32_t *samples, const char *filename, unsigned long long nwords);
void writeWitnessFile_h(uint32_t *samples, const char *filename, const unsigned long long nwords);
void readU256CircuitFileHeader_h(cirbin_hfile_t *hfile, const char *filename);
void readU256CircuitFile_h(uint32_t *samples, const char *filename, unsigned long long nwords);
void readU256PKFileHeader_h(pkbin_hfile_t *hfile, const char *filename);
void readU256PKFile_h(uint32_t *samples, const char *filename, unsigned long long nwords);
void readR1CSFileHeader_h(r1csv1_t *r1cs_hdr, const char *filename);
void readR1CSFile_h(uint32_t *samples, const char *filename, r1csv1_t *r1cs, r1cs_idx_t r1cs_idx );

void mpoly_eval_server_h(mpoly_eval_t *mpoly_args);
void *mpoly_eval_h(void *args);
void r1cs_to_mpoly_h(uint32_t *pout, uint32_t *cin, cirbin_hfile_t *header, uint32_t to_mont, uint32_t pidx, uint32_t extend);
void r1cs_to_mpoly_len_h(uint32_t *coeff_len, uint32_t *cin, cirbin_hfile_t *header, uint32_t extend);
uint32_t shlru256_h(uint32_t *y, uint32_t *x, uint32_t count);
uint32_t shllu256_h(uint32_t *y, uint32_t *x, uint32_t count);
uint32_t msbu256_h(uint32_t *x);
void mulu256_h(uint32_t *z, uint32_t *x, uint32_t *y);
void setbitu256_h(uint32_t *x, uint32_t n);
uint32_t getbitu256_h(uint32_t *x, uint32_t n);;
uint32_t getbitu256g_h(uint32_t *x, uint32_t n, uint32_t group_size);
void montinv_h(uint32_t *y, uint32_t *x,  uint32_t pidx);
void montinv_ext_h(uint32_t *y, uint32_t *x,  uint32_t pidx);
void field_roots_compute_h(uint32_t *roots, uint32_t nbits);
void mpoly_from_montgomery_h(uint32_t *x, uint32_t pidx);
void mpoly_to_montgomery_h(uint32_t *x, uint32_t pidx);
void computeIRoots_h(uint32_t *iroots, uint32_t *roots, uint32_t nroots);
void init_h(void);
void release_h(void);
uint32_t *get_Mmul_h();
uint32_t get_nprocs_h();
int createSharedMemBuf(void **shmem, shmem_t type);
void destroySharedMemBuf(void *shmem, int shmid);

#endif
