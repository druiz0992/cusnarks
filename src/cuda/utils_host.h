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

void montmult_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx);
void montmult_h2(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx);
void montmult_sos_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx);
void montsquare_h(uint32_t *U, const uint32_t *A, uint32_t pidx);
void ntt_h(uint32_t *A, const uint32_t *roots, uint32_t L, uint32_t pidx);
void intt_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t L, uint32_t pidx);
void find_roots_h(uint32_t *roots, const uint32_t *primitive_root, uint32_t nroots, uint32_t pidx);
void ntt_parallel_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, uint32_t pidx, uint32_t mode);
void intt_parallel_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t Nrows, uint32_t Ncols, uint32_t pidx, uint32_t mode);
void ntt_parallel2D_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t fft_Ny,  uint32_t Ncols, uint32_t fft_Nx, uint32_t pidx, uint32_t mode);
void intt_parallel2D_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t Nrows, uint32_t fft_Ny,  uint32_t Ncols, uint32_t fft_Nx, uint32_t pidx, uint32_t mode);
void ntt_build_h(fft_params_t *ntt_params, uint32_t nsamples);
void transpose_h(uint32_t *mout, const uint32_t *min, uint32_t in_nrows, uint32_t in_ncols);
int compu256_h(const uint32_t *x, const uint32_t *y);
bool ltu256_h(const uint32_t *x, const uint32_t *y);
void rangeu256_h(uint32_t *samples, uint32_t nsamples, const uint32_t  *start, uint32_t inc,  const uint32_t *mod);
uint32_t zpoly_norm_h(uint32_t *pin, uint32_t n_coeff);
void sortu256_idx_h(uint32_t *idx, const uint32_t *v, uint32_t len);
void setRandom(uint32_t *x, const uint32_t);
void setRandom256(uint32_t *x, uint32_t nsamples, const uint32_t *p);
void to_montgomery_h(uint32_t *z, const uint32_t *x, uint32_t pidx);
void from_montgomery_h(uint32_t *z, const uint32_t *x, uint32_t pidx);
void subm_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx);
void addm_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx);
void printU256Number(const uint32_t *x);
void readU256DataFile_h(uint32_t *samples, const char *filename, uint32_t insize, uint32_t outsize);
void writeU256CircuitFile_h(uint32_t *samples, const char *filename, uint32_t nwords);
void readU256CircuitFileHeader_h(cirbin_hfile_t *hfile, const char *filename);
void readU256CircuitFile_h(uint32_t *samples, const char *filename, uint32_t nwords);
void mpoly_eval_h(uint32_t *pout, const uint32_t *scalar, uint32_t *pin, uint32_t ncoeff, uint32_t last_idx, uint32_t pidx);
void r1cs_to_mpoly_h(uint32_t *pout, uint32_t *cin, cirbin_hfile_t *header, uint32_t extend);
void r1cs_to_mpoly_len_h(uint32_t *coeff_len, uint32_t *cin, cirbin_hfile_t *header, uint32_t extend);
void shlru256_h(uint32_t *y, uint32_t *x, uint32_t count);
void shllu256_h(uint32_t *y, uint32_t *x, uint32_t count);
void subu256_h(uint32_t *x, const uint32_t *y);
void addu256_h(uint32_t *x, const uint32_t *y);
void subu256_h(uint32_t *z, uint32_t *x, uint32_t *y);
void addu256_h(uint32_t *x, const uint32_t *y);
void setbitu256_h(uint32_t *x, uint32_t n);
uint32_t getbitu256_h(uint32_t *x, uint32_t n);;
void montinv_h(uint32_t *y, uint32_t *x,  uint32_t pidx);
#endif
