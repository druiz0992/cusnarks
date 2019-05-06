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

void montmult_h(uint32_t *U, uint32_t *A, uint32_t *B, uint32_t pidx);
void ntt_h(uint32_t *A, uint32_t *roots, uint32_t L, uint32_t pidx);
void intt_h(uint32_t *A, uint32_t *roots, uint32_t L, uint32_t pidx);
void find_roots_h(uint32_t *roots, uint32_t *primitive_root, uint32_t nroots, uint32_t pidx);
void ntt_parallel_h(uint32_t *A, uint32_t *roots, uint32_t Nrows, uint32_t Ncols, uint32_t pidx, uint32_t mode);
void intt_parallel_h(uint32_t *A, uint32_t *roots, uint32_t Nrows, uint32_t Ncols, uint32_t pidx, uint32_t mode);
void ntt_parallel2D_h(uint32_t *A, uint32_t *roots, uint32_t Nrows, uint32_t fft_Ny,  uint32_t Ncols, uint32_t fft_Nx, uint32_t pidx, uint32_t mode);
void intt_parallel2D_h(uint32_t *A, uint32_t *roots, uint32_t Nrows, uint32_t fft_Ny,  uint32_t Ncols, uint32_t fft_Nx, uint32_t pidx, uint32_t mode);
void ntt_build_h(fft_params_t *ntt_params, uint32_t nsamples);
int compu256_h(uint32_t *x, uint32_t *y);
void subu256_h(uint32_t *x, uint32_t *y);
void rangeu256_h(uint32_t *samples, uint32_t nsamples, uint32_t  *start, uint32_t inc,  uint32_t *mod);
void int_to_byte_h(char *out, uint32_t *in, uint32_t nbytes);
void byte_to_int_h(uint32_t *out, char *in, uint32_t nwords);
void zpoly_maddm_h(uint32_t *pout, uint32_t *scalar, uint32_t *pin, uint32_t ncoeff, uint32_t last_idx, uint32_t pidx);
#endif
