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
// File name  : ff.h
//
// Date       : 06/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of Number Theoretic Transform
// ------------------------------------------------------------------

*/
#ifndef _NTT_H_
#define _NTT_H_

void ntt_h(uint32_t *A, const uint32_t *roots, uint32_t L, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx);
void ntt_h(uint32_t *A, const uint32_t *roots, uint32_t L, int32_t direction, t_ff *ff);
void ntt_h(uint32_t *A, const uint32_t *roots, uint32_t L, int32_t direction, t_ff *ff, uint32_t *preA, uint32_t levels2, uint32_t *postA);
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
void ntt_build_h(fft_params_t *ntt_params, uint32_t nsamples);
uint32_t * ntt_interpolandmul_parallel_h(uint32_t *A, uint32_t *B, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, uint32_t pidx);
uint32_t * ntt_interpolandmul_server_h(ntt_interpolandmul_t *args);
void field_roots_compute_h(uint32_t *roots, uint32_t nbits);
uint32_t *get_Mmul_h();
uint32_t *get_Mtranspose_h();
void ntt_init_h(uint32_t nroots, uint32_t *M);
void ntt_free_h(void);

void computeRoots_h(uint32_t *roots, uint32_t nbits);


#endif


