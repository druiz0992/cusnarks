"""
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
// File name  : _utila_host.pxd
//
// Date       : 06/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//   utils_host cython wrapper
// ------------------------------------------------------------------

"""
cimport _types as ct

cdef extern from "../cuda/utils_host.h":
    void cmontmult_h "montmult_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, ct.uint32_t pidx)
    void cntt_h "ntt_h" (ct.uint32_t *A, ct.uint32_t *roots, ct.uint32_t L, ct.uint32_t pidx)
    void cfind_roots_h "find_roots_h" (ct.uint32_t *roots, ct.uint32_t *primitive_root, ct.uint32_t nroots, ct.uint32_t pidx)
    void cntt_parallel_h "ntt_parallel_h" (ct.uint32_t *A, ct.uint32_t *roots, ct.uint32_t Nrows, ct.uint32_t Ncols, ct.uint32_t pidx)
    void cntt_parallel2D_h "ntt_parallel2D_h" (ct.uint32_t *A, ct.uint32_t *roots, ct.uint32_t Nrows, ct.uint32_t fft_Ny,  ct.uint32_t Ncols, ct.uint32_t fft_Nx, ct.uint32_t pidx)