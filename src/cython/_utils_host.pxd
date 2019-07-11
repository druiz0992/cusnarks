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
    void caddm_h "addm_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, ct.uint32_t pidx)
    void caddu256_h "addu256_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B)
    void csubm_h "subm_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, ct.uint32_t pidx)
    void cntt_h "ntt_h" (ct.uint32_t *A, ct.uint32_t *roots, ct.uint32_t L, ct.uint32_t pidx)
    void cintt_h "intt_h" (ct.uint32_t *A, ct.uint32_t *roots, ct.uint32_t fmat, ct.uint32_t L, ct.uint32_t pidx)
    void cfind_roots_h "find_roots_h" (ct.uint32_t *roots, ct.uint32_t *primitive_root, ct.uint32_t nroots, ct.uint32_t pidx)
    void cntt_parallel_h "ntt_parallel_h" (ct.uint32_t *A, ct.uint32_t *roots, ct.uint32_t Nrows, ct.uint32_t Ncols, ct.uint32_t pidx, ct.uint32_t mode)
    void cntt_parallel2D_h "ntt_parallel2D_h" (ct.uint32_t *A, ct.uint32_t *roots, ct.uint32_t Nrows, ct.uint32_t fft_Ny,  ct.uint32_t Ncols, ct.uint32_t fft_Nx, ct.uint32_t pidx, ct.uint32_t mode)
    void cntt_build_h "ntt_build_h" (ct.fft_params_t *ntt_params, ct.uint32_t nsamples)
    void crangeu256_h "rangeu256_h" (ct.uint32_t *samples, ct.uint32_t nsamples, ct.uint32_t  *start, ct.uint32_t inc, ct.uint32_t *mod)
    ct.uint32_t czpoly_norm_h "zpoly_norm_h" (ct.uint32_t *pin, ct.uint32_t cidx)
    void csortu256_idx_h "sortu256_idx_h" (ct.uint32_t *idx, ct.uint32_t *v, ct.uint32_t l)
    void creadU256DataFile_h "readU256DataFile_h"(ct.uint32_t *samples, const char *filename, ct.uint32_t insize, ct.uint32_t outsize)
    void cwriteU256CircuitFile_h "writeU256CircuitFile_h"(ct.uint32_t *samples, const char *filename, ct.uint32_t nwords)
    void creadU256CircuitFile_h "readU256CircuitFile_h"(ct.uint32_t *samples, const char *filename, ct.uint32_t nwords)
    void creadU256CircuitFileHeader_h "readU256CircuitFileHeader_h"( ct.cirbin_hfile_t *hfile, const char *filename)
    void cmpoly_eval_h "mpoly_eval_h" (ct.uint32_t *pout, ct.uint32_t *scalar, ct.uint32_t *p,ct.uint32_t reduce_coeff, ct.uint32_t last_idx, ct.uint32_t pidx)
    void cr1cs_to_mpoly_h "r1cs_to_mpoly_h" (ct.uint32_t *pout, ct.uint32_t *cin, ct.cirbin_hfile_t *header, ct.uint32_t to_mont, ct.uint32_t pidx, ct.uint32_t extend)
    void cr1cs_to_mpoly_len_h "r1cs_to_mpoly_len_h" (ct.uint32_t *plen_out, ct.uint32_t *cin, ct.cirbin_hfile_t *header, ct.uint32_t extend)
    void cmontinv_h "montinv_h" (ct.uint32_t *y, ct.uint32_t *x, ct.uint32_t pidx)
    void cec_jac2aff_h "ec_jac2aff_h" (ct.uint32_t *y, ct.uint32_t *x, ct.uint32_t n, ct.uint32_t pidx)
    void cec2_jac2aff_h "ec2_jac2aff_h" (ct.uint32_t *y, ct.uint32_t *x, ct.uint32_t n, ct.uint32_t pidx)
    void cto_montgomeryN_h "to_montgomeryN_h"(ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t n, ct.uint32_t pidx)
    void cfrom_montgomeryN_h "from_montgomeryN_h" (ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t n, ct.uint32_t pidx)
    void cfield_roots_compute_h "field_roots_compute_h" (ct.uint32_t *roots, ct.uint32_t nbits)
    void cmpoly_from_montgomery_h "mpoly_from_montgomery_h" (ct.uint32_t *x, ct.uint32_t pidx)
    void cmpoly_to_montgomery_h "mpoly_to_montgomery_h" (ct.uint32_t *x, ct.uint32_t pidx)
  
