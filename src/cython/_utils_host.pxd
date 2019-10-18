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

cdef extern from "../cuda/utils_host.h" nogil: 
    void cmontmult_h "montmult_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, ct.uint32_t pidx)
    void cmontsquare_h "montsquare_h" (ct.uint32_t *U, ct.uint32_t *B, ct.uint32_t pidx)
    #void cmontmultN_h "montmultN_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, ct.uint32_t n, ct.uint32_t pidx)
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
    void creadWitnessFile_h "readWitnessFile_h"(ct.uint32_t *samples, const char *filename, ct.uint32_t fmt, const unsigned long long inlen)
    void cwriteU256DataFile_h "writeU256DataFile_h"(ct.uint32_t *samples, const char *filename, unsigned long long nwords)
    void cappendU256DataFile_h "appendU256DataFile_h"(ct.uint32_t *samples, const char *filename, unsigned long long nwords)
    void cwriteWitnessFile_h "writeWitnessFile_h"(ct.uint32_t *samples, const char *filename, const unsigned long longnwords)
    void creadU256CircuitFile_h "readU256CircuitFile_h"(ct.uint32_t *samples, const char *filename, unsigned long long nwords)
    void creadU256CircuitFileHeader_h "readU256CircuitFileHeader_h"( ct.cirbin_hfile_t *hfile, const char *filename)
    void creadU256PKFile_h "readU256PKFile_h"(ct.uint32_t *samples, const char *filename, unsigned long long nwords)
    void creadU256PKFileHeader_h "readU256PKFileHeader_h"( ct.pkbin_hfile_t *hfile, const char *filename)
    void creadR1CSFileHeader_h "readR1CSFileHeader_h" (ct.r1csv1_t *r1cs_hdr, const char *filename)
    void creadR1CSFile_h "readR1CSFile_h"(ct.uint32_t *samples, const char *filename, ct.r1csv1_t *r1cs, ct.r1cs_idx_t r1cs_idx )
    void *cmpoly_eval_h "mpoly_eval_h" (ct.mpoly_eval_t *args)
    void cmpoly_eval_server_h "mpoly_eval_server_h" (ct.mpoly_eval_t *args)
    void cr1cs_to_mpoly_h "r1cs_to_mpoly_h" (ct.uint32_t *pout, ct.uint32_t *cin, ct.cirbin_hfile_t *header, ct.uint32_t to_mont, ct.uint32_t pidx, ct.uint32_t extend)
    void cr1cs_to_mpoly_len_h "r1cs_to_mpoly_len_h" (ct.uint32_t *plen_out, ct.uint32_t *cin, ct.cirbin_hfile_t *header, ct.uint32_t extend)
    void cmontinv_h "montinv_h" (ct.uint32_t *y, ct.uint32_t *x, ct.uint32_t pidx)
    void cec_jac2aff_h "ec_jac2aff_h" (ct.uint32_t *y, ct.uint32_t *x, ct.uint32_t n, ct.uint32_t pidx, ct.uint32_t strip_last)
    void cec2_jac2aff_h "ec2_jac2aff_h" (ct.uint32_t *y, ct.uint32_t *x, ct.uint32_t n, ct.uint32_t pidx, ct.uint32_t strip_last)
    void cec_jacadd_h "ec_jacadd_h" (ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t *y, ct.uint32_t pidx)
    void cec_jacdouble_h "ec_jacdouble_h" (ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t pidx)
    void cec_jacscmul_h "ec_jacscmul_h" (ct.uint32_t *z, ct.uint32_t *scl, ct.uint32_t *x, ct.uint32_t n, ct.uint32_t pidx, ct.uint32_t add_last);
    ct.uint32_t cec_isoncurve_h "ec_isoncurve_h" (ct.uint32_t *x, ct.uint32_t is_affine, ct.uint32_t pidx)
    ct.uint32_t cec2_isoncurve_h "ec2_isoncurve_h" (ct.uint32_t *x, ct.uint32_t is_affine, ct.uint32_t pidx)
    void cec_jacaddreduce_h "ec_jacaddreduce_h" (ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t n,
                              ct.uint32_t pidx, ct.uint32_t to_aff, ct.uint32_t add_in, ct.uint32_t strip_last)
    void cec2_jacaddreduce_h "ec2_jacaddreduce_h" (ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t n, 
                              ct.uint32_t pidx, ct.uint32_t to_aff, ct.uint32_t add_in, ct.uint32_t strip_last);
    void cto_montgomeryN_h "to_montgomeryN_h"(ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t n, ct.uint32_t pidx)
    void cfrom_montgomeryN_h "from_montgomeryN_h" (ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t n, ct.uint32_t pidx, ct.uint32_t strip_last)
    void cec_stripc_h "ec_stripc_h" (ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t n)
    void cec2_stripc_h "ec2_stripc_h" (ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t n)
    void cfield_roots_compute_h "field_roots_compute_h" (ct.uint32_t *roots, ct.uint32_t nbits)
    void cmpoly_from_montgomery_h "mpoly_from_montgomery_h" (ct.uint32_t *x, ct.uint32_t pidx)
    void cmpoly_to_montgomery_h "mpoly_to_montgomery_h" (ct.uint32_t *x, ct.uint32_t pidx)
    void cinit_h "init_h"()
    void crelease_h "release_h"()
  
