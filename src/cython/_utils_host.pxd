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

cdef extern from "../cuda/bigint.h" nogil: 

    void csortuBI_idx_h "sortuBI_idx_h" (ct.uint32_t *idx, ct.uint32_t *v, ct.uint32_t l, ct.uint32_t biSize, ct.uint32_t sort_en)

cdef extern from "../cuda/ff.h" nogil: 

    void cmontmult_h "montmult_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, 
                                   ct.uint32_t pidx)
    void cmontmult_ext_h "montmult_ext_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, 
                                   ct.uint32_t pidx)

    void cmontsquare_h "montsquare_h" (ct.uint32_t *U, ct.uint32_t *B, ct.uint32_t pidx)
    void cmontsquare_ext_h "montsquare_ext_h" (ct.uint32_t *U, ct.uint32_t *B, ct.uint32_t pidx)

    #void cmontmultN_h "montmultN_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, 
    #                                  ct.uint32_t n, ct.uint32_t pidx)

    void caddm_h "addm_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, ct.uint32_t pidx)

    void cadduBI_h "adduBI_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, ct.uint32_t biSize)

    void csubm_h "subm_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, ct.uint32_t pidx)
    void csubm_ext_h "subm_ext_h" (ct.uint32_t *U, ct.uint32_t *A, ct.uint32_t *B, ct.uint32_t pidx)

    void crangeuBI_h "rangeuBI_h" (ct.uint32_t *samples, ct.uint32_t nsamples,
                                     ct.uint32_t  *start, ct.uint32_t inc, ct.uint32_t *mod, ct.uint32_t biSize)

    void cmontinv_h "montinv_h" (ct.uint32_t *y, ct.uint32_t *x, ct.uint32_t pidx)

    void cto_montgomeryN_h "to_montgomeryN_h"(ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t n,
                                              ct.uint32_t pidx)

    void cfrom_montgomeryN_h "from_montgomeryN_h" (ct.uint32_t *z, ct.uint32_t *x,
                                                   ct.uint32_t n, ct.uint32_t pidx,
                                                   ct.uint32_t strip_last)


cdef extern from "../cuda/constants.h":
    ct.uint32_t cCusnarksPSizeGet "CusnarksPSizeGet" (ct.mod_t pidx)

cdef extern from "../cuda/ntt.h" nogil: 

    void cntt_h "ntt_h" (ct.uint32_t *A, ct.uint32_t *roots, ct.uint32_t L, ct.t_uint64 astride,  ct.t_uint64 rstride, ct.int32_t direction, ct.uint32_t pidx)

    void cintt_h "intt_h" (ct.uint32_t *A, ct.uint32_t *roots, ct.uint32_t fmat, ct.uint32_t L, 
                           ct.t_uint64 rstride, ct.uint32_t pidx)

    void cfind_roots_h "find_roots_h" (ct.uint32_t *roots, ct.uint32_t *primitive_root,
                                       ct.uint32_t nroots, ct.uint32_t pidx)

    void cntt_parallel_h "ntt_parallel_h" (ct.uint32_t *A, ct.uint32_t *roots, 
                                           ct.uint32_t Nrows, ct.uint32_t Ncols,
                                           ct.t_uint64 rstride, ct.int32_t direction, ct.fft_mode_t fft_mode, ct.uint32_t pidx)

    void cntt_parallel2D_h "ntt_parallel2D_h" (ct.uint32_t *A, ct.uint32_t *roots,
                                               ct.uint32_t Nrows, ct.uint32_t fft_Ny,  
                                               ct.uint32_t Ncols, ct.uint32_t fft_Nx,
                                               ct.t_uint64 rstride, ct.int32_t direction, ct.uint32_t pidx)

    void cntt_build_h "ntt_build_h" (ct.fft_params_t *ntt_params, ct.uint32_t nsamples)

    ct.uint32_t * cntt_interpolandmul_server_h "ntt_interpolandmul_server_h" (ct.ntt_interpolandmul_t *args)

    ct.uint32_t * cget_Mmul_h "get_Mmul_h" ()

    ct.uint32_t * cget_Mtranspose_h "get_Mtranspose_h" ()

cdef extern from "../cuda/ec.h" nogil: 
    void cec_jac2aff_h "ec_jac2aff_h" (ct.uint32_t *y, ct.uint32_t *x, ct.t_uint64 n,
                                       ct.uint32_t pidx, ct.uint32_t strip_last)

    void cec2_jac2aff_h "ec2_jac2aff_h" (ct.uint32_t *y, ct.uint32_t *x, ct.t_uint64 n,
                                         ct.uint32_t pidx, ct.uint32_t strip_last)

    void cec_jacadd_h "ec_jacadd_h" (ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t *y,
                                     ct.uint32_t pidx)

    void cec_jacdouble_h "ec_jacdouble_h" (ct.uint32_t *z, ct.uint32_t *x, ct.uint32_t pidx)

    void cec_jacscmul_h "ec_jacscmul_h" (ct.uint32_t *z, ct.uint32_t *scl, ct.uint32_t *x,
                                         ct.uint32_t n, ct.uint32_t pidx, ct.uint32_t add_last)

    void cec_jacsc1mul_h "ec_jacsc1mul_h" (ct.uint32_t *z, ct.uint32_t *x,
                                         ct.uint32_t n, ct.uint32_t pidx, ct.uint32_t add_last)

    void cec2_jacscmul_h "ec2_jacscmul_h" (ct.uint32_t *z, ct.uint32_t *scl, ct.uint32_t *x,
                                           ct.uint32_t n, ct.uint32_t pidx,
                                           ct.uint32_t add_last)

    void cec2_jacsc1mul_h "ec2_jacsc1mul_h" (ct.uint32_t *z, ct.uint32_t *x,
                                           ct.uint32_t n, ct.uint32_t pidx,
                                           ct.uint32_t add_last)

    ct.uint32_t cec_isoncurve_h "ec_isoncurve_h" (ct.uint32_t *x, ct.uint32_t is_affine,
                                                  ct.uint32_t pidx)

    ct.uint32_t cec2_isoncurve_h "ec2_isoncurve_h" (ct.uint32_t *x, ct.uint32_t is_affine,
                                                    ct.uint32_t pidx)

    void cec_inittable_h "ec_inittable_h" (ct.uint32_t *x, ct.uint32_t *ectable, ct.uint32_t n, ct.uint32_t table_order, ct.uint32_t pidx, ct.uint32_t add_last)
    void cec2_inittable_h "ec2_inittable_h" (ct.uint32_t *x, ct.uint32_t *ectable, ct.uint32_t n, ct.uint32_t table_order, ct.uint32_t pidx, ct.uint32_t add_last)
    void cec_jacaddreduce_h "ec_jacaddreduce_h" (ct.uint32_t *z, ct.uint32_t *x,
                                                 ct.uint32_t n, ct.uint32_t pidx,
                                                 ct.uint32_t to_aff, ct.uint32_t add_in,
                                                 ct.uint32_t strip_last)

    void cec2_jacaddreduce_h "ec2_jacaddreduce_h" (ct.uint32_t *z, ct.uint32_t *x,
                                                   ct.uint32_t n, ct.uint32_t pidx,
                                                   ct.uint32_t to_aff, ct.uint32_t add_in,
                                                   ct.uint32_t strip_last);

    void cec_jacreduce_server_h "ec_jacreduce_server_h" (ct.jacadd_reduced_t *args)

    void cec2_jacreduce_h "ec2_jacreduce_h" (ct.uint32_t *z, ct.uint32_t *scl,
                                              ct.uint32_t *x, ct.uint32_t n, 
                                              ct.uint32_t pidx, ct.uint32_t to_aff,
                                              ct.uint32_t add_in, ct.uint32_t strip_last)


    void cec_isinf "ec_isinf" (ct.uint32_t *z, const ct.uint32_t *x, const ct.uint32_t n, const ct.uint32_t pidx)

    void cec2_isinf "ec2_isinf" (ct.uint32_t *z, const ct.uint32_t *x, const ct.uint32_t n, const ct.uint32_t pidx)


cdef extern from "../cuda/file_utils.h" nogil: 
    void creadU256DataFile_h "readU256DataFile_h"(ct.uint32_t *samples, 
                                                  const char *filename, ct.uint32_t insize,
                                                  ct.uint32_t outsize)

    void creadU256DataFileFromOffset_h "readU256DataFileFromOffset_h"(ct.uint32_t *samples, 
                                                  const char *filename, ct.t_uint64 woffset,
                                                  ct.t_uint64 nwords)


    void creadWitnessFile_h "readWitnessFile_h"(ct.uint32_t *samples, const char *filename, ct.uint32_t fmt, const unsigned long long inlen)

    void cwriteU256DataFile_h "writeU256DataFile_h"(ct.uint32_t *samples, 
                                                    const char *filename, 
                                                    unsigned long long nwords)

    void cappendU256DataFile_h "appendU256DataFile_h"(ct.uint32_t *samples,
                                                      const char *filename, 
                                                      unsigned long long nwords)

    void cwriteWitnessFile_h "writeWitnessFile_h"(ct.uint32_t *samples,
                                                  const char *filename,
                                                  const unsigned long longnwords)

    void creadU256CircuitFile_h "readU256CircuitFile_h"(ct.uint32_t *samples,
                                                        const char *filename,
                                                        unsigned long long nwords)

    void creadU256CircuitFileHeader_h "readU256CircuitFileHeader_h"( ct.cirbin_hfile_t *hfile,
                                                                     const char *filename)

    void creadU256PKFile_h "readU256PKFile_h"(ct.uint32_t *samples, const char *filename,
                                              unsigned long long nwords)

    void creadU256PKFileHeader_h "readU256PKFileHeader_h"( ct.pkbin_hfile_t *hfile,
                                                          const char *filename)

    void creadR1CSFileHeader_h "readR1CSFileHeader_h" (ct.r1csv1_t *r1cs_hdr, 
                                                       const char *filename)

    void creadR1CSFile_h "readR1CSFile_h"(ct.uint32_t *samples, const char *filename,
                                          ct.r1csv1_t *r1cs, ct.r1cs_idx_t r1cs_idx )

    void creadECTablesNElementsFile_h "readECTablesNElementsFile_h" (ct.ec_table_offset_t *table_offset,
                                                                     const char *filename)

    unsigned long long creadNWtnsNEls_h "readNWtnsNEls_h"(unsigned long long *start, const char *filename)

    void creadWtnsFile_h "readWtnsFile_h"(ct.uint32_t *samples, unsigned long long nElems,
                                           unsigned long long start, const char *filename)

    void creadSharedMWtnsFile_h "readSharedMWtnsFile_h"(ct.uint32_t *samples, unsigned long long nElems,
                                           unsigned long long start, const char *filename)

    void czKeyToPkFile_h "zKeyToPkFile_h" (const char *pkbin_filename, const char *zkey_filename)

cdef extern from "../cuda/mpoly.h" nogil: 
    void cmpoly_from_montgomery_h "mpoly_from_montgomery_h" (ct.uint32_t *x, ct.uint32_t pidx)

    void cmpoly_to_montgomery_h "mpoly_to_montgomery_h" (ct.uint32_t *x, ct.uint32_t pidx)

 
    void *cmpoly_eval_h "mpoly_eval_h" (ct.mpoly_eval_t *args)

    void cmpoly_eval_server_h "mpoly_eval_server_h" (ct.mpoly_eval_t *args)

    void cr1cs_to_mpoly_h "r1cs_to_mpoly_h" (ct.uint32_t *pout, ct.uint32_t *cin,
                                             ct.cirbin_hfile_t *header, 
                                             ct.uint32_t to_mont, ct.uint32_t pidx, 
                                             ct.uint32_t extend)

    void cr1cs_to_mpoly_len_h "r1cs_to_mpoly_len_h" (ct.uint32_t *plen_out, ct.uint32_t *cin,
                                                     ct.cirbin_hfile_t *header,
                                                     ct.uint32_t extend)


cdef extern from "../cuda/init.h" :
    void cinit_h "init_h"()

    void crelease_h "release_h"()

cdef extern from "../cuda/utils_host.h" :

    ct.uint32_t cget_nprocs_h "get_nprocs_h"()

    int cshared_new_h "shared_new_h" (void **shmem, unsigned long long size)

    void cshared_free_h "shared_free_h" (void *shmem, int shmid)

