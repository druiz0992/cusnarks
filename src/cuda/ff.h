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
//  Implementation of finite field arithmetic
// ------------------------------------------------------------------

*/
#ifndef _FF_H_
#define _FF_H_

void montmult_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx);
void montmultN_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t n, uint32_t pidx);
void mulN_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t n, uint32_t pidx);
void montmult_ext_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx);
void montmultN_ext_h(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t n, uint32_t pidx);
void montmult_h2(uint32_t *U, const uint32_t *A, const uint32_t *B, uint32_t pidx);
void montsquare_h(uint32_t *U, const uint32_t *A, uint32_t pidx);
void montsquare_ext_h(uint32_t *U, const uint32_t *A, uint32_t pidx);
void setRandom256(uint32_t *x, uint32_t nsamples, const uint32_t *p);
void setRandom256(uint32_t *x, const uint32_t nsamples, int32_t min_nwords, int32_t max_nwords, const uint32_t *p);
void rangeu256_h(uint32_t *samples, uint32_t nsamples, const uint32_t  *start, uint32_t inc,  const uint32_t *mod);
void to_montgomery_h(uint32_t *z, const uint32_t *x, uint32_t pidx);
void to_montgomeryN_h(uint32_t *z, const uint32_t *x, uint32_t n, uint32_t pidx);
void from_montgomery_h(uint32_t *z, const uint32_t *x, uint32_t pidx);
void from_montgomeryN_h(uint32_t *z, const uint32_t *x, uint32_t n, uint32_t pidx, uint32_t strip_last);
void subm_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx);
void subm_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx);
void addm_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx);
void addm_ext_h(uint32_t *z, const uint32_t *x, const uint32_t *y, uint32_t pidx);
void montinv_h(uint32_t *y, uint32_t *x,  uint32_t pidx);
void montinv_ext_h(uint32_t *y, uint32_t *x,  uint32_t pidx);
t_subm getcb_subm_h( uint32_t pidx);
t_addm getcb_addm_h( uint32_t pidx);
t_mulm getcb_mulm_h( uint32_t pidx);
t_sqm getcb_sqm_h( uint32_t pidx);
t_tomont getcb_tomont_h( uint32_t pidx);
t_frommont getcb_frommont_h( uint32_t pidx);




#endif
