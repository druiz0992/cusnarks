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
//  Implementation of Elliptic Curve Arithmetic
// ------------------------------------------------------------------

*/
#ifndef _EC_H_
#define _EC_H_

void ec_init_h(void);
void ec_free_h(void);
void ec_jacreduce_del_h(uint32_t *ectable, uint32_t *scmul);

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
  return ( (compuBI_h(x, y, NWORDS_FP) == 0) &&
	   (compuBI_h(&x[NWORDS_FP], &y[NWORDS_FP], NWORDS_FP)==0) );
}
inline int32_t ec2_iseq_h(const uint32_t *x, const uint32_t *y)
{
  return ( (compuBI_h(x, y, NWORDS_FP) == 0) &&
	   (compuBI_h(&x[NWORDS_FP], &y[NWORDS_FP], NWORDS_FP)==0) &&
	   (compuBI_h(&x[2*NWORDS_FP], &y[2*NWORDS_FP], NWORDS_FP)==0) &&
	   (compuBI_h(&x[3*NWORDS_FP], &y[3*NWORDS_FP], NWORDS_FP)==0) );
}

void ec_jacadd_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx);
void ec_jacaddaff_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx);
void ec_jacaddmixed_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx);
void ec_jacdouble_h(uint32_t *z, uint32_t *x, uint32_t pidx);
void ec_jacscmul_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last);
void ec_jacsc1mul_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last);
void ec_inittable_h(uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t table_order, uint32_t pidx, uint32_t add_last);
void ec_jac2aff_h(uint32_t *y, uint32_t *x, t_uint64 n, uint32_t pidx, uint32_t strip_last);
void ec_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last, uint32_t flags);
void ec_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last);
uint32_t ec_isoncurve_h(uint32_t *x, uint32_t is_affine, uint32_t pidx);
uint32_t ec_isinf(const uint32_t *x, const uint32_t pidx);
void ec_isinf(uint32_t *z, const uint32_t *x, const uint32_t n, const uint32_t pidx);
void ec_jacreduce_server_h(jacadd_reduced_t *args);
void ec_jacreduce_pippen_server_h(jacadd_reduced_t *args);


void ec2_jacadd_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx);
void ec2_jacaddmixed_h(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t pidx);
void ec2_jacdouble_h(uint32_t *z, uint32_t *x, uint32_t pidx);
void ec2_jacscmul_h(uint32_t *z, uint32_t *scl, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last);
void ec2_jacsc1mul_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t add_last);
void ec2_inittable_h(uint32_t *x, uint32_t *ectable, uint32_t n, uint32_t table_order, uint32_t pidx, uint32_t add_last);
void ec2_jac2aff_h(uint32_t *y, uint32_t *x, t_uint64 n, uint32_t pidx, uint32_t strip_last);
void ec2_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last);
void ec2_jacaddreduce_h(uint32_t *z, uint32_t *x, uint32_t n, uint32_t pidx, uint32_t to_aff, uint32_t add_in, uint32_t strip_last, uint32_t flags);
uint32_t ec2_isoncurve_h(uint32_t *x, uint32_t is_affine, uint32_t pidx);
uint32_t ec2_isinf(const uint32_t *x, const uint32_t pidx);
void ec2_isinf(uint32_t *z, const uint32_t *x, const uint32_t n, const uint32_t pidx);
void ec2_jacreduce_server_h(jacadd_reduced_t *args);

#endif


