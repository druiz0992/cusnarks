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
// File name  : _u256pxd
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   U256 Integer wrapper function wrapper
// ------------------------------------------------------------------

"""

cimport _types as ct

cdef extern from "rng_cython.h":
     pass

cdef extern from "../cuda/u256.h":
    cdef cppclass C_U256 "U256":
        C_U256(ct.uint32_t *p, ct.uint32_t len, ct.uint32_t seed) except +
        C_U256(ct.uint32_t *p, ct.uint32_t len) except +
        void rand(ct.uint32_t *samples, ct.uint32_t n_samples)
        void add(ct.uint32_t *in_vector, ct.uint32_t *out_vector, ct.uint32_t len)
        void sub(ct.uint32_t *in_vector, ct.uint32_t *out_vector, ct.uint32_t len)
        void addm(ct.uint32_t *in_vector, ct.uint32_t *out_vector, ct.uint32_t len)
        void subm(ct.uint32_t *in_vector, ct.uint32_t *out_vector, ct.uint32_t len)
        void mod(ct.uint32_t *in_vector, ct.uint32_t *out_vector, ct.uint32_t len)
        void mulmont(ct.uint32_t *in_vector, ct.uint32_t *out_vector, ct.uint32_t len)

