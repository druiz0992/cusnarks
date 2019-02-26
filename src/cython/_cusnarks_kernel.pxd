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
// File name  : _cusnarks_kernel.pxd
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Cusnarks cython wrapper
// ------------------------------------------------------------------

"""

cimport _types as ct
cimport _cusnarks_kernel

cdef extern from "rng_cython.h":
     pass

cdef extern from "../cuda/cusnarks_kernel.h":
    cdef cppclass C_CUSnarks "CUSnarks":
        C_CUSnarks(ct.uint32_t in_len, ct.uint32_t in_size, ct.uint32_t out_len, ct.uint32_t out_size, 
                        ct.uint32_t seed) except +
        C_CUSnarks(ct.uint32_t in_len, ct.uint32_t in_size, ct.uint32_t out_len, ct.uint32_t out_size) except +
        void rand(ct.uint32_t *samples, ct.uint32_t n_samples)
        double kernelLaunch(ct.uint32_t kernel_idx,
                          ct.vector_t *out_vector_host, ct.vector_t *in_vector_host,
                          ct.kernel_config_t *config, ct.kernel_params_t *params)
        void getDeviceInfo()

cdef extern from "../cuda/u256.h":
    cdef cppclass C_U256 "U256" (C_CUSnarks) :
        C_U256(ct.uint32_t len, ct.uint32_t seed) except +
        C_U256(ct.uint32_t len) except +

cdef extern from "../cuda/ecbn128.h":
    cdef cppclass C_ECBN128 "ECBN128" (C_CUSnarks) :
        C_ECBN128(ct.uint32_t len, ct.uint32_t seed) except +
        C_ECBN128(ct.uint32_t len) except +

cdef extern from "../cuda/ecbn128_2.h":
    cdef cppclass C_ECBN128_2 "ECBN128_2" (C_CUSnarks) :
        C_ECBN128_2(ct.uint32_t len, ct.uint32_t seed) except +
        C_ECBN128_2(ct.uint32_t len) except +

cdef extern from "../cuda/zpoly.h":
    cdef cppclass C_ZCUPoly "ZCUPoly" (C_CUSnarks) :
        C_ZCUPoly(ct.uint32_t len, ct.uint32_t seed) except +
        C_ZCUPoly(ct.uint32_t len) except +



