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
// Date       : 13/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   CUSnarks Kernel wrapper definition
// ------------------------------------------------------------------

"""

cimport _types as ct

cdef extern from "rng_cython.h":
     pass

cdef extern from "../cuda/cusnarks_kernel.h":
    cdef cppclass C_CUSnarks "CUSnarks":
        C_CUSnarks(ct.mod_info_t *p, ct.uint32_t len, ct.uint32_t in_size, ct.uint32_t out_size, 
                        ct.uint32_t seed) except +
        C_CUSnarks(ct.mod_info_t *p, ct.uint32_t len, ct.uint32_t in_size, ct.uint32_t out_size) except +
        void rand(ct.uint32_t *samples, ct.uint32_t n_samples, ct.uint32_t size)
        void kernelLaunch[kernel_function_t](kernel_function_t &kernel_function, 
                        ct.uint32_t *out_vector_host, ct.uint32_t *in_vector_host,
                        ct.uint32_t in_size, ct.uint32_t out_size ,
                        ct.kernel_config_t, ct.kernel_parameters_t...)
        void getDeviceInfo()

