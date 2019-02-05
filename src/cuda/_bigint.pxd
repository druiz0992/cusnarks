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
// File name  : _bigint.pxd
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Big Integer wrapper function wrapper
// ------------------------------------------------------------------

"""

cimport _types as ct

cdef extern from "../cuda/bigint.h":
    cdef cppclass C_BigInt "BigInt":
        C_BigInt(ct.uint32_t *, ct.uint32_t *, ct.uint32_t) except +
        void addm()
        void retrieve(ct.uint32_t *vector)
