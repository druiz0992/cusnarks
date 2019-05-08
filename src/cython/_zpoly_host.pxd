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
// File name  : _zpoly_host.pxd
//
// Date       : 27/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//   zpoly_host cython wrapper
// ------------------------------------------------------------------

"""
cimport _types as ct

cdef extern from "../cuda/zpoly_host.h":
      void cmaddm_h "maddm_h" (ct.uint32_t *pout, ct.uint32_t *scalar, ct.uint32_t *p,ct.uint32_t last_idx, ct.uint32_t ncoeff, ct.uint32_t pidx)

  
