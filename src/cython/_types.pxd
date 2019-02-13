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
// File name  : types.pxd
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Definition of basic data types for wrapper functions
// ------------------------------------------------------------------

"""


import numpy as np
cimport numpy as np

cdef extern from "types.h":
  
  #Constants 
  cdef uint32_t NWORDS_256BIT

  # Types
  ctypedef unsigned int uint32_t
  ctypedef int int32_t

  ctypedef struct mod_info_t
	cdef uint32_t p[NWORDS_256BIT]
        cdef uint32_t p_[NWORDS_256BIT]
        cdef uint32_t r[NWORDS_256BIT]
        cdef uint32_t r_[NWORDS_256BIT]

  ctypedef struct kernel_config_t
        int blockD
        int gridD
        int smemblockD
        int gridD
        int smemSS
  

_NWORDS_256BIT = NWORDS_256BIT
