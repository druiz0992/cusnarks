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


cdef extern from "types.h":
  
  # Types
  ctypedef unsigned char uint8_t
  ctypedef unsigned short uint16_t
  ctypedef unsigned int uint32_t
  ctypedef char int8_t
  ctypedef short int16_t
  ctypedef int int32_t
  ctypedef float float_t
  
  #Constants 
  cdef uint32_t NWORDS_256BIT
  cdef uint32_t XOFFSET
  cdef uint32_t YOFFSET
  cdef uint32_t ZOFFSET
  cdef uint32_t VWIDTH

_NWORDS_256BIT = NWORDS_256BIT
