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
*/

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : constants.h
//
// Date       : 5/09/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Header of functions used in Cusnarks constant.cpp
//
// ------------------------------------------------------------------

#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

const uint32_t * CusnarksPGet(mod_t type);
const uint32_t * CusnarksR2Get(mod_t type);
const uint32_t * CusnarksR3Get(mod_t type);
const uint32_t * CusnarksNPGet(mod_t type);
const uint32_t * CusnarksIScalerGet(fmt_t type);
const uint32_t * CusnarksZeroGet(void);
const uint32_t * CusnarksOneGet(void);
const uint32_t * CusnarksOneMontGet(uint32_t pidx);
const uint32_t * CusnarksOneMont2Get(uint32_t pidx);
const uint32_t * CusnarksEcbn128ParamsGet(void);
const uint32_t * CusnarksModInfoGet(void);
const uint32_t * CusnarksMiscKGet(void);
const uint32_t * CusnarksW32RootsGet(void);
const uint32_t * CusnarksIW32RootsGet(void);
const uint32_t * CusnarksIW32NRootsGet(void);
const uint32_t * CusnarksPrimitiveRootsFieldGet(uint32_t nbits);
uint32_t CusnarksGetNRoots(void);
void CusnarksGetFRoots(char *filename, uint32_t sizeof_f);
const uint32_t * CusnarksTidxGet(void);
void computeIRoots_h(uint32_t *iroots, uint32_t *roots, uint32_t nroots);

#endif

