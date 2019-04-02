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
// File name  : zpoly_host.h
//
// Date       : 27/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Definition of zpoly class

*/

#ifndef _ZPOLY_H_
#define _ZPOLY_H_

#define CUZPOLY_DEG_OFFSET (0)
#define CUZPOLY_COEFF_OFFSET (1)

#if 0
class CUZPolySps {
    protected:
     uint32_t *coeff_d;
     uint32_t *coeff_v;

    public:
      CUZPolySps();
      CUZPolySps(uint32_t n_coeff);
      CUZPolySps(uint32_t n_coeff, uint32_t create_coeff);
      ~CUZPolySps();
      uint32_t getNCoeff();
      uint32_t *getDCoeff();
      uint32_t *getVCoeff();
      uint32_t *getZPolySpsRep();
      void show();
      void rand(uint32_t seed);
      void mulmScalar(uint32_t *scalar, uint32_t pidx, uint32_t convert);
      void addm(CUZPolySps **p, uint32_t n_zpoly, uint32_t pidx);
      void maddm(uint32_t *scalar, uint32_t *p, uint32_t last_idx, uint32_t pidx, uint32_t convert);

};
/*
class CUZPolySpsArray {
    protected:
       uint32_t *coeff_d;
       uint32_t *coeff_v;

    public:
       CUZPolySpsArray(CUZPolySps **zpoly, uint32_t n_zpoly);
       ~CUZPolySpsArray();
       CUZPolySps *getZPolySps(uint32_t index);
       uint32_t getNZPolySps();
       uint32_t *getZPolySpsRep();
       void show(uint32_t i);

};
*/
#endif
#endif

void maddm_h(uint32_t *pout, uint32_t *scalar, uint32_t *pin, uint32_t last_idx, uint32_t ncoeff, uint32_t pidx);

