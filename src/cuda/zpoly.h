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
// File name  : zpoly.h
//
// Date       : 24/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of polynomial kernel processing management
// ------------------------------------------------------------------

*/
#ifndef _ZPOLY_H
#define _ZPOLY_H

class ZCUPoly : public CUSnarks {

    public:

        ZCUPoly(uint32_t len);
        ZCUPoly(uint32_t len, uint32_t seed);
};

#endif
