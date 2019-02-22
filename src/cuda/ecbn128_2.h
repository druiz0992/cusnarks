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
// File name  : ecbn128_2.h
//
// Date       : 22/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of Elliptic Curve kernel processing for extended fields
//
//  TODO
//  This implementation is quite embarrasing. I am practically copying
//  all ecbn128.cu functionality. I need to use C++ so that ecc over extended fields
//  is a derived class from ecc and simply let the compiler select the right
//  class. For the moment though, this will have to do :-(
// ------------------------------------------------------------------

*/
#ifndef _ECBN128_2_H_
#define _ECBN128_2_H_


class ECBN128_2 : public CUSnarks {

    public:

        ECBN128_2(uint32_t len);
        ECBN128_2(uint32_t len, uint32_t seed);
        
};

#endif
