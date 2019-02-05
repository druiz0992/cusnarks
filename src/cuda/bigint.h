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
// File name  : bigint.h
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of big integer class
// ------------------------------------------------------------------

*/

class BigInt {
    const uint32_t XOFFSET = 0;
    const uint32_t YOFFSET = 1;
    const uint32_t ZOFFSET = 2;
    const uint32_t VWIDTH = 3;

    private:
        uint256_t *array_device;    // pointer to device buffer
        uint256_t *array_host;      // pointer to host buffer
	uint256_t *p;               // prime number
        uint32_t  len;              // array len

    public:
        BigInt(uint256_t *vector, uint256_t *p, uint32_t len) ;
        ~BigInt();
        void addm();
        void retrieve(uint256_t *vector, uint32_t len);
}
