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
#ifndef _BIGINT_H_
#define _BIGINT_H_

class BigInt {
    private:
        uint32_t *in_vector_device;    // pointer to device input buffer
        uint32_t *out_vector_device;    // pointer to device output buffer
	uint32_t *p;               // prime number
        const uint32_t  in_vector_len;              // array len

        void copyVectorToDevice(uint32_t *in_vector_host, uint32_t len);
        void copyVectorFromDevice(uint32_t *out_vector_host, uint32_t len);

    public:

        BigInt(const uint32_t *p, const uint32_t len) ;
        ~BigInt();
        void addm(uint32_t *in_vector_host, uint32_t *out_vector_host, const uint32_t len);
};

#endif
