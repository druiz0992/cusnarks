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
// File name  : u256.h
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of U256 integer class
// ------------------------------------------------------------------

*/
#ifndef _U256_H_
#define _U256_H_

class U256 {

    public:

        U256(const uint32_t *p, uint32_t len);
        U256(const uint32_t *p, uint32_t len, uint32_t seed);
        void rand(uint32_t *samples, const uint32_t n_samples);
        void addm(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, uint32_t premod);
        void subm(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, uint32_t premod);
        void mod(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len);
        void mulm(uint32_t *out_vector_host, const uint32_t *in_vector_host, uint32_t len, uint32_t np, uint32_t premod);
};

#endif
