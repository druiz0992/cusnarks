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
// File name  : rng.h
//
// Date       : 08/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Random number generator class definition
// ------------------------------------------------------------------

*/

#ifndef _RNG_H_
#define _RNG_H_

#include "pcg_random.hpp"

class _RNG {
    private:
        static _RNG *instance;
        pcg32 rng; 

        // prevent instances
        _RNG();
        _RNG(uint32_t seed);

    public:
        ~_RNG();
	static _RNG* get_instance();
	static _RNG* get_instance(uint32_t seed);
        void randu32(uint32_t *samples, uint32_t n_samples);
        //void randu256(uint32_t *samples, uint32_t n_samples, uint32_t *mod);
};

#endif
