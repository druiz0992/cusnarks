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
// Date       : 08/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Random number generator
// ------------------------------------------------------------------

*/
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <climits>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <random>       // for random_device

#include "types.h"
#include "rng.h"

using namespace std;

_RNG::_RNG():rng(43u)
{
   this->rng.seed(pcg_extras::seed_seq_from<std::random_device>());
}
_RNG::_RNG(uint32_t seed) : rng(seed) { }

_RNG* _RNG::get_instance()
{
  if (instance == NULL){
     instance = new _RNG();
  }

  return instance;
}
_RNG* _RNG::get_instance(uint32_t seed)
{
  if (instance == 0){
     instance = new _RNG(seed);
  }

  return instance;
}

void _RNG::destroy()
{
  delete instance;
  instance = NULL;
}

void _RNG::randu32(uint32_t *samples, uint32_t n_samples)
{
   for (int i =0; i < n_samples; i++){
     samples[i] = rng();
   }
}

// null
_RNG* _RNG::instance=NULL;

