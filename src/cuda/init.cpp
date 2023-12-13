
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
// File name  : init.cpp
//
// Date       : 6/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Init and Release Host resources
//
// ------------------------------------------------------------------
#include <pthread.h>

#include "types.h"
#include "utils_host.h"
#include "constants.h"
#include "bigint.h"
#include "ec.h"
#include "ntt.h"
#include "mpoly.h"
#include "init.h"


/*
  General initialization function. To be called at the beginning of program
*/
void init_h(void)
{
  utils_init_h();
  ec_init_h();
  //mpoly_init_h(1 << CusnarksGetNRoots());
  ntt_init_h(1<< CusnarksGetNRoots());
}

/*
  Release resources
*/
void release_h(void)
{
  utils_free_h();
  ec_free_h();
  //mpoly_free_h();
  ntt_free_h();
}

