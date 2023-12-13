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
// File name  : gen_roots.cpp
//
// Date       : 6/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Generate roots
//
// ------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

#include "types.h"
#include "file_utils.h"
#include "ntt.h"


void test_gen_roots(uint32_t nbits, char *filename)
{
  uint32_t nsamples = (1<<nbits) * NWORDS_FR;
  uint32_t *roots = (uint32_t *)malloc( nsamples * sizeof(uint32_t));

  computeRoots_h(roots,nbits);

  writeU256DataFile_h(roots, filename, nsamples);
}


int main(int argc, char **argv )
{
  char *n;
  test_gen_roots(strtol(argv[1], &n,10),argv[2]);
  return 1;
}

