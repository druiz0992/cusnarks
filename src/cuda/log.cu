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
// File name  : log.cpp
//
// Date       : 15/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of logging functionality
// ------------------------------------------------------------------

*/

#include <stdio.h>
#include <stdarg.h>
#include "types.h"
#include "log.h"

#if LOG_LEVEL <= LOG_LEVEL_DEBUG
   #ifdef __CUDACC__
__host__ __device__ void logDebugBigNumber(char *str, uint32_t *n)
#else
void logDebugBigNumber(char *str, uint32_t *n)
#endif
{
  uint32_t i;
  char buf[500];
  memset(buf,0, 500*sizeof(char));
  logDebug("%s",str);
  
  for (i=0; i < NWORDS_256BIT; i++){
    logDebug("%u ",n[i]);
  }
  logDebug("\n");
}
#endif

#if LOG_LEVEL <= LOG_LEVEL_INFO
   #ifdef __CUDACC__
__host__ __device__ void logInfoBigNumber(char *str, uint32_t *n)
#else
void logInfoBigNumber(char *str, uint32_t *n)
#endif
{
  uint32_t i;
  char buf[500];
  memset(buf,0, 500*sizeof(char));
  logInfo("%s",str);
  
  for (i=0; i < NWORDS_256BIT; i++){
    logInfo("%u ",n[i]);
  }
  logInfo("\n");
}
#endif
