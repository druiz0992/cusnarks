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
//  Definition of logging functionality
// ------------------------------------------------------------------

*/


#ifndef _LOG_H
#define _LOG_H

#define LOG_LEVEL_NOLOG   (4)
#define LOG_LEVEL_ERROR   (3)
#define LOG_LEVEL_WARNING (2)
#define LOG_LEVEL_INFO    (1)
#define LOG_LEVEL_DEBUG   (0)

#ifndef LOG_LEVEL
  #define LOG_LEVEL (LOG_LEVEL_NOLOG)
#endif


#if LOG_LEVEL <= LOG_LEVEL_DEBUG
   #define logDebug printf
   #ifdef __CUDACC__
   __host__ __device__ void logDebugBigNumber(char *str, uint32_t *x);
   #else
   void logDebugBigNumber(char *str, uint32_t *x);
   #endif
#else
  #define logDebug(f,...) 
  #define logDebugBigNumber(STR, X)
#endif


#if LOG_LEVEL <= LOG_LEVEL_ERROR
   #define logError printf
   #ifdef __CUDACC__
   __host__ __device__ void logInfoBigNumber(char *str, uint32_t *x);
   #else
   void logInfoBigNumber(char *str, uint32_t *x);
   #endif
#else
  #define logError(f,...) 
#endif


#if LOG_LEVEL <= LOG_LEVEL_WARNING
   #define logWarning printf
#else
  #define logWarning(f,...)
#endif

#if LOG_LEVEL <= LOG_LEVEL_INFO
   #define logInfo printf
#else
  #define logInfo(f,...)
#endif


#endif
