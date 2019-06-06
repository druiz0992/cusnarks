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

#ifdef __CUDACC__
  __host__ __device__ void logBigNumber(char *str, uint32_t *n)
#else
  void logBigNumber(char *str, uint32_t *n)
#endif
{
 uint32_t i;
 //char buf[500];
 //memset(buf,0, 500*sizeof(char));
 printf("%s",str);
 
 for (i=0; i < NWORDS_256BIT; i++){
   printf("%u ",n[i]);
 }
 printf("\n");
}

#ifdef __CUDACC__
  __host__ __device__ void logBigNumberExt(char *str, uint32_t *n)
#else
  void logBigNumberExt(char *str, uint32_t *n)
#endif
{
 uint32_t i;
 //char buf[500];
 //memset(buf,0, 500*sizeof(char));
 printf("%s",str);
 
 for (i=0; i < NWORDS_256BIT; i++){
   printf("%u ",n[i]);
 }
 printf("\n");
 for (i=NWORDS_256BIT; i < 2*NWORDS_256BIT; i++){
   printf("%u ",n[i]);
 }
 printf("\n");
}
#if LOG_LEVEL <= LOG_LEVEL_DEBUG
   #ifdef __CUDACC__
     __host__ __device__ void logDebugBigNumber(char *str, uint32_t *n)
   #else
     void logDebugBigNumber(char *str, uint32_t *n)
   #endif
     {
        logBigNumber(str, n);
     }
  #if defined (LOG_TID) && defined (__CUDACC__)
    __device__ void logDebugBigNumberTid(int tid,uint32_t nelems, char *str, uint32_t *n)
    {
       if (tid == LOG_TID){
         uint32_t i;
         for (i=0; i< nelems; i++){
           logBigNumber(str, &n[i*NWORDS_256BIT]);
         }
       }
    }
    
    __device__ void logDebugBigNumberTid(int tid,uint32_t nelems, char *str, Z1_t *n)
    {
       if (tid == LOG_TID){
         uint32_t i;
         for (i=0; i< nelems; i++){
           logBigNumber(str, n->getu256(i));
         }
       }
    }
    __device__ void logDebugBigNumberTid(int tid,uint32_t nelems, char *str, Z2_t *n)
    {
       if (tid == LOG_TID){
         uint32_t i;
         for (i=0; i< nelems/2; i++){
           logBigNumberExt(str, n->getu256(i));
         }
       }
    }
    __device__ void logDebugTid(int tid, uint32_t nelems, const char *f,uint32_t args )
    {
       if (tid == LOG_TID){
          printf(f,args);
       }
    }
  #endif
#endif

#if LOG_LEVEL <= LOG_LEVEL_INFO
   #ifdef __CUDACC__
     __host__ __device__ void logInfoBigNumber(char *str, uint32_t *n)
   #else
     void logInfoBigNumber(char *str, uint32_t *n)
   #endif
     {
        logBigNumber(str, n);
     }
  #if defined (LOG_TID) && defined (__CUDACC__)
      __device__ void logInfoBigNumberTid(int tid, uint32_t nelems, char *str, uint32_t *n)
      {
         if (tid == LOG_TID){
           uint32_t i;
           for (i=0; i< nelems; i++){
             logBigNumber(str, &n[i*NWORDS_256BIT]);
           }
         }
      }
     __device__ void logInfoBigNumberTid(int tid, uint32_t nelems, char *str, Z1_t *n)
      {
         if (tid == LOG_TID){
           uint32_t i;
           for (i=0; i< nelems; i++){
             logBigNumber(str, n->getu256(i));
           }
         }
      }
    __device__ void logInfoBigNumberTid(int tid,uint32_t nelems, char *str, Z2_t *n)
    {
       if (tid == LOG_TID){
         uint32_t i;
         for (i=0; i< nelems/2; i++){
           logBigNumberExt(str, n->getu256(i));
         }
       }
    }
    //template <typename  Args> 
     __device__ char * logInfoTid(int tid, const char *f, uint32_t args)
    {
       if (tid == LOG_TID){
          printf(f,args);
       }
    }
  #endif
#endif



   
  

