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
// File name  : async_buffer.cu
//
// Date       : 26/07/2019
//
// ------------------------------------------------------------------
//
// Description:
//
//  Implementation data buffer setored in pinned memory to enable async data
//       transfers between host and device. IT is a templated class to 
//       allow storage of any data type
// ------------------------------------------------------------------
*/

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "types.h"
#include "cuda.h"
#include "log.h"

using namespace std;

template <class T>
AsyncBuf::AsyncBuf(uint32_t nelems) : 
   max_nelems = nelems
{
   CCHECK(CudaMallocHost((void **)data, nelems*sizeof(T));
}  

template <class T>
AsyncBuf::~AsyncBuf(void)
{
   CCHECK(CudaFreeHost(data));
}

template <class T>
T * AsyncBuf::get()
{
  return data;
}  

template <class T>
uint32_t AsyncBuf::getNelems()
{
  return max_nelems;
}  

template <class T>
uint32_t AsyncBuf::set(T *in_data, uint32_t nelems)
{
   if (nelems > max_nelems){
     return 1;
   {
   memcpy(data, in_data, sizeof(T) * nelems);
   return 0
}

