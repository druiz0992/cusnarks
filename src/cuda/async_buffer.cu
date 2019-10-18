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
//  Implementation of data buffer setored in pinned memory to enable async data
//       transfers between host and device
// ------------------------------------------------------------------

*/

#include <stdio.h>

#include "types.h"
#include "log.h"
#include "cuda.h"
#include "async_buffer.h"

AsyncBuf::AsyncBuf(uint32_t nelems, uint32_t elsize):max_nelems(nelems), el_size(elsize)
{
    logInfo("Nelems : %d, Elsize : %d, Buffer : %x\n",nelems, elsize, buffer);	
    CCHECK(cudaMallocHost((void **)&buffer, nelems*el_size));
}        

AsyncBuf::~AsyncBuf()
{
  logInfo("release async buffer : %x\n",buffer);
  CCHECK(cudaFreeHost(buffer));
}

void * AsyncBuf::getBuf(void)
{
    return buffer;
}

uint32_t AsyncBuf::getNelems(void)
{
    return max_nelems;
}

uint32_t AsyncBuf::setBuf(void *in_data, uint32_t nelems) 
{  
     logInfo("Set buffer: in_data : %x, n_elenms : %d, max_nelems : %d\n", in_data, nelems, max_nelems);
     if (nelems > max_nelems){
         return 1;
     }
     memcpy(buffer, in_data, el_size * nelems);
     return 0;   
}


