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
// File name  : async_buffer.h
//
// Date       : 26/07/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Definition of data buffer setored in pinned memory to enable async data
//       transfers between host and device
// ------------------------------------------------------------------

*/
#ifndef _ASYNC_BUFFER_H_
#define _ASYNC_BUFFER_H_

// Templates is the correct way to implement this function. However, I can't make it work 
// (functionalitiy implemented in h file using cuda functions) with cython, as there are lots of dependencies. 
// For now, I will use void *

/*
template <class T> 
class AsyncBuf {
    private:
        T *buffer;
        uint32_t max_nelems;

    public:
 
        AsyncBuf(uint32_t nelems):max_nelems(nelems)
        {
          printf("size of T : %d\n", sizeof(T));
          CCHECK(cudaMallocHost((void **)&buffer, nelems*sizeof(T)));
          return;
        }

        ~AsyncBuf()
        {
          CCHECK(cudaFreeHost(buffer));
          return;
        }

        T * getBuf(void)
        {
           return buffer;
        }

        uint32_t  getNelems(void)
        {
           return max_nelems;
        }

        uint32_t setBuf(T *in_data, uint32_t nelems) 
        {  
          printf("setBuf. data : %d, nelems : %d, max_nelems %d \n",in_data, nelems, max_nelems);
          if (nelems > max_nelems){
            return 1;
          }
          printf("memcpy beore : \n");
          memcpy(buffer, in_data, sizeof(T) * nelems);
          printf("memcpy after\n");
          return 0;   
       }
};

*/


class AsyncBuf {
    private:
        void *buffer;
        uint32_t max_nelems;
        uint32_t el_size;

    public:
 
        AsyncBuf(uint32_t nelems, uint32_t el_size);
        ~AsyncBuf();

        void * getBuf(void);
        uint32_t  getNelems(void);
        uint32_t setBuf(void *in_data, uint32_t nelems); 
};


#endif
