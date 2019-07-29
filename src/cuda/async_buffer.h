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

template <class T> 
class AsyncBuf {
    private:
        T *data;
        uint32_t max_nelems;

    public:
 
        T * get(void);
        uint32_t  getNelems(void);
        uint32_t set(T *in_data, uint32_t nelems);   
        AsyncBuf(uint32_t nelems)
        ~AsyncBuf();
};

#endif
