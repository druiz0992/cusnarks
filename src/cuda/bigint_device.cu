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
// File name  : bigint_kernel.cu
//
// Date       : 05/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation of biginteger kernel and device functions
// ------------------------------------------------------------------

*/

/*
   addition of two 256 bit number modulo p Z[i] = X[i] + Y[i] (mod p)

   Input vector contains intercalated X, Y and Z numbers (X[0], Y[0], Z[0], X[1], Y[1], Z[1],..
    X[N-1], Y[N-1], Z[N-1]) where X, Y and Z are 256 bit numbers represented as an array of uint32_t
   
*/

#include <stdio.h>

#include "types.h"
#include "bigint_device.h"

//void __global__ addm_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector)
void addm_kernel(uint32_t *in_vector, uint32_t *p, uint32_t len, uint32_t *out_vector)
{
    //int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = 0;

    const uint32_t *x;
    const uint32_t *y;
    uint32_t * z;
    uint32_t c = 0;
    uint32_t i;
 
    if(tid >= len) {
      return;
    }

    x = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + XOFFSET];
    y = (uint32_t *) &in_vector[tid * 2 * NWORDS_256BIT + YOFFSET];
    z = (uint32_t *) &out_vector[tid * NWORDS_256BIT];

    for (i=0; i < NWORDS_256BIT; i++){
      z[i] = x[i] + y[i] + c;
      c = (z[i] < x[i]);
    }

    return;
}

/*
__global__ void BitInt_ModSub256()
{
}

__global__ void BigInt_ModNeg256()
{
}

__global__ void ciosV2(KernelArray<unsigned int>d_a1, KernelArray<unsigned int>d_b1, KernelArray<unsigned int>d_ans, KernelArray<unsigned int>d_n, KernelArray<unsigned int>d_n1, int d_s, int blkSize)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int t[33] = { 0 };
    unsigned long long temp;
    __shared__ unsigned int shared_n[32], shared_n1[32], shared_s;
    shared_s = d_s;
    for (int i = 0; i < shared_s; i++){
        shared_n[i] = d_n._array[i];
        shared_n1[i] = d_n1._array[i];
    }
    __syncthreads();
    for (int i = 0; i < shared_s; i++){
        unsigned long long c = 0;
        for (int j = 0; j < shared_s; j++){
            temp = t[j] + (unsigned long long)d_a1._array[j * (1024 *
            blkSize) + idx] * (unsigned long long)d_b1._array[i * (1024 * blkSize) + idx] +
            c;
            t[j] = temp & 4294967295;
            c = temp >> 32;
        }
        temp = (unsigned long long)t[shared_s] + c;
        t[shared_s] = temp & 4294967295;
        t[shared_s + 1] = temp >> 32;
        unsigned long long m = ((unsigned long long)t[0] * (unsigned long
        long)shared_n1[0]) & 4294967295;
        temp = (unsigned long long)t[0] + m*(unsigned long
        long)shared_n[0];
        c = temp >> 32;
        for (int j = 1; j < shared_s; j++){
            temp = (unsigned long long)t[j] + m*(unsigned long
            long)shared_n[j] + c;
            t[j - 1] = temp & 4294967295;
            c = temp >> 32;
        }
        temp = (unsigned long long)t[shared_s] + c;
        t[shared_s - 1] = temp & 4294967295;
        c = temp >> 32;
        t[shared_s] = t[shared_s + 1] + c;
    }
    unsigned int u[33];
    for (int j = 0; j < shared_s + 1; j++){
        u[j] = t[j];
    }
    int b = 0;
    long long sub;
    for (int i = 0; i < shared_s; i++){
        sub = (long long)u[i] - shared_n[i] - b;
        if (sub < 0){
            t[i] = sub + 4294967296;
            b = 1;
        }
        else{
            t[i] = sub;
            b = 0;
        }
    }
    sub = (long long)u[shared_s] - b;
    u[shared_s] = sub;
    if (sub >= 0){
        int counter = 0;
        for (int i = 0; i < 32; i++){
            d_ans._array[i * 1024 * blkSize + idx] = t[counter++];
        }
    }
    else{
        int counter = 0;
        for (int i = 0; i < 32; i++){
            d_ans._array[i * 1024 * blkSize + idx] = u[counter++];
        }
    }
}

__global__ void BigInt_MontMul(KernelArray<unsigned int>d_a1, KernelArray<unsigned int>d_b1, KernelArray<unsigned int>d_ans, KernelArray<unsigned int>d_n, KernelArray<unsigned int>d_n1, int d_s, int blkSize)
{
    for (int i = 0; i < shared_s; i++){
        unsigned long long c = 0;
        for (int j = 0; j < shared_s; j++){
            temp = t[j] + (unsigned long long)d_a1._array[j * (1024 *
            blkSize) + idx] * (unsigned long long)d_b1._array[i * (1024 * blkSize) + idx] +
            c;
            t[j] = temp & 4294967295;
            c = temp >> 32;
        }
        temp = (unsigned long long)t[shared_s] + c;
        t[shared_s] = temp & 4294967295;
        t[shared_s + 1] = temp >> 32;
        unsigned long long m = ((unsigned long long)t[0] * (unsigned long
        long)shared_n1[0]) & 4294967295;
        temp = (unsigned long long)t[0] + m*(unsigned long
        long)shared_n[0];
        c = temp >> 32;
        for (int j = 1; j < shared_s; j++){
            temp = (unsigned long long)t[j] + m*(unsigned long
            long)shared_n[j] + c;
            t[j - 1] = temp & 4294967295;
            c = temp >> 32;
        }
        temp = (unsigned long long)t[shared_s] + c;
        t[shared_s - 1] = temp & 4294967295;
        c = temp >> 32;
        t[shared_s] = t[shared_s + 1] + c;
    }
    unsigned int u[33];
    for (int j = 0; j < shared_s + 1; j++){
        u[j] = t[j];
    }
    int b = 0;
    long long sub;
    for (int i = 0; i < shared_s; i++){
        sub = (long long)u[i] - shared_n[i] - b;
        if (sub < 0){
            t[i] = sub + 4294967296;
            b = 1;
        }
        else{
            t[i] = sub;
            b = 0;
        }
    }
    sub = (long long)u[shared_s] - b;
    u[shared_s] = sub;
    if (sub >= 0){
        int counter = 0;
        for (int i = 0; i < 32; i++){
            d_ans._array[i * 1024 * blkSize + idx] = t[counter++];
        }
    }
    else{
        int counter = 0;
        for (int i = 0; i < 32; i++){
            d_ans._array[i * 1024 * blkSize + idx] = u[counter++];
        }
    }
}

*/
