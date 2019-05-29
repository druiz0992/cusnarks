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
// File name  : test_montgomery_fios.c
//
// Date       : 6/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Test script for Montgomery FIOS routine
//
// ------------------------------------------------------------------

// NOTE Signigicant parts of this code have been taken from :
//
// https://github.com/Xilinx/embeddedsw/blob/master/XilinxProcessorIPLib/drivers/hdcp22_rx/src/xhdcp22_rx_crypt.c
// https://github.com/Xilinx/embeddedsw/blob/master/XilinxProcessorIPLib/drivers/hdcp22_common/src/bigdigits.c

/******************************************************************************
*
* Copyright (C) 2015 - 2016 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/***** BEGIN LICENSE BLOCK *****
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2001-15 David Ireland, D.I. Management Services Pty Limited
 * <http://www.di-mgt.com.au/bigdigits.html>. All rights reserved.
 *
 ***** END LICENSE BLOCK *****/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef unsigned int u32;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define NDIGITS 8
#define MAX_NDIGITS_FIOS   ((NDIGITS) + 3)
#define MAX_DIGIT 0xFFFFFFFFUL
#define NTEST 1

static uint32_t N[] = {4026531841, 1138881939, 2042196113,  674490440, 2172737629,
                             3092268470, 3778125865,  811880050};

static uint32_t NPrime[] = {4026531839, 3269588371, 1281954227, 1703315019, 2567316369,
       3818559528,  226705842, 1945644829};

//static uint32_t N[] = { 216778863,0,0,0,0,0,0,0};
//static uint32_t NPrime[] = {1782147441,0,0,0,0,0,0,0};

void MontMulFios(u32 *U, u32 *A, u32 *B,
	u32 *N, const u32 *NPrime, int NDigits);

void mpAddWithCarryProp(u32 *A, u32 C, int SDigit, int NDigits);
u32 mpAdd(u32 w[], const u32 u[], const u32 v[], size_t ndigits);
int spMultiply(uint32_t p[2], uint32_t x, uint32_t y);
int mpCompare(const u32 a[], const u32 b[], size_t ndigits);
u32 mpSubtract(u32 w[], const u32 u[], const u32 v[], size_t ndigits);
void setRandom(u32 *x, u32);
void printNumber(u32 *x);

void main()
{
        //uint32_t  a [] = {2415918983, 3929206035, 2002373462, 4286002622, 3386719223,
                             //3792637116, 2408270919,  546761651};
        //uint32_t  b [] = {2415918983, 3929206035, 2002373462, 4286002622, 3386719223,
                             //3792637116, 2408270919,  546761651};
	//uint32_t a[] = {149865143,0,0,0,0,0,0,0};
	//uint32_t b[] = {149865143,0,0,0,0,0,0,0};
	uint32_t r[MAX_NDIGITS_FIOS]; 
	//uint32_t a[MAX_NDIGITS_FIOS]; 
	//uint32_t b[MAX_NDIGITS_FIOS]; 
	uint32_t a[] = {1804289383, 846930886, 1681692777, 1714636915, 1957747793, 424238335, 719885386, 576018668};
        uint32_t b[] = { 596516649, 1189641421 ,1025202362 ,1350490027 , 783368690 ,1102520059 ,2044897763 ,803772102};

	int i, j;

	for (i=0; i < NTEST; i++){
            //setRandom(a,NDIGITS);
            //setRandom(b,NDIGITS);
	    //a[NDIGITS-1] &= 0x3FFFFFFF;
	    //b[NDIGITS-1] &= 0x3FFFFFFF;
	    

            printf("P\n");
            printNumber(N);
            printf("NP\n");
            printNumber(NPrime);
           
            printf("A\n");
            printNumber(a);
            printf("B\n");
            printNumber(b);
            MontMulFios(r, a, b, N, NPrime, NDIGITS);

            printf("R\n");
            printNumber(r);

	}

}

void printNumber(u32 *x)
{
	    for (u32 i=0; i < NDIGITS; i++){
	  	printf("%u ",x[i]);
	    }
	    printf ("\n");
}

void setRandom(u32 *x, u32 ndigits)
{
	int i;

	for (i=0; i< ndigits; i++){
		x[i] = rand(); 
	}
}
/****************************************************************************/
/**
* This function implements the Montgomery Modular Multiplication (MMM)
* Finely Integrated Operand Scanning (FIOS) algorithm. The FIOS method
* interleaves multiplication and reduction operations. Requires NDigits+3
* words of temporary storage.
*
* U = MontMult(A,B,N)
*
* Reference:
* Analyzing and Comparing Montgomery Multiplication Algorithms
* IEEE Micro, 16(3):26-33,June 1996
* By: Cetin Koc, Tolga Acar, and Burton Kaliski
*
* @param	U is the MMM result
* @param	A is the n-residue input, A' = A*R mod N
* @param	B is the n-residue input, B' = B*R mod N
* @param	N is the modulus
* @param	NPrime is a pre-computed constant, NPrime = (1-R*Rbar)/N
* @param	NDigits is the integer precision of the arguments (C,A,B,N,NPrime)
*
* @return	None.
*
* @note		None.
*****************************************************************************/
void MontMulFios(u32 *U, u32 *A, u32 *B,
	u32 *N, const u32 *NPrime, int NDigits)
{
	int i, j;
	u32 S, C, C1, C2, M[2], X[2];
	u32 T[MAX_NDIGITS_FIOS];

	memset(T, 0, 4*(NDigits+3));

	for(i=0; i<NDigits; i++)
	{
		// (C,S) = t[0] + a[0]*b[i], worst case 2 words
		spMultiply(X, A[0], B[i]);	// X[Upper,Lower] = a[0]*b[i]
		C = mpAdd(&S, T+0, X+0, 1);	// [C,S] = t[0] + X[Lower]
		mpAdd(&C, &C, X+1, 1);		// [~,C] = C + X[Upper], No carry

                printf("0 - C : %u, S: %u\n",C,S);
                printf("0 - A[0] : %u, B[i]: %u T[0] : %u\n",A[0],B[i], T[0]);
		// ADD(t[1],C)
		mpAddWithCarryProp(T, C, 1, NDigits+3);
                printf("T\n");
                printNumber(T);

		// m = S*n'[0] mod W, where W=2^32
		// Note: X[Upper,Lower] = S*n'[0], m=X[Lower]
		spMultiply(M, S, NPrime[0]);
                printf("M[0]:%u, M[1]: %u\n",M[0], M[1]);

		// (C,S) = S + m*n[0], worst case 2 words
		spMultiply(X, M[0], N[0]);	// X[Upper,Lower] = m*n[0]
		C = mpAdd(&S, &S, X+0, 1);	// [C,S] = S + X[Lower]
		mpAdd(&C, &C, X+1, 1);		// [~,C] = C + X[Upper]
                printf("1 - C : %u, S: %u\n",C,S);

		for(j=1; j<NDigits; j++)
		{
			// (C,S) = t[j] + a[j]*b[i] + C, worst case 2 words
			spMultiply(X, A[j], B[i]);	 	// X[Upper,Lower] = a[j]*b[i], double precision
			C1 = mpAdd(&S, T+j, &C, 1);		// (C1,S) = t[j] + C
                        printf("2 - C1 : %u, S: %u\n",C1,S);
			C2 = mpAdd(&S, &S, X+0, 1); 	// (C2,S) = S + X[Lower]
                        printf("3 - C2 : %u, S: %u\n",C1,S);
                        printf("X[0] : %u, X[1]: %u\n",X[0],X[1]);
			mpAdd(&C, &C1, X+1, 1);			// (~,C)  = C1 + X[Upper], doesn't produce carry
                        printf("4 - C : %u\n",C);
			mpAdd(&C, &C, &C2, 1); 			// (~,C)  = C + C2, doesn't produce carry
                        printf("5 - C : %u\n",C);

			// ADD(t[j+1],C)
			mpAddWithCarryProp(T, C, j+1, NDigits+3);
                        printf("T\n");
                        printNumber(T);

			// (C,S) = S + m*n[j]
			spMultiply(X, M[0], N[j]);	// X[Upper,Lower] = m*n[j]
			C = mpAdd(&S, &S, X+0, 1);	// [C,S] = S + X[Lower]
			mpAdd(&C, &C, X+1, 1);		// [~,C] = C + X[Upper]
                        printf("6 - C : %u, S: %u\n",C,S);

			// t[j-1] = S
			T[j-1] = S;
                        printf("T\n");
                        printNumber(T);
		}

		// (C,S) = t[s] + C
		C = mpAdd(&S, T+NDigits, &C, 1);
                printf("6 - C : %u, S: %u\n",C,S);
		// t[s-1] = S
		T[NDigits-1] = S;
		// t[s] = t[s+1] + C
		mpAdd(T+NDigits, T+NDigits+1, &C, 1);
		// t[s+1] = 0
		T[NDigits+1] = 0;
	}

	/* Step 3: if(u>=n) return u-n else return u */
	if(mpCompare(T, N, NDigits) >= 0)
	{
		mpSubtract(T, T, N, NDigits+3);
	}

	memcpy(U, T, 4*NDigits);
}

/****************************************************************************/
/**
* This function performs a carry propagation adding C to the input
* array A of size NDigits, given by the first argument starting from
* the first element SDigit, and propagates it until no further carry
* is generated.
*
* ADD(A[i],C)
*
* Reference:
* Analyzing and Comparing Montgomery Multiplication Algorithms
* IEEE Micro, 16(3):26-33,June 1996
* By: Cetin Koc, Tolga Acar, and Burton Kaliski
*
* @param	A is an input array of size NDigits
* @param	C is the value being added to the input A
* @param	SDigit is the start digit
* @param	NDigits is the integer precision of the arguments (A)
*
* @return	None.
*
* @note		None.
*****************************************************************************/
void mpAddWithCarryProp(u32 *A, u32 C, int SDigit, int NDigits)
{
	int i;
	int j=0;

	for(i=SDigit; i<NDigits; i++)
	{
		C = mpAdd(A+i, A+i, &C, 1);

		if(C == 0)
		{
			//if (j > 0) {
			       	//printf("%d\n",j);
			//}
			return;
		}
		j++;
	}
	//if (j > 0) { printf("%d\n",j);}
}

u32 mpAdd(u32 w[], const u32 u[], const u32 v[], size_t ndigits)
{
	/*	Calculates w = u + v
		where w, u, v are multiprecision integers of ndigits each
		Returns carry if overflow. Carry = 0 or 1.
		Ref: Knuth Vol 2 Ch 4.3.1 p 266 Algorithm A.
	*/

	u32 k;
	size_t j;

	/* Step A1. Initialise */
	k = 0;

	for (j = 0; j < ndigits; j++)
	{
		/*	Step A2. Add digits w_j = (u_j + v_j + k)
			Set k = 1 if carry (overflow) occurs
		*/
		w[j] = u[j] + k;
		if (w[j] < k)
			k = 1;
		else
			k = 0;

		w[j] += v[j];
		if (w[j] < v[j])
			k++;

	}	/* Step A3. Loop on j */

	return k;	/* w_n = k */
}

int spMultiply(uint32_t p[2], uint32_t x, uint32_t y)
{
	/* Use a 64-bit temp for product */
	uint64_t t = (uint64_t)x * (uint64_t)y;
	/* then split into two parts */
	p[1] = (uint32_t)(t >> 32);
	p[0] = (uint32_t)(t & 0xFFFFFFFF);

	return 0;
}

int mpCompare(const u32 a[], const u32 b[], size_t ndigits)
{
	/* All these vars are either 0 or 1 */
	unsigned int gt = 0;
	unsigned int lt = 0;
	unsigned int mask = 1;	/* Set to zero once first inequality found */
	unsigned int c;

	while (ndigits--) {
		gt |= (a[ndigits] > b[ndigits]) & mask;
		lt |= (a[ndigits] < b[ndigits]) & mask;
		c = (gt | lt);
		mask &= (c-1);	/* Unchanged if c==0 or mask==0, else mask=0 */
	}

	return (int)gt - (int)lt;	/* EQ=0 GT=+1 LT=-1 */
}
u32 mpSubtract(u32 w[], const u32 u[], const u32 v[], size_t ndigits)
{
	/*	Calculates w = u - v where u >= v
		w, u, v are multiprecision integers of ndigits each
		Returns 0 if OK, or 1 if v > u.
		Ref: Knuth Vol 2 Ch 4.3.1 p 267 Algorithm S.
	*/

	u32 k;
	size_t j;

	/* Step S1. Initialise */
	k = 0;

	for (j = 0; j < ndigits; j++)
	{
		/*	Step S2. Subtract digits w_j = (u_j - v_j - k)
			Set k = 1 if borrow occurs.
		*/
		w[j] = u[j] - k;
		if (w[j] > MAX_DIGIT - k)
			k = 1;
		else
			k = 0;

		w[j] -= v[j];
		if (w[j] > MAX_DIGIT - v[j])
			k++;

	}	/* Step S3. Loop on j */

	return k;	/* Should be zero if u >= v */
}


