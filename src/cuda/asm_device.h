
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
// File name  : asm_device.h
//
// Date       : 27/01/2020
//
// ------------------------------------------------------------------
//
// Description:
//   Field and Group arithmetic in ptx 
//
// ------------------------------------------------------------------

// NOTE Implementation of Montgomery Multiplication has been taken from
//
// https://github.com/matter-labs/belle_cuda/blob/master/sources/mont_mul.cu
//

/******************************************************************************
*  Copyright 2020 0kims association.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
******************************************************************************/


#ifndef _ASM_DEVICE_H_
#define _ASM_DEVICE_H_

// Montgomery Multiplication (FIOS)

#define ASM_MONTMULU256(out,in1,in2)  \
    ASM_MUL_START(out,in1,in2) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out, in1, in2,1) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out, in1, in2,2) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out,in1,in2,3) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out,in1,in2,4) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out,in1,in2,5) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out,in1,in2,6) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out,in1,in2,7) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out)

#if 0
#define ASM_MONTSQ256(out,in1, in2)  \
    ASM_MONTSQ_START(out,in1, in2) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MONTSQ_BLOCK1(out, in1, in2) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out, in1, in2,2) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out,in1,in2,3) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out,in1,in2,4) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out,in1,in2,5) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out,in1,in2,6) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out) \
    ASM_MUL_BLOCK(out,in1,in2,7) \
    ASM_MUL_REDUCTION_BLOCK_LAST(out)
#endif

#define ASM_MONTMULU256_64(out,in1,in2)  \
    ASM_MUL_START_64(out,in1,in2) \
    ASM_MUL_REDUCTION_BLOCK_LAST_64(out) \
    ASM_MUL_BLOCK_64(out, in1, in2,1) \
    ASM_MUL_REDUCTION_BLOCK_LAST_64(out) \
    ASM_MUL_BLOCK_64(out, in1, in2,2) \
    ASM_MUL_REDUCTION_BLOCK_LAST_64(out) \
    ASM_MUL_BLOCK_64(out,in1,in2,3) \
    ASM_MUL_REDUCTION_BLOCK_LAST_64(out) 

#define ASM_MUL_INIT \
      "{\n\t" \
           ".reg .u32 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"  \
           ".reg .u32 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"  \
           ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"  \
           ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"  \
           ".reg .u32 m, q, lo, hi;\n\t"  \
	   ".reg .pred p1;\n\t" \
                                     \
           "mov.u32         a0, %8;\n\t" \
           "mov.u32         a1, %9;\n\t" \
           "mov.u32         a2, %10;\n\t" \
           "mov.u32         a3, %11;\n\t" \
           "mov.u32         a4, %12;\n\t" \
           "mov.u32         a5, %13;\n\t" \
           "mov.u32         a6, %14;\n\t" \
           "mov.u32         a7, %15;\n\t" \
                                        \
           "mov.u32         b0, %16;\n\t" \
           "mov.u32         b1, %17;\n\t" \
           "mov.u32         b2, %18;\n\t" \
           "mov.u32         b3, %19;\n\t" \
           "mov.u32         b4, %20;\n\t" \
           "mov.u32         b5, %21;\n\t" \
           "mov.u32         b6, %22;\n\t" \
           "mov.u32         b7, %23;\n\t" \
            \
           "mov.u32         n0, %24;\n\t" \
           "mov.u32         n1, %25;\n\t" \
           "mov.u32         n2, %26;\n\t" \
           "mov.u32         n3, %27;\n\t" \
           "mov.u32         n4, %28;\n\t" \
           "mov.u32         n5, %29;\n\t" \
           "mov.u32         n6, %30;\n\t" \
           "mov.u32         n7, %31;\n\t" \
           "mov.u32         q, %32;\n\t"

#define ASM_MUL_INIT_64 \
      "{\n\t" \
           ".reg .u32 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"  \
           ".reg .u32 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"  \
           ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"  \
           ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"  \
           ".reg .u32 m, q, lo, hi;\n\t"  \
	   ".reg .pred p1;\n\t" \
                                     \
           "mov.b64         {a0,a1}, %4;\n\t" \
           "mov.b64         {a2,a3}, %5;\n\t" \
           "mov.b64         {a4,a5}, %6;\n\t" \
           "mov.b64         {a6,a7}, %7;\n\t" \
                                        \
           "mov.b64         {b0,b1}, %8;\n\t" \
           "mov.b64         {b2,b3}, %9;\n\t" \
           "mov.b64         {b4,b5}, %10;\n\t" \
           "mov.b64         {b6,b7}, %11;\n\t" \
                                        \
           "mov.b64         {n0,n1}, %12;\n\t" \
           "mov.b64         {n2,n3}, %13;\n\t" \
           "mov.b64         {n4,n5}, %14;\n\t" \
           "mov.b64         {n6,n7}, %15;\n\t" \
                                        \
           "mov.u32         q, %16;\n\t"

#define ASM_MONTSQ_INIT_64 \
      "{\n\t" \
           ".reg .u32 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"  \
           ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"  \
           ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"  \
           ".reg .u32 m, q, lo, hi;\n\t"  \
	   ".reg .pred p1;\n\t" \
                                     \
           "mov.b64         {a0,a1}, %4;\n\t" \
           "mov.b64         {a2,a3}, %5;\n\t" \
           "mov.b64         {a4,a5}, %6;\n\t" \
           "mov.b64         {a6,a7}, %7;\n\t" \
                                        \
           "mov.b64         {n0,n1}, %8;\n\t" \
           "mov.b64         {n2,n3}, %9;\n\t" \
           "mov.b64         {n4,n5}, %10;\n\t" \
           "mov.b64         {n6,n7}, %11;\n\t" \
                                        \
           "mov.u32         q, %12;\n\t"


#define ASM_MUL_INIT_64_2 \
      "{\n\t" \
           ".reg .u64 a0, a1, a2, a3;\n\t"  \
           ".reg .u64 b0, b1, b2, b3;\n\t"  \
           ".reg .u64 r0, r1, r2, r3;\n\t"  \
           ".reg .u64 n0, n1, n2, n3;\n\t"  \
           ".reg .u64 m, q, lo, hi;\n\t"  \
	   ".reg .pred p1;\n\t" \
                                     \
           "mov.b64         a0, %4;\n\t" \
           "mov.b64         a1, %5;\n\t" \
           "mov.b64         a2, %6;\n\t" \
           "mov.b64         a3, %7;\n\t" \
                                        \
           "mov.b64         b0, %8;\n\t" \
           "mov.b64         b1, %9;\n\t" \
           "mov.b64         b2, %10;\n\t" \
           "mov.b64         b3, %11;\n\t" \
                                        \
           "mov.b64         n0, %12;\n\t" \
           "mov.b64         n1, %13;\n\t" \
           "mov.b64         n2, %14;\n\t" \
           "mov.b64         n3, %15;\n\t" \
                                       \
           "mov.b64         q, %16;\n\t"


#define ASM_MULG2_INIT \
      "{\n\t" \
           ".reg .u32 ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7;\n\t"  \
           ".reg .u32 ay0, ay1, ay2, ay3, ay4, ay5, ay6, ay7;\n\t"  \
           ".reg .u32 bx0, bx1, bx2, bx3, bx4, bx5, bx6, bx7;\n\t"  \
           ".reg .u32 by0, by1, by2, by3, by4, by5, by6, by7;\n\t"  \
           ".reg .u32 rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7;\n\t"  \
           ".reg .u32 ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7;\n\t"  \
           ".reg .u32 tmulx0, tmulx1, tmulx2, tmulx3, tmulx4, tmulx5, tmulx6, tmulx7;\n\t"  \
           ".reg .u32 tmuly0, tmuly1, tmuly2, tmuly3, tmuly4, tmuly5, tmuly6, tmuly7;\n\t"  \
           ".reg .u32 tmulz0, tmulz1, tmulz2, tmulz3, tmulz4, tmulz5, tmulz6, tmulz7;\n\t"  \
           ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"  \
           ".reg .u32 m, q, lo, hi;\n\t"  \
	   ".reg .pred p1;\n\t" \
                                     \
           "mov.u32         ax0, %16;\n\t" \
           "mov.u32         ax1, %17;\n\t" \
           "mov.u32         ax2, %18;\n\t" \
           "mov.u32         ax3, %19;\n\t" \
           "mov.u32         ax4, %20;\n\t" \
           "mov.u32         ax5, %21;\n\t" \
           "mov.u32         ax6, %22;\n\t" \
           "mov.u32         ax7, %23;\n\t" \
                                        \
           "mov.u32         ay0, %24;\n\t" \
           "mov.u32         ay1, %25;\n\t" \
           "mov.u32         ay2, %26;\n\t" \
           "mov.u32         ay3, %27;\n\t" \
           "mov.u32         ay4, %28;\n\t" \
           "mov.u32         ay5, %29;\n\t" \
           "mov.u32         ay6, %30;\n\t" \
           "mov.u32         ay7, %31;\n\t" \
                                        \
           "mov.u32         bx0, %32;\n\t" \
           "mov.u32         bx1, %33;\n\t" \
           "mov.u32         bx2, %34;\n\t" \
           "mov.u32         bx3, %35;\n\t" \
           "mov.u32         bx4, %36;\n\t" \
           "mov.u32         bx5, %37;\n\t" \
           "mov.u32         bx6, %38;\n\t" \
           "mov.u32         bx7, %39;\n\t" \
           "mov.u32         by0, %40;\n\t" \
           "mov.u32         by1, %41;\n\t" \
           "mov.u32         by2, %42;\n\t" \
           "mov.u32         by3, %43;\n\t" \
           "mov.u32         by4, %44;\n\t" \
           "mov.u32         by5, %45;\n\t" \
           "mov.u32         by6, %46;\n\t" \
           "mov.u32         by7, %47;\n\t" \
            \
           "mov.u32         n0, %48;\n\t" \
           "mov.u32         n1, %49;\n\t" \
           "mov.u32         n2, %50;\n\t" \
           "mov.u32         n3, %51;\n\t" \
           "mov.u32         n4, %52;\n\t" \
           "mov.u32         n5, %53;\n\t" \
           "mov.u32         n6, %54;\n\t" \
           "mov.u32         n7, %55;\n\t" \
           "mov.u32         q, %56;\n\t"

#define ASM_MULG2_INIT_64 \
      "{\n\t" \
           ".reg .u32 ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7;\n\t"  \
           ".reg .u32 ay0, ay1, ay2, ay3, ay4, ay5, ay6, ay7;\n\t"  \
           ".reg .u32 bx0, bx1, bx2, bx3, bx4, bx5, bx6, bx7;\n\t"  \
           ".reg .u32 by0, by1, by2, by3, by4, by5, by6, by7;\n\t"  \
           ".reg .u32 rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7;\n\t"  \
           ".reg .u32 ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7;\n\t"  \
           ".reg .u32 tmulx0, tmulx1, tmulx2, tmulx3, tmulx4, tmulx5, tmulx6, tmulx7;\n\t"  \
           ".reg .u32 tmuly0, tmuly1, tmuly2, tmuly3, tmuly4, tmuly5, tmuly6, tmuly7;\n\t"  \
           ".reg .u32 tmulz0, tmulz1, tmulz2, tmulz3, tmulz4, tmulz5, tmulz6, tmulz7;\n\t"  \
           ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"  \
           ".reg .u32 m, q, lo, hi;\n\t"  \
	   ".reg .pred p1;\n\t" \
                                     \
           "mov.b64         {ax0,ax1}, %8;\n\t" \
           "mov.b64         {ax2,ax3}, %9;\n\t" \
           "mov.b64         {ax4,ax5}, %10;\n\t" \
           "mov.b64         {ax6,ax7}, %11;\n\t" \
	            \
           "mov.b64         {ay0,ay1}, %12;\n\t" \
           "mov.b64         {ay2,ay3}, %13;\n\t" \
           "mov.b64         {ay4,ay5}, %14;\n\t" \
           "mov.b64         {ay6,ay7}, %15;\n\t" \
	            \
           "mov.b64         {bx0,bx1}, %16;\n\t" \
           "mov.b64         {bx2,bx3}, %17;\n\t" \
           "mov.b64         {bx4,bx5}, %18;\n\t" \
           "mov.b64         {bx6,bx7}, %19;\n\t" \
	            \
           "mov.b64         {by0,by1}, %20;\n\t" \
           "mov.b64         {by2,by3}, %21;\n\t" \
           "mov.b64         {by4,by5}, %22;\n\t" \
           "mov.b64         {by6,by7}, %23;\n\t" \
	            \
           "mov.b64         {n0,n1}, %24;\n\t" \
           "mov.b64         {n2,n3}, %25;\n\t" \
           "mov.b64         {n4,n5}, %26;\n\t" \
           "mov.b64         {n6,n7}, %27;\n\t" \
	            \
           "mov.u32         q, %28;\n\t"

#define ASM_MUL_START(out, in1, in2) \
     "mul.lo.u32 "#out"0, "#in1"0, "#in2"0;\n\t"           \
     "mul.lo.u32 "#out"1, "#in1"0, "#in2"1;\n\t"           \
     "mul.lo.u32 "#out"2, "#in1"0, "#in2"2;\n\t"           \
     "mul.lo.u32 "#out"3, "#in1"0, "#in2"3;\n\t"           \
     "mul.lo.u32 "#out"4, "#in1"0, "#in2"4;\n\t"           \
     "mul.lo.u32 "#out"5, "#in1"0, "#in2"5;\n\t"           \
     "mul.lo.u32 "#out"6, "#in1"0, "#in2"6;\n\t"           \
     "mul.lo.u32 "#out"7, "#in1"0, "#in2"7;\n\t"           \
     "mad.hi.cc.u32 "#out"1, "#in1"0, "#in2"0, "#out"1;\n\t"           \
     "madc.hi.cc.u32 "#out"2, "#in1"0, "#in2"1, "#out"2;\n\t"           \
     "madc.hi.cc.u32 "#out"3, "#in1"0, "#in2"2, "#out"3;\n\t"           \
     "madc.hi.cc.u32 "#out"4, "#in1"0, "#in2"3, "#out"4;\n\t"           \
     "madc.hi.cc.u32 "#out"5, "#in1"0, "#in2"4, "#out"5;\n\t"           \
     "madc.hi.cc.u32 "#out"6, "#in1"0, "#in2"5, "#out"6;\n\t"           \
     "madc.hi.cc.u32 "#out"7, "#in1"0, "#in2"6, "#out"7;\n\t"           \
     "madc.hi.cc.u32 lo, "#in1"0, "#in2"7, 0;\n\t"          

#define ASM_MONTSQ_START(out, in1, in2) \
     "mul.lo.u32 "#out"0, "#in1"0, "#in2"0;\n\t"           \
     "mul.lo.u32 "#out"1, "#in1"0, "#in2"1;\n\t"           \
     "mul.lo.u32 "#out"2, "#in1"0, "#in2"2;\n\t"           \
     "mul.lo.u32 "#out"3, "#in1"0, "#in2"3;\n\t"           \
     "mul.lo.u32 "#out"4, "#in1"0, "#in2"4;\n\t"           \
     "mul.lo.u32 "#out"5, "#in1"0, "#in2"5;\n\t"           \
     "mul.lo.u32 "#out"6, "#in1"0, "#in2"6;\n\t"           \
     "mul.lo.u32 "#out"7, "#in1"0, "#in2"7;\n\t"           \
     "mul.hi.u32 tth0, "#in1"0, "#in2"1;\n\t"           \
     "mul.hi.u32 tth1, "#in1"0, "#in2"2;\n\t"           \
     "mul.hi.u32 tth2, "#in1"0, "#in2"3;\n\t"           \
     "mul.hi.u32 tth3, "#in1"0, "#in2"4;\n\t"           \
     "mul.hi.u32 tth4, "#in1"0, "#in2"5;\n\t"           \
     "mul.hi.u32 tth5, "#in1"0, "#in2"6;\n\t"           \
     "mul.hi.u32 tth6, "#in1"0, "#in2"7;\n\t"           \
     "mov.u32    ttl0, "#out"1;\n\t" \
     "mov.u32    ttl1, "#out"2;\n\t" \
     "mov.u32    ttl2, "#out"3;\n\t" \
     "mov.u32    ttl3, "#out"4;\n\t" \
     "mov.u32    ttl4, "#out"5;\n\t" \
     "mov.u32    ttl5, "#out"6;\n\t" \
     "mov.u32    ttl6, "#out"7;\n\t" \
       \
     "mad.hi.cc.u32 "#out"1, "#in1"0, "#in2"0, "#out"1;\n\t"           \
     "addc.cc.u32 "#out"2, tth0, "#out"2;\n\t"           \
     "addc.cc.u32 "#out"3, tth1, "#out"3;\n\t"           \
     "addc.cc.u32 "#out"4, tth2, "#out"4;\n\t"           \
     "addc.cc.u32 "#out"5, tth3, "#out"5;\n\t"           \
     "addc.cc.u32 "#out"6, tth4, "#out"6;\n\t"           \
     "addc.cc.u32 "#out"7, tth5, "#out"7;\n\t"           \
     "addc.cc.u32 lo, tth6, 0;\n\t"          

#define ASM_MUL_START_64(out, in1, in2) \
     "mul.lo.u64 "#out"0, "#in1"0, "#in2"0;\n\t"           \
     "mul.lo.u64 "#out"1, "#in1"0, "#in2"1;\n\t"           \
     "mul.lo.u64 "#out"2, "#in1"0, "#in2"2;\n\t"           \
     "mul.lo.u64 "#out"3, "#in1"0, "#in2"3;\n\t"           \
     "mad.hi.cc.u64 "#out"1, "#in1"0, "#in2"0, "#out"1;\n\t"           \
     "madc.hi.cc.u64 "#out"2, "#in1"0, "#in2"1, "#out"2;\n\t"           \
     "madc.hi.cc.u64 "#out"3, "#in1"0, "#in2"2, "#out"3;\n\t"           \
     "madc.hi.cc.u64 lo, "#in1"0, "#in2"3, 0;\n\t"          

#define ASM_MUL_REDUCTION_BLOCK_LAST(out) \
       "mul.lo.u32   m, "#out"0, q;\n\t" \
       "mad.lo.cc.u32 "#out"0, m, n0, "#out"0;\n\t" \
       "madc.hi.cc.u32 "#out"1, m, n0, "#out"1;\n\t" \
       "madc.hi.cc.u32 "#out"2, m, n1, "#out"2;\n\t" \
       "madc.hi.cc.u32 "#out"3, m, n2, "#out"3;\n\t" \
       "madc.hi.cc.u32 "#out"4, m, n3, "#out"4;\n\t" \
       "madc.hi.cc.u32  "#out"5, m, n4, "#out"5;\n\t" \
       "madc.hi.cc.u32  "#out"6, m, n5, "#out"6;\n\t" \
       "madc.hi.cc.u32  "#out"7, m, n6, "#out"7;\n\t" \
       "madc.hi.cc.u32  lo, m, n7, lo;\n\t" \
       "addc.u32  hi, 0, 0;\n\t" \
       "mad.lo.cc.u32 "#out"0, m, n1, "#out"1;\n\t" \
       "madc.lo.cc.u32  "#out"1, m, n2, "#out"2;\n\t" \
       "madc.lo.cc.u32  "#out"2, m, n3, "#out"3;\n\t" \
       "madc.lo.cc.u32  "#out"3, m, n4, "#out"4;\n\t" \
       "madc.lo.cc.u32  "#out"4, m, n5, "#out"5;\n\t" \
       "madc.lo.cc.u32  "#out"5, m, n6, "#out"6;\n\t" \
       "madc.lo.cc.u32  "#out"6, m, n7, "#out"7;\n\t" \
       "addc.cc.u32  "#out"7, lo, 0;\n\t" \
       "addc.u32  lo, hi, 0;\n\t"

#define ASM_MUL_REDUCTION_BLOCK_LAST_64(out) \
       "mul.lo.u64   m, "#out"0, q;\n\t" \
       "mad.lo.cc.u64 "#out"0, m, n0, "#out"0;\n\t" \
       "madc.hi.cc.u64 "#out"1, m, n0, "#out"1;\n\t" \
       "madc.hi.cc.u64 "#out"2, m, n1, "#out"2;\n\t" \
       "madc.hi.cc.u64 "#out"3, m, n2, "#out"3;\n\t" \
       "madc.hi.cc.u64  lo, m, n3, lo;\n\t" \
       "addc.u64  hi, 0, 0;\n\t" \
       "mad.lo.cc.u64 "#out"0, m, n1, "#out"1;\n\t" \
       "madc.lo.cc.u64  "#out"1, m, n2, "#out"2;\n\t" \
       "madc.lo.cc.u64  "#out"2, m, n3, "#out"3;\n\t" \
       "addc.cc.u64  "#out"3, lo, 0;\n\t" \
       "addc.u64  lo, hi, 0;\n\t" 


#define ASM_MUL_BLOCK(out, in1, in2, idx) \
    "mad.lo.cc.u32 "#out"0, "#in1#idx", "#in2"0, "#out"0;\n\t" \
    "madc.lo.cc.u32 "#out"1, "#in1#idx", "#in2"1, "#out"1;\n\t" \
    "madc.lo.cc.u32 "#out"2, "#in1#idx", "#in2"2, "#out"2;\n\t" \
    "madc.lo.cc.u32 "#out"3, "#in1#idx", "#in2"3, "#out"3;\n\t" \
    "madc.lo.cc.u32 "#out"4, "#in1#idx", "#in2"4, "#out"4;\n\t" \
    "madc.lo.cc.u32 "#out"5, "#in1#idx", "#in2"5, "#out"5;\n\t" \
    "madc.lo.cc.u32 "#out"6, "#in1#idx", "#in2"6, "#out"6;\n\t" \
    "madc.lo.cc.u32 "#out"7, "#in1#idx", "#in2"7, "#out"7;\n\t" \
    "addc.u32 lo, lo, 0;\n\t" \
    "mad.hi.cc.u32 "#out"1, "#in1#idx", "#in2"0, "#out"1;\n\t" \
    "madc.hi.cc.u32 "#out"2, "#in1#idx", "#in2"1, "#out"2;\n\t" \
    "madc.hi.cc.u32 "#out"3, "#in1#idx", "#in2"2, "#out"3;\n\t" \
    "madc.hi.cc.u32 "#out"4, "#in1#idx", "#in2"3, "#out"4;\n\t" \
    "madc.hi.cc.u32 "#out"5, "#in1#idx", "#in2"4, "#out"5;\n\t" \
    "madc.hi.cc.u32 "#out"6, "#in1#idx", "#in2"5, "#out"6;\n\t" \
    "madc.hi.cc.u32 "#out"7, "#in1#idx", "#in2"6, "#out"7;\n\t" \
    "madc.hi.cc.u32 lo, "#in1#idx", "#in2"7, lo;\n\t" \
    "addc.u32 hi, 0, 0;\n\t"  

#if 0
#define ASM_MONTSQ_BLOCK2(out, in1, in2)  \
    "add.cc.u32 "#out"0, ttl1, "#out"0;\n\t" \
    "addc.cc.u32 "#out"1, ttl7, "#out"1;\n\t" \
    "madc.lo.cc.u32 "#out"2, "#in1#idx", "#in2"2, "#out"2;\n\t" \
    "addc.cc.u32 "#out"2, ttl0, "#out"2;\n\t" \
    "mul.lo.u32  ttl0, "#in1"2, "#in2"3;\n\t" \
    "mul.lo.u32  ttl1, "#in1"2, "#in2"4;\n\t" \
    "mul.lo.u32  ttl7, "#in1"2, "#in2"5;\n\t" \
    "mul.lo.u32  ttl12, "#in1"2, "#in2"6;\n\t" \
    "mul.hi.u32  tth12, "#in1"2, "#in2"6;\n\t" \
    "mul.lo.u32  ttl13, "#in1"2, "#in2"7;\n\t" \
    "mul.hi.u32  tth13, "#in1"2, "#in2"7;\n\t" \
    "addc.cc.u32 "#out"3, ttl0, "#out"3;\n\t" \
    "addc.cc.u32 "#out"4, ttl11, "#out"4;\n\t" \
    "addc.cc.u32 "#out"5, ttl12, "#out"5;\n\t" \
    "addc.cc.u32 "#out"6, ttl13, "#out"6;\n\t" \
    "addc.cc.u32 "#out"7, ttl14, "#out"7;\n\t" \
    "addc.u32 lo, lo, 0;\n\t" \
    "add.cc.u32 "#out"1, tth1, "#out"1;\n\t" \
    "addc.cc.u32 "#out"2, tth7, "#out"2;\n\t" \
    "madc.hi.cc.u32 "#out"3, "#in1#idx", "#in2"2, "#out"3;\n\t" \
    "madc.hi.cc.u32 "#out"4, "#in1#idx", "#in2"3, "#out"4;\n\t" \
    "madc.hi.cc.u32 "#out"5, "#in1#idx", "#in2"4, "#out"5;\n\t" \
    "madc.hi.cc.u32 "#out"6, "#in1#idx", "#in2"5, "#out"6;\n\t" \
    "madc.hi.cc.u32 "#out"7, "#in1#idx", "#in2"6, "#out"7;\n\t" \
    "madc.hi.cc.u32 lo, "#in1#idx", "#in2"7, lo;\n\t" \
    "addc.u32 hi, 0, 0;\n\t"  

#define ASM_MONTSQ_BLOCK1(out, in1, in2) \
    "add.cc.u32 "#out"0, ttl0, "#out"0;\n\t" \
    "mul.lo.u32  ttl0, "#in1"1, "#in2"2;\n\t" \
    "mul.lo.u32  ttl7, "#in1"1, "#in2"3;\n\t" \
    "mul.hi.u32  tth7, "#in1"1, "#in2"3;\n\t" \
    "mul.lo.u32  ttl8, "#in1"1, "#in2"4;\n\t" \
    "mul.hi.u32  tth8, "#in1"1, "#in2"4;\n\t" \
    "mul.lo.u32  ttl9, "#in1"1, "#in2"5;\n\t" \
    "mul.hi.u32  tth9, "#in1"1, "#in2"5;\n\t" \
    "mul.lo.u32  ttl10, "#in1"1, "#in2"6;\n\t" \
    "mul.lhiu32  tth10, "#in1"1, "#in2"6;\n\t" \
    "mul.lo.u32  ttl11, "#in1"1, "#in2"7;\n\t" \
    "mul.hi.u32  tth11, "#in1"1, "#in2"7;\n\t" \
    "madc.lo.cc.u32 "#out"1, "#in1"1, "#in2"1, "#out"1;\n\t" \
    "addc.cc.u32 "#out"2, ttl0, "#out"2;\n\t" \
    "addc.cc.u32 "#out"3, ttl7, "#out"3;\n\t" \
    "addc.cc.u32 "#out"4, ttl8, "#out"4;\n\t" \
    "addc.cc.u32 "#out"5, ttl9, "#out"5;\n\t" \
    "addc.cc.u32 "#out"6, ttl10, "#out"6;\n\t" \
    "addc.cc.u32 "#out"7, ttl11, "#out"7;\n\t" \
    "addc.u32 lo, lo, 0;\n\t" \
    "add.cc.u32 "#out"1, tth0, "#out"1;\n\t" \
    "mul.hi.u32  tth0, "#in1"1, "#in2"2;\n\t" \
    "madc.hi.cc.u32 "#out"2, "#in1#idx", "#in2"1, "#out"2;\n\t" \
    "addc.u32 "#out"3, tth0, "#out"3;\n\t" \
    "addc.u32 "#out"4, tth7, "#out"4;\n\t" \
    "addc.u32 "#out"5, tth8, "#out"5;\n\t" \
    "addc.u32 "#out"6, tth9, "#out"6;\n\t" \
    "addc.u32 "#out"7, tth10, "#out"7;\n\t" \
    "addc.u32 lo, tth11, lo;\n\t" \
    "addc.u32 hi, 0, 0;\n\t" 
#endif

#define ASM_MUL_BLOCK_64(out, in1, in2, idx) \
    "mad.lo.cc.u64 "#out"0, "#in1#idx", "#in2"0, "#out"0;\n\t" \
    "madc.lo.cc.u64 "#out"1, "#in1#idx", "#in2"1, "#out"1;\n\t" \
    "madc.lo.cc.u64 "#out"2, "#in1#idx", "#in2"2, "#out"2;\n\t" \
    "madc.lo.cc.u64 "#out"3, "#in1#idx", "#in2"3, "#out"3;\n\t" \
    "addc.u64 lo, lo, 0;\n\t" \
    "mad.hi.cc.u64 "#out"1, "#in1#idx", "#in2"0, "#out"1;\n\t" \
    "madc.hi.cc.u64 "#out"2, "#in1#idx", "#in2"1, "#out"2;\n\t" \
    "madc.hi.cc.u64 "#out"3, "#in1#idx", "#in2"2, "#out"3;\n\t" \
    "madc.hi.cc.u64 lo, "#in1#idx", "#in2"3, lo;\n\t" \
    "addc.u64 hi, 0, 0;\n\t"  


#define ASM_MUL_END \
      "END_MUL:\n\t" 

#define ASM_MUL_PACK \
      "mov.u32        %0, r0;\n\t"   \
      "mov.u32        %1, r1;\n\t"   \
      "mov.u32        %2, r2;\n\t"   \
      "mov.u32        %3, r3;\n\t"   \
      "mov.u32        %4, r4;\n\t"   \
      "mov.u32        %5, r5;\n\t"   \
      "mov.u32        %6, r6;\n\t"   \
      "mov.u32        %7, r7;\n\t"   \
      "}\n\t" \
       : "=r"(U[0]), "=r"(U[1]), "=r"(U[2]), "=r"(U[3]), \
         "=r"(U[4]), "=r"(U[5]), "=r"(U[6]), "=r"(U[7]) \
       : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),   \
         "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]),   \
         "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), \
         "r"(B[4]), "r"(B[5]), "r"(B[6]), "r"(B[7]), \
         "r"(P_u256[0]), "r"(P_u256[1]), "r"(P_u256[2]), "r"(P_u256[3]), \
         "r"(P_u256[4]), "r"(P_u256[5]), "r"(P_u256[6]), "r"(P_u256[7]), \
         "r"(PN_u256[0])

#define ASM_MUL_PACK_64 \
      "mov.b64        %0, {r0,r1};\n\t"   \
      "mov.b64        %1, {r2,r3};\n\t"   \
      "mov.b64        %2, {r4,r5};\n\t"   \
      "mov.b64        %3, {r6,r7};\n\t"   \
      "}\n\t" \
       : "=l"(dU[0]), "=l"(dU[1]), "=l"(dU[2]), "=l"(dU[3]) \
       : "l"(dA[0]), "l"(dA[1]), "l"(dA[2]), "l"(dA[3]),   \
         "l"(dB[0]), "l"(dB[1]), "l"(dB[2]), "l"(dB[3]), \
         "l"(dP_u256[0]), "l"(dP_u256[1]), "l"(dP_u256[2]), "l"(dP_u256[3]), \
         "r"(PN_u256[0])

#define ASM_MONTSQ_PACK_64 \
      "mov.b64        %0, {r0,r1};\n\t"   \
      "mov.b64        %1, {r2,r3};\n\t"   \
      "mov.b64        %2, {r4,r5};\n\t"   \
      "mov.b64        %3, {r6,r7};\n\t"   \
      "}\n\t" \
       : "=l"(dU[0]), "=l"(dU[1]), "=l"(dU[2]), "=l"(dU[3]) \
       : "l"(dA[0]), "l"(dA[1]), "l"(dA[2]), "l"(dA[3]),   \
         "l"(dP_u256[0]), "l"(dP_u256[1]), "l"(dP_u256[2]), "l"(dP_u256[3]), \
         "r"(PN_u256[0])

#define ASM_MUL_PACK_64_2 \
      "mov.b64        %0, r0;\n\t"   \
      "mov.b64        %1, r1;\n\t"   \
      "mov.b64        %2, r2;\n\t"   \
      "mov.b64        %3, r3;\n\t"   \
      "}\n\t" \
       : "=l"(dU[0]), "=l"(dU[1]), "=l"(dU[2]), "=l"(dU[3]) \
       : "l"(dA[0]), "l"(dA[1]), "l"(dA[2]), "l"(dA[3]),   \
         "l"(dB[0]), "l"(dB[1]), "l"(dB[2]), "l"(dB[3]), \
         "l"(dP_u256[0]), "l"(dP_u256[1]), "l"(dP_u256[2]), "l"(dP_u256[3]), \
         "l"(dPN_u256[0])


#define ASM_MULG2_PACK \
      "mov.u32        %0, rx0;\n\t"   \
      "mov.u32        %1, rx1;\n\t"   \
      "mov.u32        %2, rx2;\n\t"   \
      "mov.u32        %3, rx3;\n\t"   \
      "mov.u32        %4, rx4;\n\t"   \
      "mov.u32        %5, rx5;\n\t"   \
      "mov.u32        %6, rx6;\n\t"   \
      "mov.u32        %7, rx7;\n\t"   \
      "mov.u32        %8, ry0;\n\t"   \
      "mov.u32        %9, ry1;\n\t"   \
      "mov.u32        %10, ry2;\n\t"   \
      "mov.u32        %11, ry3;\n\t"   \
      "mov.u32        %12, ry4;\n\t"   \
      "mov.u32        %13, ry5;\n\t"   \
      "mov.u32        %14, ry6;\n\t"   \
      "mov.u32        %15, ry7;\n\t"   \
      "}\n\t" \
       : "=r"(U[0]), "=r"(U[1]), "=r"(U[2]), "=r"(U[3]), \
         "=r"(U[4]), "=r"(U[5]), "=r"(U[6]), "=r"(U[7]), \
         "=r"(U[8]), "=r"(U[9]), "=r"(U[10]), "=r"(U[11]), \
         "=r"(U[12]), "=r"(U[13]), "=r"(U[14]), "=r"(U[15]) \
       : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),   \
         "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]),   \
         "r"(A[8]), "r"(A[9]), "r"(A[10]), "r"(A[11]),   \
         "r"(A[12]), "r"(A[13]), "r"(A[14]), "r"(A[15]),   \
         "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), \
         "r"(B[4]), "r"(B[5]), "r"(B[6]), "r"(B[7]), \
         "r"(B[8]), "r"(B[9]), "r"(B[10]), "r"(B[11]), \
         "r"(B[12]), "r"(B[13]), "r"(B[14]), "r"(B[15]), \
         "r"(P_u256[0]), "r"(P_u256[1]), "r"(P_u256[2]), "r"(P_u256[3]), \
         "r"(P_u256[4]), "r"(P_u256[5]), "r"(P_u256[6]), "r"(P_u256[7]), \
         "r"(PN_u256[0])

#define ASM_MULG2_PACK_64 \
      "mov.b64        %0, {rx0, rx1};\n\t"   \
      "mov.b64        %1, {rx2, rx3};\n\t"   \
      "mov.b64        %2, {rx4, rx5};\n\t"   \
      "mov.b64        %3, {rx6, rx7};\n\t"   \
              \
      "mov.b64        %4, {ry0, ry1};\n\t"   \
      "mov.b64        %5, {ry2, ry3};\n\t"   \
      "mov.b64        %6, {ry4, ry5};\n\t"   \
      "mov.b64        %7, {ry6, ry7};\n\t"   \
      "}\n\t" \
       : "=l"(dU[0]), "=l"(dU[1]), "=l"(dU[2]), "=l"(dU[3]), \
         "=l"(dU[4]), "=l"(dU[5]), "=l"(dU[6]), "=l"(dU[7]) \
       : "l"(dA[0]), "l"(dA[1]), "l"(dA[2]), "l"(dA[3]),   \
         "l"(dA[4]), "l"(dA[5]), "l"(dA[6]), "l"(dA[7]),   \
         "l"(dB[0]), "l"(dB[1]), "l"(dB[2]), "l"(dB[3]), \
         "l"(dB[4]), "l"(dB[5]), "l"(dB[6]), "l"(dB[7]), \
         "l"(dP_u256[0]), "l"(dP_u256[1]), "l"(dP_u256[2]), "l"(dP_u256[3]), \
         "r"(PN_u256[0])

#define ASM_SUBU256(out, in1, in2) \
      "sub.cc.u32        "#out"0, "#in1"0, "#in2"0;\n\t"      \
      "subc.cc.u32       "#out"1, "#in1"1, "#in2"1;\n\t"     \
      "subc.cc.u32       "#out"2, "#in1"2, "#in2"2;\n\t"    \
      "subc.cc.u32       "#out"3, "#in1"3, "#in2"3;\n\t"    \
      "subc.cc.u32       "#out"4, "#in1"4, "#in2"4;\n\t"    \
      "subc.cc.u32       "#out"5, "#in1"5, "#in2"5;\n\t"    \
      "subc.cc.u32       "#out"6, "#in1"6, "#in2"6;\n\t"    \
      "subc.u32          "#out"7, "#in1"7, "#in2"7;\n\t"    

#define ASM_COND_SUBU256(out, in1, in2) \
      "@p1 sub.cc.u32        "#out"0, "#in1"0, "#in2"0;\n\t"      \
      "@p1 subc.cc.u32       "#out"1, "#in1"1, "#in2"1;\n\t"     \
      "@p1 subc.cc.u32       "#out"2, "#in1"2, "#in2"2;\n\t"    \
      "@p1 subc.cc.u32       "#out"3, "#in1"3, "#in2"3;\n\t"    \
      "@p1 subc.cc.u32       "#out"4, "#in1"4, "#in2"4;\n\t"    \
      "@p1 subc.cc.u32       "#out"5, "#in1"5, "#in2"5;\n\t"    \
      "@p1 subc.cc.u32       "#out"6, "#in1"6, "#in2"6;\n\t"    \
      "@p1 subc.u32          "#out"7, "#in1"7, "#in2"7;\n\t"    

#define ASM_MODU256(in) \
	"setp.ge.u32           p1, "#in"7, n7;\n\t"  \
        "@p1 sub.cc.u32        "#in"0, "#in"0, n0;\n\t"      \
        "@p1 subc.cc.u32       "#in"1, "#in"1, n1;\n\t"     \
        "@p1 subc.cc.u32       "#in"2, "#in"2, n2;\n\t"    \
        "@p1 subc.cc.u32       "#in"3, "#in"3, n3;\n\t"    \
        "@p1 subc.cc.u32       "#in"4, "#in"4, n4;\n\t"    \
        "@p1 subc.cc.u32       "#in"5, "#in"5, n5;\n\t"    \
        "@p1 subc.cc.u32       "#in"6, "#in"6, n6;\n\t"    \
        "@p1 subc.cc.u32       "#in"7, "#in"7, n7;\n\t"    \
        "@p1 bfe.u32            m, "#in"7, 31, 1;\n\t"   \
        "@p1 setp.eq.u32        p1, m,  1;\n\t"      \
        "@p1 add.cc.u32        "#in"0, "#in"0, n0;\n\t"      \
        "@p1 addc.cc.u32       "#in"1, "#in"1, n1;\n\t"     \
        "@p1 addc.cc.u32       "#in"2, "#in"2, n2;\n\t"    \
        "@p1 addc.cc.u32       "#in"3, "#in"3, n3;\n\t"    \
        "@p1 addc.cc.u32       "#in"4, "#in"4, n4;\n\t"    \
        "@p1 addc.cc.u32       "#in"5, "#in"5, n5;\n\t"    \
        "@p1 addc.cc.u32       "#in"6, "#in"6, n6;\n\t"    \
        "@p1 addc.u32          "#in"7, "#in"7, n7;\n\t"    

#define ASM_MODU256_64(in) \
	"setp.ge.u64           p1, "#in"3, n3;\n\t"  \
        "@p1 sub.cc.u64        "#in"0, "#in"0, n0;\n\t"      \
        "@p1 subc.cc.u64       "#in"1, "#in"1, n1;\n\t"     \
        "@p1 subc.cc.u64       "#in"2, "#in"2, n2;\n\t"    \
        "@p1 subc.cc.u64       "#in"3, "#in"3, n3;\n\t"    \
        "@p1 bfe.u64            m, "#in"3, 63, 1;\n\t"   \
        "@p1 setp.eq.u64        p1, m,  1;\n\t"      \
        "@p1 add.cc.u64        "#in"0, "#in"0, n0;\n\t"      \
        "@p1 addc.cc.u64       "#in"1, "#in"1, n1;\n\t"     \
        "@p1 addc.cc.u64       "#in"2, "#in"2, n2;\n\t"    \
        "@p1 add.cc.u64        "#in"3, "#in"3, n3;\n\t"    \

#define ASM_SUBMU256(out, in1, in2) \
	ASM_SUBU256(out, in1, in2) \
	"setp.gt.u32     p1, "#out"7, n7;\n\t"  \
        ASM_COND_ADDU256(out, out, n) 

#define ASM_ADDMU256(out, in1, in2) \
	ASM_ADDU256(out, in1, in2) \
	ASM_SUBU256(out, out, n) \
	"setp.gt.u32     p1, "#out"7, n7;\n\t"  \
        ASM_COND_ADDU256(out, out, n) 

#define ASM_COND_ADDU256(out,in1, in2) \
      "@p1 add.cc.u32        "#out"0, "#in1"0, "#in2"0;\n\t"      \
      "@p1 addc.cc.u32       "#out"1, "#in1"1, "#in2"1;\n\t"     \
      "@p1 addc.cc.u32       "#out"2, "#in1"2, "#in2"2;\n\t"    \
      "@p1 addc.cc.u32       "#out"3, "#in1"3, "#in2"3;\n\t"    \
      "@p1 addc.cc.u32       "#out"4, "#in1"4, "#in2"4;\n\t"    \
      "@p1 addc.cc.u32       "#out"5, "#in1"5, "#in2"5;\n\t"    \
      "@p1 addc.cc.u32       "#out"6, "#in1"6, "#in2"6;\n\t"    \
      "@p1 addc.u32          "#out"7, "#in1"7, "#in2"7;\n\t"    

#define ASM_ADDU256(out, in1, in2) \
      "add.cc.u32        "#out"0, "#in1"0, "#in2"0;\n\t"      \
      "addc.cc.u32       "#out"1, "#in1"1, "#in2"1;\n\t"     \
      "addc.cc.u32       "#out"2, "#in1"2, "#in2"2;\n\t"    \
      "addc.cc.u32       "#out"3, "#in1"3, "#in2"3;\n\t"    \
      "addc.cc.u32       "#out"4, "#in1"4, "#in2"4;\n\t"    \
      "addc.cc.u32       "#out"5, "#in1"5, "#in2"5;\n\t"    \
      "addc.cc.u32       "#out"6, "#in1"6, "#in2"6;\n\t"    \
      "addc.u32          "#out"7, "#in1"7, "#in2"7;\n\t"    


#define ASM_ADDECJAC_INIT \
      "{\n\t" \
           ".reg .u32 ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7;\n\t"  \
           ".reg .u32 ay0, ay1, ay2, ay3, ay4, ay5, ay6, ay7;\n\t"  \
           ".reg .u32 az0, az1, az2, az3, az4, az5, az6, az7;\n\t"  \
           ".reg .u32 bx0, bx1, bx2, bx3, bx4, bx5, bx6, bx7;\n\t"  \
           ".reg .u32 by0, by1, by2, by3, by4, by5, by6, by7;\n\t"  \
           ".reg .u32 bz0, bz1, bz2, bz3, bz4, bz5, bz6, bz7;\n\t"  \
           ".reg .u32 rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7;\n\t"  \
           ".reg .u32 ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7;\n\t"  \
           ".reg .u32 rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7;\n\t"  \
           ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"  \
           ".reg .u32 m, q, lo, hi;\n\t"  \
	   ".reg .pred p1, p2;\n\t" \
                                    \
	   "mov.u32    ax0, %24;\n\t"   \
	   "mov.u32    ax1, %25;\n\t"   \
	   "mov.u32    ax2, %26;\n\t"   \
	   "mov.u32    ax3, %27;\n\t"   \
	   "mov.u32    ax4, %28;\n\t"   \
	   "mov.u32    ax5, %29;\n\t"   \
	   "mov.u32    ax6, %30;\n\t"   \
	   "mov.u32    ax7, %31;\n\t"   \
	   "mov.u32    ay0, %32;\n\t"   \
	   "mov.u32    ay1, %33;\n\t"   \
	   "mov.u32    ay2, %34;\n\t"   \
	   "mov.u32    ay3, %35;\n\t"   \
	   "mov.u32    ay4, %36;\n\t"   \
	   "mov.u32    ay5, %37;\n\t"   \
	   "mov.u32    ay6, %38;\n\t"   \
	   "mov.u32    ay7, %39;\n\t"   \
	   "mov.u32    az0, %40;\n\t"   \
	   "mov.u32    az1, %41;\n\t"   \
	   "mov.u32    az2, %42;\n\t"   \
	   "mov.u32    az3, %43;\n\t"   \
	   "mov.u32    az4, %44;\n\t"   \
	   "mov.u32    az5, %45;\n\t"   \
	   "mov.u32    az6, %46;\n\t"   \
	   "mov.u32    az7, %47;\n\t"   \
	                               \
	   "mov.u32    bx0, %48;\n\t"   \
	   "mov.u32    bx1, %49;\n\t"   \
	   "mov.u32    bx2, %50;\n\t"   \
	   "mov.u32    bx3, %51;\n\t"   \
	   "mov.u32    bx4, %52;\n\t"   \
	   "mov.u32    bx5, %53;\n\t"   \
	   "mov.u32    bx6, %54;\n\t"   \
	   "mov.u32    bx7, %55;\n\t"   \
	   "mov.u32    by0, %56;\n\t"   \
	   "mov.u32    by1, %57;\n\t"   \
	   "mov.u32    by2, %58;\n\t"   \
	   "mov.u32    by3, %59;\n\t"   \
	   "mov.u32    by4, %60;\n\t"   \
	   "mov.u32    by5, %61;\n\t"   \
	   "mov.u32    by6, %62;\n\t"   \
	   "mov.u32    by7, %63;\n\t"   \
	   "mov.u32    bz0, %64;\n\t"   \
	   "mov.u32    bz1, %65;\n\t"   \
	   "mov.u32    bz2, %66;\n\t"   \
	   "mov.u32    bz3, %67;\n\t"   \
	   "mov.u32    bz4, %68;\n\t"   \
	   "mov.u32    bz5, %69;\n\t"   \
	   "mov.u32    bz6, %70;\n\t"   \
	   "mov.u32    bz7, %71;\n\t"   \
	                                \
           "mov.u32         n0, %72;\n\t" \
           "mov.u32         n1, %73;\n\t" \
           "mov.u32         n2, %74;\n\t" \
           "mov.u32         n3, %75;\n\t" \
           "mov.u32         n4, %76;\n\t" \
           "mov.u32         n5, %77;\n\t" \
           "mov.u32         n6, %78;\n\t" \
           "mov.u32         n7, %79;\n\t" \
           "mov.u32         q, %80;\n\t"

#define ASM_ECJACADD_PACK \
      "ECADDJAC_L0:\n\t" \
      "}\n\t" \
       : "=r"(xr[0]), "=r"(xr[1]), "=r"(xr[2]), "=r"(xr[3]), "=r"(xr[4]), "=r"(xr[5]), "=r"(xr[6]), "=r"(xr[7]), \
         "=r"(xr[8]), "=r"(xr[9]), "=r"(xr[10]), "=r"(xr[11]), "=r"(xr[12]), "=r"(xr[13]), "=r"(xr[14]), "=r"(xr[15]), \
         "=r"(xr[16]), "=r"(xr[17]), "=r"(xr[18]), "=r"(xr[19]), "=r"(xr[20]), "=r"(xr[21]), "=r"(xr[22]), "=r"(xr[23]) \
       : "r"(x1[0]), "r"(x1[1]), "r"(x1[2]), "r"(x1[3]), "r"(x1[4]), "r"(x1[5]), "r"(x1[6]), "r"(x1[7]),   \
         "r"(x1[8]), "r"(x1[9]), "r"(x1[10]), "r"(x1[11]), "r"(x1[12]), "r"(x1[13]), "r"(x1[14]), "r"(x1[15]),   \
         "r"(x1[16]), "r"(x1[17]), "r"(x1[18]), "r"(x1[19]), "r"(x1[20]), "r"(x1[21]), "r"(x1[22]), "r"(x1[23]),   \
         "r"(x2[0]), "r"(x2[1]), "r"(x2[2]), "r"(x2[3]), "r"(x2[4]), "r"(x2[5]), "r"(x2[6]), "r"(x2[7]),   \
         "r"(x2[8]), "r"(x2[9]), "r"(x2[10]), "r"(x2[11]), "r"(x2[12]), "r"(x2[13]), "r"(x2[14]), "r"(x2[15]),   \
         "r"(x2[16]), "r"(x2[17]), "r"(x2[18]), "r"(x2[19]), "r"(x2[20]), "r"(x2[21]), "r"(x2[22]), "r"(x2[23]),   \
         "r"(P_u256[0]), "r"(P_u256[1]), "r"(P_u256[2]), "r"(P_u256[3]), \
         "r"(P_u256[4]), "r"(P_u256[5]), "r"(P_u256[6]), "r"(P_u256[7]), \
         "r"(PN_u256[0]), \
	 "r"(_1[0]), "r"(_1[1]), "r"(_1[2]), "r"(_1[3]), "r"(_1[4]), "r"(_1[5]), "r"(_1[6]), "r"(_1[7])

// available registers here are rx, ry, rz, ax, ay.
// restrictions : mul cannot have an input = output register
// used reg : rz(Z3), ay(R), by(U1 * H^2), rx(X3), ax(S1 * H^3)
// free : bx, ry, az, bz
// by = by - rx
// ry = ay * by
// ry = ry - ax

#define ASM_ECCHECK0(p,in1, in2) \
	ASM_EQ0U256(p, in1) \
	ASM_COND_ECMOV(p, in2 ) \

#define ASM_COND_ECMOV(p, in) \
	"@"#p" mov.u32  %0, "#in"x0;\n\t" \
	"@"#p" mov.u32  %1, "#in"x1;\n\t" \
	"@"#p" mov.u32  %2, "#in"x2;\n\t" \
	"@"#p" mov.u32  %3, "#in"x3;\n\t" \
	"@"#p" mov.u32  %4, "#in"x4;\n\t" \
	"@"#p" mov.u32  %5, "#in"x5;\n\t" \
	"@"#p" mov.u32  %6, "#in"x6;\n\t" \
	"@"#p" mov.u32  %7, "#in"x7;\n\t" \
	"@"#p" mov.u32  %8, "#in"y0;\n\t" \
	"@"#p" mov.u32  %9, "#in"y1;\n\t" \
	"@"#p" mov.u32  %10, "#in"y2;\n\t" \
	"@"#p" mov.u32  %11, "#in"y3;\n\t" \
	"@"#p" mov.u32  %12, "#in"y4;\n\t" \
	"@"#p" mov.u32  %13, "#in"y5;\n\t" \
	"@"#p" mov.u32  %14, "#in"y6;\n\t" \
	"@"#p" mov.u32  %15, "#in"y7;\n\t" \
	"@"#p" mov.u32  %16, "#in"z0;\n\t" \
	"@"#p" mov.u32  %17, "#in"z1;\n\t" \
	"@"#p" mov.u32  %18, "#in"z2;\n\t" \
	"@"#p" mov.u32  %19, "#in"z3;\n\t" \
	"@"#p" mov.u32  %20, "#in"z4;\n\t" \
	"@"#p" mov.u32  %21, "#in"z5;\n\t" \
	"@"#p" mov.u32  %22, "#in"z6;\n\t" \
	"@"#p" mov.u32  %23, "#in"z7;\n\t" \




#define ASM_ECJACADD \
	ASM_ECCHECK0(p1,a, b) \
	"@p1 bra  ECADDJAC_L0;\n\t"  \
	ASM_ECCHECK0(p1,b, a) \
	"@p1 bra  ECADDJAC_L0;\n\t"  \
	ASM_MONTMULU256(rx, bz, bz) \
	ASM_MONTMULU256(rz, ax, rx) \
        ASM_MODU256(rz) \
	ASM_MONTMULU256(ax, rx, bz) \
	ASM_MONTMULU256(rx, ay, ax) \
        ASM_MODU256(rx) \
	ASM_MONTMULU256(ry, az, az) \
	ASM_MONTMULU256(ax, bx, ry) \
        ASM_MODU256(ax) \
	ASM_MONTMULU256(ay, ry, az) \
	ASM_MONTMULU256(ry, by, ay) \
        ASM_MODU256(ry) \
	                             \
	ASM_CHECK_DOUBLE(rz, ax, rx, ry) \
"CONT_ECADDJAC_L0:\n\t"   \
	ASM_SUBMU256(ax, ax, rz) \
	ASM_SUBMU256(ay, ry, rx) \
	                          \
	ASM_MONTMULU256(ry, az, bz) \
	ASM_MONTMULU256(az, ax, ax) \
	ASM_MONTMULU256(bx, az, ax) \
        ASM_MODU256(bx) \
	ASM_MONTMULU256(bz, az, rx) \
	ASM_MONTMULU256(by, ax, bz) \
        ASM_MODU256(by) \
	ASM_MONTMULU256(bz, rz, az) \
        ASM_MODU256(bz) \
	ASM_MONTMULU256(rz, ry, ax) \
        ASM_MODU256(rz) \
	ASM_MONTMULU256(ax, ay, ay) \
        ASM_MODU256(ax) \
	ASM_SUBMU256(rx, ax, bx) \
	ASM_SUBMU256(rx, rx, bz) \
	ASM_SUBMU256(rx, rx, bz) \
	ASM_SUBMU256(ax, bz, rx) \
	ASM_MONTMULU256(ry, ax, ay) \
        ASM_MODU256(ry) \
	ASM_SUBMU256(ry, ry, by) \
	"bra ECADDJAC_FINSH;\n\t" \
        ASM_ECJACDOUBLE 


// in : bx, by, bz, n, q
// out : rx, ry, rz
// free : ax, ay, az 
#define ASM_ECJACDOUBLE \
	"ECDOUBLEJAC_L0:\n\t"   \
	ASM_MONTMULU256(ax, by, by) \
	ASM_MONTMULU256(ay, ax, ax) \
        ASM_MODU256(ay) \
	ASM_MONTMULU256(az, by, bz) \
        ASM_MODU256(az) \
	ASM_ADDMU256(rz, az, az) \
	ASM_MONTMULU256(az, ax, bx) \
        ASM_MODU256(az) \
	ASM_ADDMU256(az, az, az) \
	ASM_ADDMU256(az, az, az) \
	ASM_MONTMULU256(by, bx, bx) \
        ASM_MODU256(by) \
	ASM_ADDMU256(bz, by, by) \
	ASM_ADDMU256(by, by, bz) \
	ASM_MONTMULU256(bz, by, by) \
        ASM_MODU256(bz) \
	ASM_SUBMU256(rx, bz, az) \
	ASM_SUBMU256(rx, rx, az) \
	ASM_SUBMU256(ax, az, rx )  \
	ASM_MONTMULU256(ry, by, ax) \
        ASM_MODU256(ry) \
	ASM_ADDMU256(ay, ay, ay) \
	ASM_ADDMU256(ay, ay, ay) \
	ASM_ADDMU256(ay, ay, ay) \
	ASM_SUBMU256(ry, ry, ay )  

// in1 = U1, in2 = U2, in3 = S1,  in4 = S2
#define	ASM_CHECK_DOUBLE(in1, in2, in3, in4)  \
        ASM_EQU256(p1, in1, in2) \
        ASM_EQU256(p2, in3, in4) \
	"@p1 bra  CHECK_DOUBLE_P2;\n\t" \
	"bra CONT_ECADDJAC_L0;\n\t"  \
"CHECK_DOUBLE_P2:\n\t" \
	"@p2 bra ECDOUBLEJAC_L0;\n\t"   \
	"mov.u32  rx0, 0;\n\t" \
	"mov.u32  rx1, 0;\n\t" \
	"mov.u32  rx2, 0;\n\t" \
	"mov.u32  rx3, 0;\n\t" \
	"mov.u32  rx4, 0;\n\t" \
	"mov.u32  rx5, 0;\n\t" \
	"mov.u32  rx6, 0;\n\t" \
	"mov.u32  rx7, 0;\n\t" \
	"mov.u32  ry0, %81 ;\n\t" \
	"mov.u32  ry1, %82;\n\t" \
	"mov.u32  ry2, %83;\n\t" \
	"mov.u32  ry3, %84;\n\t" \
	"mov.u32  ry4, %85;\n\t" \
	"mov.u32  ry5, %86;\n\t" \
	"mov.u32  ry6, %87;\n\t" \
	"mov.u32  ry7, %88;\n\t" \
	"mov.u32  rz0, 0;\n\t" \
	"mov.u32  rz1, 0;\n\t" \
	"mov.u32  rz2, 0;\n\t" \
	"mov.u32  rz3, 0;\n\t" \
	"mov.u32  rz4, 0;\n\t" \
	"mov.u32  rz5, 0;\n\t" \
	"mov.u32  rz6, 0;\n\t" \
	"mov.u32  rz7, 0;\n\t" \
	"bra  ECADDJAC_L0;\n\t" 

#define ASM_EQ0U256(p, in) \
	"        setp.eq.u32     "#p", "#in"z0, 0;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in"z1, 0;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in"z2, 0;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in"z3, 0;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in"z4, 0;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in"z5, 0;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in"z6, 0;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in"z7, 0;\n\t" 

#define ASM_EQU256(p, in1, in2) \
	"        setp.eq.u32     "#p", "#in1"0, "#in2"0;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in1"1, "#in2"1;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in1"2, "#in2"2;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in1"3, "#in2"3;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in1"4, "#in2"4;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in1"5, "#in2"5;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in1"6, "#in2"6;\n\t" \
	"@"#p"   setp.eq.u32     "#p", "#in1"7, "#in2"7;\n\t" 

#define ASM_ECFINSH \
  "ECADDJAC_FINSH:\n\t" \
	    "mov.u32 %0, rx0;\n\t" \
	    "mov.u32 %1, rx1;\n\t" \
	    "mov.u32 %2, rx2;\n\t" \
	    "mov.u32 %3, rx3;\n\t" \
	    "mov.u32 %4, rx4;\n\t" \
	    "mov.u32 %5, rx5;\n\t" \
	    "mov.u32 %6, rx6;\n\t" \
	    "mov.u32 %7, rx7;\n\t" \
	    "mov.u32 %8, ry0;\n\t" \
	    "mov.u32 %9, ry1;\n\t" \
	    "mov.u32 %10, ry2;\n\t" \
	    "mov.u32 %11, ry3;\n\t" \
	    "mov.u32 %12, ry4;\n\t" \
	    "mov.u32 %13, ry5;\n\t" \
	    "mov.u32 %14, ry6;\n\t" \
	    "mov.u32 %15, ry7;\n\t" \
	    "mov.u32 %16, rz0;\n\t" \
	    "mov.u32 %17, rz1;\n\t" \
	    "mov.u32 %18, rz2;\n\t" \
	    "mov.u32 %19, rz3;\n\t" \
	    "mov.u32 %20, rz4;\n\t" \
	    "mov.u32 %21, rz5;\n\t" \
	    "mov.u32 %22, rz6;\n\t" \
	    "mov.u32 %23, rz7;\n\t"

#if 0
#define ASM_DOUBLE_ADDECJAC_INIT \
      "{\n\t" \
           ".reg .u32 ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7;\n\t"  \
           ".reg .u32 ay0, ay1, ay2, ay3, ay4, ay5, ay6, ay7;\n\t"  \
           ".reg .u32 az0, az1, az2, az3, az4, az5, az6, az7;\n\t"  \
           ".reg .u32 bx0, bx1, bx2, bx3, bx4, bx5, bx6, bx7;\n\t"  \
           ".reg .u32 by0, by1, by2, by3, by4, by5, by6, by7;\n\t"  \
           ".reg .u32 bz0, bz1, bz2, bz3, bz4, bz5, bz6, bz7;\n\t"  \
           ".reg .u32 rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7;\n\t"  \
           ".reg .u32 ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7;\n\t"  \
           ".reg .u32 rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7;\n\t"  \
           ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"  \
           ".reg .u32 m, q, lo, hi, msb, t0, t1;\n\t"  \
	   ".reg .pred p1, p2;\n\t" \
                                    \
	   "mov.u32    ax0, %24;\n\t"   \
	   "mov.u32    ax1, %25;\n\t"   \
	   "mov.u32    ax2, %26;\n\t"   \
	   "mov.u32    ax3, %27;\n\t"   \
	   "mov.u32    ax4, %28;\n\t"   \
	   "mov.u32    ax5, %29;\n\t"   \
	   "mov.u32    ax6, %30;\n\t"   \
	   "mov.u32    ax7, %31;\n\t"   \
	   "mov.u32    ay0, %32;\n\t"   \
	   "mov.u32    ay1, %33;\n\t"   \
	   "mov.u32    ay2, %34;\n\t"   \
	   "mov.u32    ay3, %35;\n\t"   \
	   "mov.u32    ay4, %36;\n\t"   \
	   "mov.u32    ay5, %37;\n\t"   \
	   "mov.u32    ay6, %38;\n\t"   \
	   "mov.u32    ay7, %39;\n\t"   \
	   "mov.u32    az0, %40;\n\t"   \
	   "mov.u32    az1, %41;\n\t"   \
	   "mov.u32    az2, %42;\n\t"   \
	   "mov.u32    az3, %43;\n\t"   \
	   "mov.u32    az4, %44;\n\t"   \
	   "mov.u32    az5, %45;\n\t"   \
	   "mov.u32    az6, %46;\n\t"   \
	   "mov.u32    az7, %47;\n\t"   \
	                               \
	   "mov.u32    bx0, %48;\n\t"   \
	   "mov.u32    bx1, %49;\n\t"   \
	   "mov.u32    bx2, %50;\n\t"   \
	   "mov.u32    bx3, %51;\n\t"   \
	   "mov.u32    bx4, %52;\n\t"   \
	   "mov.u32    bx5, %53;\n\t"   \
	   "mov.u32    bx6, %54;\n\t"   \
	   "mov.u32    bx7, %55;\n\t"   \
	   "mov.u32    by0, %56;\n\t"   \
	   "mov.u32    by1, %57;\n\t"   \
	   "mov.u32    by2, %58;\n\t"   \
	   "mov.u32    by3, %59;\n\t"   \
	   "mov.u32    by4, %60;\n\t"   \
	   "mov.u32    by5, %61;\n\t"   \
	   "mov.u32    by6, %62;\n\t"   \
	   "mov.u32    by7, %63;\n\t"   \
	   "mov.u32    bz0, %64;\n\t"   \
	   "mov.u32    bz1, %65;\n\t"   \
	   "mov.u32    bz2, %66;\n\t"   \
	   "mov.u32    bz3, %67;\n\t"   \
	   "mov.u32    bz4, %68;\n\t"   \
	   "mov.u32    bz5, %69;\n\t"   \
	   "mov.u32    bz6, %70;\n\t"   \
	   "mov.u32    bz7, %71;\n\t"   \
	                                \
           "mov.u32         n0, %72;\n\t" \
           "mov.u32         n1, %73;\n\t" \
           "mov.u32         n2, %74;\n\t" \
           "mov.u32         n3, %75;\n\t" \
           "mov.u32         n4, %76;\n\t" \
           "mov.u32         n5, %77;\n\t" \
           "mov.u32         n6, %78;\n\t" \
           "mov.u32         n7, %79;\n\t" \
           "mov.u32         q, %80;\n\t" \
                        \
           "mov.u32         msb, %89;\n\t" 

#define ASM_EC_DOUBLE_JACADD_PACK \
      "EC_DOUBLE_JACADD_L0:\n\t" \
      "}\n\t" \
       : "=r"(xr[0]), "=r"(xr[1]), "=r"(xr[2]), "=r"(xr[3]), "=r"(xr[4]), "=r"(xr[5]), "=r"(xr[6]), "=r"(xr[7]), \
         "=r"(xr[8]), "=r"(xr[9]), "=r"(xr[10]), "=r"(xr[11]), "=r"(xr[12]), "=r"(xr[13]), "=r"(xr[14]), "=r"(xr[15]), \
         "=r"(xr[16]), "=r"(xr[17]), "=r"(xr[18]), "=r"(xr[19]), "=r"(xr[20]), "=r"(xr[21]), "=r"(xr[22]), "=r"(xr[23]) \
       : "r"(x1[0]), "r"(x1[1]), "r"(x1[2]), "r"(x1[3]), "r"(x1[4]), "r"(x1[5]), "r"(x1[6]), "r"(x1[7]),   \
         "r"(x1[8]), "r"(x1[9]), "r"(x1[10]), "r"(x1[11]), "r"(x1[12]), "r"(x1[13]), "r"(x1[14]), "r"(x1[15]),   \
         "r"(x1[16]), "r"(x1[17]), "r"(x1[18]), "r"(x1[19]), "r"(x1[20]), "r"(x1[21]), "r"(x1[22]), "r"(x1[23]),   \
         "r"(x2[0]), "r"(x2[1]), "r"(x2[2]), "r"(x2[3]), "r"(x2[4]), "r"(x2[5]), "r"(x2[6]), "r"(x2[7]),   \
         "r"(x2[8]), "r"(x2[9]), "r"(x2[10]), "r"(x2[11]), "r"(x2[12]), "r"(x2[13]), "r"(x2[14]), "r"(x2[15]),   \
         "r"(x2[16]), "r"(x2[17]), "r"(x2[18]), "r"(x2[19]), "r"(x2[20]), "r"(x2[21]), "r"(x2[22]), "r"(x2[23]),   \
         "r"(P_u256[0]), "r"(P_u256[1]), "r"(P_u256[2]), "r"(P_u256[3]), \
         "r"(P_u256[4]), "r"(P_u256[5]), "r"(P_u256[6]), "r"(P_u256[7]), \
         "r"(PN_u256[0]), \
	 "r"(_1[0]), "r"(_1[1]), "r"(_1[2]), "r"(_1[3]), "r"(_1[4]), "r"(_1[5]), "r"(_1[6]), "r"(_1[7]), \
         "r"(msb), "r"(scl)
#define ASM_EC_DOUBLEJACADD \
"START_DOUBLEADD_LOOP:\n\t"  \
  for (i=msb, i < 1 << NWORDS_256BIT, i++){
        "setp.lt.u32      p1, msb, 256;\n\t"
        "@!p1  bra END_DOUBLEADD_LOOP;\n\t"   -> End of loop to return
	"add.u32          msb, msb, 1;\n\t"
	ASM_EQ0U256(p1, bz) \
	"@p1 bra  CHK_SCL_L0;\n\t"  \
	ASM_ECJACDOUBLE \  -> mov r to b

"CHK_SCL_L0:\n\t" \
        "mov.u32     t0,0;\n\t" \
        "mov.u32     t1,0;\n\t" \
	"sub.u32     m, 255, msb;\n\t" \
	"shr.u32 	hi, m, 5;\n\t"
	"and.b32  	lo, m, 31;\n\t"

	for(i=0; i< DEFAULT_U256_BSELM; i++){
"START_BSELU256:\n\t"
        "setp.lt.u32      p1, t0, 8;\n\t"
        "@!p1  bra END_BSELU256;\n\t"   -> End of loop to start addition
	"mad.lo.u32 	hi, t0, 8, hi;\n\t"
	"cvt.u64.u32	scl_offset, hi;\n\t"
	"shl.b64 	scl_offset, scl_offset, 2;\n\t"
	"add.s64 	scl_offset, %89, scl_offset;\n\t"
	"ld.u32 	scl, [scl_offset];\n\t"
        "bfe.u32        m,   scl,  lo, 1;  \n\t"      
	"shl.u32        m, m, t0;\n\t"
	"add.u32        t1, t1, m;\n\t"
	"add.u32        t0, t0, 1;\n\t"
	"braSTART_BSELU256;\n\t"
"END_BSELU256:\n\t"
   
        "setp.eq.u32      p1, t0, 0;\n\t"
	"@p1 bra  START_DOUBLEADD_LOOP;\n\t"
	// LOAD N[b*3*NWORDS_256BIT] into a 
	"mul.lo.lo.u32 	   hi, t1, 24;\n\t"
	"cvt.u64.u32	scl_offset, hi;\n\t"
	"shl.b64 	scl_offset, scl_offset, 2;\n\t"
	"add.s64 	scl_offset, %90, scl_offset;\n\t"
	"ld.u32 	ax0, [scl_offset];\n\t"
	"ld.u32 	ax1, [scl_offset+4];\n\t"
	"ld.u32 	ax2, [scl_offset+8];\n\t"
	"ld.u32 	ax3, [scl_offset+12];\n\t"
	"ld.u32 	ax4, [scl_offset+16];\n\t"
	"ld.u32 	ax5, [scl_offset+20];\n\t"
	"ld.u32 	ax6, [scl_offset+24];\n\t"
	"ld.u32 	ax7, [scl_offset+28];\n\t"
	"ld.u32 	ay0, [scl_offset+32;\n\t"
	"ld.u32 	ay1, [scl_offset+36];\n\t"
	"ld.u32 	ay2, [scl_offset+40];\n\t"
	"ld.u32 	ay3, [scl_offset+44];\n\t"
	"ld.u32 	ay4, [scl_offset+48];\n\t"
	"ld.u32 	ay5, [scl_offset+52];\n\t"
	"ld.u32 	ay6, [scl_offset+54];\n\t"
	"ld.u32 	ay7, [scl_offset+60];\n\t"
	"ld.u32 	az0, [scl_offset+64];\n\t"
	"ld.u32 	az1, [scl_offset+68];\n\t"
	"ld.u32 	az2, [scl_offset+72;\n\t"
	"ld.u32 	az3, [scl_offset+76];\n\t"
	"ld.u32 	az4, [scl_offset+80];\n\t"
	"ld.u32 	az5, [scl_offset+84];\n\t"
	"ld.u32 	az6, [scl_offset+88];\n\t"
	"ld.u32 	az7, [scl_offset+92];\n\t"
        ASM_ECJACADD
"END_DOUBLEADD_LOOP:\n\t"  
#endif

#endif
