
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
//    ASM_MUL_INIT : Register and constants initialization
//    ASM_MUL_START : Preparation
//    ASM_MUL_REDUCTION_BLOCK
//    ASM_MUL_BLOCK
//    ....  repeat 7 times ASM_MUL_REDUCTION_BLOCK and ASM_MUL_BLOCK
//    ASM_MUL_PACK \

#define ASM_MONTMULU256(out,in1,in2)  \
    ASM_MUL_START(out,in1,in2) \
    ASM_MUL_REDUCTION_BLOCK(out) \
    ASM_MUL_BLOCK(out, in1, in2, 0,1) \
    ASM_MUL_REDUCTION_BLOCK(out) \
    ASM_MUL_BLOCK(out, in1, in2, 1,2) \
    ASM_MUL_REDUCTION_BLOCK(out) \
    ASM_MUL_BLOCK(out,in1,in2, 2,3) \
    ASM_MUL_REDUCTION_BLOCK(out) \
    ASM_MUL_BLOCK(out,in1,in2,3,4) \
    ASM_MUL_REDUCTION_BLOCK(out) \
    ASM_MUL_BLOCK(out,in1,in2,4,5) \
    ASM_MUL_REDUCTION_BLOCK(out) \
    ASM_MUL_BLOCK(out,in1,in2,5,6) \
    ASM_MUL_REDUCTION_BLOCK(out) \
    ASM_MUL_BLOCK(out,in1,in2,6,7) \
    ASM_MUL_REDUCTION_BLOCK(out) \
    ASM_MUL_MUL_BLOCK_LAST(out,in1,in2,7)

#define ASM_MUL_INIT \
      "{\n\t" \
           ".reg .u32 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"  \
           ".reg .u32 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"  \
           ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"  \
           ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"  \
           ".reg .u32 m, q, prefix_low, prefix_high;\n\t"  \
	   ".reg .u64 p_address, p_offset;\n\t" \
                                     \
                                                            \
	   "mov.u64         p_address, %26;\n\t" \
	   "cvta.const.u64  p_address, p_address;\n\t" \
	   "mov.u32         prefix_high, %24;\n\t"   \
	   "mul.lo.u32     prefix_high, prefix_high, %25;\n\t" \
	   "cvt.u64.u32     p_offset, prefix_high; \n\t"   \
	   "add.u64         p_address, p_address, p_offset;\n\t" \
           "ld.const.u32    n0, [p_address];\n\t"  \
           "ld.const.u32    n1, [p_address + 4];\n\t"  \
           "ld.const.u32    n2, [p_address + 8];\n\t"  \
           "ld.const.u32    n3, [p_address + 12];\n\t"  \
           "ld.const.u32    n4, [p_address + 16];\n\t"  \
           "ld.const.u32    n5, [p_address + 20];\n\t"  \
           "ld.const.u32    n6, [p_address + 24];\n\t"  \
           "ld.const.u32    n7, [p_address + 28];\n\t"  \
           "ld.const.u32    q,  [p_address + 32];\n\t"        

#define ASM_MUL_START(out, in1, in2) \
     "mov.u32    "#in1"0, %8;\n\t"  \
     "mov.u32    "#in1"1, %9;\n\t"  \
     "mov.u32    "#in2"0, %16;\n\t"  \
     "mul.lo.u32 "#out"0, "#in1"0, "#in2"0;\n\t"           \
     "mov.u32    "#in2"1, %17;\n\t"  \
     "mul.lo.u32 "#out"1, "#in1"0, "#in2"1;\n\t"           \
     "mov.u32    "#in2"1, %18;\n\t"  \
     "mul.lo.u32 "#out"2, "#in1"0, "#in2"2;\n\t"           \
     "mov.u32    "#in2"1, %19;\n\t"  \
     "mul.lo.u32 "#out"3, "#in1"0, "#in2"3;\n\t"           \
     "mov.u32    "#in2"1, %20;\n\t"  \
     "mul.lo.u32 "#out"4, "#in1"0, "#in2"4;\n\t"           \
     "mov.u32    "#in2"1, %21;\n\t"  \
     "mul.lo.u32 "#out"5, "#in1"0, "#in2"5;\n\t"           \
     "mov.u32    "#in2"1, %22;\n\t"  \
     "mul.lo.u32 "#out"6, "#in1"0, "#in2"6;\n\t"           \
     "mov.u32    "#in2"1, %23;\n\t"  \
     "mul.lo.u32 "#out"7, "#in1"0, "#in2"7;\n\t"           \
     "mov.u32    "#in2"1, %24;\n\t"  \
     "mad.hi.cc.u32 "#out"1, "#in1"0, "#in2"0, "#out"1;\n\t"           \
     "madc.hi.cc.u32 "#out"2, "#in1"0, "#in2"1, "#out"2;\n\t"           \
     "madc.hi.cc.u32 "#out"3, "#in1"0, "#in2"2, "#out"3;\n\t"           \
     "madc.hi.cc.u32 "#out"4, "#in1"0, "#in2"3, "#out"4;\n\t"           \
     "madc.hi.cc.u32 "#out"5, "#in1"0, "#in2"4, "#out"5;\n\t"           \
     "madc.hi.cc.u32 "#out"6, "#in1"0, "#in2"5, "#out"6;\n\t"           \
     "madc.hi.cc.u32 "#out"7, "#in1"0, "#in2"6, "#out"7;\n\t"           \
     "madc.hi.cc.u32 prefix_low, "#in1"0, "#in2"7, 0;\n\t"          

#define ASM_MUL_REDUCTION_BLOCK(out) \
       "mul.lo.u32   m, "#out"0, q;\n\t" \
       "mad.lo.cc.u32 "#out"0, m, n0, "#out"0;\n\t" \
       "madc.hi.cc.u32 "#out"1, m, n0, "#out"1;\n\t" \
       "madc.hi.cc.u32 "#out"2, m, n1, "#out"2;\n\t" \
       "madc.hi.cc.u32 "#out"3, m, n2, "#out"3;\n\t" \
       "madc.hi.cc.u32 "#out"4, m, n3, "#out"4;\n\t" \
       "madc.hi.cc.u32  "#out"5, m, n4, "#out"5;\n\t" \
       "madc.hi.cc.u32  "#out"6, m, n5, "#out"6;\n\t" \
       "madc.hi.cc.u32  "#out"7, m, n6, "#out"7;\n\t" \
       "madc.hi.cc.u32  prefix_low, m, n7, prefix_low;\n\t" \
       "addc.u32  prefix_high, 0, 0;\n\t" \
       "mad.lo.cc.u32 "#out"0, m, n1, "#out"1;\n\t" \
       "madc.lo.cc.u32  "#out"1, m, n2, "#out"2;\n\t" \
       "madc.lo.cc.u32  "#out"2, m, n3, "#out"3;\n\t" \
       "madc.lo.cc.u32  "#out"3, m, n4, "#out"4;\n\t" \
       "madc.lo.cc.u32  "#out"4, m, n5, "#out"5;\n\t" \
       "madc.lo.cc.u32  "#out"5, m, n6, "#out"6;\n\t" \
       "madc.lo.cc.u32  "#out"6, m, n7, "#out"7;\n\t" \
       "addc.cc.u32  "#out"7, prefix_low, 0;\n\t" \
       "addc.u32  prefix_low, prefix_high, 0;\n\t" 

#define ASM_MUL_BLOCK(out, in1, in2, idx,idx1) \
    "mov.u32    "#in1#idx1", %9;\n\t"  \
    "mad.lo.cc.u32 "#out"0, "#in1#idx", "#in2"0, "#out"0;\n\t" \
    "madc.lo.cc.u32 "#out"1, "#in1#idx", "#in2"1, "#out"1;\n\t" \
    "madc.lo.cc.u32 "#out"2, "#in1#idx", "#in2"2, "#out"2;\n\t" \
    "madc.lo.cc.u32 "#out"3, "#in1#idx", "#in2"3, "#out"3;\n\t" \
    "madc.lo.cc.u32 "#out"4, "#in1#idx", "#in2"4, "#out"4;\n\t" \
    "madc.lo.cc.u32 "#out"5, "#in1#idx", "#in2"5, "#out"5;\n\t" \
    "madc.lo.cc.u32 "#out"6, "#in1#idx", "#in2"6, "#out"6;\n\t" \
    "madc.lo.cc.u32 "#out"7, "#in1#idx", "#in2"7, "#out"7;\n\t" \
    "addc.u32 prefix_low, prefix_low, 0;\n\t" \
    "mad.hi.cc.u32 "#out"1, "#in1#idx", "#in2"0, "#out"1;\n\t" \
    "madc.hi.cc.u32 "#out"2, "#in1#idx", "#in2"1, "#out"2;\n\t" \
    "madc.hi.cc.u32 "#out"3, "#in1#idx", "#in2"2, "#out"3;\n\t" \
    "madc.hi.cc.u32 "#out"4, "#in1#idx", "#in2"3, "#out"4;\n\t" \
    "madc.hi.cc.u32 "#out"5, "#in1#idx", "#in2"4, "#out"5;\n\t" \
    "madc.hi.cc.u32 "#out"6, "#in1#idx", "#in2"5, "#out"6;\n\t" \
    "madc.hi.cc.u32 "#out"7, "#in1#idx", "#in2"6, "#out"7;\n\t" \
    "madc.hi.cc.u32 prefix_low, "#in1#idx", "#in2"7, prefix_low;\n\t" \
    "addc.u32 prefix_high, 0, 0;\n\t"   

#define ASM_MUL_BLOCK_LAST(out, in1, in2, idx) \
    "mad.lo.cc.u32 "#out"0, "#in1#idx", "#in2"0, "#out"0;\n\t" \
    "mov.u32        %0, "#out"0;\n\t"   \
    "madc.lo.cc.u32 "#out"1, "#in1#idx", "#in2"1, "#out"1;\n\t" \
    "madc.lo.cc.u32 "#out"2, "#in1#idx", "#in2"2, "#out"2;\n\t" \
    "madc.lo.cc.u32 "#out"3, "#in1#idx", "#in2"3, "#out"3;\n\t" \
    "madc.lo.cc.u32 "#out"4, "#in1#idx", "#in2"4, "#out"4;\n\t" \
    "madc.lo.cc.u32 "#out"5, "#in1#idx", "#in2"5, "#out"5;\n\t" \
    "madc.lo.cc.u32 "#out"6, "#in1#idx", "#in2"6, "#out"6;\n\t" \
    "madc.lo.cc.u32 "#out"7, "#in1#idx", "#in2"7, "#out"7;\n\t" \
    "addc.u32 prefix_low, prefix_low, 0;\n\t" \
    "mad.hi.cc.u32 "#out"1, "#in1#idx", "#in2"0, "#out"1;\n\t" \
    "mov.u32        %1, "#out"1;\n\t"   \
    "madc.hi.cc.u32 "#out"2, "#in1#idx", "#in2"1, "#out"2;\n\t" \
    "mov.u32        %2, "#out"2;\n\t"   \
    "madc.hi.cc.u32 "#out"3, "#in1#idx", "#in2"2, "#out"3;\n\t" \
    "mov.u32        %3, "#out"3;\n\t"   \
    "madc.hi.cc.u32 "#out"4, "#in1#idx", "#in2"3, "#out"4;\n\t" \
    "mov.u32        %4, "#out"4;\n\t"   \
    "madc.hi.cc.u32 "#out"5, "#in1#idx", "#in2"4, "#out"5;\n\t" \
    "mov.u32        %5, "#out"5;\n\t"   \
    "madc.hi.cc.u32 "#out"6, "#in1#idx", "#in2"5, "#out"6;\n\t" \
    "mov.u32        %6, "#out"6;\n\t"   \
    "madc.hi.cc.u32 "#out"7, "#in1#idx", "#in2"6, "#out"7;\n\t" \
    "mov.u32        %7, "#out"7;\n\t"   \
    "madc.hi.cc.u32 prefix_low, "#in1#idx", "#in2"7, prefix_low;\n\t" \
    "addc.u32 prefix_high, 0, 0;\n\t"   




#define ASM_MUL_END \
      "END_MUL:\n\t" 

#define ASM_MUL_PACK \
      "}\n\t" \
       : "=r"(U[0]), "=r"(U[1]), "=r"(U[2]), "=r"(U[3]), \
         "=r"(U[4]), "=r"(U[5]), "=r"(U[6]), "=r"(U[7]) \
       : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),   \
         "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]),   \
         "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), \
         "r"(B[4]), "r"(B[5]), "r"(B[6]), "r"(B[7]), \
	 "r"(midx), "r"((uint32_t) (MOD_INFO_N * NWORDS_256BIT * sizeof(uint32_t)), \
	 "l"(mod_info_ct) ) 

// u = u - P
#define ASM_SUBU256(out, in1, in2) \
      "sub.cc.u32        "#out"0, "#in1"0, "#in2"0;\n\t"      \
      "subc.cc.u32       "#out"1, "#in1"1, "#in2"1;\n\t"     \
      "subc.cc.u32       "#out"2, "#in1"2, "#in2"2;\n\t"    \
      "subc.cc.u32       "#out"3, "#in1"3, "#in2"3;\n\t"    \
      "subc.cc.u32       "#out"4, "#in1"4, "#in2"4;\n\t"    \
      "subc.cc.u32       "#out"5, "#in1"5, "#in2"5;\n\t"    \
      "subc.cc.u32       "#out"6, "#in1"6, "#in2"6;\n\t"    \
      "subc.u32          "#out"7, "#in1"7, "#in2"7;\n\t"    

#define ASM_SUBMU256(out, in1, in2) \
	ASM_SUBU256(out, in1, in2) 

#define ASM_LTU256 \
      ".reg  .pred      p1;\n\t"  \
      "bfe.u32          q, r7, 31, 1;\n\t"   \
      "setp.eq.u32      p1, q,  1;\n\t"     \
      "@!p1  bra  END_MUL;\n\t"   

#define ASM_ADDU256 \
      "add.cc.u32        r0, r0, n0;\n\t"      \
      "addc.cc.u32       r1, r1, n1;\n\t"     \
      "addc.cc.u32       r2, r2, n2;\n\t"    \
      "addc.cc.u32       r3, r3, n3;\n\t"    \
      "addc.cc.u32       r4, r4, n4;\n\t"    \
      "addc.cc.u32       r5, r5, n5;\n\t"    \
      "addc.cc.u32       r6, r6, n6;\n\t"    \
      "addc.u32          r7, r7, n7;\n\t"    

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
           ".reg .u32 m, q, prefix_low, prefix_high;\n\t"  \
	   ".reg .u64 p_address, p_offset;\n\t" \
                                     \
	   "mov.u64         p_address, %74;\n\t" \
	   "cvta.const.u64  p_address, p_address;\n\t" \
	   "mov.u32         prefix_high, %72;\n\t"   \
	   "mul.lo.u32     prefix_high, prefix_high, %73;\n\t" \
	   "cvt.u64.u32     p_offset, prefix_high; \n\t"   \
	   "add.u64         p_address, p_address, p_offset;\n\t" \
           "ld.const.u32    n0, [p_address];\n\t"  \
           "ld.const.u32    n1, [p_address + 4];\n\t"  \
           "ld.const.u32    n2, [p_address + 8];\n\t"  \
           "ld.const.u32    n3, [p_address + 12];\n\t"  \
           "ld.const.u32    n4, [p_address + 16];\n\t"  \
           "ld.const.u32    n5, [p_address + 20];\n\t"  \
           "ld.const.u32    n6, [p_address + 24];\n\t"  \
           "ld.const.u32    n7, [p_address + 28];\n\t"  \
           "ld.const.u32    q,  [p_address + 32];\n\t"        

#define ASM_ADDECJAC_PACK \
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
	 "r"(midx), "r"((uint32_t) (MOD_INFO_N * NWORDS_256BIT * sizeof(uint32_t))), "l"(mod_info_ct) 

// available registers here are rx, ry, rz, ax, ay.
// restrictions : mul cannot have an input = output register
// rx = bz * bz
// rz = ax * rx   -> U1

// ax = rx * bz
// rx = ay * ax  -> S1

// ry = az * az
// ax = bx * ry -> U2

// ay = ry * az
// ry = by * ay -> S2

// if (rz == ax)
//   if (rx != ry)
//      return inf
//   else
//      return double(bx, by, bz)
// ax = ax - rz -> H
// ay = ry - rx  -> R
// used reg : ax(H), ay(R), az(Z1), bz(Z2), rz(U1), rx(S1)
// free reg : ry, bx, by
// ry = az * bz  -> Z1 * Z2
// az = ax * ax -> H^2
// bx = az * ax -> H^3
// by = az * rz -> U1 * H^2
// ax = bz * rx -> S1 * H^3
// rz = ry * ax  -> Z3
// used reg : bx(H^3), rz(Z3), ay(R), by(U1 * H^2), ax(S1 * H^3)
// free : ry, az, bz, rx
// rx = ay * ay
// rx = rx - bx
// rx = rx - by
// rx = rx - by  -> X3

// used reg : rz(Z3), ay(R), by(U1 * H^2), rx(X3), ax(S1 * H^3)
// free : bx, ry, az, bz
// by = by - rx
// ry = ay * by
// ry = ry - ax

#define ASM_ADDECJAC_START \
	ASM_MONTMULU256(rx, bz, bz)  \
	ASM_MONTMULU256(rz, ax, rx) \
	ASM_MONTMULU256(ax, rx, bz) \
	ASM_MONTMULU256(rx, ay, ax) \
	ASM_MONTMULU256(ry, az, az) \
	ASM_MONTMULU256(ax, bx, ry) \
	ASM_MONTMULU256(ay, ry, az) \
	ASM_MONTMULU256(ry, by, ay) \
	ASM_CHECK_DOUBLE(rz, ax, rx, ry) \
	ASM_SUBMU256(ax, ax, rz) \
	ASM_SUBMU256(ay, ry, rx) \
	ASM_MONTMULU256(ry, az, bz) \
	ASM_MONTMULU256(az, ax, ax) \
	ASM_MONTMULU256(bx, az, ax) \
	ASM_MONTMULU256(by, az, rx) \
	ASM_MONTMULU256(ax, bz, rx) \
	ASM_MONTMULU256(rz, ry, ax) \
	ASM_MONTMULU256(rx, ay, ay) \
	ASM_SUBMU256(rx, rx, bx) \
	ASM_SUBMU256(rx, rx, by) \
	ASM_SUBMU256(rx, rx, by) \
	ASM_SUBMU256(by, by, rx) \
	ASM_MONTMULU256(ry, ay, by) \
	ASM_SUBMU256(ry, ry, ax) 

#define	ASM_CHECK_DOUBLE(in1, in2, in3, in4)

#endif
