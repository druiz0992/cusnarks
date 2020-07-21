
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
// File name  : ntt.cpp
//
// Date       : 6/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//
// ------------------------------------------------------------------

#include <stdio.h>
#include <omp.h>
#include <math.h>

#include "types.h"
#include "rng.h"
#include "log.h"
#include "utils_host.h"
#include "bigint.h"
#include "ff.h"
#include "constants.h"
#include "transpose.h"
#include "ntt.h"

#ifdef PARALLEL_EN
static  uint32_t parallelism_enabled =  1;
#else
static  uint32_t parallelism_enabled =  0;
#endif

static char rootsBN256[] = {
 251,255,255,79,28,52,150,172,41,205,96,159,149,118,252,54,46,70,121,120,111,163,110,102,47,223,7,154,193,119,10,14, //0
  6,0,0,160,119,193,75,151,103,163,88,218,178,113,55,241,46,18,8,9,71,162,225,81,250,192,41,71,177,214,89,34,        //1 
  139,239,220,158,151,61,117,127,32,145,71,177,44,23,63,95,110,108,9,116,121,98,177,141,207,8,193,57,53,123,55,43,   //2
 63,124,173,181,226,74,173,248,190,133,203,131,255,198,96,45,247,41,148,93,43,253,118,217,169,217,154,63,231,124,64,36, //3
 3,143,47,116,124,125,182,244,204,104,208,99,220,45,27,104,106,87,251,27,239,188,229,140,254,60,182,210,81,41,124,22,  //4
  100,76,87,191,177,247,20,34,242,125,49,247,47,35,249,40,205,117,173,176,168,132,117,229,3,109,23,220,89,251,129,43,  //5
  191,97,143,129,229,3,144,142,194,254,248,155,52,191,155,140,78,83,1,63,205,238,220,83,60,170,41,229,107,150,144,38,  //6
 177,123,129,38,48,196,121,10,240,125,83,153,124,204,178,123,222,230,65,2,213,39,202,182,76,240,50,54,63,179,122,0,   //7
 204,74,162,131,63,184,175,162,110,83,93,82,217,85,242,146,25,221,134,2,8,102,117,94,73,37,45,197,166,177,123,24,    //8
 222,35,164,34,231,59,83,156,13,110,223,124,18,157,42,100,5,192,154,64,70,117,188,13,130,80,61,178,141,76,240,0,    //9
 132,17,12,40,180,179,244,30,44,42,94,174,194,212,122,207,24,101,163,197,108,59,6,184,140,192,223,101,185,196,72,35, //10
 178,207,79,174,137,33,231,72,7,90,248,141,60,251,3,10,10,46,155,234,53,138,77,255,119,29,156,205,46,140,169,40,  //11
 211,219,236,179,47,82,212,29,173,243,85,208,147,42,34,104,232,85,213,179,102,125,156,190,70,248,148,97,184,246,146,27 ,  //12
 214,78,160,121,190,220,76,137,135,7,211,68,106,222,108,149,95,193,219,215,43,182,161,89,78,111,128,154,16,228,235,18,  //13
 184,234,5,77,199,160,19,186,22,49,171,17,99,93,1,46,90,160,165,140,44,146,3,181,218,148,227,254,215,21,190,6,  //14
 84,184,253,91,5,247,78,128,242,234,206,64,113,107,167,122,203,137,254,178,104,90,201,252,199,6,196,241,53,28,70,29, //15
 51,116,57,57,89,231,179,71,209,36,28,13,146,58,58,109,67,95,247,116,81,18,52,161,86,213,106,238,1,31,130,27,   //16
 124,220,4,18,216,184,5,218,65,141,48,6,230,42,50,72,44,137,158,132,39,142,53,53,146,213,45,214,251,202,15,4,   //17
 132,11,112,9,47,198,102,37,96,134,191,160,118,58,24,51,241,88,80,87,89,143,57,217,52,205,209,57,206,46,109,5,  //18
 54,122,162,230,183,163,158,4,188,219,62,5,3,230,235,239,212,158,206,58,90,180,36,132,94,121,136,166,144,131,124,40,  //19
 26,147,141,170,101,212,50,218,156,143,128,97,133,246,105,38,133,176,200,228,70,171,123,36,26,2,214,129,135,102,59,13,  //20
 60,47,50,245,146,33,234,39,167,233,143,101,233,132,24,177,105,192,83,160,188,35,134,58,166,57,225,37,240,243,143,18,   //21
 242,26,239,188,110,34,142,155,96,107,64,223,171,241,69,158,61,187,167,213,87,210,141,83,188,163,130,120,3,147,56,10,   //22
 0,145,158,192,4,36,72,110,178,37,0,89,199,145,117,13,17,190,94,58,121,39,2,164,168,76,169,193,195,166,100,1,      //23
 48,208,79,216,105,189,34,199,44,22,82,207,38,74,14,96,233,167,243,69,215,126,114,251,92,39,251,105,178,167,82,22,  //24
 226,7,92,87,255,250,14,64,197,154,143,75,73,115,35,85,55,173,231,129,237,171,121,170,57,46,77,8,184,229,198,26,   //25
 254,32,138,201,34,148,162,160,157,92,147,101,202,98,212,115,247,130,69,212,110,74,186,225,182,130,58,12,192,20,252,40,  //26
 103,2,137,128,20,100,89,135,73,3,192,228,181,120,58,74,126,177,166,82,221,79,0,73,18,234,230,101,221,23,69,40,  //27
 156,61,209,128,85,115,110,99,214,255,69,36,116,243,43,162,216,3,178,30,192,42,69,86,231,249,99,41,148,239,96,24};  //28
 

#define MIN(X,Y)  ((X)<(Y) ? (X) : (Y))
#define MAX(X,Y)  ((X)>(Y) ? (X) : (Y))

static char config_curve_filename[]="./.curve";

static uint32_t *M_transpose = NULL;
static uint32_t *M_mul = NULL;

static void _ntt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx);
static void _ntt_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx);
static void _ntt_h(uint32_t *A, const uint32_t *roots, uint32_t levels, int32_t direction, t_ff *ff);
static void _intt_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx);
static void _intt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx);
static void ntt_reorder_h(uint32_t *A, uint32_t levels, uint32_t astride);
static void ntt_parallel_T_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, int32_t direction, fft_mode_t fft_mode, uint32_t pidx);
static void _ntt_parallel_T_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, int32_t direction, fft_mode_t fft_mode, uint32_t pidx);
static void montmult_reorder_h(uint32_t *A, const uint32_t *roots, uint32_t levels, uint32_t pidx);
static void montmult_parallel_reorder_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, uint32_t rstride, uint32_t direction, uint32_t pidx);
static void ntt_interpolandmul_init_h(uint32_t *A, uint32_t *B, uint32_t *mNrows, uint32_t *mNcols, uint32_t nRows, uint32_t nCols);
void *interpol_and_mul_h(void *args);
void *interpol_and_muls_h(void *args);



/*
  Bit reverse 32 bit number

  uint32_t x : input number
  uint32_t bits : number of bits
  
  returns bit reversed input number
*/
inline uint32_t reverse32(uint32_t x, uint32_t bits)
{
  // from http://graphics.stanford.edu/~seander/bithacks.html
  // swap odd and even bits
  x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
  // swap consecutixe pairs
  x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
  // swap nibbles ... 
  x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4);
  // swap bytes
  x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8);
  // swap 2-byte long pairs
  x = ( x >> 16             ) | ( x               << 16);

  x = ( x >> (32 - bits));
  return x;
}
inline uint32_t reverse16(uint32_t x, uint32_t bits)
{
  // from http://graphics.stanford.edu/~seander/bithacks.html
  // swap odd and even bits
  x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
  // swap consecutixe pairs
  x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
  // swap nibbles ... 
  x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4);
  // swap bytes
  x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8);

  x = ( x >> (16 - bits));
  return x;
}

inline uint32_t reverse8(uint32_t x, uint32_t bits)
{
  // from http://graphics.stanford.edu/~seander/bithacks.html
  // swap odd and even bits
  x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
  // swap consecutixe pairs
  x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
  // swap nibbles ... 
  x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4);

  x = ( x >> (8 - bits));
  return x;
}


/*
  Recursive 4 step N 256 bit sample IFFT. Read https://www.davidhbailey.com/dhbpapers/fftq.pdf for more info
   1) Get input samples in a N=N1xN2 matrix, filling the matrix by columns. Compute N1 N2 point FFT (FFT of every row)
   2) Multiply resulting N1xN2 Ajk matrix by inv_root[j*k]
   3) Transpose resulting matrix to N2xN1 matrix
   4) Perform N2 N1 point FFT
   5) Divide each coefficient by number of samples

   Retrieve data columnwise from the resulting N2xN1 matrix
   Function is 2D because when computing FFT of rows/columns, the 4-step procedure is repeated

   uint32_t *A     : Input vector containing ordered samples in Montgomery format. Input vector has 1<<(Nrows+Ncols) samples. Resulting FFT is returned in vector A
                        Output samples are ordered
   uint32_t *roots : input roots (first root is 1) in Montgomery. If roots are inverse, IFFT is computed
   uint32_t *format : if 0, output is in normal format. If 1, outout is montgomery
   uint32_t Nrows  : Number of rows in starting matrix (N1)
   uint32_t fft_Nyx   : Number of columns N12 in secondary matrix (N1=N11xN12)
   uint32_t Ncols  : Number of columns in starting matrix (N2)
   uint32_t fft_Nxx : Number of columns N22 in secondary matrix (N2=N21xN22)
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/

void intt_parallel2D_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, t_uint64 rstride, uint32_t pidx)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);

  ntt_parallel2D_h(A, roots, Nrows, fft_Nyx,  Ncols, fft_Nxx, rstride,1,pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&A[i*NWORDS_FR], &A[i*NWORDS_FR], &scaler[(Nrows + Ncols)*NWORDS_FR], pidx);
  }
}


void intt_parallel3D_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t N_fftx, uint32_t N_ffty, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, uint32_t pidx)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);

  ntt_parallel3D_h(A, roots, N_fftx, N_ffty, Nrows, fft_Nyx,  Ncols, fft_Nxx, 1,pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0;i < 1 << (N_fftx + N_ffty); i++){
      montmult_h(&A[i*NWORDS_FR], &A[i*NWORDS_FR], &scaler[(N_fftx + N_ffty)*NWORDS_FR], pidx);
  }
}

/*
  4 step N 256 bit sample FFT. Read https://www.davidhbailey.com/dhbpapers/fftq.pdf for more info
   1) Get input samples in a N=N1xN2 matrix, filling the matrix by columns. Compute N1 N2 point FFT (FFT of every row)
   2) Multiply resulting N1xN2 Ajk matrix by root[j*k]
   3) Transpose resulting matrix to N2xN1 matrix
   4) Perform N2 N1 point FFT

   Retrieve data columnwise from the resulting N2xN1 matrix
   Function is 2D because when computing FFT of rows/columns, the 4-step procedure is repeated
   uint32_t *A     : Input vector containing ordered samples in Montgomery format. Input vector has 1<<(Nrows+Ncols) samples. Resulting FFT is returned in vector A
                        Output samples are ordered
   uint32_t *roots : input roots (first root is 1) in Montgomery. If roots are inverse, IFFT is computed
   uint32_t Nrows  : Number of rows in starting matrix (N1)
   uint32_t fft_Nyx   : Number of columns N12 in secondary matrix (N1=N11xN12)
   uint32_t Ncols  : Number of columns in starting matrix (N2)
   uint32_t fft_Nxx : Number of columns N22 in secondary matrix (N2=N21xN22)
   uint32_t rstride : Root stride
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void ntt_parallel2D_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, t_uint64 rstride, int32_t direction, uint32_t pidx)
{
  uint32_t Anrows = (1<<Nrows);
  uint32_t Ancols = (1<<Ncols);
  uint32_t Mnrows = Ancols;
  uint32_t Mncols = Anrows;
  uint32_t *M = (uint32_t *) malloc (Anrows * Ancols * NWORDS_FR * sizeof(uint32_t));

  transpose_h(M,A,Anrows, Ancols);

  for (uint32_t i=0;i < Mnrows; i++){
    ntt_parallel_h(&M[i*NWORDS_FR*Mncols], roots, fft_Nyx, Nrows - fft_Nyx, rstride*Mnrows, direction, FFT_T_DIT,pidx);
    for (uint32_t j=0;j < Mncols; j++){   
        montmult_h(&M[i*NWORDS_FR*Mncols+j*NWORDS_FR], &M[i*NWORDS_FR*Mncols+j*NWORDS_FR], &roots[rstride*i*j*NWORDS_FR], pidx);
    }
  }

  transpose_h(A,M,Mnrows, Mncols);

  for (uint32_t i=0;i < Anrows; i++){
    ntt_parallel_h(&A[i*NWORDS_FR*Ancols], roots, fft_Nxx,Ncols - fft_Nxx, rstride*Mncols, direction, FFT_T_DIT, pidx);
  }

  transpose_h(M,A,Anrows, Ancols);
  memcpy(A,M,Ancols * Anrows * NWORDS_FR * sizeof(uint32_t));

  free(M);
}

void ntt_parallel3D_h(uint32_t *A, const uint32_t *roots, uint32_t Nfft_x, uint32_t Nfft_y, uint32_t Nrows, uint32_t fft_Nyx,  uint32_t Ncols, uint32_t fft_Nxx, int32_t direction, uint32_t pidx)
{
  uint32_t Anrows = (1<<Nfft_y);
  uint32_t Ancols = (1<<Nfft_x);
  uint32_t Mnrows = Ancols;
  uint32_t Mncols = Anrows;
  uint32_t *M = (uint32_t *) malloc (Anrows * Ancols * NWORDS_FR * sizeof(uint32_t));
  uint32_t i,j;

  transpose_h(M,A,Anrows, Ancols);

  for (i=0;i < Mnrows; i++){
    ntt_parallel2D_h(&M[i*NWORDS_FR*Mncols], roots, Nrows, fft_Nyx, Nfft_y - Nrows, Nrows - fft_Nyx, Mnrows, direction,pidx);
    for (j=0;j < Mncols; j++){   
        montmult_h(&M[i*NWORDS_FR*Mncols+j*NWORDS_FR], &M[i*NWORDS_FR*Mncols+j*NWORDS_FR], &roots[i*j*NWORDS_FR], pidx);
    }
  }
  
  transpose_h(A,M,Mnrows, Mncols);

  for (i=0;i < Anrows; i++){
    ntt_parallel2D_h(&A[i*NWORDS_FR*Ancols], roots, Ncols,fft_Nxx,  Nfft_x- Ncols, Ncols - fft_Nxx, Mncols, direction, pidx);
  }

  transpose_h(M,A,Anrows, Ancols);
  memcpy(A,M,Ancols * Anrows * NWORDS_FR * sizeof(uint32_t));

  free(M);
}

/*
  4 step N 256 bit sample FFT. Read https://www.davidhbailey.com/dhbpapers/fftq.pdf for more info
   1) Get input samples in a N=N1xN2 matrix, filling the matrix by columns. Compute N1 N2 point FFT (FFT of every row)
   2) Multiply resulting N1xN2 Ajk matrix by root[j*k]
   3) Transpose resulting matrix to N2xN1 matrix
   4) Perform N2 N1 point FFT
   Retrieve data columnwise from the resulting N2xN1 matrix

   uint32_t *A     : Input vector containing ordered samples in Montgomery format. Input vector has 1<<(Nrows+Ncols) samples. Resulting FFT is returned in vector A
                        Output samples are ordered
   uint32_t *roots : input roots (first root is 1) in Montgomery. If roots are inverse, IFFT is computed
   uint32_t Nrows  : Number of rows in starting matrix (N1)
   uint32_t Ncols  : Number of columns in starting matrix (N2)
   uint32_t rstride : roots stride
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void ntt_parallel_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, int32_t direction, fft_mode_t fft_mode,  uint32_t pidx)
{
  ntt_parallel_T_h(A, roots, Nrows, Ncols, rstride, direction, fft_mode, pidx);

  transpose_h(M_transpose,A,1<<Nrows, 1<<Ncols);
  memcpy(A,M_transpose, (1ull << (Nrows + Ncols)) * NWORDS_FR * sizeof(uint32_t));
}

static void ntt_parallel_T_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, int32_t direction, fft_mode_t fft_mode, uint32_t pidx)
{
  uint32_t Anrows = (1<<Nrows);
  uint32_t Ancols = (1<<Ncols);
  int64_t ridx;
  void (*fft_ptr)(uint32_t *, const uint32_t *, uint32_t , t_uint64 , t_uint64 , int32_t , uint32_t ) = ntt_h;
  if (fft_mode == FFT_T_DIF){
      fft_ptr = ntt_dif_h;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < Ancols; i++){
    fft_ptr(&A[i * NWORDS_FR], roots, Nrows,Ancols, rstride<<Ncols, direction,  pidx);
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << (Nrows + Ncols); i++){
    ridx = (rstride * (i >> Ncols) * (i & (Ancols - 1)) * direction ) & (rstride * Anrows * Ancols - 1);
    montmult_h(&A[i * NWORDS_FR],
               &A[i * NWORDS_FR],
               &roots[ridx * NWORDS_FR],
               pidx);
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < Anrows; i++){
    fft_ptr(&A[(i * NWORDS_FR ) <<Ncols], roots, Ncols,1, rstride<<Nrows, direction, pidx);
  }
}

static void _ntt_parallel_T_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, int32_t direction, fft_mode_t fft_mode, uint32_t pidx)
{
  void (*fft_ptr)(uint32_t *, const uint32_t *, uint32_t , t_uint64 , t_uint64 , int32_t , uint32_t ) = _ntt_h;
  if (fft_mode == FFT_T_DIF){
      fft_ptr = _ntt_dif_h;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << Ncols; i++){
    fft_ptr(&A[i * NWORDS_FR], roots, Nrows,1 << Ncols, rstride<<Ncols, direction,  pidx);
  }

  montmult_parallel_reorder_h(A, roots, Nrows, Ncols, rstride, direction, pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << Nrows; i++){
    fft_ptr(&A[(i*NWORDS_FR) <<Ncols], roots, Ncols,1, rstride<<Nrows, direction, pidx);
  }
}


/*
  4 step N 256 bit sample IFFT. Read https://www.davidhbailey.com/dhbpapers/fftq.pdf for more info
   1) Get input samples in a N=N1xN2 matrix, filling the matrix by columns. Compute N1 N2 point FFT (FFT of every row)
   2) Multiply resulting N1xN2 Ajk matrix by inv_root[j*k]
   3) Transpose resulting matrix to N2xN1 matrix
   4) Perform N2 N1 point FFT
   5) Divide each coefficient by number of samples

   Retrieve data columnwise from the resulting N2xN1 matrix

   uint32_t *A     : Input vector containing ordered samples in Montgomery format. Input vector has 1<<(Nrows+Ncols) samples. Resulting FFT is returned in vector A
                        Output samples are ordered
   uint32_t *roots : input roots (first root is 1) in Montgomery. If roots are inverse, IFFT is computed
   uint32_t *format : if 0, output is in normal format. If 1, outout is montgomery
   uint32_t Nrows  : Number of rows in starting matrix (N1)
   uint32_t Ncols  : Number of columns in starting matrix (N2)
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void intt_parallel_h(uint32_t *A, const uint32_t *roots,uint32_t format, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, fft_mode_t fft_mode, uint32_t pidx)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);

  ntt_parallel_h(A, roots, Nrows, Ncols, rstride, -1, fft_mode, pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&A[i * NWORDS_FR], &A[i * NWORDS_FR], &scaler[(Nrows + Ncols) * NWORDS_FR], pidx);
  }
}


/*
   Computes the forward number-theoretic transform of the given vector in place,
   with respect to the given primitive nth root of unity under the given modulus.
   The length of the vector must be a power of 2.
   NOTE https://www.nayuki.io/page/number-theoretic-transform-integer-dft

   NOTE https://www.nayuki.io/page/number-theoretic-transform-integer-dft

   uint32_t *A     : ordered input vector of length 1<<levels in montgomery format. Ordered result is 
                  is stored in A as well.
   uint32_t *roots : input roots (first root is 1) in Montgomery. If roots are inverse, IFFT is computed. 
   uint32_t levels : 1<<levels is the number of samples in the FFT
   uint32_t rstride : roots stride
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
static void _ntt_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx)
{
   int32_t n = 1 << levels;
   uint32_t i,j,l, halfsize;
   int32_t k, tablestep, size, ridx;
   uint32_t right[NWORDS_FR];
   uint32_t lidx, jidx;

   t_mulm mulm_cb =  getcb_mulm_h(pidx);
   t_addm addm_cb =  getcb_addm_h(pidx);
   t_subm subm_cb =  getcb_subm_h(pidx);

   for(size=2; size <= n; size *=2){
     halfsize = size >> 1; 
     tablestep = (1 * direction)*n/size;
     for (i=0; i<n; i+=size){
        k = 0;
        for (j=i; j<i+halfsize; j++){
           l = j + halfsize;
           ridx = (rstride*k) & (rstride * n-1);
           lidx = (astride*l) * NWORDS_FR;
           jidx = (astride*j) * NWORDS_FR;
           mulm_cb(right,&A[lidx], &roots[ridx * NWORDS_FR]);
           subm_cb(&A[lidx], &A[jidx], right);
           addm_cb(&A[jidx], &A[jidx], right);
           k += tablestep;
        }
     }
  }
}
static void _ntt_h(uint32_t *A, const uint32_t *roots, uint32_t levels, int32_t direction, t_ff *ff)
{
   int32_t n = 1 << levels;
   uint32_t i,j,l, halfsize;
   int32_t k, tablestep, size, ridx;
   uint32_t right[NWORDS_FR];
   uint32_t lidx, jidx;

   t_mulm mulm_cb =  ff->mulm_cb;
   t_addm addm_cb =  ff->addm_cb;
   t_subm subm_cb =  ff->subm_cb;

   for(size=2; size <= n; size *=2){
     halfsize = size >> 1; 
     tablestep = (1 * direction)*n/size;
     for (i=0; i<n; i+=size){
        k = 0;
        for (j=i; j<i+halfsize; j++){
           l = j + halfsize;
           ridx = k & (n-1);
           lidx = l * NWORDS_FR;
           jidx = j * NWORDS_FR;
           mulm_cb(right,&A[lidx], &roots[ridx * NWORDS_FR]);
           subm_cb(&A[lidx], &A[jidx], right);
           addm_cb(&A[jidx], &A[jidx], right);
           k += tablestep;
        }
     }
  }
}

void ntt_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx)
{
   ntt_reorder_h(A, levels, astride);
   _ntt_h(A, roots, levels, astride, rstride, direction, pidx);
}

void ntt_h(uint32_t *A, const uint32_t *roots, uint32_t levels, int32_t direction, t_ff *ff)
{
   ntt_reorder_h(A, levels, 1);
   _ntt_h(A, roots, levels, direction, ff);
}

static void _ntt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx)
{
   uint32_t i,j,k ,s, t;
   uint32_t right[NWORDS_FR];

   for (i=0; i<levels; i++){
     for (j=0; j < 1<< i; j++) {
       for (k=0; k < 1 << (levels - i - 1); k++){
          s = j * (1 << (levels - i)) + k;
          t = s + (1 << (levels - i - 1));
          subm_h(right,
                 &A[astride*s * NWORDS_FR],
                 &A[astride*t * NWORDS_FR],
                 pidx);
          addm_h(&A[astride*s * NWORDS_FR],
                 &A[astride*s * NWORDS_FR],
                 &A[astride*t * NWORDS_FR], pidx);
          montmult_h(&A[astride*t * NWORDS_FR],
                     right,
                     &roots[((rstride*(1ull<<i)*k*direction) & (rstride*(1ull << levels)-1)) * NWORDS_FR], pidx);
       }
     }
   }
}

static void ntt_reorder_h(uint32_t *A, uint32_t levels, uint32_t astride)
{
  uint32_t (*reverse_ptr) (uint32_t, uint32_t);
  
   if (levels <= 8){
      reverse_ptr = reverse8;
   } else if (levels <= 16) {
      reverse_ptr = reverse16;
   } else {
      reverse_ptr = reverse32;
   }

   #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
   #endif
   for (uint32_t i=0; i < 1 << levels ; i++){
      uint32_t j = reverse_ptr(i, levels);
      if (j > i){
         swapuBI_h(&A[astride* i * NWORDS_FR],&A[astride*j * NWORDS_FR], NWORDS_FR);
      }
   }
}

static void montmult_reorder_h(uint32_t *A, const uint32_t *roots, uint32_t levels, uint32_t pidx)
{
  uint32_t (*reverse_ptr) (uint32_t, uint32_t);

  if (levels <= 8){
      reverse_ptr = reverse8;
   } else if (levels <= 16) {
      reverse_ptr = reverse16;
   } else {
      reverse_ptr = reverse32;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i<1 << levels; i++){
    uint32_t j = reverse_ptr(i, levels);
    if (j > i){
      montmult_h(&A[i * NWORDS_FR], &A[i * NWORDS_FR], &roots[j * NWORDS_FR], pidx);
      montmult_h(&A[j * NWORDS_FR], &A[j * NWORDS_FR], &roots[i * NWORDS_FR], pidx);
    } else if (j == i) {
      montmult_h(&A[i * NWORDS_FR], &A[i * NWORDS_FR], &roots[i * NWORDS_FR], pidx);
    } 
  }
}

static void montmult_parallel_reorder_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, uint32_t rstride, uint32_t direction, uint32_t pidx)
{
  uint32_t (*reverse_ptr) (uint32_t, uint32_t);
  uint32_t levels = Nrows + Ncols;

  if ((levels+1) <= 8){
      reverse_ptr = reverse8;
   } else if ((levels+1) <= 16) {
      reverse_ptr = reverse16;
   } else {
      reverse_ptr = reverse32;
  }

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (t_uint64 i=0;i<1 << levels; i++){
    int32_t ridx = (rstride * (i >> Ncols) * (i & ( (1 << Ncols) - 1)) * direction ) & ( (rstride  << (Nrows + Ncols)) - 1);
    t_uint64 j = reverse_ptr(ridx, levels+1);
    montmult_h(&A[i * NWORDS_FR], &A[i * NWORDS_FR], &roots[j * NWORDS_FR], pidx);
  }
}


void ntt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t levels, t_uint64 astride, t_uint64 rstride, int32_t direction, uint32_t pidx)
{
   _ntt_dif_h(A, roots, levels, astride, rstride, direction, pidx);
   ntt_reorder_h(A, levels, astride);
}

/*
   Computes the inverse number-theoretic transform of the given vector in place,
   with respect to the given primitive nth root of unity under the given modulus.
   The length of the vector must be a power of 2.

   NOTE https://www.nayuki.io/page/number-theoretic-transform-integer-dft

   uint32_t *A     : ordered input vector of length 1<<levels in montgomery format. Ordered result is 
                  is stored in A as well.
   uint32_t *roots : input inverse roots (first root is 1) in Montgomery. 
   uint32_t *format : if 0, output is in normal format. If 1, outout is montgomery
   uint32_t levels : 1<<levels is the number of samples in the FFT
   uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void intt_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx)
{
  ntt_reorder_h(A, levels, 1);
  _intt_h(A, roots, format, levels, rstride, pidx);
}

static void _intt_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);
  
  _ntt_h(A, roots, levels, 1,rstride, -1, pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i< (1<<levels); i++){
     montmult_h(&A[i * NWORDS_FR], &A[i * NWORDS_FR], &scaler[levels * NWORDS_FR], pidx);
  }
  
}
void intt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx)
{
  _intt_dif_h(A, roots, format, levels, rstride, pidx);
  ntt_reorder_h(A, levels, 1);
}

static void _intt_dif_h(uint32_t *A, const uint32_t *roots, uint32_t format, uint32_t levels, t_uint64 rstride,  uint32_t pidx)
{
  uint32_t i;
  const uint32_t *scaler = CusnarksIScalerGet((fmt_t)format);

  _ntt_dif_h(A, roots, levels, 1,rstride,-1, pidx);
  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (i=0; i< (1<<levels); i++){
     montmult_h(&A[i * NWORDS_FR], &A[i * NWORDS_FR], &scaler[levels * NWORDS_FR], pidx);
  }
}

void interpol_odd_h(uint32_t *A, const uint32_t *roots, uint32_t levels,t_uint64 rstride, uint32_t pidx)
{ 
  _intt_dif_h(A, roots, 1, levels,rstride, pidx);
  montmult_reorder_h(A,roots,levels, pidx);
  _ntt_h(A, roots, levels,1, rstride,1, pidx);
}

void ntt_init_h(uint32_t nroots)
{
  if (M_transpose == NULL){
    M_transpose = (uint32_t *) malloc ( (t_uint64) (nroots) * NWORDS_FR * sizeof(uint32_t));
  }
  if (M_mul == NULL){
    M_mul = (uint32_t *) malloc ( (t_uint64)(nroots) * NWORDS_FR * sizeof(uint32_t));
  }
  
}

void ntt_free_h(void)
{
  free (M_transpose);
  free (M_mul);
  M_transpose = NULL;
  M_mul = NULL;
}

uint32_t *get_Mmul_h()
{
  return M_mul;
}

uint32_t *get_Mtranspose_h()
{
  return M_transpose;
}

void interpol_parallel_odd_h(uint32_t *A, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, uint32_t pidx)
{
  intt_parallel_h(A, roots, 1, Nrows, Ncols, rstride, FFT_T_DIF, pidx);
  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&A[i * NWORDS_FR], &A[i * NWORDS_FR], &roots[i * NWORDS_FR], pidx);
  }
  ntt_parallel_h(A, roots, Nrows, Ncols, rstride, 1, FFT_T_DIT,  pidx);
}

uint32_t * ntt_interpolandmul_server_h(ntt_interpolandmul_t *args)
{
  if ((!args->max_threads) || (!utils_isinit_h())) {
    return ntt_interpolandmul_parallel_h(args->A, args->B, args->roots, args->Nrows, args->Ncols, args->rstride, args->pidx);
  }
  #ifndef PARALLEL_EN
    return ntt_interpolandmul_parallel_h(args->A, args->B, args->roots, args->Nrows, args->Ncols, args->rstride, args->pidx);
  #endif

  uint32_t nprocs = get_nprocs_h();
  args->max_threads = MIN(args->max_threads, MIN(nprocs, 1<<MIN(args->Nrows, args->Ncols)));

  unsigned long long nvars = 1ull<<(args->Nrows+args->Ncols);
  unsigned long long start_idx, last_idx;
  unsigned long long vars_per_thread = nvars/args->max_threads;

  if  ((vars_per_thread & (vars_per_thread-1)) != 0){
    vars_per_thread = sizeof(uint32_t) * NBITS_BYTE - __builtin_clz(args->max_threads/nvars) - 1;
    vars_per_thread = 1 << (vars_per_thread-1);
  }
   
  pthread_t *workers = (pthread_t *) malloc(args->max_threads * sizeof(pthread_t));
  ntt_interpolandmul_t *w_args  = (ntt_interpolandmul_t *)malloc(args->max_threads * sizeof(ntt_interpolandmul_t));

  init_barrier_h(args->max_threads);

  /*
  printf("N threads : %d\n", args->max_threads);
  printf("N vars    : %d\n", nvars);
  printf("Vars per thread : %d\n", vars_per_thread);
  printf("Nrows : %d\n", args->Nrows);
  printf("Ncols : %d\n", args->Ncols);
  printf("nroots : %d\n", args->nroots);
  printf("rstride : %d\n", args->rstride);
  printf("pidx : %d\n", args->pidx);
  */
 
  
  ntt_interpolandmul_init_h(args->A, args->B, &args->mNrows,&args->mNcols, args->Nrows, args->Ncols);

  for(uint32_t i=0; i< args->max_threads; i++){
     start_idx = i * vars_per_thread;
     last_idx = (i+1) * vars_per_thread;
     if ( (i == args->max_threads - 1) && (last_idx != nvars) ){
         last_idx = nvars;
     }
     memcpy(&w_args[i], args, sizeof(ntt_interpolandmul_t));

     w_args[i].start_idx = start_idx;
     w_args[i].last_idx = last_idx;
     w_args[i].thread_id = i;
     
    /* 
     printf("Thread : %d, start_idx : %d, end_idx : %d\n",
             w_args[i].thread_id, 
             w_args[i].start_idx,
             w_args[i].last_idx);   
   */
    
  }

  /*
  printf("mNrows : %d\n", args->mNrows);
  printf("mNcols : %d\n", args->mNcols);
  */
 
  /*
  for (uint32_t i=0; i < 1 << (args->Nrows +args->Ncols); i++){
    printf("[%d] : ",i);
    printUBINumber(&args->A[i*NWORDS_FR], NWORDS_FR);
    printUBINumber(&args->B[i*NWORDS_FR], NWORDS_FR);
  }
  */

  if (args->mode == 0) {
    launch_client_h(interpol_and_mul_h, workers, (void *)w_args, sizeof(ntt_interpolandmul_t), args->max_threads,0);
  } else {
    launch_client_h(interpol_and_muls_h, workers, (void *)w_args, sizeof(ntt_interpolandmul_t), args->max_threads,0);
  }

  /*
  for (uint32_t i=0; i < 1 << (args->Nrows +args->Ncols); i++){
    printf("[%d] : ",i);
    printUBINumber(&args->A[i*NWORDS_FR], NWORDS_FR);
    printUBINumber(&args->B[i*NWORDS_FR], NWORDS_FR);
  }
  for (uint32_t i=0; i < 1 << (args->Nrows +args->Ncols+1); i++){
    printf("[%d] : ",i);
    printUBINumber(&M_mul[i*NWORDS_FR], NWORDS_FR);
  }
  */

  del_barrier_h();
  free(workers);
  free(w_args);

  //return M_mul;
  return M_transpose;
}

static void ntt_interpolandmul_init_h(uint32_t *A, uint32_t *B, uint32_t *mNrows, uint32_t *mNcols, uint32_t Nrows, uint32_t Ncols)
{
  if (Nrows < Ncols){
    *mNcols = Ncols;
    *mNrows = Nrows+1;
  } else {
    *mNrows = Nrows;
    *mNcols = Ncols+1;
  }
}

void *interpol_and_mul_h(void *args)
{
  ntt_interpolandmul_t *wargs = (ntt_interpolandmul_t *)args;
  unsigned long long start_col = wargs->start_idx>>wargs->Nrows, last_col = wargs->last_idx>>wargs->Nrows;
  unsigned long long start_row = wargs->start_idx>>wargs->Ncols, last_row = wargs->last_idx>>wargs->Ncols;
  uint32_t Ancols = 1 << wargs->Ncols;
  uint32_t Anrows = 1 << wargs->Nrows;
  uint32_t Amncols = 1 << wargs->mNcols;
  uint32_t Amnrows = 1 << wargs->mNrows;
  uint32_t *save_A, *save_B, *save_M;
  const uint32_t *scaler_mont = CusnarksIScalerGet((fmt_t)1);
  const uint32_t *scaler_ext = CusnarksIScalerGet((fmt_t)0);
  int64_t ridx;
  unsigned long long i;
  unsigned long long ridx1;
  unsigned long long B_offset = (1ull*NWORDS_FR)<<(wargs->Nrows+wargs->Ncols);
  uint32_t *M_transpose1_ptr = M_transpose;
  uint32_t *M_transpose2_ptr = &M_transpose[B_offset];
  unsigned long long roffset1, roffset2, roffset3, roffset4;
  t_ff  ff = {
             .subm_cb = getcb_subm_h(wargs->pidx),
             .addm_cb = getcb_addm_h(wargs->pidx),
             .mulm_cb = getcb_mulm_h(wargs->pidx)
  };

  t_mulm mulm_cb =  ff.mulm_cb;
  
  roffset1 = wargs->Ncols + wargs->Nrows;
  if  (roffset1 % 2){
     roffset1 = 1 << (roffset1+1);
     roffset2 = roffset1 + (1<<wargs->Nrows);
     roffset3  = roffset1;
     roffset4 =  roffset1;
  } else {
     roffset1 = 1 << (roffset1+1); 
     roffset2 = roffset1;
     roffset3  = roffset1;
     roffset4 =  roffset1 + (1<<(wargs->Nrows));
  }
  // multiply M_mul[2*i] = A * B
  for (i=wargs->start_idx; i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_mul[(2*(i+1)) * NWORDS_FR]);
      __builtin_prefetch(&wargs->A[(i+1) * NWORDS_FR]);
      __builtin_prefetch(&wargs->B[(i+1) * NWORDS_FR]);
    }
    mulm_cb(&M_mul[(2*i) * NWORDS_FR],
               &wargs->A[i * NWORDS_FR],
               &wargs->B[i * NWORDS_FR]);
  }
  wait_h(wargs->thread_id, NULL, NULL);

  // A = IFFT_N/2(A); B = IFFT_N/2(B)
  transposeBlock_h(M_transpose1_ptr, wargs->A,
              start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_transpose2_ptr,
              wargs->B, start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);

  
  wait_h(wargs->thread_id, NULL, NULL);

  for (i=start_col; i<last_col; i++){
    ntt_h(&M_transpose1_ptr[(i*NWORDS_FR)<<wargs->Nrows], &wargs->roots[roffset1*NWORDS_FR], wargs->Nrows, -1,  &ff);
  }
  for (i=start_col; i<last_col; i++){
    ntt_h(&M_transpose2_ptr[(i*NWORDS_FR)<<wargs->Nrows], &wargs->roots[roffset1*NWORDS_FR], wargs->Nrows, -1,  &ff);
  }

  wait_h(wargs->thread_id, NULL, NULL);

  // A[i] = A[i] * l2_IW[i]

  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_transpose1_ptr[(i+1) * NWORDS_FR]);
    }
    ridx1 = (wargs->rstride * (i >> wargs->Nrows) * (i & (Anrows - 1)) * -1) & (wargs->rstride * Anrows * Ancols - 1);
    mulm_cb(&M_transpose1_ptr[i * NWORDS_FR],
               &M_transpose1_ptr[i * NWORDS_FR],
               &wargs->roots[ridx1 * NWORDS_FR]);
  }
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_transpose2_ptr[(i+1) * NWORDS_FR]);
    }
    ridx1 = (wargs->rstride * (i >> wargs->Nrows) * (i & (Anrows - 1)) * -1) & (wargs->rstride * Anrows * Ancols - 1);
    mulm_cb(&M_transpose2_ptr[i * NWORDS_FR],
               &M_transpose2_ptr[i * NWORDS_FR],
               &wargs->roots[ridx1 * NWORDS_FR]);
  }
  wait_h(wargs->thread_id, NULL, NULL);

  transposeBlock_h(wargs->A, M_transpose1_ptr,
              start_col, last_col,
              1<<wargs->Ncols, 1<<wargs->Nrows,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(wargs->B,M_transpose2_ptr,
              start_col, last_col,
              1<<wargs->Ncols, 1<<wargs->Nrows,
              TRANSPOSE_BLOCK_SIZE);

  wait_h(wargs->thread_id, NULL, NULL);


  // A[i] = IFFT_N/2(A).T; B[i] = IFFT_N/2(B).T
  for (i=start_row;i < last_row; i++){
    ntt_h(&wargs->A[(i*NWORDS_FR)<<wargs->Ncols], &wargs->roots[roffset2*NWORDS_FR], wargs->Ncols, -1, &ff);
  }
  for (i=start_row;i < last_row; i++){
    ntt_h(&wargs->B[(i*NWORDS_FR)<<wargs->Ncols],&wargs->roots[roffset2*NWORDS_FR], wargs->Ncols, -1, &ff);
  }

  transposeBlock_h(M_transpose1_ptr, wargs->A,
              start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_transpose2_ptr,
              wargs->B, start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  wait_h(wargs->thread_id, NULL, NULL);

  save_A = wargs->A;
  save_B = wargs->B;
  wargs->A = M_transpose1_ptr;
  wargs->B = M_transpose2_ptr;
  M_transpose1_ptr = save_A;
  M_transpose2_ptr = save_B;

  // A = A * scaler * l3W; B = B * scaler * l3W
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&wargs->A[(i+1) * NWORDS_FR]);
      __builtin_prefetch(&wargs->roots[(i+1) * NWORDS_FR]);
    }
      mulm_cb(&wargs->A[i * NWORDS_FR],
                 &wargs->A[i * NWORDS_FR],
                 &scaler_mont[(wargs->Nrows + wargs->Ncols) * NWORDS_FR]);
      mulm_cb(&wargs->A[i * NWORDS_FR],
                 &wargs->A[i * NWORDS_FR],
                 &wargs->roots[i * NWORDS_FR]);
  }
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&wargs->B[(i+1) * NWORDS_FR]);
      __builtin_prefetch(&wargs->roots[(i+1) * NWORDS_FR]);
    }
      mulm_cb(&wargs->B[i * NWORDS_FR],
                 &wargs->B[i * NWORDS_FR],
                 &scaler_mont[(wargs->Nrows + wargs->Ncols) * NWORDS_FR]);
      mulm_cb(&wargs->B[i * NWORDS_FR],
                 &wargs->B[i * NWORDS_FR],
                 &wargs->roots[i * NWORDS_FR]);
  }
  wait_h(wargs->thread_id, NULL, NULL);

  // A = FFT_N/2(A); B = FFT_N/2(B)
  transposeBlock_h(M_transpose1_ptr, wargs->A,
              start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_transpose2_ptr,
              wargs->B, start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  
  wait_h(wargs->thread_id, NULL, NULL);

  for (i=start_col;i < last_col; i++){
    ntt_h(&M_transpose1_ptr[(i*NWORDS_FR)<<wargs->Nrows], &wargs->roots[roffset1*NWORDS_FR], wargs->Nrows, 1,  &ff);
  }
  for (i=start_col;i < last_col; i++){
    ntt_h(&M_transpose2_ptr[(i*NWORDS_FR)<<wargs->Nrows], &wargs->roots[roffset1*NWORDS_FR], wargs->Nrows, 1,  &ff);
  }
  wait_h(wargs->thread_id, NULL, NULL);

  
  // A[i] = A[i] * l2_W[i]
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_transpose1_ptr[(i+1) * NWORDS_FR]);
    }
    ridx1 = (wargs->rstride * (i >> wargs->Nrows) * (i & (Anrows - 1))) & (wargs->rstride * Anrows * Ancols - 1);
    mulm_cb(&M_transpose1_ptr[i * NWORDS_FR],
               &M_transpose1_ptr[i * NWORDS_FR],
               &wargs->roots[ridx1 * NWORDS_FR]);
  }
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_transpose2_ptr[(i+1) * NWORDS_FR]);
    }
    ridx1 = (wargs->rstride * (i >> wargs->Nrows) * (i & (Anrows - 1))) & (wargs->rstride * Anrows * Ancols - 1);
    mulm_cb(&M_transpose2_ptr[i * NWORDS_FR],
               &M_transpose2_ptr[i * NWORDS_FR],
               &wargs->roots[ridx1 * NWORDS_FR]);
  }
  wait_h(wargs->thread_id, NULL, NULL);

  transposeBlock_h(wargs->A, M_transpose1_ptr,
              start_col, last_col,
              1<<wargs->Ncols, 1<<wargs->Nrows,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(wargs->B,M_transpose2_ptr,
              start_col, last_col,
              1<<wargs->Ncols, 1<<wargs->Nrows,
              TRANSPOSE_BLOCK_SIZE);



  wait_h(wargs->thread_id, NULL, NULL);

  // A = FFT_N/2(A).T; B = FFT_N/2(B).T
  for (i=start_row;i < last_row; i++){
    ntt_h(&wargs->A[(i*NWORDS_FR)<<wargs->Ncols], &wargs->roots[roffset2*NWORDS_FR], wargs->Ncols, 1, &ff);
  }
  for (i=start_row;i < last_row; i++){
    ntt_h(&wargs->B[(i*NWORDS_FR)<<wargs->Ncols], &wargs->roots[roffset2*NWORDS_FR], wargs->Ncols, 1, &ff);
  }

  M_transpose1_ptr = wargs->A;
  M_transpose2_ptr = wargs->B;
  wargs->A=save_A;
  wargs->B=save_B;

  transposeBlock_h(wargs->A,M_transpose1_ptr, start_row, last_row,1<<wargs->Nrows, 1<<wargs->Ncols, TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(wargs->B,M_transpose2_ptr, start_row, last_row,1<<wargs->Nrows, 1<<wargs->Ncols, TRANSPOSE_BLOCK_SIZE);
  wait_h(wargs->thread_id, NULL, NULL);


  // M_mul[2*i+1] = A * B
  for (i=wargs->start_idx; i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_mul[(2*(i+1)+1) * NWORDS_FR]);
    }
    mulm_cb(&M_mul[(2*i+1) * NWORDS_FR],
               &wargs->A[i * NWORDS_FR],
               &wargs->B[i * NWORDS_FR]);
  }
  wait_h(wargs->thread_id, NULL, NULL);

  start_col = wargs->start_idx>>(wargs->mNrows-1), last_col = wargs->last_idx>>(wargs->mNrows-1);
  start_row = wargs->start_idx>>(wargs->mNcols-1), last_row = wargs->last_idx>>(wargs->mNcols-1);

  transposeBlock_h(M_transpose, M_mul,
              start_row, last_row,
              1<<wargs->mNrows, 1<<wargs->mNcols,
              TRANSPOSE_BLOCK_SIZE);

 
  wait_h(wargs->thread_id, NULL, NULL);

  // A = IFFT_N(A); B = IFFT_N(B)
  for (i=start_col;i < last_col; i++){
    ntt_h(&M_transpose[(i*NWORDS_FR)<<wargs->mNrows], &wargs->roots[roffset3*NWORDS_FR], wargs->mNrows, -1, &ff);
  }
  wait_h(wargs->thread_id, NULL, NULL);

  // A[i] = A[i] * l2_IW[i]
  for (i=wargs->start_idx*2;i < wargs->last_idx*2; i++){
    ridx1 = ( (wargs->rstride>>1) * (i >> wargs->mNrows) * (i & (Amnrows - 1)) * -1) & ((wargs->rstride>>1) * Amnrows * Amncols - 1);
    mulm_cb(&M_transpose[i * NWORDS_FR],
               &M_transpose[i * NWORDS_FR],
               &wargs->roots[ridx1 * NWORDS_FR]);
  }
  wait_h(wargs->thread_id, NULL, NULL);

  transposeBlock_h(M_mul, M_transpose,
              start_col, last_col,
              1<<wargs->mNcols, 1<<wargs->mNrows,
              TRANSPOSE_BLOCK_SIZE);

  wait_h(wargs->thread_id, NULL, NULL);

  for (i=start_row;i < last_row; i++){
    ntt_h(&M_mul[(i*NWORDS_FR)<<wargs->mNcols], &wargs->roots[roffset4*NWORDS_FR], wargs->mNcols, -1, &ff);
  }
  wait_h(wargs->thread_id, NULL, NULL);

  for (i=wargs->start_idx*2;i < wargs->last_idx*2; i++){
      mulm_cb(&M_mul[i * NWORDS_FR],
                 &M_mul[i * NWORDS_FR],
                 &scaler_ext[(wargs->mNrows + wargs->mNcols) * NWORDS_FR]);
  }

  transposeBlock_h(M_transpose,M_mul,start_row, last_row, 1<<wargs->mNrows, 1<<wargs->mNcols, TRANSPOSE_BLOCK_SIZE);
  wait_h(wargs->thread_id, NULL, NULL);

  return NULL;
}

void *interpol_and_muls_h(void *args)
{
  ntt_interpolandmul_t *wargs = (ntt_interpolandmul_t *)args;
  unsigned long long start_col = wargs->start_idx>>wargs->Nrows, last_col = wargs->last_idx>>wargs->Nrows;
  unsigned long long start_row = wargs->start_idx>>wargs->Ncols, last_row = wargs->last_idx>>wargs->Ncols;
  uint32_t Ancols = 1 << wargs->Ncols;
  uint32_t Anrows = 1 << wargs->Nrows;
  uint32_t *save_A, *save_B, *save_C;
  const uint32_t *scaler_mont = CusnarksIScalerGet((fmt_t)1);
  int64_t ridx;
  unsigned long long i;
  unsigned long long ridx1;
  unsigned long long B_offset = (1ull*NWORDS_FR)<<(wargs->Nrows+wargs->Ncols);
  unsigned long long C_offset = (1ull*NWORDS_FR)<<(wargs->Nrows+wargs->Ncols);
  uint32_t *M_transpose1_ptr = M_transpose;
  uint32_t *M_transpose2_ptr = &M_transpose[B_offset];
  uint32_t *M_transpose3_ptr = &M_mul[C_offset];
  unsigned long long roffset1, roffset2;
  t_ff  ff = {
             .subm_cb = getcb_subm_h(wargs->pidx),
             .addm_cb = getcb_addm_h(wargs->pidx),
             .mulm_cb = getcb_mulm_h(wargs->pidx)
  };

  t_mulm mulm_cb =  ff.mulm_cb;
  t_addm subm_cb =  ff.subm_cb;
  t_frommont fromm_cb = getcb_frommont_h(wargs->pidx);
  
  roffset1 = wargs->Ncols + wargs->Nrows;
  if  (roffset1 % 2){
     roffset1 = 1 << (roffset1+1);
     roffset2 = roffset1 + (1<<wargs->Nrows);
  } else {
     roffset1 = 1 << (roffset1+1); 
     roffset2 = roffset1;
  }
  wait_h(wargs->thread_id, NULL, NULL);

  // multiply M_mul[i] = A * B = polC
  for (i=wargs->start_idx; i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_mul[((i+1)) * NWORDS_FR]);
      __builtin_prefetch(&wargs->A[(i+1) * NWORDS_FR]);
      __builtin_prefetch(&wargs->B[(i+1) * NWORDS_FR]);
    }
    mulm_cb(&M_mul[i * NWORDS_FR],
               &wargs->A[i * NWORDS_FR],
               &wargs->B[i * NWORDS_FR]);
  }

  wait_h(wargs->thread_id, NULL, NULL);

  // A = IFFT_N/2(A); B = IFFT_N/2(B); C = IFFT_N/2(M) . step 1
  transposeBlock_h(M_transpose1_ptr, wargs->A,
              start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_transpose2_ptr,
              wargs->B, start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_transpose3_ptr,
              M_mul, start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);

  wait_h(wargs->thread_id, NULL, NULL);

  //printf("N rows %d\n", wargs->Nrows);
  for (i=start_col; i<last_col; i++){
    ntt_h(&M_transpose1_ptr[(i*NWORDS_FR)<<wargs->Nrows], &wargs->roots[roffset1*NWORDS_FR], wargs->Nrows, -1,  &ff);
  }
  for (i=start_col; i<last_col; i++){
    ntt_h(&M_transpose2_ptr[(i*NWORDS_FR)<<wargs->Nrows], &wargs->roots[roffset1*NWORDS_FR], wargs->Nrows, -1,  &ff);
  }
  for (i=start_col; i<last_col; i++){
    ntt_h(&M_transpose3_ptr[(i*NWORDS_FR)<<wargs->Nrows], &wargs->roots[roffset1*NWORDS_FR], wargs->Nrows, -1,  &ff);
  }

  wait_h(wargs->thread_id, NULL, NULL);
  // A[i] = A[i] * l2_IW[i]

  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_transpose1_ptr[(i+1) * NWORDS_FR]);
    }
    ridx1 = (wargs->rstride * (i >> wargs->Nrows) * (i & (Anrows - 1)) * -1) & (wargs->rstride * Anrows * Ancols - 1);
    mulm_cb(&M_transpose1_ptr[i * NWORDS_FR],
               &M_transpose1_ptr[i * NWORDS_FR],
               &wargs->roots[ridx1 * NWORDS_FR]);
  }
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_transpose2_ptr[(i+1) * NWORDS_FR]);
    }
    ridx1 = (wargs->rstride * (i >> wargs->Nrows) * (i & (Anrows - 1)) * -1) & (wargs->rstride * Anrows * Ancols - 1);
    mulm_cb(&M_transpose2_ptr[i * NWORDS_FR],
               &M_transpose2_ptr[i * NWORDS_FR],
               &wargs->roots[ridx1 * NWORDS_FR]);
  }
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_transpose3_ptr[(i+1) * NWORDS_FR]);
    }
    ridx1 = (wargs->rstride * (i >> wargs->Nrows) * (i & (Anrows - 1)) * -1) & (wargs->rstride * Anrows * Ancols - 1);
    mulm_cb(&M_transpose3_ptr[i * NWORDS_FR],
               &M_transpose3_ptr[i * NWORDS_FR],
               &wargs->roots[ridx1 * NWORDS_FR]);
  }


  wait_h(wargs->thread_id, NULL, NULL);
  transposeBlock_h(wargs->A, M_transpose1_ptr,
              start_col, last_col,
              1<<wargs->Ncols, 1<<wargs->Nrows,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(wargs->B,M_transpose2_ptr,
              start_col, last_col,
              1<<wargs->Ncols, 1<<wargs->Nrows,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_mul,M_transpose3_ptr,
              start_col, last_col,
              1<<wargs->Ncols, 1<<wargs->Nrows,
              TRANSPOSE_BLOCK_SIZE);

  wait_h(wargs->thread_id, NULL, NULL);


  // A[i] = IFFT_N/2(A).T; B[i] = IFFT_N/2(B).T; C[i] = IFFT_N/2(M).T step 2
  for (i=start_row;i < last_row; i++){
    ntt_h(&wargs->A[(i*NWORDS_FR)<<wargs->Ncols], &wargs->roots[roffset2*NWORDS_FR], wargs->Ncols, -1, &ff);
  }
  for (i=start_row;i < last_row; i++){
    ntt_h(&wargs->B[(i*NWORDS_FR)<<wargs->Ncols],&wargs->roots[roffset2*NWORDS_FR], wargs->Ncols, -1, &ff);
  }
  for (i=start_row;i < last_row; i++){
    ntt_h(&M_mul[(i*NWORDS_FR)<<wargs->Ncols],&wargs->roots[roffset2*NWORDS_FR], wargs->Ncols, -1, &ff);
  }


  wait_h(wargs->thread_id, NULL, NULL);
  transposeBlock_h(M_transpose1_ptr, wargs->A,
              start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_transpose2_ptr,
              wargs->B, start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_transpose3_ptr,
              M_mul, start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);

  wait_h(wargs->thread_id, NULL, NULL);

  save_A = wargs->A;
  save_B = wargs->B;
  save_C = M_mul;
  wargs->A = M_transpose1_ptr;
  wargs->B = M_transpose2_ptr;
  uint32_t *M_mul_ptr = M_transpose3_ptr;
  M_transpose1_ptr = save_A;
  M_transpose2_ptr = save_B;
  M_transpose3_ptr = save_C;

  // A = A * scaler * l3W; B = B * scaler * l3W; C= C * scale * l3V
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&wargs->A[(i+1) * NWORDS_FR]);
      __builtin_prefetch(&wargs->roots[(i+1) * NWORDS_FR]);
    }
      mulm_cb(&wargs->A[i * NWORDS_FR],
                 &wargs->A[i * NWORDS_FR],
                 &scaler_mont[(wargs->Nrows + wargs->Ncols) * NWORDS_FR]);
      mulm_cb(&wargs->A[i * NWORDS_FR],
                 &wargs->A[i * NWORDS_FR],
                 &wargs->roots[i * NWORDS_FR]);
  }

  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&wargs->B[(i+1) * NWORDS_FR]);
      __builtin_prefetch(&wargs->roots[(i+1) * NWORDS_FR]);
    }
      mulm_cb(&wargs->B[i * NWORDS_FR],
                 &wargs->B[i * NWORDS_FR],
                 &scaler_mont[(wargs->Nrows + wargs->Ncols) * NWORDS_FR]);
      mulm_cb(&wargs->B[i * NWORDS_FR],
                 &wargs->B[i * NWORDS_FR],
                 &wargs->roots[i * NWORDS_FR]);
  }
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_mul[(i+1) * NWORDS_FR]);
      __builtin_prefetch(&wargs->roots[(i+1) * NWORDS_FR]);
    }
      mulm_cb(&M_mul_ptr[i * NWORDS_FR],
                 &M_mul_ptr[i * NWORDS_FR],
                 &scaler_mont[(wargs->Nrows + wargs->Ncols) * NWORDS_FR]);
      mulm_cb(&M_mul_ptr[i * NWORDS_FR],
                 &M_mul_ptr[i * NWORDS_FR],
                 &wargs->roots[i * NWORDS_FR]);
  }
  wait_h(wargs->thread_id, NULL, NULL);


  // A = FFT_N/2(A); B = FFT_N/2(B); C = FFT_N/2(C). Step 1
  transposeBlock_h(M_transpose1_ptr, wargs->A,
              start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_transpose2_ptr,
              wargs->B, start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_transpose3_ptr,
              M_mul_ptr, start_row, last_row,
              1<<wargs->Nrows, 1<<wargs->Ncols,
              TRANSPOSE_BLOCK_SIZE);
  
  wait_h(wargs->thread_id, NULL, NULL);

  for (i=start_col;i < last_col; i++){
    ntt_h(&M_transpose1_ptr[(i*NWORDS_FR)<<wargs->Nrows], &wargs->roots[roffset1*NWORDS_FR], wargs->Nrows, 1,  &ff);
  }
  for (i=start_col;i < last_col; i++){
    ntt_h(&M_transpose2_ptr[(i*NWORDS_FR)<<wargs->Nrows], &wargs->roots[roffset1*NWORDS_FR], wargs->Nrows, 1,  &ff);
  }
  for (i=start_col;i < last_col; i++){
    ntt_h(&M_transpose3_ptr[(i*NWORDS_FR)<<wargs->Nrows], &wargs->roots[roffset1*NWORDS_FR], wargs->Nrows, 1,  &ff);
  }
  wait_h(wargs->thread_id, NULL, NULL);
  
  // A[i] = A[i] * l2_W[i]
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_transpose1_ptr[(i+1) * NWORDS_FR]);
    }
    ridx1 = (wargs->rstride * (i >> wargs->Nrows) * (i & (Anrows - 1))) & (wargs->rstride * Anrows * Ancols - 1);
    mulm_cb(&M_transpose1_ptr[i * NWORDS_FR],
               &M_transpose1_ptr[i * NWORDS_FR],
               &wargs->roots[ridx1 * NWORDS_FR]);
  }
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_transpose2_ptr[(i+1) * NWORDS_FR]);
    }
    ridx1 = (wargs->rstride * (i >> wargs->Nrows) * (i & (Anrows - 1))) & (wargs->rstride * Anrows * Ancols - 1);
    mulm_cb(&M_transpose2_ptr[i * NWORDS_FR],
               &M_transpose2_ptr[i * NWORDS_FR],
               &wargs->roots[ridx1 * NWORDS_FR]);
  }
  for (i=wargs->start_idx;i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_transpose3_ptr[(i+1) * NWORDS_FR]);
    }
    ridx1 = (wargs->rstride * (i >> wargs->Nrows) * (i & (Anrows - 1))) & (wargs->rstride * Anrows * Ancols - 1);
    mulm_cb(&M_transpose3_ptr[i * NWORDS_FR],
               &M_transpose3_ptr[i * NWORDS_FR],
               &wargs->roots[ridx1 * NWORDS_FR]);
  }
  wait_h(wargs->thread_id, NULL, NULL);

  transposeBlock_h(wargs->A, M_transpose1_ptr,
              start_col, last_col,
              1<<wargs->Ncols, 1<<wargs->Nrows,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(wargs->B,M_transpose2_ptr,
              start_col, last_col,
              1<<wargs->Ncols, 1<<wargs->Nrows,
              TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_mul_ptr,M_transpose3_ptr,
              start_col, last_col,
              1<<wargs->Ncols, 1<<wargs->Nrows,
              TRANSPOSE_BLOCK_SIZE);



  wait_h(wargs->thread_id, NULL, NULL);

  // A = FFT_N/2(A).T; B = FFT_N/2(B).T ; C= FFT_N/2(C).T . Step 2
  for (i=start_row;i < last_row; i++){
    ntt_h(&wargs->A[(i*NWORDS_FR)<<wargs->Ncols], &wargs->roots[roffset2*NWORDS_FR], wargs->Ncols, 1, &ff);
  }
  for (i=start_row;i < last_row; i++){
    ntt_h(&wargs->B[(i*NWORDS_FR)<<wargs->Ncols], &wargs->roots[roffset2*NWORDS_FR], wargs->Ncols, 1, &ff);
  }
  for (i=start_row;i < last_row; i++){
    ntt_h(&M_mul_ptr[(i*NWORDS_FR)<<wargs->Ncols], &wargs->roots[roffset2*NWORDS_FR], wargs->Ncols, 1, &ff);
  }

  wait_h(wargs->thread_id, NULL, NULL);
  M_transpose1_ptr = wargs->A;
  M_transpose2_ptr = wargs->B;
  M_transpose3_ptr = M_mul_ptr;
  wargs->A=save_A;
  wargs->B=save_B;
  M_mul_ptr=save_C;

  transposeBlock_h(wargs->A,M_transpose1_ptr, start_row, last_row,1<<wargs->Nrows, 1<<wargs->Ncols, TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(wargs->B,M_transpose2_ptr, start_row, last_row,1<<wargs->Nrows, 1<<wargs->Ncols, TRANSPOSE_BLOCK_SIZE);
  transposeBlock_h(M_mul_ptr,M_transpose3_ptr, start_row, last_row,1<<wargs->Nrows, 1<<wargs->Ncols, TRANSPOSE_BLOCK_SIZE);
  wait_h(wargs->thread_id, NULL, NULL);

  // tmp = A * B
  for (i=wargs->start_idx; i < wargs->last_idx; i++){
    if (i < wargs->last_idx -1){
      __builtin_prefetch(&M_mul_ptr[(i+1) * NWORDS_FR]);
      __builtin_prefetch(&wargs->A[(i+1) * NWORDS_FR]);
      __builtin_prefetch(&wargs->B[(i+1) * NWORDS_FR]);
    }
    mulm_cb(&wargs->A[i*NWORDS_FR],
               &wargs->A[i * NWORDS_FR],
               &wargs->B[i * NWORDS_FR]);
    subm_cb(&M_mul_ptr[i*NWORDS_FR],
               &wargs->A[i * NWORDS_FR],
               &M_mul_ptr[i * NWORDS_FR]);
    fromm_cb(&M_transpose[i*NWORDS_FR], &M_mul_ptr[i*NWORDS_FR]);
  }

  return NULL;
}

uint32_t *ntt_interpolandmul_parallel_h(uint32_t *A, uint32_t *B, const uint32_t *roots, uint32_t Nrows, uint32_t Ncols, t_uint64 rstride, uint32_t pidx)
{
  uint32_t mNrows, mNcols;
  ntt_interpolandmul_init_h(A, B, &mNrows, &mNcols, Nrows, Ncols);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&M_mul[(2*i) * NWORDS_FR], &A[i * NWORDS_FR], &B[i * NWORDS_FR], pidx);
  }

  interpol_parallel_odd_h(A, roots, Nrows, Ncols, rstride, pidx);
  interpol_parallel_odd_h(B, roots, Nrows, Ncols, rstride, pidx);

  #ifndef TEST_MODE
    #pragma omp parallel for if(parallelism_enabled)
  #endif
  for (uint32_t i=0;i < 1 << (Nrows + Ncols); i++){
      montmult_h(&M_mul[(2*i+1) * NWORDS_FR], &A[i * NWORDS_FR], &B[i * NWORDS_FR], pidx);
  }

  intt_parallel_h(M_mul, roots, 0, mNrows, mNcols, 1, FFT_T_DIF, pidx);
  
  /*
  for(uint32_t i=0; i < 1<<(mNcols + mNrows); i++){
    printf("[%d] : ",i);
    printUBINumber(&M_mul[i*NWORDS_FR], NWORDS_FR);
  }
  */
 
 
  return M_mul;
}

/*

  Generate format of FFT from number of samples. Parameters include 1D FFT/2D FFT/ 3D FFT/ 4D FFT, size of 
   matrix for multi D FFT,...

  fft_params_t *ntt_params : pointer to structure containing resulting FFT format
  uint32_t nsamples : number of samples of FFT
  
*/
void ntt_build_h(fft_params_t *ntt_params, uint32_t nsamples)
{
  uint32_t min_samples = 1 << 6;
  uint32_t levels = (uint32_t) ceil(log2(MAX(nsamples, min_samples)));
  memset(ntt_params,0,sizeof(fft_params_t));
  ntt_params->padding = (1 << levels) - nsamples;
  ntt_params->levels = levels;
  
  if (MAX(min_samples, nsamples) <= 32){
    ntt_params->fft_type =  FFT_T_1D;
    ntt_params->fft_N[0] = levels;

// } else if (nsamples <= 1024) {
//    ntt_params->fft_type =  FFT_T_2D;
//    ntt_params->fft_N[(1<<FFT_T_2D)-1] = levels/2;
//    ntt_params->fft_N[(1<<FFT_T_2D)-2] = levels - levels/2;

  } else if (MAX(min_samples, nsamples) <= (1<<20)){
  //} else if (MAX(min_samples, nsamples) <= (1<<12)){
    ntt_params->fft_type =  FFT_T_3D;
    ntt_params->fft_N[(1<<FFT_T_3D)-1] = levels/2;
    ntt_params->fft_N[(1<<FFT_T_3D)-2] = levels - levels/2;
    ntt_params->fft_N[(1<<FFT_T_3D)-3] = levels/4;
    ntt_params->fft_N[(1<<FFT_T_3D)-4] = (levels - levels/2)/2;

  } else {
    ntt_params->fft_type =  FFT_T_4D;
    ntt_params->fft_N[(1<<FFT_T_4D)-1] = levels/2;
    ntt_params->fft_N[(1<<FFT_T_4D)-2] = levels - levels/2;
    ntt_params->fft_N[(1<<FFT_T_4D)-3] = levels/4;
    ntt_params->fft_N[(1<<FFT_T_4D)-4] = (levels - levels/2)/2;
    
    levels = ntt_params->fft_N[(1<<FFT_T_4D)-1];
    ntt_params->fft_N[(1<<FFT_T_4D)-5] = levels/2;
    ntt_params->fft_N[(1<<FFT_T_4D)-6] = levels - levels/2;
    ntt_params->fft_N[(1<<FFT_T_4D)-7] = levels/4;
    ntt_params->fft_N[(1<<FFT_T_4D)-8] = (levels - levels/2)/2;
  }
}

/*
  Computes N roots of unity from a given primitive root. Roots are in montgomery format

  uint32_t *roots : Output vector containing computed roots. Size of vector is nroots
  uint32_t *primitive_root : Primitive root 
  uint32_t nroots : Number of roots
  uint32_t pidx    : index of 256 modulo to be used. Modulos are retrieved from CusnarksNPGet(pidx)
*/
void find_roots_h(uint32_t *roots, const uint32_t *primitive_root, uint32_t nroots, uint32_t pidx)
{
  uint32_t i;
  const uint32_t *_1 = CusnarksOneMontGet((mod_t)pidx);
  
  memcpy(roots,_1,sizeof(uint32_t)*NWORDS_FR);
  memcpy(&roots[NWORDS_FR],primitive_root,sizeof(uint32_t)*NWORDS_FR);
  for (i=2;i<nroots; i++){
    montmult_h(&roots[i*NWORDS_FR], &roots[(i-1)*NWORDS_FR], primitive_root, pidx);
  }

  return;
}

/*
  Compute Roots of unity
*/
void computeRoots_h(uint32_t *roots, uint32_t nbits)
{
  uint32_t i, pidx = MOD_FR;
  const  uint32_t *One = CusnarksOneMontGet((mod_t)pidx);
  const uint32_t *proots = CusnarksPrimitiveRootsFieldGet(nbits);
  FILE *ifp = fopen(config_curve_filename,"r");
  char curve[100];
  fgets(curve,sizeof(curve) , ifp);
  fclose(ifp);

  if (strcmp(curve,"_BN256\n")==0) {
       uint32_t prootBN256[NWORDS_FR];
       memcpy(prootBN256, &rootsBN256[nbits*NWORDS_FR*sizeof(uint32_t)],NWORDS_FR*sizeof(uint32_t));
       proots = prootBN256;
  } 
  
  memcpy(roots, One, NWORDS_FR*sizeof(uint32_t));
  if (nbits > 1){
     memcpy(&roots[NWORDS_FR], proots, NWORDS_FR*sizeof(uint32_t));
  }

  for(i=2; i < (1 << nbits); i++){
    montmult_h(&roots[i*NWORDS_FR],&roots[(i-1)*NWORDS_FR], proots, pidx);
  }
}


