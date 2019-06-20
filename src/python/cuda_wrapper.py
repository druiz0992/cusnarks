"""
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

#  NOTES:
//
//
// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : cuda_wrapper
//
// Date       : 06/06/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Python wrapper for some cuda functions
# 


// Description:
//    
//   TODO
//    
// ------------------------------------------------------------------

"""
import os.path
import numpy as np
import time

from zutils import ZUtils
from random import randint
from zfield import *
from ecc import *
from zpoly import *
from constants import *
from pysnarks_utils import *

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
except ImportError:
  sys.exit()

def zpoly_div_cuda(pysnark, poly ,n, fidx):
     nd = (1<<  int(math.ceil(math.log(n+1, 2))) )- 1 - n
     ne = n + nd
     nsamples = len(poly) + nd
     zpoly_vector = np.zeros((nsamples,NWORDS_256BIT),dtype=np.uint32)
     zpoly_vector[nd:] = poly

     kernel_config={}
     kernel_params={}
     kernel_params['in_length'] = [nsamples]
     kernel_params['out_length'] = nsamples - 2*ne + nd 
     kernel_params['stride'] = [1]
     kernel_params['premod'] = [0]
     kernel_params['midx'] = [fidx]
     kernel_params['padding_idx'] = [ne]
     kernel_params['forward'] = [nd]

     kernel_config['smemS'] = [0]
     kernel_config['blockD'] = [256]
     kernel_config['gridD'] = \
                 [(kernel_config['blockD'][0] + \
                   kernel_params['in_length'][0]-2*ne+nd - 1)/ kernel_config['blockD'][0]]
     kernel_config['kernel_idx']= [CB_ZPOLY_DIVSNARKS]

     result_snarks,t = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,1)

     result_snarks_complete = np.zeros((nsamples-ne,NWORDS_256BIT),dtype=np.uint32)
     result_snarks_complete[:nsamples-2*ne+nd] = result_snarks
     result_snarks_complete[nsamples-2*ne+nd:] = zpoly_vector[-ne+nd:]

     return result_snarks_complete, t
 
def ec2_sc1mul_cuda(pysnark, vector, fidx):
     kernel_params={}
     kernel_config={}
     nsamples = len(vector)

     kernel_params['stride'] = [ECP2_JAC_INDIMS]
     kernel_config['smemS'] =  [0]
     kernel_config['blockD'] = [256]
     kernel_params['premul'] = [0]
     kernel_params['premod'] = [0]
     kernel_params['midx'] = [fidx]
     kernel_config['kernel_idx'] = [CB_EC2_MUL1]
     kernel_params['in_length'] = [nsamples]
     kernel_params['out_length'] = nsaples-1
     kernel_params['padding_idx'] = [0]
     kernel_config['gridD'] = [0]
     kernel_config['return_val']=[1]

     result,t = pysnark.kernelLaunch(vector, kernel_config, kernel_params,1 )
     
     return result,t
     
def ec_sc1mul_cuda(pysnark, vector, fidx):
     kernel_params={}
     kernel_config={}
     nsamples = len(vector)

     kernel_params['stride'] = [ECP_JAC_INDIMS]
     kernel_config['smemS'] =  [0]
     kernel_config['blockD'] = [256]
     kernel_params['premul'] = [0]
     kernel_params['premod'] = [0]
     kernel_params['midx'] = [fidx]
     kernel_config['kernel_idx'] = [CB_EC_MUL1]
     kernel_params['in_length'] = [nsamples]
     kernel_params['out_length'] = nsaples-1
     kernel_params['padding_idx'] = [0]
     kernel_config['gridD'] = [0]
     kernel_config['return_val']=[1]

     result,t = pysnark.kernelLaunch(vector, kernel_config, kernel_params,1 )

def ec2_mad_cuda(pysnark, vector, fidx):
     kernel_params={}
     kernel_config={}
     nsamples = len(vector)/5

     kernel_params['stride'] = [ECP2_JAC_INDIMS+U256_NDIMS, ECP2_JAC_OUTDIMS, ECP2_JAC_OUTDIMS]
     #kernel_config['blockD'] = [256,32]
     kernel_config['blockD'] = [256,128,32]
     kernel_params['premul'] = [1,0, 0]
     kernel_params['premod'] = [0,0, 0]
     kernel_params['midx'] = [fidx, fidx, fidx]
     kernel_config['smemS'] = [kernel_config['blockD'][0]/32 * NWORDS_256BIT * ECP2_JAC_OUTDIMS * 4, \
                               kernel_config['blockD'][1]/32 * NWORDS_256BIT * ECP2_JAC_OUTDIMS * 4, \
                               kernel_config['blockD'][2]/32 * NWORDS_256BIT * ECP2_JAC_OUTDIMS * 4]
     kernel_config['kernel_idx'] = [CB_EC2_JAC_MAD_SHFL, CB_EC2_JAC_MAD_SHFL, CB_EC2_JAC_MAD_SHFL]
     out_len1 = ECP2_JAC_OUTDIMS * ((nsamples + (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP2_JAC_OUTDIMS) -1) /
                                   (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP2_JAC_OUTDIMS))
     out_len2 = ECP2_JAC_OUTDIMS * ((out_len1 + (kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP2_JAC_OUTDIMS) -1) /
                                   (kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP2_JAC_OUTDIMS))
     kernel_params['in_length'] = [nsamples * (ECP2_JAC_INDIMS+U256_NDIMS), out_len1, out_len2]
     kernel_params['out_length'] = 1 * ECP2_JAC_OUTDIMS
     kernel_params['padding_idx'] = [0,0, 0]
     kernel_config['gridD'] = [0,1, 1]
     kernel_config['return_val']=[1,1,1]
     min_length = [ECP2_JAC_OUTDIMS * \
             (kernel_config['blockD'][idx] * kernel_params['stride'][idx]/ECP2_JAC_OUTDIMS) for idx in range(len(kernel_params['stride']))]

     v_mad = np.copy(vector)
     for bidx, l in enumerate(kernel_params['in_length']):
        if l < min_length[bidx]:
           if bidx == 0:
              zeros = np.zeros((min_length[bidx] - kernel_params['in_length'][bidx],NWORDS_256BIT), dtype=np.uint32)
              v_mad = np.concatenate((vector,zeros))
              kernel_params['in_length'][bidx] = min_length[bidx]
           else:
              kernel_params['in_length'][bidx] = min_length[bidx]
              kernel_params['padding_idx'][bidx] = l/ECP2_JAC_OUTDIMS

     result,t = pysnark.kernelLaunch(v_mad, kernel_config, kernel_params,3 )
     
     return result,t
     
def ec_mad_cuda(pysnark, vector, fidx):
   kernel_params={}
   kernel_config={}
   nsamples = len(vector)/3

   kernel_params['stride'] = [ECP_JAC_OUTDIMS, ECP_JAC_OUTDIMS, ECP_JAC_OUTDIMS]
   #kernel_config['blockD'] = [256,32]
   kernel_config['blockD'] = [256,128,32]
   kernel_params['premul'] = [1,0,0]
   kernel_params['premod'] = [0,0,0]
   kernel_params['midx'] = [fidx, fidx, fidx]
   kernel_config['smemS'] = [kernel_config['blockD'][0]/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4, \
                             kernel_config['blockD'][1]/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4, \
                             kernel_config['blockD'][2]/32 * NWORDS_256BIT * ECP_JAC_OUTDIMS * 4]
   kernel_config['kernel_idx'] = [CB_EC_JAC_MAD_SHFL, CB_EC_JAC_MAD_SHFL, CB_EC_JAC_MAD_SHFL]
   out_len1 = ECP_JAC_OUTDIMS * ((nsamples + (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS) -1) /
                                 (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS))
   out_len2 = ECP_JAC_OUTDIMS * ((out_len1 + (kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP_JAC_OUTDIMS) -1) /
                                 (kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP_JAC_OUTDIMS))
   kernel_params['in_length'] = [nsamples * (ECP_JAC_INDIMS+U256_NDIMS), out_len1, out_len2]
   kernel_params['out_length'] = 1 * ECP_JAC_OUTDIMS
   kernel_params['padding_idx'] = [0,0,0]
   kernel_config['gridD'] = [0,1,1]
   min_length = [ECP_JAC_OUTDIMS * \
           (kernel_config['blockD'][idx] * kernel_params['stride'][idx]/ECP_JAC_OUTDIMS) for idx in range(len(kernel_params['stride']))]

   v_mad = np.copy(vector)
   for bidx, l in enumerate(kernel_params['in_length']):
      if l < min_length[bidx]:
         if bidx == 0:
            zeros = np.zeros((min_length[bidx] - kernel_params['in_length'][bidx],NWORDS_256BIT), dtype=np.uint32)
            v_mad = np.concatenate((vector,zeros))
            kernel_params['in_length'][bidx] = min_length[bidx]
         else:
            kernel_params['in_length'][bidx] = min_length[bidx]
            kernel_params['padding_idx'][bidx] = l/ECP_JAC_OUTDIMS

   result,t = pysnark.kernelLaunch(v_mad, kernel_config, kernel_params,3 )
   
   return result, t

def zpoly_fft_cuda(pysnark, vector, roots, fidx ):
        nsamples = len(vector)

        n_cols = 10
        n_rows = 10
        kernel_config={}
        kernel_params={}

        # Test FFT kernel:
        kernel_params['in_length'] = [2*nsamples,nsamples, nsamples, nsamples]
        kernel_params['out_length'] = nsamples
        kernel_params['stride'] = [2,1,1,1]
        kernel_params['premod'] = [0,0,0,0]
        kernel_params['midx'] = [fidx, fidx, fidx, fidx]
        kernel_params['fft_Nx'] = [5,5,5,5]
        kernel_params['fft_Ny'] = [5,5,5,5]
        kernel_params['N_fftx'] = [n_cols, n_cols, n_cols, n_cols]
        kernel_params['N_ffty'] = [n_rows, n_rows, n_rows, n_rows]
        kernel_params['forward'] = [1,1,1,1]

        kernel_config['smemS'] = [0,0,0,0]
        kernel_config['blockD'] = [256,256,256,256]
        kernel_config['gridD'] = [0, (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][1] + nsamples-1)/kernel_config['blockD'][1], \
                                     (kernel_config['blockD'][2] + nsamples-1)/kernel_config['blockD'][2]]
        kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]
        zpoly_vector = np.concatenate((vector, roots))
        result,t = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,4)

        return result, t


def zpoly_ifft_cuda(pysnark, vector, ifft_params, fidx, roots=None, as_mont=1, return_val=1, out_extra_len=0 ):
        nsamples = 1<<ifft_params['levels']
        expanded_vector = np.zeros((nsamples,NWORDS_256BIT),dtype=np.uint32)
        expanded_vector[:len(vector)] = vector
        if roots is not None:
             expanded_roots = roots[::1<<(20-ifft_params['levels'])]
             scalerMont = ZFieldElExt(len(expanded_roots)).inv().reduce().as_uint256()
             scalerExt = ZFieldElExt(len(expanded_roots)).inv().as_uint256()
             zpoly_vector = np.concatenate((expanded_vector, expanded_roots, [scalerExt],[scalerMont]))
        else :
             zpoly_vector = expanded_vector

        Nrows = ifft_params['fft_N'][(1<<FFT_T_3D)-1]
        Ncols = ifft_params['fft_N'][(1<<FFT_T_3D)-2]
        fft_yx = ifft_params['fft_N'][(1<<FFT_T_3D)-3]
        fft_yy = Nrows - fft_yx
        fft_xx = ifft_params['fft_N'][(1<<FFT_T_3D)-4]
        fft_xy = Ncols - fft_xx
        n_kernels1 = 4
        kernel_config={}
        kernel_params={}
        
        kernel_params['in_length'] = [nsamples] * n_kernels1
        kernel_params['in_length'][0] = 2*nsamples+1
        kernel_params['out_length'] = nsamples+out_extra_len
        kernel_params['stride'] = [1] * n_kernels1
        kernel_params['stride'][0] = 2
        kernel_params['premod'] = [0] * n_kernels1
        kernel_params['midx'] = [MOD_FIELD]  * n_kernels1
        kernel_params['N_fftx'] = [Ncols] * n_kernels1
        kernel_params['N_ffty'] = [Nrows] * n_kernels1
        kernel_params['fft_Nx'] = [fft_xx, fft_xx, fft_yx, fft_yx] #xx,xx,yx,yx
        kernel_params['fft_Ny'] = [fft_xy, fft_xy, fft_yy, fft_yy] #xy,xy,yy,yy
        kernel_params['forward'] = [0] * n_kernels1
        kernel_params['as_mont'] = [as_mont] * n_kernels1
  
        kernel_config['smemS'] = [0] * n_kernels1
        kernel_config['blockD'] = [256] * n_kernels1
        kernel_config['gridD'] = [(kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]*n_kernels1
        kernel_config['gridD'][0] = 0
        kernel_config['return_val'] = [return_val] * n_kernels1

        kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]

        result,t = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,4)
        if return_val == 0:
           result = nsamples

        return result,t

def zpoly_mul_cuda(pysnark, vectorA, vectorB, mul_params, fidx, roots=None, return_val=0, as_mont=1):
    nsamples = 1<<mul_params['levels']
    expanded_vectorA = np.zeros((nsamples,NWORDS_256BIT),dtype=np.uint32)
    expanded_vectorB = np.zeros((nsamples,NWORDS_256BIT),dtype=np.uint32)
    expanded_vectorA[:len(vectorA)] = vectorA
    expanded_vectorB[:len(vectorB)] = vectorB

    kernel_config={}
    kernel_params={}

    if roots is not None:
          expanded_roots = roots[::1<<(20-mul_params['levels'])]
          scalerMont = ZFieldElExt(len(expanded_roots)).inv().reduce().as_uint256()
          scalerExt = ZFieldElExt(len(expanded_roots)).inv().as_uint256()
          zpoly_vectorA = np.concatenate((expanded_vectorA, expanded_roots,[scalerExt], [scalerMont]))
    else :
          zpoly_vectorA = expanded_vectorA
    zpoly_vectorB = expanded_vectorB

    Nrows = mul_params['fft_N'][(1<<FFT_T_3D)-1]
    Ncols = mul_params['fft_N'][(1<<FFT_T_3D)-2]
    fft_yx = mul_params['fft_N'][(1<<FFT_T_3D)-3]
    fft_yy = Nrows - fft_yx
    fft_xx = mul_params['fft_N'][(1<<FFT_T_3D)-4]
    fft_xy = Ncols - fft_xx
    n_kernels1 = 4
    n_kernels2= 5

    kernel_params['in_length'] = [nsamples] * n_kernels1
    kernel_params['in_length'][0] = 2*nsamples+1
    kernel_params['out_length'] = nsamples
    kernel_params['stride'] = [1] * n_kernels1
    kernel_params['stride'][0] = 2
    kernel_params['premod'] = [0] * n_kernels1
    kernel_params['midx'] = [MOD_FIELD]  * n_kernels1
    kernel_params['N_fftx'] = [Ncols] * n_kernels1
    kernel_params['N_ffty'] = [Nrows] * n_kernels1
    kernel_params['fft_Nx'] = [fft_xx, fft_xx, fft_yx, fft_yx] #xx,xx,yx,yx
    kernel_params['fft_Ny'] = [fft_xy, fft_xy, fft_yy, fft_yy] #xy,xy,yy,yy
    kernel_params['forward'] = [1] * n_kernels1
  
    kernel_config['smemS'] = [0] * n_kernels1
    kernel_config['blockD'] = [256] * n_kernels1
    kernel_config['gridD'] = [(kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]*n_kernels1
    kernel_config['gridD'][0] = 0
    kernel_config['return_val'] = [1] * n_kernels1

    kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]

    X1S,t1 = pysnark.kernelLaunch(zpoly_vectorA, kernel_config, kernel_params,n_kernels1)

    kernel_params['in_length'][0] = nsamples
    kernel_params['stride'][0] = 1
    kernel_config['return_val'][0] = 0

    Y1S,t2 = pysnark.kernelLaunch(zpoly_vectorB, kernel_config, kernel_params,n_kernels1)

    kernel_params['in_length'] = [nsamples] * n_kernels2
    kernel_params['out_length'] = nsamples
    kernel_params['stride'] = [1] * n_kernels2
    kernel_params['premod'] = [0] * n_kernels2
    kernel_params['midx'] = [MOD_FIELD]  * n_kernels2
    kernel_params['N_fftx'] = [Ncols] * n_kernels2
    kernel_params['N_ffty'] = [Nrows] * n_kernels2
    kernel_params['fft_Nx'] = [0,fft_xx, fft_xx, fft_yx, fft_yx] #xx,xx,yx,yx
    kernel_params['fft_Ny'] = [0,fft_xy, fft_xy, fft_yy, fft_yy] #xy,xy,yy,yy
    kernel_params['forward'] = [0,0,0,0,0]
    kernel_params['as_mont'] = [as_mont] * n_kernels2
  
    kernel_config['smemS'] = [0] * n_kernels2
    kernel_config['blockD'] = [256] * n_kernels2
    kernel_config['gridD'] = [(kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]*n_kernels2
    kernel_config['gridD'][0] = 0
    kernel_config['return_val'] = [return_val] * n_kernels2
    kernel_config['kernel_idx']= [CB_ZPOLY_MULCPREV,
                                  CB_ZPOLY_FFT3DXXPREV, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY ]

    fftmul_result,t3 = pysnark.kernelLaunch(X1S, kernel_config, kernel_params,n_kernels2)
    if return_val == 0:
      fftmul_result = nsamples
  
    return fftmul_result, t1+t2+t3

def zpoly_sub_cuda(pysnark, vectorA, vectorB, fidx, vectorA_len=1, return_val=0):  

     kernel_config={}
     kernel_params={}

     if vectorA_len is 0:

        #TODO revie
        if len(vectorB) < len(vectorA): 
           vector =np.concatenate((vectorA, vectorB))
        else:
           vector =np.concatenate((vectorB, vectorA))
        nsamples = len(vector)
        kernel_params['out_length'] = nsamples/2
        kernel_params['stride'] = [2]
        kernel_params['padding_idx'] = [min(len(vectorA),len(vectorB))]
 
     else :
        vector = vectorB
        nsamples = len(vector)
        kernel_params['out_length'] = vectorA_len
        kernel_params['stride'] = [1]
         
     kernel_params['premod'] = [0]
     kernel_params['in_length'] = [nsamples]
     kernel_params['midx'] = [fidx]
     kernel_config['smemS'] = [0]
     kernel_config['blockD'] = [U256_BLOCK_DIM]
     #TODO
     #kernel_config['kernel_idx'] = [CB_ZPOLY_SUBPREV]
     kernel_config['kernel_idx'] = [CB_ZPOLY_SUB]
     kernel_config['return_val'] = [return_val] 
     result,t = pysnark.kernelLaunch(vector, kernel_config, kernel_params )

     if return_val == 0:
        result = kernel_params['out_length']

     return result,t

def u256_mul_cuda(pysnark, vectorA, vectorB, fidx):
     vector = np.zeros((int(2*len(vectorA)), NWORDS_256BIT),dtype=np.uint32)
     vector[::2] = vectorA
     vector[1::2] = vectorB
  
     kernel_params={}
     kernel_config={}

     kernel_params['in_length'] = [len(vector)]
     kernel_params['midx'] = [fidx]
     kernel_params['out_length'] = len(vector)/2
     kernel_params['stride'] = [2]
     kernel_config['smemS'] = [0]
     kernel_config['blockD'] = [U256_BLOCK_DIM]
     kernel_config['kernel_idx'] = [CB_U256_MULM]

     result, t = pysnark.kernelLaunch(vector, kernel_config, kernel_params,2 )

     return result, t


def zpoly_mulK_cuda(pysnark, vectorA, K, fidx):  
     #TODO revie
     vector =np.concatenate((K, vector))
     nsamples = len(vectorA)
     kernel_config={}
     kernel_params={}

     kernel_params['in_length'] = [nsamples+1]
     kernel_params['out_length'] = nsamples/2
     kernel_params['stride'] = [1]
     kernel_params['midx'] = [fidx]
     kernel_params['premod'] = [0]
     kernel_config['smemS'] = [0]
     kernel_config['blockD'] = [U256_BLOCK_DIM]
     kernel_config['kernel_idx'] = [CB_ZPOLY_MULK]
     result,t= pysnark.kernelLaunch(vector, kernel_config, kernel_params )

     return result,t

def zpoly_mad_cuda(pysnark, vectors, fidx):  

     kernel_config={}
     kernel_params={}
    
     """
     #TODO revie
     max_len = 0
     max_v = None
     new_v = []
     for v in vectors:
       if len(v[0]) > max_len: 
          max_len = len(v[0])
          if max_v is not None:
            new_v.append(max_v)
          max_v = np.copy(v)
       else :
          new_v.append(v)

     for v in new_v:
        vector =np.concatenate((max_v, v))


        kernel_params['in_length'] = [len(vector)]
        kernel_params['out_length'] = len(max_v)
        kernel_params['stride'] = [2]
        kernel_params['padding_idx'] = [len(v)]
        kernel_params['premod'] = [0]
        kernel_params['midx'] = [fidx]
        kernel_config['smemS'] = [0]
        kernel_config['blockD'] = [U256_BLOCK_DIM]
        kernel_config['kernel_idx'] = [CB_ZPOLY_ADD]
        kernel_config['return_val'] = [1]
        kernel_config['gridD'] = \
                 [(kernel_config['blockD'][0] + \
                   2*kernel_params['padding_idx'][0]/kernel_params['stride'][0] - 1)/ kernel_config['blockD'][0]]
        vector,t = pysnark.kernelLaunch(vector, kernel_config, kernel_params,1 )

     return vector,t
     """
