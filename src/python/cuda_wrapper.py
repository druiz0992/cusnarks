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
import math
from subprocess import call
import multiprocessing as mp
import nvgpu


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
     #kernel_params['out_length'] = nsamples - 2*ne + nd 
     kernel_params['out_length'] = len(poly) - n
     kernel_params['stride'] = [1]
     kernel_params['premod'] = [0]
     kernel_params['midx'] = [fidx]
     kernel_params['padding_idx'] = [ne]
     kernel_params['forward'] = [nd]

     kernel_config['smemS'] = [0]
     kernel_config['blockD'] = [256]
     kernel_config['gridD'] = [int((kernel_config['blockD'][0] + kernel_params['in_length'][0]-1) / kernel_config['blockD'][0])]
     #kernel_config['gridD'] = \
                 #[int((kernel_config['blockD'][0] + \
                   #kernel_params['in_length'][0]-2*ne+nd - 1)/ kernel_config['blockD'][0])]
     kernel_config['kernel_idx']= [CB_ZPOLY_DIVSNARKS]

     result_snarks,t = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,1)

     result_snarks_complete = np.zeros((nsamples-ne,NWORDS_256BIT),dtype=np.uint32)
     result_snarks_complete[:nsamples-2*ne+nd] = result_snarks
     result_snarks_complete[nsamples-2*ne+nd:] = zpoly_vector[-ne+nd:]

     ### DEBUG
     #ZUtils.NROOTS = 8192 * 2
     #proot = 20619701001583904760601357484951574588621083236087856586626117568842480512645
     #roots_ext, inv_roots_ext = ZField.find_roots(ZUtils.NROOTS, rformat_ext=True, primitive_root=proot) 
     zpoly_1 = ZPoly.from_uint256(poly)
     #zpoly_2 = np.zeros(n,dtype=np.int32)
     #zpoly_2[0] = -1
     #zpoly_2[-1] = 1
     #zpoly_2 = ZPoly(zpoly_2.tolist())
     #r = zpoly_1.poly_div(zpoly_2)
     r = zpoly_1.poly_div_snarks(n-1)
     r2 = zpoly_div_h(zpoly_vector, n, fidx);
     

     return result_snarks_complete, t
 
def ec_sc1mul_cuda(pysnark, vector, fidx, ec2=False, premul=False ):
    kernel_params={}
    kernel_config={}
   
    if ec2:
      outdims = ECP2_JAC_OUTDIMS
      indims = ECP2_JAC_INDIMS
      indims_e = ECP2_JAC_INDIMS + U256_NDIMS
      factor = 2
      kernel = CB_EC2_JAC_MUL1
    else:
      outdims = ECP_JAC_OUTDIMS 
      indims = ECP_JAC_INDIMS
      indims_e = ECP_JAC_INDIMS + U256_NDIMS
      factor = 1
      kernel = CB_EC_JAC_MUL1

    nsamples = len(vector)

    kernel_params['stride'] = [1]
    kernel_config['smemS'] =  [0]
    kernel_config['blockD'] = [256]
    kernel_params['premul'] = [0]
    if premul:
      kernel_params['premul'] = [1]

    kernel_params['premod'] = [0]
    kernel_params['midx'] = [fidx]
    kernel_config['kernel_idx'] = [kernel]
    kernel_params['in_length'] = [nsamples]
    kernel_params['out_length'] = (nsamples-indims)*outdims
    kernel_params['padding_idx'] = [0]
    kernel_config['gridD'] = [int((kernel_config['blockD'][0] + kernel_params['in_length'][0]-indims-1) /
                                kernel_config['blockD'][0])]
    kernel_config['return_val']=[1]

    result,t = pysnark.kernelLaunch(vector, kernel_config, kernel_params,1 )

    return result,t

def ec_mad_cuda(pysnark, vector, fidx, ec2=False):
   kernel_params={}
   kernel_config={}
   
   if ec2:
      outdims = ECP2_JAC_OUTDIMS
      indims = ECP2_JAC_INDIMS
      indims_e = ECP2_JAC_INDIMS + U256_NDIMS
      factor = 2
      kernel = CB_EC2_JAC_MAD_SHFL
   else:
      outdims = ECP_JAC_OUTDIMS 
      indims = ECP_JAC_INDIMS
      indims_e = ECP_JAC_INDIMS + U256_NDIMS
      factor = 1
      kernel = CB_EC_JAC_MAD_SHFL

 
   nsamples = int(len(vector)/indims_e)
   
   kernel_config['blockD']    = get_shfl_blockD(nsamples)
   nkernels = len(kernel_config['blockD'])
   new_nsamples = np.product(kernel_config['blockD'])
   new_vector = np.zeros((indims_e*new_nsamples,NWORDS_256BIT), dtype=np.uint32)
   new_vector[new_nsamples-nsamples:new_nsamples] = vector[:nsamples]
   new_vector[new_nsamples+indims*(new_nsamples -nsamples):] = vector[nsamples:]
   nsamples = np.product(kernel_config['blockD'])
   #new_vector = np.copy(vector)
   kernel_params['stride']    = [outdims] * nkernels
   kernel_params['stride'][0]    =  indims_e
   kernel_params['premul']    = [0] * nkernels
   kernel_params['premul'][0] = 1
   kernel_params['premod']    = [0] * nkernels
   kernel_params['midx']      = [fidx] * nkernels
   kernel_config['smemS']     = [int(blockD/32 * NWORDS_256BIT * outdims * 4) for blockD in kernel_config['blockD']]
   kernel_config['kernel_idx'] =[kernel] * nkernels
   #out_len1 = ECP_JAC_OUTDIMS * ((nsamples + (kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS) -1) /
                                 #(kernel_config['blockD'][0]*kernel_params['stride'][0]/ECP_JAC_OUTDIMS))
   #out_len2 = ECP_JAC_OUTDIMS * ((out_len1 + (kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP_JAC_OUTDIMS) -1) /
                                 #(kernel_config['blockD'][1]*kernel_params['stride'][1]/ECP_JAC_OUTDIMS))
   kernel_params['in_length'] = [nsamples* indims_e]*nkernels 
   for l in xrange(1,nkernels):
      #kernel_params['in_length'][l] = int(kernel_params['in_length'][l-1]/(kernel_config['blockD'][l-1]))
      kernel_params['in_length'][l] = outdims * (
             int((kernel_params['in_length'][l-1]/outdims + (kernel_config['blockD'][l-1] * kernel_params['stride'][l-1] / outdims) - 1) /
             (kernel_config['blockD'][l-1] * kernel_params['stride'][l-1] / (outdims))))

   kernel_params['out_length'] = 1 * outdims
   #kernel_params['out_length'] = nsamples * outdims
   kernel_params['padding_idx'] = [0] * nkernels
   kernel_config['gridD'] = [0] * nkernels
   kernel_config['gridD'][nkernels-1] = 1
   min_length = [outdims * \
                    int(kernel_config['blockD'][idx]) for idx in range(nkernels)]
   for i in xrange(1,nkernels):
       if min_length[i] > kernel_params['in_length'][i]:
           kernel_params['padding_idx'][i] = int(kernel_params['in_length'][i]/outdims)
           kernel_params['in_length'][i] = min_length[i]
    
   result,t = pysnark.kernelLaunch(new_vector, kernel_config, kernel_params,nkernels )
   #result,t = pysnark.kernelLaunch(new_vector, kernel_config, kernel_params,1 )

   #new_vector = new_vector.reshape((-1,8))
   #Kp = [ZFieldElExt(BigInt.from_uint256(k)) for k in new_vector[:new_nsamples]]
   #Pp = np.zeros((new_nsamples, 3, 8), dtype=np.uint32)
   #Pp[:,0] = np.reshape(new_vector[new_nsamples:],(-1,2,8))[:,0]
   #Pp[:, 1] = np.reshape(new_vector[new_nsamples:], (-1, 2, 8))[:, 1]
   #Pp[:, 2, 0] = np.ones(new_nsamples,dtype=np.uint32)
   #Pp = ECC.from_uint256(np.reshape(Pp,(-1,NWORDS_256BIT)), in_ectype=1, out_ectype=2, reduced=True)
   #NPp = np.multiply(Kp, Pp)
   #NPp = np.sum(np.multiply(Kp, Pp))
   
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

        #np.savez_compressed('../../test/python/aux_data/ifft_data.npz',ifft_data=vector)
        Nrows = ifft_params['fft_N'][(1<<FFT_T_3D)-1]
        Ncols = ifft_params['fft_N'][(1<<FFT_T_3D)-2]
        fft_yx = ifft_params['fft_N'][(1<<FFT_T_3D)-3]
        fft_yy = Nrows - fft_yx
        fft_xx = ifft_params['fft_N'][(1<<FFT_T_3D)-4]
        fft_xy = Ncols - fft_xx
        n_kernels1 = 4
        kernel_config={}
        kernel_params={}
       
        #kernel_params['padding_idx'] = [2*nsamples+2] * n_kernels1 
        kernel_params['in_length'] = [nsamples] * n_kernels1
        kernel_params['in_length'][0] = 2*nsamples+2
        kernel_params['out_length'] = nsamples+out_extra_len
        kernel_params['stride'] = [1] * n_kernels1
        kernel_params['stride'][0] = 2
        kernel_params['premod'] = [0] * n_kernels1
        kernel_params['midx'] = [fidx]  * n_kernels1
        kernel_params['N_fftx'] = [Ncols] * n_kernels1
        kernel_params['N_ffty'] = [Nrows] * n_kernels1
        kernel_params['fft_Nx'] = [fft_xx, fft_xx, fft_yx, fft_yx] #xx,xx,yx,yx
        kernel_params['fft_Ny'] = [fft_xy, fft_xy, fft_yy, fft_yy] #xy,xy,yy,yy
        kernel_params['forward'] = [0] * n_kernels1
        kernel_params['as_mont'] = [as_mont] * n_kernels1
  
        kernel_config['smemS'] = [0] * n_kernels1
        kernel_config['blockD'] = [256] * n_kernels1
        kernel_config['gridD'] = [int((kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0])]*n_kernels1
        kernel_config['return_val'] = [return_val] * n_kernels1

        kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]

        result,t = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,4)
        if return_val == 0:
           result = nsamples

        ### DEBUG
        #ROOTS_1M_filename = '../../data/zpoly_roots_1M.bin'
        #roots = readU256DataFile_h(ROOTS_1M_filename.encode("UTF-8"), 1<<20, 1<<20)
        #expanded_roots = roots[::1<<(20-ifft_params['levels'])]
        #inv_roots = np.copy(expanded_roots)
        #inv_roots[1:] = expanded_roots[::-1][:-1]
        #result2 = intt_h(expanded_vector,inv_roots,1,fidx)
        #result3 = ntt_h(result2,expanded_roots,fidx)
        #result3 = ntt_h(result,expanded_roots,fidx)
         


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
    kernel_params['in_length'][0] = 2*nsamples+2
    kernel_params['padding_idx'] = [2*nsamples+2] * n_kernels1
    kernel_params['out_length'] = nsamples
    kernel_params['stride'] = [1] * n_kernels1
    kernel_params['stride'][0] = 2
    kernel_params['premod'] = [0] * n_kernels1
    kernel_params['midx'] = [fidx]  * n_kernels1
    kernel_params['N_fftx'] = [Ncols] * n_kernels1
    kernel_params['N_ffty'] = [Nrows] * n_kernels1
    kernel_params['fft_Nx'] = [fft_xx, fft_xx, fft_yx, fft_yx] #xx,xx,yx,yx
    kernel_params['fft_Ny'] = [fft_xy, fft_xy, fft_yy, fft_yy] #xy,xy,yy,yy
    kernel_params['forward'] = [1] * n_kernels1
    kernel_params['as_mont'] = [1] * n_kernels1
  
    kernel_config['smemS'] = [0] * n_kernels1
    kernel_config['blockD'] = [256] * n_kernels1
    kernel_config['gridD'] = [int((kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0])]*n_kernels1
    #kernel_config['gridD'][0] = 0
    kernel_config['return_val'] = [1] * n_kernels1

    kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]

    X1S,t1 = pysnark.kernelLaunch(zpoly_vectorA, kernel_config, kernel_params,n_kernels1)

    kernel_params['in_length'][0] = nsamples
    kernel_params['stride'][0] = 1
    kernel_config['return_val'][0] = 1

    Y1S,t2 = pysnark.kernelLaunch(zpoly_vectorB, kernel_config, kernel_params,n_kernels1)

    kernel_params['padding_idx'] = [2*nsamples+2] * n_kernels2
    kernel_params['in_length'] = [nsamples] * n_kernels2
    kernel_params['out_length'] = nsamples
    kernel_params['stride'] = [1] * n_kernels2
    kernel_params['premod'] = [0] * n_kernels2
    kernel_params['midx'] = [fidx]  * n_kernels2
    kernel_params['N_fftx'] = [Ncols] * n_kernels2
    kernel_params['N_ffty'] = [Nrows] * n_kernels2
    kernel_params['fft_Nx'] = [0,fft_xx, fft_xx, fft_yx, fft_yx] #xx,xx,yx,yx
    kernel_params['fft_Ny'] = [0,fft_xy, fft_xy, fft_yy, fft_yy] #xy,xy,yy,yy
    kernel_params['forward'] = [0,0,0,0,0]
    kernel_params['as_mont'] = [as_mont] * n_kernels2
  
    kernel_config['smemS'] = [0] * n_kernels2
    kernel_config['blockD'] = [256] * n_kernels2
    kernel_config['gridD'] = [int((kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0])]*n_kernels2
    kernel_config['return_val'] = [return_val] * n_kernels2
    kernel_config['kernel_idx']= [CB_ZPOLY_MULCPREV,
                                  CB_ZPOLY_FFT3DXXPREV, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY ]

    fftmul_result,t3 = pysnark.kernelLaunch(X1S, kernel_config, kernel_params,n_kernels2)
    if return_val == 0:
      fftmul_result = nsamples

    ## DEBUG
    #X2S = ntt_h(expanded_vectorA,expanded_roots,fidx)
    #Y2S = ntt_h(expanded_vectorB,expanded_roots,fidx)
    #C = np.copy(Y2S)
    #idx = 0
    #inv_roots = np.copy(expanded_roots)
    #inv_roots[1:] = expanded_roots[::-1][:-1]
    #for c1,c2 in zip(X2S, Y2S):
         #C[idx] = montmult_h(c1, c2,1)
         #idx+=1
    #result = intt_h(C,inv_roots,1,fidx)
  
    return fftmul_result, t1+t2+t3

def zpoly_sub_cuda(pysnark, vectorA, vectorB, fidx, vectorA_len=1, return_val=0):  

     kernel_config={}
     kernel_params={}

     """
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
     """
         
     if len(vectorB) <= len(vectorA): 
         vector =np.concatenate((vectorA[:int(len(vectorB))], vectorB))
         copy = 1
     else:
         vector = np.zeros((2*vectorB.shape[0],vectorB.shape[1]), dtype=np.uint32)
         vector[:len(vectorA)] = vectorA
         vector[len(vectorB):] = vectorB
         copy = 0

     nsamples = len(vector)
     kernel_params['out_length'] = nsamples/2
     kernel_params['stride'] = [2]
     #kernel_params['padding_idx'] = [min(len(vectorA),len(vectorB))]
     kernel_params['padding_idx'] = [nsamples/2]
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
     elif copy:
        result = np.concatenate((result,vectorA[int(len(vectorB)):]))

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

def get_shfl_blockD(nsamples):

   l = max(math.ceil(math.log2(nsamples)),5)
   nb = math.ceil(l/8)
   lpb = math.ceil(l/nb)

   blockD = [1<<lpb] 
   rsamples = l - lpb
   while rsamples > lpb:
      blockD.append(1<<lpb)
      rsamples -= lpb

   if rsamples :
      lastb = max(rsamples,5)
      blockD.append(1<<lastb)

   blockD.sort()
 
   return blockD[::-1]
    
def get_gpu_affinity_cuda():
   available_gpus = nvgpu.available_gpus()
   n_cores = mp.cpu_count()
   gpu_affinity = {}
   for gpu in available_gpus:
     gpu_affinity[gpu] = []
   for core in xrange(n_cores):
      r = call(['nvidia-smi' , 'topo', '-c' , str(core)])
      gpu_affinity[str(r)].append(core)

   return gpu_affinity
