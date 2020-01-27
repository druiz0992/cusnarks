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
from subprocess import call, Popen, PIPE
import multiprocessing as mp
import nvgpu


from zutils import ZUtils
from random import randint
from zfield import *
from ecc import *
from zpoly import *
from constants import *
from pysnarks_utils import *


sys.path.append(os.path.abspath(os.path.dirname('../../config/')))
import cusnarks_config as cfg

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
except ImportError:
  sys.exit()

def zpoly_div_cuda(pysnark, poly ,n, fidx, gpu_id=0, stream_id = 0):
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

     result_snarks,t = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,gpu_id, stream_id, n_kernels=1)

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
 
def ec_sc1mul_cuda(pysnark, vector, fidx, ec2=False, premul=False, batch_size=0, gpu_id=0, stream_id=0 ):
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


    kernel_params['stride'] = [1]
    kernel_config['smemS'] =  [0]
    kernel_config['blockD'] = [256]
    kernel_params['premul'] = [0]
    if premul:
      kernel_params['premul'] = [1]

    kernel_params['premod'] = [0]
    kernel_params['midx'] = [fidx]
    kernel_params['padding_idx'] = [0]
    kernel_config['kernel_idx'] = [kernel]
    kernel_config['return_val']=[1]

    nsamples = len(vector)
    if batch_size == 0:

       kernel_params['in_length'] = [nsamples]
       kernel_params['out_length'] = (nsamples-indims)*outdims
       kernel_config['gridD'] =\
            [int((kernel_config['blockD'][0] +
                   kernel_params['in_length'][0]-indims-1) /
                                         kernel_config['blockD'][0])]

       result,t = pysnark.kernelLaunch(vector, kernel_config, kernel_params,
                                       gpu_id, stream_id, n_kernels=1 )
       pysnark.streamDel(gpu_id,stream_id)
   
    else:
      new_vector = np.zeros((batch_size  + indims, NWORDS_256BIT), dtype=np.uint32)
      result = np.zeros(((nsamples-indims) * outdims ,NWORDS_256BIT), dtype=np.uint32)
      t=0.0

      for start_idx in xrange(0,nsamples-indims, batch_size-indims):
          new_vector[:min(batch_size-indims, nsamples-indims-start_idx)] =\
                    vector[start_idx:min(start_idx+batch_size-indims,nsamples-indims)]
          new_vector[min(batch_size-indims, nsamples-indims-start_idx):
                     min(batch_size-indims, nsamples-indims-start_idx) + indims] = vector[-indims:]

          kernel_params['in_length'] = [min(batch_size, nsamples-start_idx)]
          kernel_params['out_length'] = min(batch_size-indims, nsamples-indims-start_idx)*outdims
          kernel_config['gridD'] =\
            [int((kernel_config['blockD'][0] +
                   min(batch_size-indims, nsamples-start_idx-indims)-1) /
                                         kernel_config['blockD'][0])]
          result[start_idx*outdims:min(start_idx + batch_size - indims , nsamples-indims)*outdims], t1=\
                 pysnark.kernelLaunch(new_vector[:min(batch_size-indims, nsamples-indims-start_idx)+indims], kernel_config, kernel_params, gpu_id, stream_id, n_kernels=1 )
          pysnark.streamDel(gpu_id,stream_id)
          t+=t1
            
    return result,t

def ec_mad_cuda2(pysnark, vector, fidx, ec2=False, shamir_en=0, gpu_id=0, stream_id = 0):
   kernel_params={}
   kernel_config={}
   
   if ec2:
      outdims = ECP2_JAC_OUTDIMS
      indims_e = ECP2_JAC_INDIMS + U256_NDIMS
      kernel = CB_EC2_JAC_MAD_SHFL
   else:
      outdims = ECP_JAC_OUTDIMS 
      indims_e = ECP_JAC_INDIMS + U256_NDIMS
      kernel = CB_EC_JAC_MAD_SHFL

 
   nsamples = int(len(vector)/indims_e)
   
   #if shamir_en == 0 or nsamples < 32 * U256_BSELM :
   if shamir_en == 0 :
     kernel_config['blockD']    = get_shfl_blockD(nsamples)
     shamir_en = 0
   else:
     kernel_config['blockD']    = get_shfl_blockD(math.ceil(nsamples/U256_BSELM))

   nkernels = len(kernel_config['blockD'])
   kernel_params['stride']    = [outdims] * nkernels
   kernel_params['stride'][0]    =  indims_e
   kernel_params['premul']    = [0] * nkernels
   kernel_params['premul'][0] = 1
   kernel_params['premod']    = [0] * nkernels
   kernel_params['midx']      = [fidx] * nkernels
   kernel_config['smemS']     = [int(blockD/32 * NWORDS_256BIT * outdims * 4) for blockD in kernel_config['blockD']]
   kernel_config['kernel_idx'] =[kernel] * nkernels
   kernel_params['in_length'] = [nsamples* indims_e]*nkernels
   for l in xrange(1,nkernels):
      kernel_params['in_length'][l] = outdims * (
             int((kernel_params['in_length'][l-1]/outdims + (kernel_config['blockD'][l-1] * kernel_params['stride'][l-1] / outdims) - 1) /
             (kernel_config['blockD'][l-1] * kernel_params['stride'][l-1] / (outdims))))

   kernel_params['out_length'] = 1 * outdims
   """
   if not shamir_en:
     kernel_params['out_length'] = int(nsamples * outdims)
   else:
     kernel_params['out_length'] = int(nsamples/8 * outdims)
   """
   #kernel_params['out_length'] = np.product(kernel_config['blockD'][1:]) * outdims
   kernel_params['padding_idx'] = [shamir_en] * nkernels
   kernel_config['gridD'] = [0] * nkernels
   kernel_config['gridD'][0] = int(np.product(kernel_config['blockD'])/kernel_config['blockD'][0])
   kernel_config['gridD'][nkernels-1] = 1
   #kernel_params['out_length'] = kernel_config['gridD'][0] * outdims
    
   result,t = pysnark.kernelLaunch(vector, kernel_config, kernel_params, gpu_id, stream_id, n_kernels=nkernels )
   #result,t = pysnark.kernelLaunch(vector, kernel_config, kernel_params, gpu_id, stream_id, n_kernels= 1)

   return vector, result, t



def ec_mad_cuda(pysnark, vector, fidx, ec2=False, gpu_id=0, stream_id = 0):
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
   kernel_params['in_length'] = [nsamples* indims_e]*nkernels 
   for l in xrange(1,nkernels):
      kernel_params['in_length'][l] = outdims * (
             int((kernel_params['in_length'][l-1]/outdims + (kernel_config['blockD'][l-1] * kernel_params['stride'][l-1] / outdims) - 1) /
             (kernel_config['blockD'][l-1] * kernel_params['stride'][l-1] / (outdims))))

   kernel_params['out_length'] = 1 * outdims
   #kernel_params['out_length'] = nsamples * outdims
   #kernel_params['out_length'] = np.product(kernel_config['blockD'][1:]) * outdims
   kernel_params['padding_idx'] = [0] * nkernels
   kernel_config['gridD'] = [0] * nkernels
   kernel_config['gridD'][nkernels-1] = 1
    
   result,t = pysnark.kernelLaunch(new_vector, kernel_config, kernel_params, gpu_id, stream_id, n_kernels=nkernels )
   #result,t = pysnark.kernelLaunch(new_vector, kernel_config, kernel_params, 1)

   #new_vector = new_vector.reshape((-1,8))
   #Kp = [ZFieldElExt(BigInt.from_uint256(k)) for k in new_vector[:new_nsamples]]
   #Pp = np.zeros((new_nsamples, 3, 8), dtype=np.uint32)
   #Pp[:,0] = np.reshape(new_vector[new_nsamples:],(-1,2,8))[:,0]
   #Pp[:, 1] = np.reshape(new_vector[new_nsamples:], (-1, 2, 8))[:, 1]
   #Pp[:, 2, 0] = np.ones(new_nsamples,dtype=np.uint32)
   #Pp = ECC.from_uint256(np.reshape(Pp,(-1,NWORDS_256BIT)), in_ectype=1, out_ectype=2, reduced=True)
   #NPp = np.multiply(Kp, Pp)
   #NPp = np.sum(np.multiply(Kp, Pp))
   
   return new_vector, result, t

def zpoly_fft_cuda2(pysnark, vector, roots, fidx, gpu_id=0, stream_id=0 ):
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
        result,t = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,gpu_id, stream_id, n_kernels=4)

        return result, t


def zpoly_fft_cuda(pysnark, vector, ifft_params, fidx, roots=None, as_mont=1, return_val=1, out_extra_len=0, gpu_id=0, stream_id=0, fft=1 ):
        nsamples = 1<<ifft_params['levels']
        expanded_vector = np.zeros((nsamples,NWORDS_256BIT),dtype=np.uint32)
        expanded_vector[:len(vector)] = vector
        if roots is not None:
             n_bits = int(math.log(roots.shape[0],2))
             expanded_roots = roots[::1<<(n_bits-ifft_params['levels'])]
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
       
        kernel_params['padding_idx'] = [nsamples] * n_kernels1 
        kernel_params['in_length'] = [nsamples] * n_kernels1
        kernel_params['in_length'][0] = len(zpoly_vector)
        kernel_params['out_length'] = nsamples+out_extra_len
        kernel_params['stride'] = [1] * n_kernels1
        kernel_params['stride'][0] = 2
        kernel_params['premod'] = [0] * n_kernels1
        kernel_params['midx'] = [fidx]  * n_kernels1
        kernel_params['N_fftx'] = [Ncols] * n_kernels1
        kernel_params['N_ffty'] = [Nrows] * n_kernels1
        kernel_params['fft_Nx'] = [fft_xx, fft_xx, fft_yx, fft_yx] #xx,xx,yx,yx
        kernel_params['fft_Ny'] = [fft_xy, fft_xy, fft_yy, fft_yy] #xy,xy,yy,yy
        kernel_params['forward'] = [fft] * n_kernels1
        kernel_params['as_mont'] = [as_mont] * n_kernels1
        kernel_params['premul'] = [0] * n_kernels1
  
        kernel_config['smemS'] = [0] * n_kernels1
        kernel_config['blockD'] = [256] * n_kernels1
        kernel_config['gridD'] = [int((kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0])]*n_kernels1
        kernel_config['return_val'] = [return_val] * n_kernels1

        kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]

        result,t = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,gpu_id, stream_id,n_kernels=4)
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


def zpoly_interp3d_kernel_get(interp_params, nsamples, nsamples0, fidx, roots_2d_len, return_offset=0, n_pass=0, fft=-1):
        Npoints_pass1 = 1 << (interp_params['fft_N'][-1])
        Npoints_pass2 = 1 << (interp_params['fft_N'][-2])

        out_length = 2*nsamples

        if n_pass == 0 or n_pass == 2:
           fft_pass_params = ntt_build_h(Npoints_pass1)
           premul = 1 # First pass
           as_mont = 1
           if n_pass == 0:
             fft = 0
             out_length = 3*nsamples
           else :
             fft = 1

           kernel_idx = [CB_ZPOLY_INTERP4DXX, CB_ZPOLY_INTERP4DXY, CB_ZPOLY_INTERP4DYX, CB_ZPOLY_INTERP4DYY]

           n_kernels_interp = 4
           n_kernels_ifft = 0

        elif n_pass == 1 or n_pass == 3:
           fft_pass_params = ntt_build_h(Npoints_pass2)

           premul = 0
           as_mont = 1
           if n_pass == 1:
             fft = 0
           else :
             fft = 1
             out_length = nsamples

           kernel_idx = [CB_ZPOLY_INTERP4DXX, CB_ZPOLY_INTERP4DXY, CB_ZPOLY_INTERP4DYX, CB_ZPOLY_INTERP4DYY, CB_ZPOLY_INTERP4DFINISH]

           n_kernels_interp = 5
           n_kernels_ifft = 0

        elif n_pass == 4 :
           fft_pass_params = ntt_build_h(Npoints_pass1)

           if fft==-1:
             fft = 0
           premul = 1
           as_mont = 0
           out_length = nsamples

           kernel_idx = [CB_ZPOLY_FFT4DXX, CB_ZPOLY_FFT4DXY, CB_ZPOLY_FFT4DYX, CB_ZPOLY_FFT4DYY]

           n_kernels_interp = 0
           n_kernels_ifft = 4

        elif n_pass == 5:
           fft_pass_params = ntt_build_h(Npoints_pass2)

           if fft == -1:
             fft = 0
           premul = 0
           as_mont = 0
           out_length = nsamples

           n_kernels_interp = 0
           n_kernels_ifft = 4

           kernel_idx = [CB_ZPOLY_FFT4DXX, CB_ZPOLY_FFT4DXY, CB_ZPOLY_FFT4DYX, CB_ZPOLY_FFT4DYY]


        total_kernels = n_kernels_ifft + n_kernels_interp

        kernel_config={}
        kernel_params={}

        Nrows = fft_pass_params['fft_N'][(1<<FFT_T_3D)-1]
        Ncols = fft_pass_params['fft_N'][(1<<FFT_T_3D)-2]
        fft_yx = fft_pass_params['fft_N'][(1<<FFT_T_3D)-3]
        fft_yy = Nrows - fft_yx
        fft_xx = fft_pass_params['fft_N'][(1<<FFT_T_3D)-4]
        fft_xy = Ncols - fft_xx

        # Comon parameters
        kernel_params['out_length'] = out_length
        kernel_params['stride'] = [roots_2d_len] * total_kernels
        kernel_params['midx'] = [fidx]  * total_kernels
        kernel_params['padding_idx'] = [nsamples] * total_kernels
        kernel_params['in_length'] = [nsamples] * total_kernels
        kernel_params['in_length'][0] = nsamples0

        kernel_config['smemS'] = [0] * total_kernels
        kernel_config['blockD'] = [256] * total_kernels
        kernel_config['gridD'] = [int((kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0])]*total_kernels
        kernel_config['return_val'] = [1] * total_kernels
        kernel_config['return_offset'] = [return_offset] * total_kernels
        kernel_params['as_mont'] = [as_mont] * n_kernels_interp + [0] * n_kernels_ifft
        kernel_params['forward'] = [fft] * total_kernels


        kernel_params['N_fftx'] = [Ncols] * total_kernels
        kernel_params['N_ffty'] = [Nrows] * total_kernels
        kernel_params['fft_Nx'] = [fft_xx, fft_xx, fft_yx, fft_yx, 0] 
        kernel_params['fft_Ny'] = [fft_xy, fft_xy, fft_yy, fft_yy, 0]
        kernel_params['premul'] = [premul] * total_kernels 
        kernel_params['premod'] = [0] * total_kernels 
        kernel_config['kernel_idx'] = kernel_idx
     
        return kernel_config, kernel_params

def zpoly_interp_cuda(pysnark, zpoly_vector, interp_params, fidx, gpu_id=0, stream_id=0 ):
        nsamples = 1 << (interp_params['levels'])

        Nrows = interp_params['fft_N'][(1<<FFT_T_3D)-1]
        Ncols = interp_params['fft_N'][(1<<FFT_T_3D)-2]
        fft_yx = interp_params['fft_N'][(1<<FFT_T_3D)-3]
        fft_yy = Nrows - fft_yx
        fft_xx = interp_params['fft_N'][(1<<FFT_T_3D)-4]
        fft_xy = Ncols - fft_xx
        n_kernels1 = 5
        n_kernels2 = 5
        n_kernels = n_kernels1 +n_kernels2
        kernel_config={}
        kernel_params={}
       
        kernel_params['padding_idx'] = [nsamples] * n_kernels 
        kernel_params['in_length'] = [nsamples] * n_kernels
        kernel_params['in_length'][0] = zpoly_vector.shape[0]
        kernel_params['out_length'] = 2*nsamples

        kernel_params['premod'] = [0] * n_kernels
        kernel_params['stride'] = [nsamples] * n_kernels
        kernel_params['midx'] = [fidx]  * n_kernels
        kernel_params['N_fftx'] = [Ncols] * n_kernels
        kernel_params['N_ffty'] = [Nrows] * n_kernels
        kernel_params['fft_Nx'] = [fft_xx, fft_xx, fft_yx, fft_yx, 0,
                                   fft_xx, fft_xx, fft_yx, fft_yx, 0]
        kernel_params['fft_Ny'] = [fft_xy, fft_xy, fft_yy, fft_yy, 0, 
                                   fft_xy, fft_xy, fft_yy, fft_yy, 0] 
        kernel_params['forward'] = [0] * n_kernels1 + [1] * n_kernels2
        kernel_params['as_mont'] = [1] * n_kernels
        kernel_params['premul'] = [0] * n_kernels
  
        kernel_config['smemS'] = [0] * n_kernels
        kernel_config['blockD'] = [256] * n_kernels
        kernel_config['gridD'] = [int((kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0])]*n_kernels
        kernel_config['return_val'] = [1] * n_kernels
        kernel_config['return_offset'] = [0] * n_kernels

        kernel_config['kernel_idx']= [CB_ZPOLY_INTERP3DXX, CB_ZPOLY_INTERP3DXY, CB_ZPOLY_INTERP3DYX,
                                      CB_ZPOLY_INTERP3DYY, CB_ZPOLY_INTERP3DFINISH,
                                      CB_ZPOLY_INTERP3DXX, CB_ZPOLY_INTERP3DXY, CB_ZPOLY_INTERP3DYX,
                                      CB_ZPOLY_INTERP3DYY, CB_ZPOLY_INTERP3DFINISH]

        result,t = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,gpu_id, stream_id, n_kernels=10)

        return result, t
        #return kernel_params ,kernel_config, zpoly_vector

        ### DEBUG
        #ROOTS_1M_filename = '../../data/zpoly_roots_1M.bin'
        #roots = readU256DataFile_h(ROOTS_1M_filename.encode("UTF-8"), 1<<20, 1<<20)
        #expanded_roots = roots[::1<<(20-ifft_params['levels'])]
        #inv_roots = np.copy(expanded_roots)
        #inv_roots[1:] = expanded_roots[::-1][:-1]
        #result2 = intt_h(expanded_vector,inv_roots,1,fidx)
        #result3 = ntt_h(result2,expanded_roots,fidx)
        #result3 = ntt_h(result,expanded_roots,fidx)
         

def zpoly_mul_cuda(pysnark, vectorA, vectorB, mul_params, fidx, roots=None, return_val=0, as_mont=1, gpu_id=0, stream_id=0):
    nsamples = 1<<mul_params['levels']
    expanded_vectorA = np.zeros((nsamples,NWORDS_256BIT),dtype=np.uint32)
    expanded_vectorB = np.zeros((nsamples,NWORDS_256BIT),dtype=np.uint32)
    expanded_vectorA[:len(vectorA)] = vectorA
    expanded_vectorB[:len(vectorB)] = vectorB

    kernel_config={}
    kernel_params={}

    if roots is not None:
          n_bits = int(math.log(roots.shape[0],2))
          expanded_roots = roots[::1<<(n_bits-mul_params['levels'])]
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
    kernel_params['padding_idx'] = [nsamples] * n_kernels1
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

    X1S,t1 = pysnark.kernelLaunch(zpoly_vectorA, kernel_config, kernel_params,gpu_id, stream_id, n_kernels = n_kernels1)

    kernel_params['in_length'][0] = nsamples
    kernel_params['stride'][0] = 1
    kernel_config['return_val'][0] = 1

    Y1S,t2 = pysnark.kernelLaunch(zpoly_vectorB, kernel_config, kernel_params,gpu_id, stream_id, n_kernels = n_kernels1)

    kernel_params['padding_idx'] = [nsamples] * n_kernels2
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

    fftmul_result,t3 = pysnark.kernelLaunch(X1S, kernel_config, kernel_params,gpu_id, stream_id, n_kernels = n_kernels2)
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

def zpoly_interp_and_mul_single_cuda(pysnark, vector, interp_params, fidx, roots, mul=1):
     # Data fits in single FFT
     nsamples = 1 << (interp_params['levels'])
     nBitsRoots = cfg.get_n_roots()
     t = 0.0
     t_ifft = 0.0

     zpoly_vector = np.zeros((4*nsamples + 2,NWORDS_256BIT),dtype=np.uint32)
     zpoly_vector[:nsamples] = vector[:nsamples]
     zpoly_vector[nsamples : 2*nsamples] = vector[nsamples:]
     zpoly_vector[2*nsamples:3*nsamples] =\
                 roots[::1<<(nBitsRoots - interp_params['levels'])]
     zpoly_vector[3*nsamples:4*nsamples] =\
                roots[::1<<(nBitsRoots - interp_params['levels'] - 1)][:nsamples]
     scalerMont = ZFieldElExt(nsamples).inv().reduce().as_uint256()
     scalerExt = ZFieldElExt(nsamples).inv().as_uint256()
     zpoly_vector[-2] = scalerExt
     zpoly_vector[-1] = scalerMont


     # Data fits in single kernel lot, so no need to use streams
     zpoly_vector, t_interp = zpoly_interp_cuda(pysnark, zpoly_vector, interp_params,
                                  fidx, gpu_id=0, stream_id=0)
     try:
       pysnark.streamDel(0,0)
     except ValueError:
       print("Error releasing stream")
       sys.exit(1)

     if mul == 1:
        ifft_params = ntt_build_h(zpoly_vector.shape[0])
        zpoly_vector,t_ifft = zpoly_fft_cuda(pysnark, zpoly_vector,ifft_params, fidx,
                                 as_mont=0, roots=roots, fft=0, gpu_id=0, stream_id=0)
        try:
          pysnark.streamDel(0,0)
        except ValueError:
          print("Error releasing stream")
          sys.exit(1)

     t = t_interp + t_ifft

     return zpoly_vector, t

def zpoly_interp_and_mul_test(vector, interp_params, fidx, roots):

     vlen = int(vector.shape[0]/2)
     pA = np.copy(vector[:vlen])
     pB = np.copy(vector[vlen:])
     nBitsRoots = cfg.get_n_roots()


     roots_W11 = np.copy(roots[::(1<<(nBitsRoots-interp_params['fft_N'][-1]))])
     Iroots_W11 = np.copy(roots_W11)
     Iroots_W11[1:] = roots_W11[::-1][:-1]
     
     roots_W12 = np.copy(roots[::(1<<(nBitsRoots-interp_params['fft_N'][-2]))])
     Iroots_W12 = np.copy(roots_W12)
     Iroots_W12[1:] = roots_W12[::-1][:-1]

     roots_W2 = np.copy(roots[::(1<<(nBitsRoots-interp_params['levels']))])
     Iroots_W2 = np.copy(roots_W2)
     Iroots_W2[1:] = roots_W2[::-1][:-1]

     roots_W22 = np.copy(roots[::(1<<(nBitsRoots-interp_params['levels']-1))])
     Iroots_W22 = np.copy(roots_W22)
     Iroots_W22[1:] = roots_W22[::-1][:-1]

     roots_W3 = roots[::1<<(nBitsRoots - interp_params['levels'] - 1)][:vlen]


     ifft_mul_params = ntt_build_h(pA.shape[0]*2)
     Npoints_passB1 = 1 << (ifft_mul_params['fft_N'][-1])
     Npoints_passB2 = 1 << (ifft_mul_params['fft_N'][-2])

     roots_WB11 = np.copy(roots[::(1<<(nBitsRoots-ifft_mul_params['fft_N'][-1]))])
     Iroots_WB11 = np.copy(roots_WB11)
     Iroots_WB11[1:] = roots_WB11[::-1][:-1]
     
     roots_WB12 = np.copy(roots[::(1<<(nBitsRoots-ifft_mul_params['fft_N'][-2]))])
     Iroots_WB12 = np.copy(roots_WB12)
     Iroots_WB12[1:] = roots_WB12[::-1][:-1]

     pA_S = intt_h(pA, Iroots_W2,1, fidx)
     pA1_S = montmultN_h(pA_S.reshape(-1),
                    np.reshape(roots_W3,-1),fidx)
     pA2 = ntt_h(pA1_S, roots_W2, fidx)

     pB_S = intt_h(pB, Iroots_W2,1, fidx)
     pB1_S = montmultN_h(pB_S.reshape(-1),
                    np.reshape(roots_W3,-1),fidx)

     pB2 = ntt_h(pB1_S, roots_W2, fidx)
    
     pA3 = np.zeros((2*pA2.shape[0], NWORDS_256BIT), dtype=np.uint32)
     pA3[1::2] = montmultN_h(pA2.reshape(-1),
                       pB2.reshape(-1),fidx)
     pA3[::2] = montmultN_h(pA.reshape(-1),
                         pB.reshape(-1),fidx)
     r_pA = intt_h(pA3, Iroots_W22,0, fidx)

     return pA3, r_pA


def zpoly_fft4d_test(pA, fft_params, fidx, roots, fft=1, as_mont=1):
     Npoints_pass1 = 1 << (fft_params['fft_N'][-1])
     Npoints_pass2 = 1 << (fft_params['fft_N'][-2])
     nBitsRoots = cfg.get_n_roots()
     voffset1=0

     scalerMont = ZFieldElExt(pA.shape[0]).inv().reduce().as_uint256()
     scalerExt = ZFieldElExt(pA.shape[0]).inv().as_uint256()

     roots_W11 = np.copy(roots[::(1<<(nBitsRoots-fft_params['fft_N'][-1]))])
     Iroots_W11 = np.copy(roots_W11)
     Iroots_W11[1:] = roots_W11[::-1][:-1]
     
     roots_W12 = np.copy(roots[::(1<<(nBitsRoots-fft_params['fft_N'][-2]))])
     Iroots_W12 = np.copy(roots_W12)
     Iroots_W12[1:] = roots_W12[::-1][:-1]

     roots_W2 = np.copy(roots[::(1<<(nBitsRoots-fft_params['levels']))])
     Iroots_W2 = np.copy(roots_W2)
     Iroots_W2[1:] = roots_W2[::-1][:-1]

     pAT_S = np.zeros(pA.shape, dtype=np.uint32)
     pAT3_S = np.zeros(pA.shape, dtype=np.uint32)
 
     pAT = zpoly_transpose(pA, Npoints_pass1, Npoints_pass2)
     
     if fft:
         W11R = roots_W11 
         W12R = roots_W12 
         W2R  = roots_W2
     else:
         W11R = Iroots_W11 
         W12R = Iroots_W12 
         W2R  = Iroots_W2

     for i in xrange(Npoints_pass2):
        pAT_S[voffset1:voffset1+Npoints_pass1] = ntt_h(pAT[voffset1:voffset1+Npoints_pass1], W11R, fidx)
        if i == 0:
            pAT_S[voffset1:voffset1+Npoints_pass1] = montmultN_h(pAT_S[voffset1:voffset1+Npoints_pass1].reshape(-1),
                    np.reshape(np.tile(W2R[0],(Npoints_pass1,1)),(-1)),fidx)
        else:   
            pAT_S[voffset1:voffset1+Npoints_pass1] = montmultN_h(pAT_S[voffset1:voffset1+Npoints_pass1].reshape(-1),
                    W2R[::i][:Npoints_pass1].reshape(-1),fidx)

        voffset1+=Npoints_pass1

     voffset1=0
     pAT2_S = zpoly_transpose(pAT_S, Npoints_pass2, Npoints_pass1)
     for i in xrange(Npoints_pass1):
        pAT3_S[voffset1:voffset1+Npoints_pass2] = ntt_h(pAT2_S[voffset1:voffset1+Npoints_pass2], W12R, fidx)
        voffset1+=Npoints_pass2

     if fft==0:
       if as_mont==1:
          pAT3_S = montmultN_h(pAT3_S.reshape(-1),
                     np.reshape(np.tile(scalerMont,pA.shape[0]),-1) ,fidx) 
       else:
          pAT3_S = montmultN_h(pAT3_S.reshape(-1),
                     np.reshape(np.tile(scalerExt,pA.shape[0]),-1) ,fidx)

     pAT3_S = zpoly_transpose(pAT3_S, Npoints_pass1, Npoints_pass2)

     return pAT2_S, pAT3_S


def zpoly_interp_batch_cuda(pysnark, vector, interp_params, fidx, roots, batch_size, n_gpu=1):
     Npoints_pass1 = 1 << (interp_params['fft_N'][-1])
     Npoints_pass2 = 1 << (interp_params['fft_N'][-2])
     vlen = 1<< interp_params['levels']
     nBitsRoots = cfg.get_n_roots()
     nbatches = math.ceil(vlen/batch_size)
     pAB_vector = np.zeros((vlen,NWORDS_256BIT),dtype=np.uint32)
     t=0.0
     nsamples = batch_size
     voffset1 = 0
     voffset2 = vlen

     scalerMont = ZFieldElExt(vlen).inv().reduce().as_uint256()
     scalerExt = ZFieldElExt(vlen).inv().as_uint256()

    
     # Build basic vector:
     #  * -> Means param changes

     # IFFT1 Fist pass
     #   PA[nsamples]* | PB[nsamples]* | W2[nsamples]* | W11[first pass ncols]  

     # IFFT1 Secpnd pass
     # PA[nsamples]* | PB[nsamples]* | W3[nsamples]* | W12[second pass ncols] | ScalersA[2]

     # FFT First pass
     # PA[nsamples]* | PB[nsamples]* | W2[nsamples]* | W11[first pass ncols] 

     # FFT second pass
     # PA[nsamples]* | PB[nsamples]* | W12[second pass ncols]  

     # IFFT2 first pass
     # PAB[2*nsamples]* | W2[2*nsamples]* | W11[first pass ncols]  

     # IFFT2 Second pass
     # PAB[2*nsamples]* |  W12[second pass ncols]  | ScalersB[2]

     ## Summary
     #
     # PA/PB - 0:2 nsamples - always
     # W2      2:3/4 nsamples - IFFT1/2 first
     # W3      2:3 nsamples - IFFT1 seconds
     # 
     #  IFFT1-1: PA[nsamples]* | PB[nsamples]* | W2[nsamples]* |   W11[n] | W12[m] | ScalerA[2] 
     #  IFFT1-2: PA[nsamples]* | PB[nsamples]* | W3[nsamples]* |          | W12
     #  FFT-1    PA[nsamples]* | PB[nsamples]* | W2[nsamples]* |   W11 
     #  FFT-2    PA[nsamples]* | PB[nsamples]* |                          | W12
     #
     #  IFFT2-1:       PAB[2nsamples]*         |   W2[2*nsamples]*                |  W11  |            |  ScalerB[2] 
     #  IFFT2-2        PAB[2nsamples]*         |                                  |       | W12ncols[  |    ScalerB[2] 
     
     # Two configurations:
     #   For interp -> IFFT-1
     #   For ifft2   > 
     #  Max size : 4 batch_size + 1<<11 + 2

     roots_W11 = roots[::1<<(nBitsRoots - interp_params['fft_N'][-1])]
     roots_W12 = roots[::1<<(nBitsRoots - interp_params['fft_N'][-2])]
     roots_W1_len = roots_W11.shape[0] + roots_W12.shape[0]
     roots_W2 = roots[::(1<<(nBitsRoots- interp_params['levels']))]
     roots_W3 = roots[::1<<(nBitsRoots - interp_params['levels'] - 1)][:vlen]
     roots_W3T = zpoly_transpose(roots_W3,Npoints_pass2, Npoints_pass1)

     # Init zpoly
     zpoly_vector = np.zeros((3*nsamples + roots_W1_len + 2,NWORDS_256BIT),dtype=np.uint32)

     # Add W11/W12 roots -> Fixed
     zpoly_vector[3*nsamples:3*nsamples + roots_W1_len] =\
                 np.concatenate((roots_W11, roots_W12))
     # Add Scaler -> Fixed
     zpoly_vector[-2] = scalerExt
     zpoly_vector[-1] = scalerMont

     #Transpose and take IFFT (first pass)
     voffset1 = 0
     voffset2 = vlen

     """
     pA = np.copy(vector[:vlen])
     pB = np.copy(vector[vlen: ])
     pAB = montmultN_h(pA.reshape(-1), pB.reshape(-1),fidx)
     pA_S1, pA_S2 = zpoly_fft4d_test(pA, interp_params, fidx, roots, fft=0, as_mont=1)
     pA_S3 = montmultN_h(pA_S2.reshape(-1), roots_W3.reshape(-1),fidx)
     pB_S1, pB_S2 = zpoly_fft4d_test(pB, interp_params, fidx, roots, fft=0, as_mont=1)
     pB_S3 = montmultN_h(pB_S2.reshape(-1), roots_W3.reshape(-1),fidx)
     #writeU256DataFile_h(np.reshape(pAB,-1), "../../test/c/aux_data/pAB_samples.bin".encode("UTF-8"))
     #writeU256DataFile_h(np.reshape(pA_S1,-1), "../../test/c/aux_data/pA_S1_samples.bin".encode("UTF-8"))
     #writeU256DataFile_h(np.reshape(pA_S2,-1), "../../test/c/aux_data/pA_S2_samples.bin".encode("UTF-8"))
     #writeU256DataFile_h(np.reshape(pA_S3,-1), "../../test/c/aux_data/pA_S3_samples.bin".encode("UTF-8"))
     #writeU256DataFile_h(np.reshape(pB_S1,-1), "../../test/c/aux_data/pB_S1_samples.bin".encode("UTF-8"))
     #writeU256DataFile_h(np.reshape(pB_S2,-1), "../../test/c/aux_data/pB_S2_samples.bin".encode("UTF-8"))
     #writeU256DataFile_h(np.reshape(pB_S3,-1), "../../test/c/aux_data/pB_S3_samples.bin".encode("UTF-8"))
     #p_len = interp_params['fft_N'][-1] + interp_params['fft_N'][-2]
     #pAB = readU256DataFile_h("../../test/c/aux_data/pAB_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     #pA_S1 = readU256DataFile_h("../../test/c/aux_data/pA_S1_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     #pA_S2 = readU256DataFile_h("../../test/c/aux_data/pA_S2_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     #pA_S3 = readU256DataFile_h("../../test/c/aux_data/pA_S3_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     #pB_S1 = readU256DataFile_h("../../test/c/aux_data/pB_S1_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     #pB_S2 = readU256DataFile_h("../../test/c/aux_data/pB_S2_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     #pB_S3 = readU256DataFile_h("../../test/c/aux_data/pB_S3_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     """

     vector[:vlen] = zpoly_transpose(vector[:vlen], Npoints_pass1, Npoints_pass2)
     vector[vlen:] = zpoly_transpose(vector[vlen:], Npoints_pass1, Npoints_pass2) 

     n_streams = get_nstreams()
     #n_gpu = 1
     #n_streams = 1
     dispatch_table = buildDispatchTable( nbatches, 1, n_gpu, n_streams, nsamples, 0, vlen)
     pending_dispatch_table = []
     n_dispatch = 0
     n_par_batches = n_gpu * max((n_streams - 1),1)

     cols_idx = np.arange(Npoints_pass1)
     for i in range(nbatches):
        de = dispatch_table[i]
        gpu_id = de[3]
        stream_id = de[4]


        # Add samples
        zpoly_vector[:nsamples] = np.copy(vector[voffset1:voffset1+nsamples])
        zpoly_vector[nsamples : 2*nsamples] = np.copy(vector[voffset2:voffset2+nsamples])
        # Add W2 roots
        root_idx = np.outer(cols_idx,-1*np.arange(int(i*batch_size/Npoints_pass1),int((i+1)*batch_size/Npoints_pass1))).T
        zpoly_vector[2*nsamples : 3*nsamples] = np.copy(np.reshape(roots_W2[root_idx],(-1,NWORDS_256BIT)))

        kernel_config, kernel_params = zpoly_interp3d_kernel_get(interp_params, nsamples,
                                                                 len(zpoly_vector), fidx, roots_W1_len,
                                                                 return_offset=0, n_pass=0)
        result,t_fft = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params,gpu_id, stream_id, n_kernels=4)

        if stream_id == 0:
          pAB_vector[voffset1:voffset1+nsamples] = result[nsamples:2*nsamples]
          vector[voffset1:voffset1+nsamples] = result[:nsamples]
          vector[voffset2:voffset2+nsamples] = result[2*nsamples:]
          try:
            pysnark.streamDel(gpu_id,stream_id)
          except ValueError:
            print("Error releasing stream")
            sys.exit(1)
        else :
           pending_dispatch_table.append(de)
           n_dispatch+=1
     
           if n_dispatch == n_par_batches:
             getFFTResults(pysnark, pending_dispatch_table, vector, 0, vector2=pAB_vector)
             pending_dispatch_table = []
             n_dispatch = 0

        voffset1+=nsamples
        voffset2+=nsamples

     getFFTResults(pysnark, pending_dispatch_table, vector, 0, vector2=pAB_vector)
     pending_dispatch_table = []
     n_dispatch = 0

     #Transpose and Take IFFT (second pass)
     voffset1 = 0
     voffset2 = vlen
     vector[:vlen] = zpoly_transpose(vector[:vlen], Npoints_pass2, Npoints_pass1) 
     vector[vlen:] = zpoly_transpose(vector[vlen:], Npoints_pass2, Npoints_pass1) 

     for i in range(nbatches):
        de = dispatch_table[i]
        gpu_id    = de[3]
        stream_id = de[4]


        zpoly_vector[:nsamples] = vector[voffset1:voffset1+nsamples]
        zpoly_vector[nsamples : 2*nsamples] = vector[voffset2:voffset2+nsamples]

        zpoly_vector[2*nsamples:3*nsamples] =\
                 roots_W3T[voffset1:voffset1+nsamples]


        kernel_config, kernel_params = zpoly_interp3d_kernel_get(interp_params, nsamples,
                                                                 3*nsamples, fidx, roots_W1_len,
                                                                 return_offset=0, n_pass=1)

        result,t_fft = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params, gpu_id, stream_id, n_kernels=5)

        if stream_id == 0:
           vector[voffset1:voffset1+nsamples] = result[:nsamples]
           vector[voffset2:voffset2+nsamples] = result[nsamples:]
           try:
              pysnark.streamDel(gpu_id,stream_id)
           except ValueError:
              print("Error releasing stream")
              sys.exit(1)

        else:
           pending_dispatch_table.append(de)
           n_dispatch+=1

           if n_dispatch == n_par_batches:
             getFFTResults(pysnark, pending_dispatch_table, vector, 1)
             pending_dispatch_table = []
             n_dispatch = 0
        
        t+=t_fft

        voffset1+=nsamples
        voffset2+=nsamples
 

     getFFTResults(pysnark, pending_dispatch_table, vector, 1)
     pending_dispatch_table = []
     n_dispatch = 0

     vector[:vlen] = zpoly_transpose(vector[:vlen], Npoints_pass1, Npoints_pass2) 
     vector[vlen:] = zpoly_transpose(vector[vlen:], Npoints_pass1, Npoints_pass2)

     """
     pA_S4, pA_S5 = zpoly_fft4d_test(pA_S3, interp_params, fidx, roots, fft=1, as_mont=1)
     pB_S4, pB_S5 = zpoly_fft4d_test(pB_S3, interp_params, fidx, roots, fft=1, as_mont=1)
     pAB_S = montmultN_h(pA_S5.reshape(-1), pB_S5.reshape(-1),fidx)
     #writeU256DataFile_h(np.reshape(pA_S4,-1), "../../test/c/aux_data/pA_S4_samples.bin".encode("UTF-8"))
     #writeU256DataFile_h(np.reshape(pA_S5,-1), "../../test/c/aux_data/pA_S5_samples.bin".encode("UTF-8"))
     #writeU256DataFile_h(np.reshape(pB_S4,-1), "../../test/c/aux_data/pB_S4_samples.bin".encode("UTF-8"))
     #writeU256DataFile_h(np.reshape(pB_S5,-1), "../../test/c/aux_data/pB_S5_samples.bin".encode("UTF-8"))
     #writeU256DataFile_h(np.reshape(pAB_S,-1), "../../test/c/aux_data/pAB_S_samples.bin".encode("UTF-8"))
     #p_len = interp_params['fft_N'][-1] + interp_params['fft_N'][-2]
     #pAB_S = readU256DataFile_h("../../test/c/aux_data/pAB_S_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     #pA_S4 = readU256DataFile_h("../../test/c/aux_data/pA_S4_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     #pA_S5 = readU256DataFile_h("../../test/c/aux_data/pA_S5_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     #pB_S4 = readU256DataFile_h("../../test/c/aux_data/pB_S4_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     #pB_S5 = readU256DataFile_h("../../test/c/aux_data/pB_S5_samples.bin".encode("UTF-8"), 1<<p_len , 1<<p_len )
     """

     #Transpose and take FFT (first pass)
     voffset1 = 0
     voffset2 = vlen
     vector[:vlen] = zpoly_transpose(vector[:vlen], Npoints_pass1, Npoints_pass2) 
     vector[vlen:] = zpoly_transpose(vector[vlen:], Npoints_pass1, Npoints_pass2) 

     cols_idx = np.arange(Npoints_pass1)
     for i in range(nbatches):
        de = dispatch_table[i]
        gpu_id    = de[3]
        stream_id = de[4]

        zpoly_vector[:nsamples] = vector[voffset1:voffset1+nsamples]
        zpoly_vector[nsamples : 2*nsamples] = vector[voffset2:voffset2+nsamples]

        root_idx = np.outer(cols_idx,np.arange(int(i*batch_size/Npoints_pass1),int((i+1)*batch_size/Npoints_pass1))).T
        zpoly_vector[2*nsamples : 3*nsamples] = np.reshape(roots_W2[root_idx],(-1,NWORDS_256BIT))

        kernel_config, kernel_params = zpoly_interp3d_kernel_get(interp_params, nsamples,
                                                                 3*nsamples, fidx, roots_W1_len,
                                                                 return_offset=0, n_pass=2)
        result,t_fft = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params, gpu_id, stream_id, n_kernels=4)

        if stream_id==0:
          vector[voffset1:voffset1+nsamples] = result[:nsamples]
          vector[voffset2:voffset2+nsamples] = result[nsamples:]
          try:
              pysnark.streamDel(gpu_id,stream_id)
          except ValueError:
              print("Error releasing stream")
              sys.exit(1)
        else:
           pending_dispatch_table.append(de)
           n_dispatch+=1
           if n_dispatch == n_par_batches:
             getFFTResults(pysnark, pending_dispatch_table, vector, 2)
             pending_dispatch_table = []
             n_dispatch = 0

        t+=t_fft

        voffset1+=nsamples
        voffset2+=nsamples


     getFFTResults(pysnark, pending_dispatch_table, vector, 2)
     pending_dispatch_table = []
     n_dispatch = 0


     #Transpose and Take FFT (second pass)
     voffset1 = 0
     voffset2 = vlen
     vector[:vlen] = zpoly_transpose(vector[:vlen], Npoints_pass2, Npoints_pass1) 
     vector[vlen:] = zpoly_transpose(vector[vlen:], Npoints_pass2, Npoints_pass1) 

     for i in range(nbatches):
        de = dispatch_table[i]
        gpu_id    = de[3]
        stream_id = de[4]


        zpoly_vector[:nsamples] = vector[voffset1:voffset1+nsamples]
        zpoly_vector[nsamples : 2*nsamples] = vector[voffset2:voffset2+nsamples]

        kernel_config, kernel_params = zpoly_interp3d_kernel_get(interp_params, nsamples,
                                                                 2*nsamples, fidx, roots_W1_len,
                                                                 return_offset=0, n_pass=3)

        result,t_fft = pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params, gpu_id, stream_id, n_kernels = 5)

        if stream_id == 0:
           vector[voffset1:voffset1+nsamples] = result[:nsamples]
           try:
              pysnark.streamDel(gpu_id,stream_id)
           except ValueError:
              print("Error releasing stream")
              sys.exit(1)
        else:
           pending_dispatch_table.append(de)
           n_dispatch+=1
           if n_dispatch == n_par_batches:
             getFFTResults(pysnark, pending_dispatch_table, vector, 3)
             pending_dispatch_table = []
             n_dispatch = 0


        voffset1+=nsamples
        voffset2+=nsamples

        t+=t_fft


     getFFTResults(pysnark, pending_dispatch_table, vector, 3)

     vector[1::2] = zpoly_transpose(vector[:vlen], Npoints_pass1, Npoints_pass2)
     vector[::2] = zpoly_transpose(pAB_vector, Npoints_pass2, Npoints_pass1)
     
     return vector, t

#TODO review this function
def  getFFTResults(pysnark, dispatch_table, vector, n_pass, vector2=None):
       for bidx,p in enumerate(dispatch_table):
          gpu_id = p[3]
          stream_id = p[4]
          nsamples = p[2] - p[1]
          start_idx = p[1]
          end_idx = p[2]
          result, t = pysnark.streamSync(gpu_id,stream_id)
          
          # Step 0 interpolation : output is 3 vectors
          if n_pass == 0:
            vlen = int(vector.shape[0]/2)
            vector[start_idx:end_idx] = result[:nsamples]
            vector[vlen+start_idx:vlen+end_idx] = result[2*nsamples:]
            vector2[start_idx:end_idx] = result[nsamples:2*nsamples]
          elif n_pass <= 2:
            vlen = int(vector.shape[0]/2)
            vector[start_idx:end_idx] = result[:nsamples]
            vector[vlen+start_idx:vlen+end_idx] = result[nsamples:]
          elif n_pass == 3:
             vector[start_idx:end_idx] = result[:nsamples]
          # Step 4,5 interpolation : output is 1 vector
          elif n_pass >= 4:
              vector[start_idx:end_idx] = result

def zpoly_mul_batch_cuda(pysnark, vector, mul_params, fidx, roots, batch_size, n_gpu=1):
     #vector = readU256DataFile_h("../../test/c/aux_data/zpoly_samples_tmp2.bin".encode("UTF-8"), 1<<21 , 1<<21 )
     ### Start Final IFFT with combined results
     Npoints_pass1 = 1 << (mul_params['fft_N'][-1])
     Npoints_pass2 = 1 << (mul_params['fft_N'][-2])
     vlen = 1<< mul_params['levels']
     nBitsRoots = cfg.get_n_roots()
     nbatches = math.ceil(vlen/batch_size)
     nsamples = batch_size
     t=0.0

     roots_W11 = roots[::1<<(nBitsRoots - mul_params['fft_N'][-1])]
     roots_W12 = roots[::1<<(nBitsRoots - mul_params['fft_N'][-2])]
     roots_W1_len = roots_W11.shape[0] + roots_W12.shape[0]
     roots_W2 = roots[::(1<<(nBitsRoots- mul_params['levels']))]

     scalerMont2 = ZFieldElExt(vlen).inv().reduce().as_uint256()
     scalerExt2 = ZFieldElExt(vlen).inv().as_uint256()

     zpoly_vector = np.zeros((2*nsamples + roots_W1_len + 2,NWORDS_256BIT),dtype=np.uint32)

     # Add W11/W12 roots -> Fixed
     zpoly_vector[2*nsamples:2*nsamples + roots_W1_len] =\
                 np.concatenate((roots_W11, roots_W12))
     # Add Scaler -> Fixed
     zpoly_vector[2*nsamples + roots_W1_len] = scalerExt2
     zpoly_vector[2*nsamples + roots_W1_len + 1] = scalerMont2

     # Test code
     """
     pA = np.copy(vector)
     pA_S1, pA_S2 = zpoly_fft4d_test(pA, mul_params, fidx, roots, fft=0, as_mont=0)
     """

     #Transpose and take IFFT (first pass)
     voffset1 = 0
     vector = zpoly_transpose(vector, Npoints_pass1, Npoints_pass2)

     n_streams = get_nstreams()
     dispatch_table = buildDispatchTable( nbatches, 1, n_gpu, n_streams, nsamples, 0, vlen)
     pending_dispatch_table = []
     n_dispatch = 0
     n_par_batches = n_gpu * max((n_streams - 1),1)
                                    
     cols_idx = np.arange(Npoints_pass1)
     for i in range(nbatches):
        de = dispatch_table[i]
        gpu_id    = de[3]
        stream_id = de[4]

        zpoly_vector[:nsamples] = vector[voffset1:voffset1+nsamples]

        root_idx = np.outer(cols_idx,-1*np.arange(int(i*batch_size/Npoints_pass1),int((i+1)*batch_size/Npoints_pass1))).T
        zpoly_vector[nsamples : 2*nsamples] = np.reshape(roots_W2[root_idx],(-1,NWORDS_256BIT))

        kernel_config, kernel_params = zpoly_interp3d_kernel_get(mul_params, nsamples,
                                                         len(zpoly_vector),
                                                         fidx, roots_W1_len,
                                               return_offset=nsamples*NWORDS_256BIT, n_pass=4,fft=0)
                                                                 
        result,t_fft =\
               pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params, gpu_id, stream_id, n_kernels= 4)

        if stream_id == 0:
          vector[voffset1:voffset1+nsamples] = result
          try:
              pysnark.streamDel(gpu_id,stream_id)
          except ValueError:
              print("Error releasing stream")
              sys.exit(1)
        else:
           pending_dispatch_table.append(de)
           n_dispatch+=1
           if n_dispatch == n_par_batches:
             getFFTResults(pysnark, pending_dispatch_table, vector, 4)
             pending_dispatch_table = []
             n_dispatch = 0

        t+=t_fft

        voffset1+=nsamples

     getFFTResults(pysnark, pending_dispatch_table, vector, 4)
     pending_dispatch_table = []


     #Transpose and Take IFFT (second pass)
     voffset1 = 0
     vector = zpoly_transpose(vector, Npoints_pass2, Npoints_pass1) 

     for i in range(nbatches):
        de = dispatch_table[i]
        gpu_id    = de[3]
        stream_id = de[4]

        zpoly_vector[:nsamples] = vector[voffset1:voffset1+nsamples]

        kernel_config, kernel_params = zpoly_interp3d_kernel_get(mul_params, nsamples,
                                                                 nsamples, fidx, roots_W1_len,
                                           return_offset=nsamples*NWORDS_256BIT, n_pass=5, fft=0)

        result,t_fft =\
               pysnark.kernelLaunch(zpoly_vector, kernel_config, kernel_params, gpu_id, stream_id, n_kernels=4)

        if stream_id == 0:
          vector[voffset1:voffset1+nsamples] = result
          try:
              pysnark.streamDel(gpu_id,stream_id)
          except ValueError:
              print("Error releasing stream")
              sys.exit(1)
        else:
           pending_dispatch_table.append(de)
           n_dispatch+=1
           if n_dispatch == n_par_batches:
             getFFTResults(pysnark, pending_dispatch_table, vector, 5)
             pending_dispatch_table = []
             n_dispatch = 0


        t+=t_fft

        voffset1+=nsamples

     getFFTResults(pysnark, pending_dispatch_table, vector, 5)
     pending_dispatch_table = []

     vector = zpoly_transpose(vector, Npoints_pass1, Npoints_pass2)

     return vector, t

def zpoly_interp_and_mul_cuda(pysnark, vector, interp_params, fidx, roots, batch_size, n_gpu=1):
     vlen = int(vector.shape[0]/2)
     
     # if in vector's length is not power of two, append zeros
     if vlen != 1<<interp_params['levels']:
       expanded_vector = np.zeros(2<<interp_params['levels'],NWORDS_256BIT)
       expanded_vector[:vlen] = vector[:vlen]
       expanded_vector[1<<interp_params['levels']:] = vector[vlen:]
       vector = expanded_vector
       vlen = 1<< interp_params['levels']

     #p_interp, p_mul = zpoly_interp_and_mul_test(vector, interp_params,fidx, roots)
     #writeU256DataFile_h(np.reshape(p_interp,-1), "../../test/c/aux_data/p_interp_samples.bin".encode("UTF-8"))
     #writeU256DataFile_h(np.reshape(p_mul,-1), "../../test/c/aux_data/p_mul_samples.bin".encode("UTF-8"))
     #p_interp = readU256DataFile_h("../../test/c/aux_data/p_interp_samples.bin".encode("UTF-8"), 1<<23 , 1<<23 )
     #p_mul = readU256DataFile_h("../../test/c/aux_data/p_mul_samples.bin".encode("UTF-8"), 1<<23 , 1<<23 )

     if batch_size >= vlen*2:
        # Data fits in single FFT
        vector, t = zpoly_interp_and_mul_single_cuda(pysnark, vector, interp_params, fidx, roots, mul=1)

     elif batch_size >= vlen :
        # Data fits in single FFT but not after interpolation
        vector, t = zpoly_interp_and_mul_single_cuda(pysnark, vector, interp_params, fidx, roots,
                                                     mul=0)
        mul_params = ntt_build_h(vector.shape[0])
        vector, t1 = zpoly_mul_batch_cuda(pysnark, vector, mul_params, fidx, roots, batch_size, n_gpu=n_gpu)
        
        t += t1

     else :
        # Data doesnt fit in single FFT
        vector, t = zpoly_interp_batch_cuda(pysnark, vector, interp_params, fidx, roots,
                                            batch_size, n_gpu=n_gpu)

        """
        if not all(np.concatenate(vector == p_interp)) : 
           print("Interp incorrect  ",)
           vector = p_interp
        """

        mul_params = ntt_build_h(vector.shape[0])
        vector, t1 = zpoly_mul_batch_cuda(pysnark, vector, mul_params, fidx, roots, batch_size,n_gpu=n_gpu)
        """
        if not all(np.concatenate(vector == p_mul)) : 
           print("Mul incorrect  ",)   
           vector = p_mul
        """

        t +=t1

     return vector, t



def zpoly_transpose(zpoly_in, nrows_in,  ncols_in):
      #TODO : Change to c function xform (similar to transpose, but passing input and output dims)
      return  np.reshape(
                       np.swapaxes(
                           np.reshape(
                               zpoly_in,
                               (nrows_in, ncols_in,NWORDS_256BIT)),
                           0,1),
                       (-1,NWORDS_256BIT))

        
def zpoly_sub_cuda(pysnark, vectorA, vectorB, fidx, vectorA_len=1, return_val=0, gpu_id=0, stream_id=0):  

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
     result,t = pysnark.kernelLaunch(vector, kernel_config, gpu_id, stream_id, kernel_params )

     if return_val == 0:
        result = kernel_params['out_length']
     elif copy:
        result = np.concatenate((result,vectorA[int(len(vectorB)):]))

     return result,t

def u256_mul_cuda(pysnark, vectorA, vectorB, fidx, gpu_id=0, stream_id=0):
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

     result, t = pysnark.kernelLaunch(vector, kernel_config, kernel_params,gpu_id, stream_id, n_kernels=2 )

     return result, t


def zpoly_mulK_cuda(pysnark, vectorA, K, fidx, gpu_id=0, stream_id=0):  
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
     result,t= pysnark.kernelLaunch(vector, kernel_config, gpu_id, stream_id,  kernel_params )

     return result,t

def zpoly_mad_cuda(pysnark, vectors, fidx, gpu_id=0, stream_id=0):  

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

def get_shfl_blockD(nsamples, max_block_size=8):

   l = max(math.ceil(math.log2(nsamples)),5)
   nb = math.ceil(l/max_block_size)
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

def get_ngpu(max_used_percent=20.):
   return len(nvgpu.available_gpus(max_used_percent))

def get_nstreams():
    return (N_STREAMS_PER_GPU)

def get_affinity(r_cpu=0, add_single_cpu=1):
    n_cores = mp.cpu_count()
    n_gpu = get_ngpu(max_used_percent=95.)
    affinity = {}
    _stop = False
    for i in xrange(n_gpu):
        affinity[str(i)] = []
    for i in xrange(n_cores):
        _stop = False
        if i < r_cpu: 
            continue
        tmp = Popen('nvidia-smi topo -c '+str(i),
                shell=True,
                stdout=PIPE).stdout.read().decode('utf-8')
        tmp = tmp.split('\n')
        for gpu_idx in xrange(len(tmp)-1):
            if len(tmp[gpu_idx+1]) and gpu_idx < n_gpu:
                for j in tmp[gpu_idx+1].split(','):
                  if add_single_cpu and len(affinity[str(int(j))]) :
                    break
                  affinity[str(int(j))].append(i)
                  if add_single_cpu:
                    _stop = True
                    break
            if _stop:
              _stop = False
              break

    return affinity

def buildDispatchTable(nbatches, nP, ngpu, nstreams, step, start_idx, end_idx,
                           start_pidx=0, start_gpu_idx=0, start_stream_idx=1, ec_lable=None):
      dispatch_table = np.zeros( (nP * nbatches, 5), dtype=object)

      # Add EC point : Column 0
      if ec_lable is None:
          dispatch_table[:,0] = (np.arange(nP * nbatches) % nP) + start_pidx
      else:
          dispatch_table[:,0] = ec_lable[(np.arange(nP * nbatches) % nP) + start_pidx]

      # Add GPU : Column 3
      if ngpu > 1:
         dispatch_table[:,3] = (np.arange(nP * nbatches) + start_gpu_idx )% ngpu

      # Add Stream : Colum 4
      nvalid_streams = max(nstreams-1,1)
      dispatch_table[:,4] = \
               np.reshape(
                       np.tile(
                          np.reshape(
                             np.tile(
                                 (np.arange(nvalid_streams) + start_stream_idx) % nstreams, 
                                 (ngpu,1)).T,
                             -1),
                          (math.ceil(nP*nbatches/(ngpu*nvalid_streams)),1)),
                       -1)[:dispatch_table.shape[0]]

      # Add starting and ending batch indexes -> Colum 1 and 2
      idx = np.reshape(np.tile(np.arange(start_idx, end_idx+step, step) ,(nP,1)).T,-1)
      idx [-nP:] = end_idx
      
      dispatch_table[:,1] = idx[:-nP]
      dispatch_table[:,2] = idx[nP:]

      return dispatch_table


