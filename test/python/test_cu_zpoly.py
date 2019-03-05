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

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : test_cu_zpoly.py
//
// Date       : 26/02/2019
//
// ------------------------------------------------------------------
//
// Description:
//   ZPoly CUDA modulo test
//
// ------------------------------------------------------------------

"""
import os,sys, os.path
import unittest
import numpy as np
from random import randint, sample

sys.path.append('../../src/python')

from bigint import *
from zutils import *
from zfield import *
from zpoly import *
from constants import *


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

sys.path.append('../../src/python')
ZPOLY_datafile = './aux_data/zpoly_data_1M.npz'

class CUZPolyTest(unittest.TestCase):
    TEST_ITER = 1000
    prime = ZUtils.CURVE_DATA['BN128']['prime_r']
    nsamples = 32 * 1024   # 1024 FFT 32 points
    ntest_points = 100 
    u256_p = BigInt(prime).as_uint256()
    cu_zpoly = ZCUPoly(nsamples, seed=560)
    u256 = U256(nsamples, seed=560)
    ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
    ZPoly(1,force_init=True)
    roots_ext, inv_roots_ext = ZField.find_roots(ZUtils.NROOTS, rformat_ext=True)
    roots_rdc = [r.reduce() for r in roots_ext]
    inv_roots_rdc = [r.reduce() for r in inv_roots_ext]

    def test_0fft32(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc
        ZField.roots[0] = roots_rdc
        ZField.inv_roots[0] = inv_roots_rdc

        kernel_config = {'blockD' : 256 }
        kernel_params = {'midx' : MOD_FIELD ,'premod' : 0, 'in_length' : CUZPolyTest.nsamples, 'stride' : 1, 'out_length' : CUZPolyTest.nsamples}

        for iter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.rand(CUZPolyTest.nsamples)
            # do mod operation
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 1
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            zpoly_vector,_ = u256.kernelLaunch(CB_U256_MOD, zpoly_vector, kernel_config, kernel_params )

            # Test FFT kernel:
            test_points = sample(xrange(CUZPolyTest.nsamples/32-1), ntest_points)
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 1
            kernel_params['premod'] = 0
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = 256
            result_fft,_ = cu_zpoly.kernelLaunch(CB_ZPOLY_FFT32, zpoly_vector, kernel_config, kernel_params )
            result_ifft,_ = cu_zpoly.kernelLaunch(CB_ZPOLY_IFFT32, result_fft, kernel_config, kernel_params )

            for i in test_points:
               p_rdc = ZPoly.from_uint256(zpoly_vector[i*32:i*32+32], reduced=True)
               zpoly_fft = ZPoly.from_uint256(result_fft[i*32:i*32+32], reduced=True)
               p_rdc.ntt_DIF()
               self.assertTrue(p_rdc == zpoly_fft)
               p_rdc.intt_DIT()
               zpoly_ifft = ZPoly.from_uint256(result_ifft[i*32:i*32+32], reduced=True)
               self.assertTrue(p_rdc == zpoly_ifft)

    def test_1fft_mul32(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        nsamples = CUZPolyTest.nsamples
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc
        ZField.roots[0] = roots_rdc
        ZField.inv_roots[0] = inv_roots_rdc

        kernel_config = {'blockD' : 256 }
        kernel_params = {'midx' : MOD_FIELD ,'premod' : 0, 'in_length' : CUZPolyTest.nsamples, 'stride' : 1, 'out_length' : CUZPolyTest.nsamples}
        for iter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.rand(CUZPolyTest.nsamples)
            for st in xrange(16,32):
               zpoly_vector[st::32] = np.zeros((nsamples/32,8),dtype=np.uint32)
            # do mod operation
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 1
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            zpoly_vector,_ = u256.kernelLaunch(CB_U256_MOD, zpoly_vector, kernel_config, kernel_params )

            # Test FFT kernel:
            test_points = sample(xrange(CUZPolyTest.nsamples/64-1), ntest_points)
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples/2
            kernel_params['stride'] = 2
            kernel_params['premod'] = 0
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = 256
            result_mul,_ = cu_zpoly.kernelLaunch(CB_ZPOLY_MUL32, zpoly_vector, kernel_config, kernel_params )

            """
            p1_rdc = ZPoly.from_uint256(zpoly_vector[0*32:0*32+32], reduced=True)
            p2_rdc = ZPoly.from_uint256(zpoly_vector[nsamples/2 + 0*32:nsamples/2 + 0*32+32], reduced=True)
            a = ZPoly(p1_rdc)
            b = ZPoly(p2_rdc)
            a.ntt_DIF()
            b.ntt_DIF()
            c = ZPoly([x_ * y_ for x_, y_ in zip(a.get_coeff(), b.get_coeff())])
            p1_rdc.poly_mul_fft(p2_rdc)
            zpoly_mul = ZPoly.from_uint256(result_mul[0*32:0*32+32], reduced=True)
            """

            for i in test_points:
               p1_rdc = ZPoly.from_uint256(zpoly_vector[i*32:i*32+16], reduced=True)
               p2_rdc = ZPoly.from_uint256(zpoly_vector[nsamples/2 + i*32:nsamples/2 + i*32+16], reduced=True)
               p1_rdc.poly_mul_fft(p2_rdc)
               zpoly_mul = ZPoly.from_uint256(result_mul[i*32:i*32+32], reduced=True)
               self.assertTrue(p1_rdc == zpoly_mul)

    def test_2fftN(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc
        ZField.roots[0] = roots_rdc
        ZField.inv_roots[0] = inv_roots_rdc

        kernel_config = {'blockD' : 256 }
        kernel_params = {'midx' : MOD_FIELD ,'premod' : 0, 'in_length' : CUZPolyTest.nsamples, 'stride' : 1, 'out_length' : CUZPolyTest.nsamples}
        for iter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.rand(CUZPolyTest.nsamples)
            # do mod operation
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 1
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            zpoly_vector,_ = u256.kernelLaunch(CB_U256_MOD, zpoly_vector, kernel_config, kernel_params )

            # Test FFT kernel:
            N = 1 + (iter % 5)

            test_points = sample(xrange(CUZPolyTest.nsamples/32-1), ntest_points)
            #test_points = sample(xrange(CUZPolyTest.nsamples/32-1), ntest_points)
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 1
            kernel_params['premod'] = 0
            kernel_params['fft_Nx'] = N
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = 256
            result_fft,_ = cu_zpoly.kernelLaunch(CB_ZPOLY_FFTN, zpoly_vector, kernel_config, kernel_params )
            result_ifft,_ = cu_zpoly.kernelLaunch(CB_ZPOLY_IFFTN, result_fft, kernel_config, kernel_params )

            for i in test_points:
               p_rdc_ntt_all = []
               p_rdc_intt_all = []
               for j in range(32/(1<<N)):
                  p_rdc = ZPoly.from_uint256(zpoly_vector[i*32+j*(1<<N):i*32+j*(1<<N)+(1<<N)], reduced=True)
                  p_rdc.ntt_DIF()
                  p_rdc_ntt_all.append(p_rdc.as_uint256())

                  p_rdc.intt_DIT()
                  p_rdc_intt_all.append(p_rdc.as_uint256())

               p_rdc_ntt_all = ZPoly.from_uint256(np.concatenate(p_rdc_ntt_all), reduced=True)
               p_rdc_intt_all = ZPoly.from_uint256(np.concatenate(p_rdc_intt_all), reduced=True)
               zpoly_fft = ZPoly.from_uint256(result_fft[i*32:i*32+32], reduced=True)
               self.assertTrue(p_rdc_ntt_all == zpoly_fft)
               zpoly_ifft = ZPoly.from_uint256(result_ifft[i*32:i*32+32], reduced=True)
               self.assertTrue(p_rdc_intt_all == zpoly_ifft)

    """
    def test_00fft_mulN(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        nsamples = CUZPolyTest.nsamples
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc
        ZField.roots[0] = roots_rdc
        ZField.inv_roots[0] = inv_roots_rdc

        kernel_config = {'blockD' : 256 }
        kernel_params = {'midx' : MOD_FIELD ,'premod' : 0, 'in_length' : CUZPolyTest.nsamples, 'stride' : 1, 'out_length' : CUZPolyTest.nsamples}
        for iter in xrange(CUZPolyTest.TEST_ITER):
            N = 1 + (iter % 5)
            # add 0s to some of the coefficients
            for st in xrange(1<<(N-1),1<<N):
               zpoly_vector[st::1<<N] = np.zeros((nsamples/(1<<N),8),dtype=np.uint32)

            # do mod operation
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 1
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            zpoly_vector,_ = u256.kernelLaunch(CB_U256_MOD, zpoly_vector, kernel_config, kernel_params )

            # Test FFT kernel:
            test_points = sample(xrange(CUZPolyTest.nsamples/64-1), ntest_points)
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 2
            kernel_params['premod'] = 0
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = 256
            kernel_params['fft_Nx'] = N
            result_mul,_ = cu_zpoly.kernelLaunch(CB_ZPOLY_MULN, zpoly_vector, kernel_config, kernel_params )

            for i in test_points:
               p1_rdc = ZPoly.from_uint256(zpoly_vector[i*(1<<N):i*(1<<N)+(1<<N)], reduced=True)
               p2_rdc = ZPoly.from_uint256(zpoly_vector[nsamples/2 + i*(1<<N):nsamples/2 + i*(1<<N)+(1<<N)], reduced=True)
               p1_rdc.poly_mul_fft(p2_rdc)
               zpoly_mul = ZPoly.from_uint256(result_mul[i*(1<<N):i*(1<<N)+(1<<N)], reduced=True)
               self.assertTrue(p1_rdc == zpoly_mul)
    """

    def test_4fft2D_1024(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc
        ZField.roots[0] = roots_rdc
        ZField.inv_roots[0] = inv_roots_rdc
        CUZPolyTest.nsamples = 1024 
        nsamples = CUZPolyTest.nsamples
        roots_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc[::ZUtils.NROOTS/1024]])

        kernel_config = {}
        kernel_params = {}

        for iter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.rand(CUZPolyTest.nsamples)
            # do mod operatio
            kernel_config['gridD'] = 0
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 1
            kernel_params['midx']=0
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            zpoly_vector,_ = u256.kernelLaunch(CB_U256_MOD, zpoly_vector, kernel_config, kernel_params )

            # Test FFT kernel:
            kernel_params['in_length'] = [2*CUZPolyTest.nsamples,CUZPolyTest.nsamples]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [2,1]
            kernel_params['premod'] = [0,0]
            kernel_params['midx'] = [MOD_FIELD, MOD_FIELD]
            kernel_params['fft_Nx'] = [5,5]
            kernel_params['fft_Ny'] = [5,5]
            kernel_params['forward'] = [1,1]

            kernel_config['smemS'] = [0,0]
            kernel_config['blockD'] = [256,256]
            kernel_config['gridD'] = [0, (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_FFT2DX, CB_ZPOLY_FFT2DY]
            zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256))
            result_fft2d,_ = cu_zpoly.kernelMultipleLaunch(zpoly_vector1, kernel_config, kernel_params,2)

            p_rdc = ZPoly.from_uint256(zpoly_vector, reduced=True)
            #zpoly_fft2d = []
            #for i in range(32): 
               #zpoly_fft2d.append(ZPoly.from_uint256(result_fft2d[i*32:i*32+32], reduced=True))
            p_rdc.ntt_parallel2D(1<<5,1<<5)
            #p_rdc.intt_parallel2D(1<<5,1<<5)
            self.assertTrue(p_rdc == ZPoly.from_uint256(result_fft2d, reduced=True))
            
            #p_rdc.ntt_DIF()
            #self.assertTrue(p_rdc == zpoly_fft2d)
            #p_rdc.intt_DIT()
            #zpoly_ifft = ZPoly.from_uint256(result_ifft[i*32:i*32+32], reduced=True)
            #self.assertTrue(p_rdc == zpoly_ifft)

    def test_5ifft2D_1024(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc
        ZField.roots[0] = roots_rdc
        ZField.inv_roots[0] = inv_roots_rdc
        CUZPolyTest.nsamples = 1024 
        nsamples = CUZPolyTest.nsamples
        inv_roots_rdc_u256 = np.asarray([x.as_uint256() for x in inv_roots_rdc[::ZUtils.NROOTS/1024]])

        kernel_config={}
        kernel_params={}

        for iter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.rand(CUZPolyTest.nsamples)
            # do mod operatio
            kernel_config['gridD'] = 0
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 1
            kernel_params['midx']=0
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            zpoly_vector,_ = u256.kernelLaunch(CB_U256_MOD, zpoly_vector, kernel_config, kernel_params )

            # Test FFT kernel:
            kernel_params['in_length'] = [2*CUZPolyTest.nsamples,CUZPolyTest.nsamples]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [2,1]
            kernel_params['premod'] = [0,0]
            kernel_params['midx'] = [MOD_FIELD, MOD_FIELD]
            kernel_params['fft_Nx'] = [5,5]
            kernel_params['fft_Ny'] = [5,5]
            kernel_params['forward'] = [0,0]

            kernel_config['smemS'] = [0,0]
            kernel_config['blockD'] = [256,256]
            kernel_config['gridD'] = [0, (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_FFT2DX, CB_ZPOLY_FFT2DY]
            zpoly_vector1 = np.concatenate((zpoly_vector, inv_roots_rdc_u256))
            result_ifft2d,_ = cu_zpoly.kernelMultipleLaunch(zpoly_vector1, kernel_config, kernel_params,2)


            p_rdc = ZPoly.from_uint256(zpoly_vector, reduced=True)
            #zpoly_fft2d = []
            #for i in range(32): 
               #zpoly_fft2d.append(ZPoly.from_uint256(result_fft2d[i*32:i*32+32], reduced=True))
            p_rdc.intt_parallel2D(1<<5,1<<5)
            #p_rdc.intt_parallel2D(1<<5,1<<5)
            self.assertTrue(p_rdc == ZPoly.from_uint256(result_ifft2d, reduced=True))
            
            #p_rdc.ntt_DIF()
            #self.assertTrue(p_rdc == zpoly_fft2d)
            #p_rdc.intt_DIT()
            #zpoly_ifft = ZPoly.from_uint256(result_ifft[i*32:i*32+32], reduced=True)
            #self.assertTrue(p_rdc == zpoly_ifft)

    def test_00fft2D_1M(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        CUZPolyTest.nsamples = 1 << 10
        nsamples = CUZPolyTest.nsamples
        ZUtils.NROOTS = CUZPolyTest.nsamples

        if os.path.exists(ZPOLY_datafile):
           npzfile = np.load(ZPOLY_datafile)
           roots_rdc_u256 = npzfile['roots_rdc_u256']
           roots_rdc = npzfile['roots_rdc']
           inv_roots_rdc = npzfile['inv_roots_rdc']
           ZField.roots[0] = roots_rdc
           ZField.inv_roots[0] = inv_roots_rdc
       
        else:
           roots_ext, inv_roots_ext = ZField.find_roots(ZUtils.NROOTS, rformat_ext=True)
           roots_rdc = [r.reduce() for r in roots_ext]
           inv_roots_rdc = [r.reduce() for r in inv_roots_ext]
           ZField.roots[0] = roots_rdc
           ZField.inv_roots[0] = roots_rdc
           roots_ext_u256 = np.asarray([x.as_uint256() for x in roots_ext])
           inv_roots_ext_u256 = np.asarray([x.as_uint256() for x in inv_roots_ext])
           roots_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc])
           inv_roots_rdc_u256 = np.asarray([x.as_uint256() for x in inv_roots_rdc])
          
           np.savez_compressed(ZPOLY_datafile, roots_ext=roots_ext, roots_rdc=roots_rdc, inv_roots_ext=inv_roots_ext, inv_roots_rdc=inv_roots_rdc,
                                               roots_ext_u256 = roots_ext_u256, roots_rdc_u256=roots_rdc_u256, inv_roots_ext_u256=inv_roots_ext_u256,
                                               inv_roots_rdc_u256=inv_roots_rdc_u256)


        kernel_config = {}
        kernel_params = {}

        for iter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.rand(CUZPolyTest.nsamples)
            # do mod operatio
            kernel_config['gridD'] = 0
            kernel_params['in_length'] = CUZPolyTest.nsamples
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = 1
            kernel_params['midx']=0
            kernel_config['smemS'] = 0
            kernel_config['blockD'] = U256_BLOCK_DIM 
            zpoly_vector,_ = u256.kernelLaunch(CB_U256_MOD, zpoly_vector, kernel_config, kernel_params )

            # Test FFT kernel:
            kernel_params['in_length'] = [2*CUZPolyTest.nsamples,CUZPolyTest.nsamples, 2*CUZPolyTest.nsamples, CUZPolyTest.nsamples ]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [2,1,2,1]
            kernel_params['premod'] = [0,0,0,0]
            kernel_params['midx'] = [MOD_FIELD, MOD_FIELD, MOD_FIELD, MOD_FIELD]
            kernel_params['fft_Nx'] = [5,5,5,5]
            kernel_params['fft_Ny'] = [5,5,5,5]
            kernel_params['N_fftx'] = [10,10,10,10]
            kernel_params['N_ffty'] = [10,10,10,10]
            kernel_params['forward'] = [1,1,1,1]

            kernel_config['smemS'] = [0,0,0,0]
            kernel_config['blockD'] = [256,256,256,256]
            kernel_config['gridD'] = [0, (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                         (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                         (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DX, CB_ZPOLY_FFT3DY, CB_ZPOLY_FFT3DX, CB_ZPOLY_FFT3DY]
            zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256))
            result_fft2d,_ = cu_zpoly.kernelMultipleLaunch(zpoly_vector1, kernel_config, kernel_params,2)

            p_rdc = ZPoly.from_uint256(zpoly_vector, reduced=True)
            #zpoly_fft2d = []
            #for i in range(32): 
               #zpoly_fft2d.append(ZPoly.from_uint256(result_fft2d[i*32:i*32+32], reduced=True))
            p_rdc.ntt_parallel2D(1<<10,1<<10)
            #p_rdc.intt_parallel2D(1<<5,1<<5)
            self.assertTrue(p_rdc == ZPoly.from_uint256(result_fft2d, reduced=True))
            
            #p_rdc.ntt_DIF()
            #self.assertTrue(p_rdc == zpoly_fft2d)
            #p_rdc.intt_DIT()
            #zpoly_ifft = ZPoly.from_uint256(result_ifft[i*32:i*32+32], reduced=True)
            #self.assertTrue(p_rdc == zpoly_ifft)
 


if __name__ == "__main__":
    unittest.main()
