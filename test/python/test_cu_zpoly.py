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
from random import randint, sample, shuffle

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
ZPOLY_datafile_1M = '../../data/zpoly_data_1M.npz'
ZPOLY_input_datafile_1M = '../c/aux_data/zpoly_input_data_1M.bin'
ZPOLY_inputxx_datafile_1M = '../c/aux_data/zpoly_input_dataxx_1M.bin'
ZPOLY_inputxy_datafile_1M = '../c/aux_data/zpoly_input_dataxy_1M.bin'
ZPOLY_inputyx_datafile_1M = '../c/aux_data/zpoly_input_datayx_1M.bin'
ZPOLY_inputyy_datafile_1M = '../c/aux_data/zpoly_input_datayy_1M.bin'
ZPOLY_input_npz_datafile_1M = './aux_data/zpoly_input_data_1M.npz'
ZPOLY_output_datafile_1M = '../c/aux_data/zpoly_output_data_1M.bin'

ZPOLY_input_datafile_65K = '../c/aux_data/zpoly_input_data_65K.bin'
ZPOLY_inputxx_datafile_65K = '../c/aux_data/zpoly_input_dataxx_65K.bin'
ZPOLY_inputxy_datafile_65K = '../c/aux_data/zpoly_input_dataxy_65K.bin'
ZPOLY_inputyx_datafile_65K = '../c/aux_data/zpoly_input_datayx_65K.bin'
ZPOLY_inputyy_datafile_65K = '../c/aux_data/zpoly_input_datayy_65K.bin'
ZPOLY_input_npz_datafile_65K = './aux_data/zpoly_input_data_65K.npz'
ZPOLY_output_datafile_65K = '../c/aux_data/zpoly_output_data_65K.bin'

primitive_roots=[ 1,
             21888242871839275222246405745257275088548364400416034343698204186575808495616,
             21888242871839275217838484774961031246007050428528088939761107053157389710902,
             19540430494807482326159819597004422086093766032135589407132600596362845576832,
             14940766826517323942636479241147756311199852622225275649687664389641784935947,
             4419234939496763621076330863786513495701855246241724391626358375488475697872,
             9088801421649573101014283686030284801466796108869023335878462724291607593530,
             10359452186428527605436343203440067497552205259388878191021578220384701716497,
             3478517300119284901893091970156912948790432420133812234316178878452092729974,
             6837567842312086091520287814181175430087169027974246751610506942214842701774,
             3161067157621608152362653341354432744960400845131437947728257924963983317266,
             1120550406532664055539694724667294622065367841900378087843176726913374367458,
             4158865282786404163413953114870269622875596290766033564087307867933865333818,
             197302210312744933010843010704445784068657690384188106020011018676818793232,
             20619701001583904760601357484951574588621083236087856586626117568842480512645,
             20402931748843538985151001264530049874871572933694634836567070693966133783803,
             421743594562400382753388642386256516545992082196004333756405989743524594615,
             12650941915662020058015862023665998998969191525479888727406889100124684769509,
             11699596668367776675346610687704220591435078791727316319397053191800576917728,
             15549849457946371566896172786938980432421851627449396898353380550861104573629,
             17220337697351015657950521176323262483320249231368149235373741788599650842711,
             13536764371732269273912573961853310557438878140379554347802702086337840854307,
             12143866164239048021030917283424216263377309185099704096317235600302831912062,
             934650972362265999028062457054462628285482693704334323590406443310927365533,
             5709868443893258075976348696661355716898495876243883251619397131511003808859,
             19200870435978225707111062059747084165650991997241425080699860725083300967194,
             7419588552507395652481651088034484897579724952953562618697845598160172257810,
             2082940218526944230311718225077035922214683169814847712455127909555749686340,
             19103219067921713944291392827692070036145651957329286315305642004821462161904]

class CUZPolyTest(unittest.TestCase):
    TEST_ITER = 100
    prime = ZUtils.CURVE_DATA['BN128']['prime_r']
    nsamples = int(32 * 1024)   # 1024 FFT 32 points
    ntest_points = 100 
    u256_p = BigInt(prime).as_uint256()
    if use_pycusnarks:
      cu_zpoly = ZCUPoly(2*nsamples+2, seed=560)
      u256 = U256(nsamples, seed=560)
    ZField(prime, ZUtils.CURVE_DATA['BN128']['curve'])
    ZPoly(1,force_init=True)
    roots_ext, inv_roots_ext = ZField.find_roots(ZUtils.NROOTS, rformat_ext=True)
    roots_rdc = [r.reduce() for r in roots_ext]
    inv_roots_rdc = [r.reduce() for r in inv_roots_ext]

    def test_00fft32(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc
        ZField.roots[0] = roots_rdc
        ZField.inv_roots[0] = inv_roots_rdc

        kernel_config = {'blockD' : [256] }
        kernel_params = {'midx' : [MOD_FIELD] ,'premod' : [0], 'in_length' : [CUZPolyTest.nsamples], 'stride' : [1], 'out_length' : CUZPolyTest.nsamples}

        for niter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.randu256(CUZPolyTest.nsamples,CUZPolyTest.u256_p)

            # Test FFT kernel:
            test_points = sample(xrange(int(CUZPolyTest.nsamples/32)-1), ntest_points)
            #try:
             # test_points = sample(xrange(CUZPolyTest.nsamples/32-1), ntest_points)
            #except ValueError:
              #print "Test 0"
              #print ntest_points, CUZPolyTest.nsamples/32  - 1
              #return
            #print "Test 0 : " + str(niter)+"                        \r",
            kernel_params['in_length'] = [CUZPolyTest.nsamples]
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = [1]
            kernel_params['premod'] = [0]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['kernel_idx'] = [CB_ZPOLY_FFT32]

            result_fft,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params )
            kernel_config['kernel_idx'] = [CB_ZPOLY_IFFT32]
            result_ifft,_ = cu_zpoly.kernelLaunch(result_fft, kernel_config, kernel_params )
            self.assertTrue(all(np.concatenate(result_ifft == zpoly_vector)))


            for i in test_points:
               p_rdc = ZPoly.from_uint256(zpoly_vector[i*32:i*32+32], reduced=True)
               zpoly_fft = ZPoly.from_uint256(result_fft[i*32:i*32+32], reduced=True)
               p_rdc.ntt_DIF()
               self.assertTrue(p_rdc == zpoly_fft)
               p_rdc.intt_DIT()
               zpoly_ifft = ZPoly.from_uint256(result_ifft[i*32:i*32+32], reduced=True)
               self.assertTrue(p_rdc == zpoly_ifft)

    def test_01fft_mul32(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
 
        nsamples = CUZPolyTest.nsamples
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc
        ZField.roots[0] = roots_rdc
        ZField.inv_roots[0] = inv_roots_rdc

        kernel_config = {'blockD' : [256] }
        kernel_params = {'midx' : [MOD_FIELD] ,'premod' : [0], 'in_length' : [CUZPolyTest.nsamples], 'stride' : [1], 'out_length' : CUZPolyTest.nsamples}
        for niter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.randu256(CUZPolyTest.nsamples,CUZPolyTest.u256_p)
            for st in xrange(16,32):
               zpoly_vector[st::32] = np.zeros((int(nsamples/32),8),dtype=np.uint32)

            # Test FFT kernel:
            #print "Test 1 : " + str(niter)+"                               \r",
            test_points = sample(xrange(int(CUZPolyTest.nsamples/64)-1), ntest_points)
            kernel_params['in_length'] = [CUZPolyTest.nsamples]
            kernel_params['out_length'] = CUZPolyTest.nsamples/2
            kernel_params['stride'] = [2]
            kernel_params['premod'] = [0]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['kernel_idx'] = [CB_ZPOLY_MUL32]
            result_mul,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params )

            #TODO last step (IFFT) cannot be done within the same kernel. I don't understand why, but
            # this FFT is not used for anything so it is not a big deal having to call IFFT32 in a separate step
            kernel_params['in_length'] = [CUZPolyTest.nsamples/2]
            kernel_params['out_length'] = CUZPolyTest.nsamples/2
            kernel_params['stride'] = [1]
            kernel_config['kernel_idx'] = [CB_ZPOLY_IFFT32]
            result_mul,_ = cu_zpoly.kernelLaunch(result_mul, kernel_config, kernel_params )
            #p1_rdc = ZPoly.from_uint256(zpoly_vector[0*32:0*32+32], reduced=True)
            #p2_rdc = ZPoly.from_uint256(zpoly_vector[nsamples/2 + 0*32:nsamples/2 + 0*32+32], reduced=True)
            #a = ZPoly(p1_rdc)
            #b = ZPoly(p2_rdc)
            #a.ntt_DIF()
            #b.ntt_DIF()
            #c = ZPoly([x_ * y_ for x_, y_ in zip(a.get_coeff(), b.get_coeff())])
            #p1_rdc.poly_mul_fft(p2_rdc)
            #zpoly_mul = ZPoly.from_uint256(result_mul[0*32:0*32+32], reduced=True)


            for i in test_points:
            #for i in range(len(zpoly_vector)/64):

               p1_rdc = ZPoly.from_uint256(zpoly_vector[i*32:i*32+16], reduced=True)
               p2_rdc = ZPoly.from_uint256(zpoly_vector[int(nsamples/2) + i*32:int(nsamples/2) + i*32+16], reduced=True)
               p1_rdc.poly_mul_fft(p2_rdc)
               zpoly_mul = ZPoly.from_uint256(result_mul[i*32:i*32+32], reduced=True)

               #if p1_rdc != zpoly_mul:
                   #print niter, i
               self.assertTrue(p1_rdc == zpoly_mul)

    def test_02fftN(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc
        ZField.roots[0] = roots_rdc
        ZField.inv_roots[0] = inv_roots_rdc

        kernel_config = {'blockD' : [256] }
        kernel_params = {'midx' : [MOD_FIELD] ,'premod' : [0], 'in_length' : [CUZPolyTest.nsamples], 'stride' : [1], 'out_length' : CUZPolyTest.nsamples}
        for niter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.randu256(CUZPolyTest.nsamples,CUZPolyTest.u256_p)

            # Test FFT kernel:
            N = 1 + (niter % 5)
            #print "Test 2 : " + str(niter)+"                                     \r",
            #try:
            test_points = sample(xrange(int(CUZPolyTest.nsamples/32)-1), ntest_points)
            #except ValueError:
              #print "2 sample larger than population " + str(niter)
              #print 
              #continue
            #test_points = sample(xrange(CUZPolyTest.nsamples/32-1), ntest_points)
            kernel_params['in_length'] = [CUZPolyTest.nsamples]
            kernel_params['out_length'] = CUZPolyTest.nsamples
            kernel_params['stride'] = [1]
            kernel_params['premod'] = [0]
            kernel_params['fft_Nx'] = [N]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['kernel_idx'] = [CB_ZPOLY_FFTN]
            result_fft,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params )
            kernel_config['kernel_idx'] = [CB_ZPOLY_IFFTN]
            result_ifft,_ = cu_zpoly.kernelLaunch(result_fft, kernel_config, kernel_params )
            self.assertTrue(all(np.concatenate(result_ifft == zpoly_vector)))

            for i in test_points:
               p_rdc_ntt_all = []
               p_rdc_intt_all = []
               for j in range(int(32/(1<<N))):
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

    def test_04fft2D_1024(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc
        ZField.roots[0] = roots_rdc
        ZField.inv_roots[0] = inv_roots_rdc
        CUZPolyTest.nsamples = 1024 
        nsamples = CUZPolyTest.nsamples
        roots_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc[::int(ZUtils.NROOTS/1024)]])

        kernel_config = {}
        kernel_params = {}

        for niter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.randu256(CUZPolyTest.nsamples,CUZPolyTest.u256_p)

            # Test FFT kernel:
            #print "Test 4 : " + str(niter)+"                                     \r",
            kernel_params['in_length'] = [2*CUZPolyTest.nsamples+2,CUZPolyTest.nsamples]
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
            scalerMont = ZFieldElExt(len(roots_rdc_u256)).inv().reduce().as_uint256()
            scalerExt = ZFieldElExt(len(roots_rdc_u256)).inv().as_uint256()
            zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256, [scalerExt], [scalerMont]))
            result_fft2d,_ = cu_zpoly.kernelLaunch(zpoly_vector1, kernel_config, kernel_params,2)

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

    def test_05ifft2D_1024(self):

        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        ntest_points = CUZPolyTest.ntest_points
        roots_rdc = CUZPolyTest.roots_rdc
        inv_roots_rdc = CUZPolyTest.inv_roots_rdc
        ZField.roots[0] = roots_rdc
        ZField.inv_roots[0] = inv_roots_rdc
        CUZPolyTest.nsamples = 1024 
        nsamples = CUZPolyTest.nsamples
        inv_roots_rdc_u256 = np.asarray([x.as_uint256() for x in inv_roots_rdc[::int(ZUtils.NROOTS/1024)]])
        roots_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc[::int(ZUtils.NROOTS/1024)]])

        kernel_config={}
        kernel_params={}

        for niter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector = cu_zpoly.randu256(CUZPolyTest.nsamples,CUZPolyTest.u256_p)

            # Test FFT kernel:
            #print "Test 5 : " + str(niter)+"                                \r",
            kernel_params['in_length'] = [2*CUZPolyTest.nsamples+2,CUZPolyTest.nsamples]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [2,1]
            kernel_params['premod'] = [0,0]
            kernel_params['midx'] = [MOD_FIELD, MOD_FIELD]
            kernel_params['fft_Nx'] = [5,5]
            kernel_params['fft_Ny'] = [5,5]
            kernel_params['forward'] = [0,0]
            kernel_params['as_mont'] = [1,1]

            kernel_config['smemS'] = [0,0]
            kernel_config['blockD'] = [256,256]
            kernel_config['gridD'] = [0, (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_FFT2DX, CB_ZPOLY_FFT2DY]
            scalerMont = ZFieldElExt(len(inv_roots_rdc_u256)).inv().reduce().as_uint256()
            scalerExt = ZFieldElExt(len(inv_roots_rdc_u256)).inv().as_uint256()
            zpoly_vector1 = np.concatenate((zpoly_vector, inv_roots_rdc_u256,[scalerExt], [scalerMont]))
            result_ifft2d,_ = cu_zpoly.kernelLaunch(zpoly_vector1, kernel_config, kernel_params,2)

            zpoly_vector_copy = np.copy(zpoly_vector)
            ntt_p = ZPoly.from_uint256(zpoly_vector, reduced=True)

            ntt_h_r = ntt_h(zpoly_vector_copy,roots_rdc_u256, MOD_FIELD)
            ntt_p.ntt()
            self.assertTrue(ntt_p == ZPoly.from_uint256(ntt_h_r, reduced=True))

            intt_h_r = intt_h(ntt_h_r,inv_roots_rdc_u256,1,MOD_FIELD)
            ntt_p.intt()
            self.assertTrue(ntt_p == ZPoly.from_uint256(intt_h_r, reduced=True))
            self.assertTrue(all(np.concatenate(intt_h_r == zpoly_vector)))
            ntt_p.intt()


            p_rdc = ZPoly.from_uint256(zpoly_vector, reduced=True)
            p_rdc.intt_parallel2D(1<<5,1<<5)
            self.assertTrue(p_rdc == ZPoly.from_uint256(result_ifft2d, reduced=True))
            self.assertTrue(p_rdc == ntt_p)
            
    def test_06fft3D_1M(self):

        ntest_points = CUZPolyTest.ntest_points
        CUZPolyTest.nsamples = 1 << 20
        nsamples = CUZPolyTest.nsamples
        ZUtils.NROOTS = CUZPolyTest.nsamples

        if os.path.exists(ZPOLY_datafile_1M):
           npzfile = np.load(ZPOLY_datafile_1M)
           roots_rdc_u256 = npzfile['roots_rdc_u256']
           inv_roots_rdc_u256 = npzfile['inv_roots_rdc_u256']
           #roots_rdc = npzfile['roots_rdc']
           #inv_roots_rdc = npzfile['inv_roots_rdc']
           #ZField.roots[0] = roots_rdc
           #ZField.inv_roots[0] = inv_roots_rdc
       
        else:
           roots_ext, inv_roots_ext = ZField.find_roots(1<<20, rformat_ext=True, primitive_root=primitive_roots[20])
           roots_rdc = [r.reduce() for r in roots_ext]
           inv_roots_rdc = [r.reduce() for r in inv_roots_ext]
           ZField.roots[0] = roots_rdc
           ZField.inv_roots[0] = roots_rdc
           roots_ext_u256 = np.asarray([x.as_uint256() for x in roots_ext])
           inv_roots_ext_u256 = np.asarray([x.as_uint256() for x in inv_roots_ext])
           roots_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc])
           inv_roots_rdc_u256 = np.asarray([x.as_uint256() for x in inv_roots_rdc])
          
           np.savez_compressed(ZPOLY_datafile_1M, roots_ext=roots_ext, roots_rdc=roots_rdc, inv_roots_ext=inv_roots_ext, inv_roots_rdc=inv_roots_rdc,
                                               roots_ext_u256 = roots_ext_u256, roots_rdc_u256=roots_rdc_u256, inv_roots_ext_u256=inv_roots_ext_u256,
                                               inv_roots_rdc_u256=inv_roots_rdc_u256)


        kernel_config = {}
        kernel_params = {}
        CUZPolyTest.cu_zpoly = ZCUPoly(2*nsamples+2, seed=580)
        cu_zpoly = CUZPolyTest.cu_zpoly
        u256_p = CUZPolyTest.u256_p
        n_rows=10
        n_cols = 10
        n_kernels = 4
        fft_N = 5

        #for niter in xrange(CUZPolyTest.TEST_ITER/CUZPolyTest.TEST_ITER):
        for niter in xrange(5):
            zpoly_vector = cu_zpoly.randu256(CUZPolyTest.nsamples,u256_p)
            npzfile = np.load('./aux_data/main_tvector.npz')
            zpoly_vector = npzfile['in_data']

            # Test FFT kernel:

            kernel_params['in_length'] = [2*CUZPolyTest.nsamples+2, nsamples, nsamples, nsamples]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [2,1,1,1]
            kernel_params['premod'] = [0,0,0,0]
            kernel_params['midx'] = [MOD_FIELD, MOD_FIELD, MOD_FIELD, MOD_FIELD]
            kernel_params['fft_Nx'] = [fft_N, fft_N, fft_N, fft_N] #xx,xx,yx,yx
            kernel_params['fft_Ny'] = [fft_N, fft_N, fft_N, fft_N] #xy,xy,yy,yy
            kernel_params['N_fftx'] = [n_cols, n_cols, n_cols, n_cols]
            kernel_params['N_ffty'] = [n_rows, n_rows, n_rows, n_rows]
            kernel_params['forward'] = [1, 1, 1, 1]
            kernel_params['as_mont'] = [1,1,1,1]

            kernel_config['smemS'] = [0, 0, 0, 0]
            kernel_config['blockD'] = [256, 256, 256, 256]
            kernel_config['gridD'] = [(kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]
            zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256))
            scalerMont = ZFieldElExt(len(roots_rdc_u256)).inv().reduce().as_uint256()
            scalerExt = ZFieldElExt(len(roots_rdc_u256)).inv().as_uint256()
            zpoly_vector1 = np.concatenate((zpoly_vector1, [scalerExt],[scalerMont]))


            #if os.path.exists(ZPOLY_input_npz_datafile_1M):
              #npzfile = np.load(ZPOLY_input_npz_datafile_1M)
              #zpoly_vector = npzfile['in_data']
              #zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256))


            result_fft3d,_ = cu_zpoly.kernelLaunch(zpoly_vector1, kernel_config, kernel_params,4)
            #cu_zpoly.saveFile(result_fft3d.reshape(-1), ZPOLY_inputyy_datafile_1M)
            #f = open("./testxx.txt","w")
            #for r in result_fft3d[:64 * 1024]:
            #   f.write(np.array2string(r,max_line_width=90))
            #   f.write("\n")
            #f.close()

            r_u256 = ntt_parallel2D_h(zpoly_vector, roots_rdc_u256, n_rows, fft_N, n_cols, fft_N, 1)
            #r_u256 = np.reshape(np.swapaxes(np.reshape(r_u256,(1024,1024,8)),0,1),(-1,8))
            #f = open("./testxx2.txt", "w")
            #for r in r_u256[:64 * 1024]:
            #    f.write(np.array2string(r, max_line_width=90))
            #    f.write("\n")
            #f.close()


            r1_u256 = intt_h(r_u256,inv_roots_rdc_u256,1,MOD_FIELD)
            self.assertTrue(all(np.concatenate(result_fft3d == r_u256)))
            self.assertTrue(all(np.concatenate(zpoly_vector == r1_u256)))

            if not os.path.exists(ZPOLY_input_datafile_1M):
               cu_zpoly.saveFile(zpoly_vector.reshape(-1), ZPOLY_input_datafile_1M.encode("UTF-8"))
               np.savez_compressed(ZPOLY_input_npz_datafile_1M, in_data=zpoly_vector)

            if not os.path.exists(ZPOLY_output_datafile_1M):
               cu_zpoly.saveFile(zpoly_vector.reshape(-1), ZPOLY_input_datafile_1M.encode("UTF-8"))
               cu_zpoly.saveFile(result_fft3d.reshape(-1), ZPOLY_output_datafile_1M.encode("UTF-8"))

    def test_07ifft3D_1M(self):

        ntest_points = CUZPolyTest.ntest_points
        CUZPolyTest.nsamples = 1 << 20
        nsamples = CUZPolyTest.nsamples
        ZUtils.NROOTS = CUZPolyTest.nsamples

        if os.path.exists(ZPOLY_datafile_1M):
           npzfile = np.load(ZPOLY_datafile_1M)
           inv_roots_rdc_u256 = npzfile['inv_roots_rdc_u256']
           roots_rdc_u256 = npzfile['roots_rdc_u256']
           #roots_rdc = npzfile['roots_rdc']
           #inv_roots_rdc = npzfile['inv_roots_rdc']
           #ZField.roots[0] = roots_rdc
           #ZField.inv_roots[0] = inv_roots_rdc
       
        else:
           roots_ext, inv_roots_ext = ZField.find_roots(1<<20, rformat_ext=True, primitive_root=primitive_roots[20])
           roots_rdc = [r.reduce() for r in roots_ext]
           inv_roots_rdc = [r.reduce() for r in inv_roots_ext]
           ZField.roots[0] = roots_rdc
           ZField.inv_roots[0] = roots_rdc
           roots_ext_u256 = np.asarray([x.as_uint256() for x in roots_ext])
           inv_roots_ext_u256 = np.asarray([x.as_uint256() for x in inv_roots_ext])
           roots_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc])
           inv_roots_rdc_u256 = np.asarray([x.as_uint256() for x in inv_roots_rdc])
          
           np.savez_compressed(ZPOLY_datafile_1M, roots_ext=roots_ext, roots_rdc=roots_rdc, inv_roots_ext=inv_roots_ext, inv_roots_rdc=inv_roots_rdc,
                                               roots_ext_u256 = roots_ext_u256, roots_rdc_u256=roots_rdc_u256, inv_roots_ext_u256=inv_roots_ext_u256,
                                               inv_roots_rdc_u256=inv_roots_rdc_u256)


        kernel_config = {}
        kernel_params = {}
        CUZPolyTest.cu_zpoly = ZCUPoly(2*nsamples+2, seed=560)
        CUZPolyTest.u256 = U256(nsamples, seed=560)
        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        u256_p = CUZPolyTest.u256_p
        n_rows=10
        n_cols = 10
        n_kernels = 4
        fft_N = 5

        for niter in xrange(5):
            zpoly_vector = cu_zpoly.randu256(CUZPolyTest.nsamples,u256_p)

            # Test FFT kernel:

            kernel_params['in_length'] = [2*CUZPolyTest.nsamples+2, nsamples, nsamples, nsamples]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [2,1,1,1]
            kernel_params['premod'] = [0,0,0,0]
            kernel_params['midx'] = [MOD_FIELD, MOD_FIELD, MOD_FIELD, MOD_FIELD]
            kernel_params['fft_Nx'] = [fft_N, fft_N, fft_N, fft_N] #xx,xx,yx,yx
            kernel_params['fft_Ny'] = [fft_N, fft_N, fft_N, fft_N] #xy,xy,yy,yy
            kernel_params['N_fftx'] = [n_cols, n_cols, n_cols, n_cols]
            kernel_params['N_ffty'] = [n_rows, n_rows, n_rows, n_rows]
            kernel_params['forward'] = [0, 0, 0, 0]
            #kernel_params['forward'] = [1, 1, 1, 1]
            kernel_params['as_mont'] = [1,1,1,1]

            kernel_config['smemS'] = [0, 0, 0, 0]
            kernel_config['blockD'] = [256, 256, 256, 256]
            kernel_config['gridD'] = [(kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]
            zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256))
            scalerMont = ZFieldElExt(len(inv_roots_rdc_u256)).inv().reduce().as_uint256()
            scalerExt = ZFieldElExt(len(inv_roots_rdc_u256)).inv().as_uint256()
            zpoly_vector1 = np.concatenate((zpoly_vector1, [scalerExt],[scalerMont]))


            result_ifft3d,_ = cu_zpoly.kernelLaunch(zpoly_vector1, kernel_config, kernel_params,4)
            zpoly_vector_copy = np.copy(zpoly_vector)
            #intt_h_r = ntt_parallel2D_h(zpoly_vector, roots_rdc_u256, n_rows, fft_N, n_cols, fft_N, 1)
            intt_h_r = intt_h(zpoly_vector_copy,inv_roots_rdc_u256,1,MOD_FIELD)
            #intt_h_r = ntt_h(zpoly_vector_copy,roots_rdc_u256,MOD_FIELD)
            self.assertTrue(all(np.concatenate(result_ifft3d == intt_h_r)))


            kernel_params['forward'] = [1, 1, 1, 1]
            kernel_params['in_length'] = [2*nsamples+2, nsamples, nsamples, nsamples]
            kernel_params['stride'] = [2,1,1,1]
            zpoly_vector2 = np.concatenate((result_ifft3d, roots_rdc_u256, [scalerExt], [scalerMont]))
            result_fft3d,_ = cu_zpoly.kernelLaunch(result_ifft3d, kernel_config, kernel_params,4)

            self.assertTrue(all(np.concatenate(result_fft3d == zpoly_vector)))

    def test_08fft3D_65K(self):

        ntest_points = CUZPolyTest.ntest_points
        Nx = 9
        Ny = 7
        CUZPolyTest.nsamples = 1 << (Nx+Ny)
        nsamples = CUZPolyTest.nsamples
        ZUtils.NROOTS = CUZPolyTest.nsamples

        if os.path.exists(ZPOLY_datafile_1M):
           npzfile = np.load(ZPOLY_datafile_1M)
           roots_rdc_u256 = npzfile['roots_rdc_u256']
           
           #roots_rdc = npzfile['roots_rdc']
           #inv_roots_rdc = npzfile['inv_roots_rdc']
           #ZField.roots[0] = roots_rdc
           #ZField.inv_roots[0] = inv_roots_rdc
       
        else:
           roots_ext, inv_roots_ext = ZField.find_roots(1<<20, rformat_ext=True, primitive_root=primitive_roots[20])
           roots_rdc = [r.reduce() for r in roots_ext]
           inv_roots_rdc = [r.reduce() for r in inv_roots_ext]
           ZField.roots[0] = roots_rdc
           ZField.inv_roots[0] = roots_rdc
           roots_ext_u256 = np.asarray([x.as_uint256() for x in roots_ext])
           inv_roots_ext_u256 = np.asarray([x.as_uint256() for x in inv_roots_ext])
           roots_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc])
           inv_roots_rdc_u256 = np.asarray([x.as_uint256() for x in inv_roots_rdc])
          
           np.savez_compressed(ZPOLY_datafile_1M, roots_ext=roots_ext, roots_rdc=roots_rdc, inv_roots_ext=inv_roots_ext, inv_roots_rdc=inv_roots_rdc,
                                               roots_ext_u256 = roots_ext_u256, roots_rdc_u256=roots_rdc_u256, inv_roots_ext_u256=inv_roots_ext_u256,
                                               inv_roots_rdc_u256=inv_roots_rdc_u256)


        kernel_config = {}
        kernel_params = {}
        roots_rdc_u256 = roots_rdc_u256[::1<<(20-Nx-Ny)]
        CUZPolyTest.cu_zpoly = ZCUPoly(2*nsamples+2, seed=560)
        CUZPolyTest.u256 = U256(nsamples, seed=560)
        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        u256_p = CUZPolyTest.u256_p
        n_kernels = 4
        fft_xx = 5
        fft_xy = 4
        fft_yx = 4
        fft_yy = 3

        for niter in xrange(20):
            zpoly_vector = cu_zpoly.randu256(CUZPolyTest.nsamples,u256_p)

            # Test FFT kernel:

            #print "Test 8 : " + str(niter)+"                             \r",
            kernel_params['in_length'] = [2*CUZPolyTest.nsamples+2, nsamples, nsamples, nsamples]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [2,1,1,1]
            kernel_params['premod'] = [0,0,0,0]
            kernel_params['midx'] = [MOD_FIELD, MOD_FIELD, MOD_FIELD, MOD_FIELD]
            kernel_params['fft_Nx'] = [fft_xx, fft_xx, fft_yx, fft_yx] #xx,xx,yx,yx
            kernel_params['fft_Ny'] = [fft_xy, fft_xy, fft_yy, fft_yy] #xy,xy,yy,yy
            kernel_params['N_fftx'] = [Nx, Nx, Nx, Nx]
            kernel_params['N_ffty'] = [Ny, Ny, Ny, Ny]
            kernel_params['forward'] = [1, 1, 1, 1]
            kernel_params['as_mont'] = [1,1,1,1]

            kernel_config['smemS'] = [0, 0, 0, 0]
            kernel_config['blockD'] = [256, 256, 256, 256]
            kernel_config['gridD'] = [(kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]
            zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256))
            scalerMont = ZFieldElExt(len(roots_rdc_u256)).inv().reduce().as_uint256()
            scalerExt = ZFieldElExt(len(roots_rdc_u256)).inv().as_uint256()
            zpoly_vector1 = np.concatenate((zpoly_vector1, [scalerExt],[scalerMont]))

            #if os.path.exists(ZPOLY_input_npz_datafile_65K):
              #npzfile = np.load(ZPOLY_input_npz_datafile_65K)
              #zpoly_vector = npzfile['in_data']
              #zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256))

            result_fft3d,_ = cu_zpoly.kernelLaunch(zpoly_vector1, kernel_config, kernel_params,4)
            #cu_zpoly.saveFile(result_fft3d.reshape(-1), ZPOLY_inputyy_datafile_65K)
            #f = open("./testyy_65K.txt","w")
            #for r in result_fft3d[:64*1024]:
               #f.write(np.array2string(r,max_line_width=90))
               #f.write("\n")
            #f.close()

            r_u256 = ntt_parallel2D_h(zpoly_vector, roots_rdc_u256, Ny, fft_yy, Nx, fft_xx, 1)
            self.assertTrue(all(np.concatenate(result_fft3d == r_u256)))

            if not os.path.exists(ZPOLY_output_datafile_65K):
               cu_zpoly.saveFile(zpoly_vector.reshape(-1), ZPOLY_input_datafile_65K.encode("UTF-8"))
               np.savez_compressed(ZPOLY_input_npz_datafile_65K, in_data=zpoly_vector)
               cu_zpoly.saveFile(result_fft3d.reshape(-1), ZPOLY_output_datafile_65K.encode("UTF-8"))

    def test_09ifft3D_65K(self):

        ntest_points = CUZPolyTest.ntest_points
        Nx = 9
        Ny = 7
        CUZPolyTest.nsamples = 1 << (Nx+Ny)
        nsamples = CUZPolyTest.nsamples
        ZUtils.NROOTS = CUZPolyTest.nsamples

        if os.path.exists(ZPOLY_datafile_1M):
           npzfile = np.load(ZPOLY_datafile_1M)
           inv_roots_rdc_u256 = npzfile['inv_roots_rdc_u256']
           roots_rdc_u256 = npzfile['roots_rdc_u256']
           #roots_rdc = npzfile['roots_rdc']
           #inv_roots_rdc = npzfile['inv_roots_rdc']
           #ZField.roots[0] = roots_rdc
           #ZField.inv_roots[0] = inv_roots_rdc
       
        else:
           roots_ext, inv_roots_ext = ZField.find_roots(1<<20, rformat_ext=True, primitive_root=primitive_roots[20])
           roots_rdc = [r.reduce() for r in roots_ext]
           inv_roots_rdc = [r.reduce() for r in inv_roots_ext]
           ZField.roots[0] = roots_rdc
           ZField.inv_roots[0] = roots_rdc
           roots_ext_u256 = np.asarray([x.as_uint256() for x in roots_ext])
           inv_roots_ext_u256 = np.asarray([x.as_uint256() for x in inv_roots_ext])
           roots_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc])
           inv_roots_rdc_u256 = np.asarray([x.as_uint256() for x in inv_roots_rdc])
          
           np.savez_compressed(ZPOLY_datafile_1M, roots_ext=roots_ext, roots_rdc=roots_rdc, inv_roots_ext=inv_roots_ext, inv_roots_rdc=inv_roots_rdc,
                                               roots_ext_u256 = roots_ext_u256, roots_rdc_u256=roots_rdc_u256, inv_roots_ext_u256=inv_roots_ext_u256,
                                               inv_roots_rdc_u256=inv_roots_rdc_u256)


        kernel_config = {}
        kernel_params = {}
        inv_roots_rdc_u256 = inv_roots_rdc_u256[::1<<(20-Nx-Ny)]
        roots_rdc_u256 = roots_rdc_u256[::1<<(20-Nx-Ny)]
        CUZPolyTest.cu_zpoly = ZCUPoly(2*nsamples+2, seed=560)
        CUZPolyTest.u256 = U256(nsamples, seed=560)
        cu_zpoly = CUZPolyTest.cu_zpoly
        u256 = CUZPolyTest.u256
        u256_p = CUZPolyTest.u256_p
        n_kernels = 4
        fft_xx = 5
        fft_xy = 4
        fft_yx = 4
        fft_yy = 3

        for niter in xrange(20):
            zpoly_vector = cu_zpoly.randu256(CUZPolyTest.nsamples,u256_p)

            # Test FFT kernel:

            #print "Test 9 : " + str(niter)+"                                   \r",
            kernel_params['in_length'] = [2*CUZPolyTest.nsamples+2, nsamples, nsamples, nsamples]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [2,1,1,1]
            kernel_params['premod'] = [0,0,0,0]
            kernel_params['midx'] = [MOD_FIELD, MOD_FIELD, MOD_FIELD, MOD_FIELD]
            kernel_params['fft_Nx'] = [fft_xx, fft_xx, fft_yx, fft_yx] #xx,xx,yx,yx
            kernel_params['fft_Ny'] = [fft_xy, fft_xy, fft_yy, fft_yy] #xy,xy,yy,yy
            kernel_params['N_fftx'] = [Nx, Nx, Nx, Nx]
            kernel_params['N_ffty'] = [Ny, Ny, Ny, Ny]
            kernel_params['forward'] = [0, 0, 0, 0]
            kernel_params['as_mont'] = [1,1,1,1]

            kernel_config['smemS'] = [0, 0, 0, 0]
            kernel_config['blockD'] = [256, 256, 256, 256]
            kernel_config['gridD'] = [(kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0], \
                                     (kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]
            #zpoly_vector1 = np.concatenate((zpoly_vector, inv_roots_rdc_u256))
            zpoly_vector1 = np.concatenate((zpoly_vector, roots_rdc_u256))
            scalerMont = ZFieldElExt(len(inv_roots_rdc_u256)).inv().reduce().as_uint256()
            scalerExt = ZFieldElExt(len(inv_roots_rdc_u256)).inv().as_uint256()
            zpoly_vector1 = np.concatenate((zpoly_vector1, [scalerExt],[scalerMont]))


            result_ifft3d,_ = cu_zpoly.kernelLaunch(zpoly_vector1, kernel_config, kernel_params,4)
            zpoly_vector_copy = np.copy(zpoly_vector)
            intt_h_r = intt_h(zpoly_vector_copy,inv_roots_rdc_u256,1,MOD_FIELD)
            self.assertTrue(all(np.concatenate(result_ifft3d == intt_h_r)))


            kernel_params['forward'] = [1, 1, 1, 1]
            kernel_params['in_length'] = [2*nsamples+2, nsamples, nsamples, nsamples]
            kernel_params['stride'] = [2,1,1,1]
            zpoly_vector2 = np.concatenate((result_ifft3d, roots_rdc_u256, [scalerExt], [scalerMont]))
            result_fft3d,_ = cu_zpoly.kernelLaunch(zpoly_vector2, kernel_config, kernel_params,4)

            self.assertTrue(all(np.concatenate(result_fft3d == zpoly_vector)))

    def test_10mul_zpoly(self):

        ZUtils.NROOTS = CUZPolyTest.nsamples

        if os.path.exists(ZPOLY_datafile_1M):
           npzfile = np.load(ZPOLY_datafile_1M)
           inv_rootsM_rdc_u256 = npzfile['inv_roots_rdc_u256']
           rootsM_rdc_u256 = npzfile['roots_rdc_u256']
           #roots_rdc = npzfile['roots_rdc']
           #inv_roots_rdc = npzfile['inv_roots_rdc']
           #ZField.roots[0] = roots_rdc
           #ZField.inv_roots[0] = inv_roots_rdc
       
        else:
           roots_ext, inv_roots_ext = ZField.find_roots(1<<20, rformat_ext=True, primitive_root=primitive_roots[20])
           roots_rdc = [r.reduce() for r in roots_ext]
           inv_roots_rdc = [r.reduce() for r in inv_roots_ext]
           ZField.roots[0] = roots_rdc
           ZField.inv_roots[0] = roots_rdc
           roots_ext_u256 = np.asarray([x.as_uint256() for x in roots_ext])
           inv_roots_ext_u256 = np.asarray([x.as_uint256() for x in inv_roots_ext])
           rootsM_rdc_u256 = np.asarray([x.as_uint256() for x in roots_rdc])
           inv_rootsM_rdc_u256 = np.asarray([x.as_uint256() for x in inv_roots_rdc])
          
           np.savez_compressed(ZPOLY_datafile_1M, roots_ext=roots_ext, roots_rdc=roots_rdc, inv_roots_ext=inv_roots_ext, inv_roots_rdc=inv_roots_rdc,
                                               roots_ext_u256 = roots_ext_u256, roots_rdc_u256=roots_rdc_u256, inv_roots_ext_u256=inv_roots_ext_u256,
                                               inv_roots_rdc_u256=inv_roots_rdc_u256)


        for k in range(11,21):
          npoints_raw = randint((1<<(k-1))+1, 1<<k)
          fft_params = ntt_build_h(npoints_raw);
          CUZPolyTest.nsamples = npoints_raw + fft_params['padding']
          nsamples = CUZPolyTest.nsamples

          Nrows = fft_params['fft_N'][(1<<FFT_T_3D)-1]
          Ncols = fft_params['fft_N'][(1<<FFT_T_3D)-2]
          fft_yx = fft_params['fft_N'][(1<<FFT_T_3D)-3]
          fft_yy = Nrows - fft_yx
          fft_xx = fft_params['fft_N'][(1<<FFT_T_3D)-4]
          fft_xy = Ncols - fft_xx
          n_kernels1 = 4
          n_kernels2= 5

          CUZPolyTest.cu_zpoly = ZCUPoly(2*nsamples+2, seed=560)
          cu_zpoly = CUZPolyTest.cu_zpoly
          u256_p = CUZPolyTest.u256_p
          inv_roots_rdc_u256 = inv_rootsM_rdc_u256[::1<<(20-Nrows-Ncols)]
          roots_rdc_u256 = rootsM_rdc_u256[::1<<(20-Ncols-Nrows)]

          if k > 15:
            ntests = 1
          else :
             ntests = 10

          for niter in xrange(ntests):
               #print "Test 10 : " + str(k) + " " + str(niter)+"                               \r",
               kernel_config = {}
               kernel_params = {}
               X1 = cu_zpoly.randu256(CUZPolyTest.nsamples,u256_p)
               Y1 = cu_zpoly.randu256(CUZPolyTest.nsamples,u256_p)
               X1[int(CUZPolyTest.nsamples/2):] = np.zeros((int(CUZPolyTest.nsamples/2),NWORDS_256BIT),dtype=np.uint32)
               Y1[int(CUZPolyTest.nsamples/2):] = np.zeros((int(CUZPolyTest.nsamples/2),NWORDS_256BIT),dtype=np.uint32)
               X2 = np.copy(X1)
               Y2 = np.copy(Y1)

               # Test FFT kernel:
               kernel_params['in_length'] = [nsamples] * n_kernels1
               kernel_params['in_length'][0] = 2*CUZPolyTest.nsamples+2
               kernel_params['padding_idx'] = [2*CUZPolyTest.nsamples+2] * n_kernels1
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
               kernel_params['as_mont'] = [1] * n_kernels1
 
               kernel_config['smemS'] = [0] * n_kernels1
               kernel_config['blockD'] = [256] * n_kernels1
               kernel_config['gridD'] = [int((kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0])]*n_kernels1
               #kernel_config['gridD'][0] = 0
               kernel_config['return_val'] = [1] * n_kernels1

               kernel_config['kernel_idx']= [CB_ZPOLY_FFT3DXX, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY]


               zpoly_vector = np.concatenate((X1, roots_rdc_u256))
               scalerMont = ZFieldElExt(len(roots_rdc_u256)).inv().reduce().as_uint256()
               scalerExt = ZFieldElExt(len(roots_rdc_u256)).inv().as_uint256()
               zpoly_vector = np.concatenate((zpoly_vector, [scalerExt],[scalerMont]))

               X1S,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params,n_kernels1)

               kernel_params['in_length'][0] = nsamples
               kernel_params['stride'][0] = 1
               kernel_config['return_val'][0] = 0

               Y1S,_ = cu_zpoly.kernelLaunch(Y1, kernel_config, kernel_params,n_kernels1)

               kernel_params['padding_idx'] = [2*CUZPolyTest.nsamples+2] * n_kernels2
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
               kernel_params['as_mont'] = [1] * n_kernels2
  
               kernel_config['smemS'] = [0] * n_kernels2
               kernel_config['blockD'] = [256] * n_kernels2
               kernel_config['gridD'] = [(kernel_config['blockD'][0] + nsamples-1)/kernel_config['blockD'][0]]*n_kernels2
               kernel_config['return_val'] = [1] * n_kernels2
               kernel_config['kernel_idx']= [CB_ZPOLY_MULCPREV,
                                             CB_ZPOLY_FFT3DXXPREV, CB_ZPOLY_FFT3DXY, CB_ZPOLY_FFT3DYX, CB_ZPOLY_FFT3DYY ]

               fftmul_result,_ = cu_zpoly.kernelLaunch(X1S, kernel_config, kernel_params,n_kernels2)

  
               X2S = ntt_h(X2,roots_rdc_u256,MOD_FIELD)
               Y2S = ntt_h(Y2,roots_rdc_u256,MOD_FIELD)
               idx = 0
               for c1,c2 in zip(X2S, Y2S):
                  Y2S[idx] = montmult_h(c1, c2,1)
                  idx+=1
               result = intt_h(Y2S,inv_roots_rdc_u256,1,MOD_FIELD)

               self.assertTrue(all(np.concatenate(fftmul_result == result)))

    def test_11add_zpoly(self):

        u256_p = CUZPolyTest.u256_p
        CUZPolyTest.nsamples = 1024 
        nsamples = CUZPolyTest.nsamples
        cu_zpoly = ZCUPoly(2*nsamples, seed=560)


        for niter in xrange(CUZPolyTest.TEST_ITER):
            #print "Test 11 : " + str(niter)+ "                 \r",
            zpoly_vector1 = cu_zpoly.randu256(CUZPolyTest.nsamples, u256_p )
            zpoly_vector2 = cu_zpoly.randu256(CUZPolyTest.nsamples/2, u256_p)
            zpoly_vector3 = cu_zpoly.randu256(CUZPolyTest.nsamples/4, u256_p)
            kernel_config={}
            kernel_params={}

            # Test zpoly_add kernel two poly same length:
            kernel_params['in_length'] = [2*CUZPolyTest.nsamples]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [2]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['padding_idx'] = [nsamples]

            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['gridD'] = [0]
            kernel_config['kernel_idx']= [CB_ZPOLY_ADD]
            zpoly_vector = np.concatenate((zpoly_vector1, zpoly_vector1))
            result_add,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params,1)
   
            zpoly_1 = ZPoly.from_uint256(zpoly_vector1, reduced=True)
            zpoly_r = zpoly_1 + zpoly_1

            result_add_zpoly = ZPoly.from_uint256(result_add, reduced=True)
            self.assertTrue(zpoly_r == result_add_zpoly)

            # Test zpoly_add kernel two poly different length
            kernel_params['in_length'] = [len(zpoly_vector1) + len(zpoly_vector2)]
            kernel_params['out_length'] = len(zpoly_vector1)
            kernel_params['stride'] = [2]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['padding_idx'] = [len(zpoly_vector2)]

            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['gridD'] = \
                 [(kernel_config['blockD'][0] + \
                   2*kernel_params['padding_idx'][0]/kernel_params['stride'][0] - 1)/ kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_ADD]
            zpoly_vector = np.concatenate((zpoly_vector1, zpoly_vector2))
            result_add,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params,1)
   
            zpoly_1 = ZPoly.from_uint256(zpoly_vector1, reduced=True)
            zpoly_2 = ZPoly.from_uint256(zpoly_vector2, reduced=True)
            zpoly_r = zpoly_1 + zpoly_2

            result_add_zpoly = ZPoly.from_uint256(result_add, reduced=True)
            self.assertTrue(zpoly_r == result_add_zpoly)

            # Test zpoly_sub kernel two poly same length:
            kernel_params['in_length'] = [2*CUZPolyTest.nsamples]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [2]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['padding_idx'] = [nsamples]

            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['gridD'] = [0]
            kernel_config['kernel_idx']= [CB_ZPOLY_SUB]
            zpoly_vector = np.concatenate((zpoly_vector1, zpoly_vector1))
            result_sub,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params,1)
   
            zpoly_1 = ZPoly.from_uint256(zpoly_vector1, reduced=True)
            zpoly_r = zpoly_1 - zpoly_1

            result_sub_zpoly = ZPoly.from_uint256(result_sub, reduced=True)
            self.assertTrue(zpoly_r == result_sub_zpoly)

            # Test zpoly_sub kernel two poly different length
            kernel_params['in_length'] = [len(zpoly_vector1) + len(zpoly_vector2)]
            kernel_params['out_length'] = len(zpoly_vector1)
            kernel_params['stride'] = [2]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['padding_idx'] = [len(zpoly_vector2)]

            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['gridD'] = \
                 [(kernel_config['blockD'][0] + \
                   2*kernel_params['padding_idx'][0]/kernel_params['stride'][0] - 1)/ kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_SUB]
            zpoly_vector = np.concatenate((zpoly_vector1, zpoly_vector2))
            result_sub,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params,1)
   
            zpoly_1 = ZPoly.from_uint256(zpoly_vector1, reduced=True)
            zpoly_2 = ZPoly.from_uint256(zpoly_vector2, reduced=True)
            zpoly_r = zpoly_1 - zpoly_2

            result_sub_zpoly = ZPoly.from_uint256(result_sub, reduced=True)
            self.assertTrue(zpoly_r == result_sub_zpoly)

            # Test zpoly_mul kernel two poly same length:
            kernel_params['in_length'] = [2*CUZPolyTest.nsamples]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [2]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['padding_idx'] = [nsamples]

            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['gridD'] = [0]
            kernel_config['kernel_idx']= [CB_ZPOLY_MULC]
            zpoly_vector = np.concatenate((zpoly_vector1, zpoly_vector1))
            result_mulc,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params,1)
   
            zpoly_1 = ZPoly.from_uint256(zpoly_vector1, reduced=True)
            zpoly_r = ZPoly([x*x for x in zpoly_1.get_coeff()])

            result_mulc_zpoly = ZPoly.from_uint256(result_mulc, reduced=True)
            self.assertTrue(zpoly_r == result_mulc_zpoly)

            # Test zpoly_mulK kernel 
            kernel_params['in_length'] = [CUZPolyTest.nsamples+1]
            kernel_params['out_length'] = nsamples
            kernel_params['stride'] = [1]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['padding_idx'] = [nsamples]

            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['gridD'] = \
                 [(kernel_config['blockD'][0] + \
                   (kernel_params['in_length'][0]-1)/kernel_params['stride'][0] - 1)/ kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_MULK]
            zpoly_vector = np.concatenate(([zpoly_vector1[0]], zpoly_vector1))
            result_mulK,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params,1)
   
            zpoly_1 = ZPoly.from_uint256(zpoly_vector1, reduced=True)
            zpoly_r = zpoly_1.get_coeff()[0] * zpoly_1
            zpoly_e = ZFieldElRedc(zpoly_1.get_coeff()[0]).extend() * zpoly_1.extend()

            result_mulK_zpoly = ZPoly.from_uint256(result_mulK, reduced=True)

            self.assertTrue(zpoly_r == result_mulK_zpoly)
            self.assertTrue(zpoly_e.reduce() == result_mulK_zpoly)

            # Test zpoly_mulc kernel 
            kernel_params['in_length'] = [CUZPolyTest.nsamples]
            kernel_params['out_length'] = nsamples/2
            kernel_params['stride'] = [2]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]

            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['gridD'] = \
                 [(kernel_config['blockD'][0] + \
                   (kernel_params['in_length'][0]-1)/kernel_params['stride'][0] - 1)/ kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_MULC]
            result_mulC,_ = cu_zpoly.kernelLaunch(zpoly_vector1, kernel_config, kernel_params,1)
   
            zpoly_1 = ZPoly.from_uint256(zpoly_vector1, reduced=True)
            n_coeff = zpoly_1.get_degree()+1
            zpoly_r = ZPoly([c1 * c2 for c1, c2 in zip(zpoly_1.get_coeff()[0:int(n_coeff/2)],zpoly_1.get_coeff()[int(n_coeff/2):])])

            result_mulC_zpoly = ZPoly.from_uint256(result_mulC, reduced=True)

            self.assertTrue(zpoly_r == result_mulC_zpoly)

            # Test zpoly_mulc prev kernel 
            kernel_params['in_length'] = [CUZPolyTest.nsamples/2]
            kernel_params['out_length'] = nsamples/2
            kernel_params['stride'] = [1]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]

            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['gridD'] = \
                 [(kernel_config['blockD'][0] + \
                   (kernel_params['in_length'][0]-1)/kernel_params['stride'][0] - 1)/ kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_MULCPREV]
            result_mulCprev,_ = cu_zpoly.kernelLaunch(zpoly_vector1[:int(nsamples/2)], kernel_config, kernel_params,1)
   
            zpoly_1 = ZPoly.from_uint256(zpoly_vector1[:int(nsamples/2)], reduced=True)
            zpoly_r = ZPoly([c1 * c2 for c1, c2 in zip(zpoly_1.get_coeff(),zpoly_r.get_coeff())])

            result_mulCprev_zpoly = ZPoly.from_uint256(result_mulCprev, reduced=True)

            self.assertTrue(zpoly_r == result_mulCprev_zpoly)


             # Test zpoly_madd prev
            kernel_params['in_length'] = [2*CUZPolyTest.nsamples]
            kernel_params['out_length'] = len(zpoly_vector1)
            kernel_params['stride'] = [2]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['padding_idx'] = [nsamples]

            kernel_config['return_val'] = [0]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['multiple_inload'] = 1
            kernel_config['gridD'] = \
                 [(kernel_config['blockD'][0] + \
                   2*kernel_params['padding_idx'][0]/kernel_params['stride'][0] - 1)/ kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_ADD]
            zpoly_vector = np.concatenate((zpoly_vector1, zpoly_vector1))
            result_add,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params,1)
   
            kernel_params['in_length'] = [len(zpoly_vector2)+1]
            kernel_params['out_length'] = len(zpoly_vector1)
            kernel_params['stride'] = [1]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['padding_idx'] = [len(zpoly_vector2)]

            kernel_config['return_val'] = [0]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['gridD'] = \
                 [(kernel_config['blockD'][0] + \
                   2*kernel_params['padding_idx'][0]/kernel_params['stride'][0] - 1)/ kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_MADPREV]
            zpoly_vector = np.concatenate(([zpoly_vector1[0]],zpoly_vector2))
            result_add,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params,1)
   
            kernel_params['in_length'] = [len(zpoly_vector3)]
            kernel_params['out_length'] = len(zpoly_vector1)
            kernel_params['stride'] = [2]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['padding_idx'] = [len(zpoly_vector3) ]

            kernel_config['return_val'] = [1]
            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['gridD'] = \
                 [(kernel_config['blockD'][0] + \
                   2*kernel_params['padding_idx'][0]/kernel_params['stride'][0] - 1)/ kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_ADDPREV]
            zpoly_vector = zpoly_vector3
            result_add,_ = cu_zpoly.kernelLaunch(zpoly_vector, kernel_config, kernel_params,1)
   
            zpoly_1 = ZPoly.from_uint256(zpoly_vector1, reduced=True)
            zpoly_2 = ZPoly.from_uint256(zpoly_vector2, reduced=True)
            zpoly_3 = ZPoly.from_uint256(zpoly_vector3, reduced=True)
            zpoly_r = zpoly_1 + zpoly_1 + zpoly_1.get_coeff()[0] * zpoly_2 + zpoly_3

            result_add_zpoly = ZPoly.from_uint256(result_add, reduced=True)
            self.assertTrue(zpoly_r == result_add_zpoly)

    def test_12divsnarks(self):

        u256_p = CUZPolyTest.u256_p
        CUZPolyTest.nsamples = 1024 
        nd = 2047
        n = 2048
        m = 20000
        ne = nd + n
        nsamples = m + nd
        cu_zpoly = ZCUPoly(nsamples+1, seed=560)


        for niter in xrange(CUZPolyTest.TEST_ITER):
            zpoly_vector1 = np.zeros((nsamples+1,NWORDS_256BIT),dtype=np.uint32)
            zpoly_vector1[nd:] = cu_zpoly.randu256(m+1, u256_p )
            kernel_config={}
            kernel_params={}

            # Test zpoly_add kernel two poly same length:
            kernel_params['in_length'] = [nsamples+1]
            #kernel_params['out_length'] = nsamples - 2*ne + nd + 1
            kernel_params['out_length'] = m - n + 1
            kernel_params['stride'] = [1]
            kernel_params['premod'] = [0]
            kernel_params['midx'] = [MOD_FIELD]
            kernel_params['padding_idx'] = [ne]
            kernel_params['forward'] = [nd]

            kernel_config['smemS'] = [0]
            kernel_config['blockD'] = [256]
            kernel_config['gridD'] = \
                 [(kernel_config['blockD'][0] + \
                   kernel_params['in_length'][0]-2*ne+nd - 1)/ kernel_config['blockD'][0]]
            kernel_config['kernel_idx']= [CB_ZPOLY_DIVSNARKS]

            result_snarks,_ = cu_zpoly.kernelLaunch(zpoly_vector1, kernel_config, kernel_params,1)
            result_snarks_complete = np.zeros((nsamples-ne+1,NWORDS_256BIT),dtype=np.uint32)
            result_snarks_complete[:nsamples-2*ne+nd+1] = result_snarks
            result_snarks_complete[nsamples-2*ne+nd+1:] = zpoly_vector1[-ne+nd:]
  
             
            zpoly_1 = ZPoly.from_uint256(zpoly_vector1[nd:], reduced=True)
            zpoly_r = zpoly_1.poly_div_snarks(n)

            result_snarks_zpoly = ZPoly.from_uint256(result_snarks_complete, reduced=True)

            if zpoly_r != result_snarks_zpoly:
               print (niter)
            #self.assertTrue(zpoly_r == result_snarks_zpoly)


if __name__ == "__main__":
    unittest.main()
