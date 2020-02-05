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
// File name  : prof_zpoly.py
//
// Date       : 13/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Profile zpoly CUDA Module
//
// ------------------------------------------------------------------

NOTE : To launch for i in {21..24}; do  CUDA_VISIBLE_DEVICES=1 python3 prof_zpoly.py $((2**$i)); done
"""
import os,sys, os.path
import numpy as np
import logging

sys.path.append('../../src/python')

from bigint import *
from zutils import *
from zfield import *
from zpoly import *
from constants import *
from cuda_wrapper import *

sys.path.append(os.path.abspath(os.path.dirname('../../config/')))

import cusnarks_config as cfg

logging.basicConfig(level=logging.DEBUG,
                    format='[%(processName)s] %(asctime)s %(levelname)s %(message)s')


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
from pycusnarks import *

def profile_zpoly(nsamples):

    niter = 2
    kernel_stats = []
    roots_f = cfg.get_roots_file()
    n_bits_roots = cfg.get_n_roots()
    batch_size = 1<< 20
    curve_data = ZUtils.CURVE_DATA['BN128']
    # Initialize Group 
    ZField(curve_data['prime'])
    # Initialize Field 
    ZField.add_field(curve_data['prime_r'],curve_data['factor_data'])
    ECC.init(curve_data)
    ZPoly.init(MOD_FIELD)
    cuzpoly = ZCUPoly(5*batch_size  + 2, seed=560)
    roots_rdc_u256 = readU256DataFile_h(
                    roots_f.encode("UTF-8"),
                    1<<n_bits_roots, 1<<n_bits_roots) 


    print("Input Samples : "+str(nsamples))

    zpoly_vector_A = cuzpoly.rand(nsamples)
    zpoly_vector_B = cuzpoly.rand(nsamples)

    ifft_params = ntt_build_h(zpoly_vector_A.shape[0])

    for i in range(niter):
       pH,t1 = zpoly_interp_and_mul_cuda(
                                       cuzpoly,
                                       np.concatenate((zpoly_vector_A, zpoly_vector_B)),
                                       ifft_params, 
                                       ZField.get_field(), 
                                       roots_rdc_u256,
                                       batch_size, n_gpu=1)
       if i :
           kernel_stats.append(t1)

    if niter > 1:
      logging.info("Max : %s [s], Min : %s [s], Mean : %s[s]" % (np.max(kernel_stats), np.min(kernel_stats), np.mean(kernel_stats)))

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        nsamples = 1<<21
    else :
        nsamples = sys.argv[1]

    profile_zpoly(int(nsamples))
