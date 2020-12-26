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
// File name  : groth_prover
//
// Date       : 27/01/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Zero Kowledge Groth prover implementation
# 


// Description:
//    
//   TODO
//    
// ------------------------------------------------------------------

"""

import json,ast
import os.path
import signal
import numpy as np
import time
import linecache
from subprocess import call, run, PIPE
import logging
import logging.handlers as handlers
from multiprocessing import RawArray, Process, Pipe
from ctypes import c_uint32


from zutils import ZUtils
import random
from zfield import *
from ecc import *
from zpoly import *
from constants import *
from pysnarks_utils import *
import multiprocessing as mp
from cuda_wrapper import *
import socket
import sys
import json_socket


sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False

sys.path.append(os.path.abspath(os.path.dirname('../../config/')))

import cusnarks_config as cfg

class GrothProver(object):
    def __init__(self, proving_key_f, verification_key_f=None,curve='BN128', out_pk_f=None, out_pk_format=FMT_MONT, 
                 n_gpus=1,start_server=1, max_batch_size=20, seed=None, snarkjs=None, keep_f=None):
        # Check valid folder exists
        if keep_f is None:
            print ("Repo directory needs to be provided\n")
            sys.exit(1)

        timestamp = str(int(time.time()))
        self.keep_f = gen_reponame(keep_f, sufix="_PROVER")

        # Logger setup
        self.logger = logging.getLogger('cusnarks')
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ### Create new log file every day. Keep latest 7
        logHandler = handlers.TimedRotatingFileHandler(
                                self.keep_f + '/log'+'_'+timestamp,
                                when='D',
                                interval=1,
                                backupCount=7)
        logHandler.setLevel(logging.INFO)
        logHandler.setFormatter(formatter)
        self.logger.addHandler(logHandler)

        if not use_pycusnarks :
          self.logger.error('PyCUSnarks shared library not found. Exiting...')
          sys.exit(1)

        if seed is not None:
          self.seed = seed
        else:
          x = os.urandom(4)
          self.seed = int(x.hex(),16)

        random.seed(self.seed) 

        # Configures who (CPU or GPU) computes NTT, first mexp block and last mexp block
        # Default setup is that NTT is computed by CPU and mexps are computed by GPU if exists.
        # We leave this configurable because depending on the HW, it may be possible to speed
        # process is last mexp is computed by CPU
        self.compute_first_mexp_gpu = True
        self.compute_last_mexp_gpu = True

        # Two modes available, which define format of pk_bin variable
        #  0 : Trusted setup is done with cusnarks => pk format is .bin
        #  1 : Truested setup is done with snarkjs => pk format is .zkey or zkey2
        self.pkbin_mode = 0

        # There is one mexp configuration mode that allows to have tables precomputed.
        # This methods is nor used because even though method is faster than pippenger's,
        #   it requires losts of hard drive space and access to the HDD is the bottleneck.

        # file with the roots of unity
        self.roots_f = cfg.get_roots_file()
        # number roots. Limits the size of the circuit
        self.n_bits_roots = cfg.get_n_roots()

        # N streams define the number of parallel instances per GPU. The more the better,
        #  although, it consumes more memory (proportional to the number of streams)
        self.max_n_streams = get_nstreams()
        self.n_streams = self.max_n_streams

        # Number of GPUS. For a CPU only platform, this number will automatically be set to 0.
        self.n_cpu = get_nprocs_h()
        self.max_cpu = get_nprocs_h()
        self.max_gpu = get_ngpu(max_used_percent=99.)
        self.n_gpu = min(self.max_gpu,n_gpus)
        if 'CUDA_VISIBLE_DEVICES' in os.environ and \
           len(os.environ['CUDA_VISIBLE_DEVICES']) > 0:
              self.n_gpu = min(
                                 self.n_gpu,
                                 len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
                              )
        # Depending on whether or not GPUS are available, we configure where NTT and 
        # first and last mexp blocks are computed.
        if self.n_gpu == 0 :
          self.logger.info('No available GPUs')
          self.compute_first_mexp_gpu = False
          self.compute_last_mexp_gpu = False
        elif not self.compute_first_mexp_gpu and not self.compute_last_mexp_gpu:
          self.n_gpu = 0
          self.n_streams = 1

        self.last_n_streams = self.n_streams
        if self.n_gpu > 0:
           # When working with GPU, we work with batches of elliptic points.  
           # 1 << batch_size  defines the size of the batch.
           # It is left as a parameter to check if speed improves with larger batches. 
           #  Batches cannot be arbitrarily large beacuse GPU memory restrictions.
           info = get_gpu_info() 
           available_gpu_buffer = min([info[i]['mem_total'] for i in range(len(info))]) << 20 
           required_gpu_buffer = self.n_streams  \
                                    * 2  * 4 * NWORDS_FP * ECP_JAC_OUTDIMS * 2 
           batch_size = min(int(math.log(available_gpu_buffer / required_gpu_buffer, 2)),max_batch_size)

           self.max_batch_size = 1 << batch_size
           self.batch_size = min(1<<20, self.max_batch_size)
           self.last_batch_size = self.batch_size
           self.ecbn128_buffer_size = max(2*self.max_batch_size,2<<(8+8+4))

           required_gpu_buffer = self.n_streams  \
                                    * self.ecbn128_buffer_size \
                                    * 2  * 4 * NWORDS_FP * ECP_JAC_OUTDIMS 

        # If mexp is done in GPU, initialize GPU memories
        if self.compute_first_mexp_gpu :
          self.ecbn128  = ECBN128(self.ecbn128_buffer_size,   seed=self.seed)
          self.ec2bn128 = self.ecbn128
        else:
          self.ecbn128  = None
          self.ec2bn128 = None
        
        # Initialize  pk, vk, proof and public data files
        self.out_proving_key_f = out_pk_f
        self.out_proving_key_format = out_pk_format
        self.proving_key_f = proving_key_f
        self.verification_key_f = verification_key_f
        self.out_proof_f = None
        self.out_public_f = None

        # Initialize Curve parameters
        self.curve_data = ZUtils.CURVE_DATA[curve]
        ZField(self.curve_data['prime'])
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data)
        ZPoly.init(MOD_FR)
        ZField.set_field(MOD_FP)

        self.pk = getPK()
        self.verify = 2
        self.stop_client = mp.Value('i',0)
        self.active_client = mp.Value('i',0)
        self.status_client = mp.Value('i',0)

        self.public_signals = None
        self.witness_f = None
        self.snarkjs = snarkjs
        self.verify_en = None

        ZField.set_field(MOD_FR)
        self.t_GP = {}

        init_h()
        
        # ZK set to one enables zero knowledge to the proof. If Zero knowledge is not 
        # needed, scalars r and s are set to 0, and B1 mexp can be bypassed. Final value of ZK is set when client is launched
        self.zk = 1
        
        self.logger.info('Initializing memories...')

        if self.proving_key_f.endswith('.zkey'):
          self.logger.info("Regenerating verification key")
          #generate verification_key
          self.launch_snarkjs("verification_key")

          #generate pkbin
          out_fname = self.proving_key_f[:-5]+".zkey2"
          zKeyToPkFile_h(out_fname.encode("UTF-8"),self.proving_key_f.encode("UTF-8"))
          self.proving_key_f = out_fname
          self.pkbin_mode = 1
        elif self.proving_key_f.endswith('.zkey2'):
          self.pkbin_mode = 1

        # pis
        self.pi_a_eccf1 = np.zeros((ECP_JAC_INDIMS, NWORDS_256BIT), dtype=np.uint32)
        self.pi_b_eccf2 = np.zeros((ECP2_JAC_INDIMS, NWORDS_256BIT), dtype=np.uint32)
        self.pi_c_eccf1 = np.zeros((ECP_JAC_INDIMS, NWORDS_256BIT), dtype=np.uint32)
        self.pi_b1_eccf1 = np.zeros((ECP_JAC_INDIMS, NWORDS_256BIT), dtype=np.uint32)
        self.init_ec_val = np.zeros((3*ECP_JAC_INDIMS+ECP2_JAC_INDIMS)*max(self.n_gpu,1)*self.n_streams*NWORDS_256BIT, dtype=np.uint32)

        #scl r,s, rs
        self.r_scl = np.zeros(NWORDS_256BIT, dtype=np.uint32)
        self.s_scl = np.zeros(NWORDS_256BIT, dtype=np.uint32)
        self.neg_rs_scl = np.zeros(NWORDS_256BIT, dtype=np.uint32)
         
        # Shared variables
        # We define a set of shared variables that can be used by different processes without need to 
        # copy them. We put large arrays

        # self.pk -> stores PK. PK can be read using pkbinsh_get method
        # self.scl_array -> witness
        # self.pA_T : poly A
        # self.pB_T : poly B
        # self.roots_rdc_u256 : roots of unity
        # pi's  results of multiexponentiations (pi_a, pi_b1, pi_b2, pi_c)

        pkbin_nWords = int(os.path.getsize(self.proving_key_f)/4)
        self.pk_sh = RawArray(c_uint32, pkbin_nWords)
        self.pk = np.frombuffer(self.pk_sh, dtype=np.uint32)
        self.logger.info('Reading Proving Key...')
        readU256PKFileTo_h(self.proving_key_f.encode("UTF-8"), self.pk)
        #readU256PKFileTo_h(self.proving_key_f.encode("UTF-8"), self.pk_sh, self.pk.shape[0])
            
        pkbin_vars = pkbinsh_get(self.pk_sh,['nVars','domainSize', 'delta_1', 'hExps', 'nPublic'])
        self.nVars = int(pkbin_vars[0][0])
        self.domainSize = int(pkbin_vars[1][0])
        delta_1 = pkbin_vars[2]
        hExps = pkbin_vars[3]
        hExps[2*(self.domainSize+1)*NWORDS_256BIT:2*(self.domainSize+2)*NWORDS_256BIT] = delta_1
        nPublic = int(pkbin_vars[4][0])
        if self.pkbin_mode == 1:
            self.m = self.domainSize +1
        else:
            self.m = self.domainSize 

        if self.domainSize > 1<<self.n_bits_roots:
          self.logger.error('Insufficient number of roots (%s) for a domainSize of %s',
                            1<<self.n_bits_roots, self.domainSize)
          sys.exit(1)

        # scl_array (witness)
        witLen = max(self.nVars, 2*self.domainSize + 8 )
        self.scl_array_sh = RawArray(c_uint32, witLen * NWORDS_256BIT)     
        self.scl_array = np.frombuffer(
                     self.scl_array_sh, dtype=np.uint32).reshape((witLen, NWORDS_256BIT))
        self.scl_array_shape = self.scl_array.shape

        # pA_T
        self.pA_T_sh = RawArray(c_uint32, 2 * self.domainSize * NWORDS_256BIT)
        self.pA_T = np.frombuffer(
                     self.pA_T_sh, dtype=np.uint32).reshape((2 * self.domainSize, NWORDS_256BIT))
        self.pA_T_shape = self.pA_T.shape
        np.copyto(
                self.pA_T,
                np.zeros((2 * self.domainSize, NWORDS_256BIT), dtype=np.uint32))
        self.pB_T = self.pA_T[self.domainSize:]

        # Roots
        # Format roots so that they can be easily used during NTT (NTT, interpolation and multiplication).
        ifft_params = ntt_build_h(self.domainSize)
        nroots = ifft_params['levels'] + 1
        nroots2 = int(nroots/2)
        if int(nroots) % 2 :
          nroots3 = nroots2+1
        else:
          nroots3 = nroots2-1
        self.roots_rdc_u256_sh = RawArray(c_uint32,  ((1 << nroots) + (1<< nroots2) + (1 << nroots3)) * NWORDS_256BIT)
        self.roots_rdc_u256 =\
              np.frombuffer(
                      self.roots_rdc_u256_sh,
                      dtype=np.uint32).reshape(( (1<< nroots) + (1 << nroots2) + (1 << nroots3), NWORDS_256BIT))
        self.roots_rdc_u256_shape = self.roots_rdc_u256.shape
        if nroots > self.n_bits_roots:
             print("Insufficient precomputed roots... Aborting")
             sys.exit(1)
        np.copyto(
              self.roots_rdc_u256[:1<<nroots],
              readU256DataFile_h(
                  self.roots_f.encode("UTF-8"),
                  1<<self.n_bits_roots, 1<<nroots) )
        np.copyto(
              self.roots_rdc_u256[1<<nroots:(1<<nroots)+(1<<nroots2)],
              readU256DataFile_h(
                  self.roots_f.encode("UTF-8"),
                  1<<self.n_bits_roots, 1<<nroots2) )
        np.copyto(
              self.roots_rdc_u256[(1<<nroots)+(1<<nroots2):(1<<nroots)+(1<<nroots2)+(1<<nroots3)],
              readU256DataFile_h(
                  self.roots_f.encode("UTF-8"),
                  1<<self.n_bits_roots, 1<<nroots3) )



        self.logger.info('#################################### ')
        self.logger.info('Initializing Groth prover with the following parameters :')
        self.logger.info(' - curve : %s',curve)
        self.logger.info(' - proving_key_f : %s', proving_key_f)
        self.logger.info(' - verification_key_f : %s',verification_key_f)
        self.logger.info(' - out_pk_f : %s',out_pk_f)
        self.logger.info(' - out_pk_format : %s',out_pk_format) 
        self.logger.info(' - seed : %s', self.seed)
        self.logger.info(' - snarkjs : %s', snarkjs)
        self.logger.info(' - keep_f : %s', keep_f)
        self.logger.info(' - n available GPUs : %s', self.n_gpu)
        self.logger.info(' - n available CPUs : %s', get_nprocs_h())
        self.logger.info(' - compute first Mexp in GPU : %s', self.compute_first_mexp_gpu)
        self.logger.info(' - N Constraints : %s', self.nVars)
        self.logger.info(' - Domain Size : %s', self.domainSize)
        self.logger.info(' - N Public : %s', nPublic)
        self.logger.info('#################################### ')
 
        if self.out_proving_key_f is not None:
             if self.out_proving_key_f.endswith('.bin'):
               pk_bin = pkvars_to_bin(self.out_proving_key_format, EC_T_AFFINE, self.pk, ext=False)
               writeU256DataFile_h(pk_bin, self.out_proving_key_f.encode("UTF-8"))
               del pk_bin

        self.ec_lable = np.asarray(['A', 'B2', 'B1', 'C','hExps'])
                             # Point Name, cuda pointer, step, idx, ec2, pi
        self.ec_type_dict = {'A'     : [self.ecbn128,  2, 0, 0, 0],
                             'B2'    : [self.ec2bn128, 4, 1, 1, 1],
                             'B1'    : [self.ecbn128,  2, 2, 0, 2 ],
                             'C'     : [self.ecbn128,  2, 3, 0, 3 ],
                             'hExps' : [self.ecbn128,  2, 4, 0, 3 ] }

        if self.compute_first_mexp_gpu:
          # Init Mexp tables
          self.init_p_Mexp()
          self.parent_conn_CPU, self.child_conn_CPU = Pipe()

          if not start_server:
             self.startProcesses(self.nVars)

    def init_p_Mexp(self):
        pk_bin = pkbinsh_get(self.pk_sh,['nVars', 'nPublic'])

        nVars = pk_bin[0][0]
        nPublic = pk_bin[1][0]

        next_gpu_idx = 0
        first_stream_idx = min(self.n_streams-1,1)

        nsamples = self.nVars+2
        nsamplesC = self.nVars - nPublic -1

        self.tableA = buildDispatchTable( math.ceil(nsamples/self.batch_size),
                                         1,
                                         self.n_gpu, self.n_streams, self.batch_size,
                                         0, nsamples,
                                         start_pidx=0,
                                         start_gpu_idx=0,
                                         ec_lable = np.asarray(['A']))

        self.tableB1 = buildDispatchTable( math.ceil(nsamples/self.batch_size),
                                         1,
                                         self.n_gpu, self.n_streams, self.batch_size,
                                         0, nsamples,
                                         start_pidx=0,
                                         start_gpu_idx=0,
                                         ec_lable = np.asarray(['B1']))

        self.tableC = buildDispatchTable( math.ceil(nsamplesC/self.batch_size),
                                         1,
                                         self.n_gpu, self.n_streams, self.batch_size,
                                         0, nsamplesC,
                                         start_pidx=0,
                                         start_gpu_idx=0,
                                         ec_lable = np.asarray(['C']))

        self.tableB2 = buildDispatchTable( math.ceil(nsamples/self.batch_size),
                                         1,
                                         self.n_gpu, self.n_streams, self.batch_size,
                                         0, nsamples,
                                         start_pidx=0,
                                         start_gpu_idx=0,
                                         ec_lable = np.asarray(['B2']))

        self.mexp1Batch = np.zeros((self.batch_size*3, NWORDS_FP),dtype=np.uint32)
        self.mexp2Batch = np.zeros((self.batch_size*5, NWORDS_FP),dtype=np.uint32)

        if self.pkbin_mode:
            m = self.domainSize
        else:
            m = self.domainSize -1

        nsamplesH = m + 1 +1 +1 +1  # a + b1 + delta_1 + c
        self.tableH = buildDispatchTable( math.ceil(nsamplesH/self.batch_size),
                                         1,
                                         self.n_gpu, self.n_streams, self.batch_size,
                                         0, nsamplesH,
                                         start_pidx=0,
                                         start_gpu_idx=0,
                                         ec_lable = np.asarray(['hExps']))

    def FFT_CPU(self, w, wnElems):

        self.logger.info(' Evaluating QAP')
        start = time.time()
        pk_bin = pkbinsh_get(self.pk_sh,['nVars', 'polsA', 'polsB'])
        roots_rdc_u256 = np.frombuffer(self.roots_rdc_u256_sh, dtype=np.uint32).reshape(self.roots_rdc_u256_shape)
        pA_T = np.frombuffer(self.pA_T_sh, dtype=np.uint32).reshape(self.pA_T_shape)
        m = self.domainSize
        nVars = pk_bin[0][0]

        if self.pkbin_mode == 0:
          pA = pk_bin[1][:m*NWORDS_256BIT]
          pB = pk_bin[2][:m*NWORDS_256BIT]
          np.copyto(pA_T, np.zeros(self.pA_T.shape, dtype=np.uint32))
          self.logger.info(' Process server - Evaluating Poly A...')
          self.evalPoly(pA_T[:m],w[:wnElems], pA, nVars, m, MOD_FR)
          self.logger.info(' Process server - Evaluating Poly B...')
          self.evalPoly(pA_T[m:],w[:wnElems], pB, nVars, m, MOD_FR)
          self.logger.info(' Process server - Completed Evaluating Poly B...')

        else :
          pA = pk_bin[1]
          self.logger.info(' Process server - Evaluating Polys...')
          mpoly_evals_h(np.reshape(pA_T,-1),np.reshape(w,-1), pA, m, self.n_cpu,MOD_FR)

        t_eval = time.time()
        self.t_GP['Eval'] = time.time()-start


        self.logger.info(' Process server - Calculate H...')
        """
        print(m)
        writeU256DataFile_h(np.reshape(pA_T,-1), "/usr/src/app/cusnarksdata/pA.bin".encode("UTF-8"))
        writeU256DataFile_h(np.reshape(roots_rdc_u256,-1), "/usr/src/app/cusnarksdata/roots.bin".encode("UTF-8"))
        """
        # TODO : add polH to share vars so that i do not need to copy polH.
        # I probably need to pass polH as input param to ntt_interpol and use that instead of
        #  get_Mtranspose. I may save 2-4 seconds 
        polH = ntt_interpolandmul_h(
                   np.reshape(pA_T,-1),
                   np.reshape(roots_rdc_u256, -1),
                   m,
                   2,
                   self.pkbin_mode,
                   self.n_cpu,
                   MOD_FR)
        self.logger.info(' Process server - Calculate H done')
        if self.pkbin_mode == 0:
          polH = polH[m:-1]
        else :
          m=m+1
        self.t_GP['H'] = time.time()-t_eval

        return polH, m


    def Mexp1_CPU(self,w):

        offset = 0
        total_words = 0
        if self.compute_first_mexp_gpu is False:
          start2 = time.time()
          self.logger.info(' Process server - Starting First Mexp...')
          pk_bin = pkbinsh_get(self.pk_sh,['A','B2','B1','C','nPublic', 'nVars'])
          nPublic = pk_bin[4][0]
          nVars = pk_bin[5][0]

          ep_vector = pk_bin[0][:(nVars+2)*NWORDS_256BIT*ECP_JAC_INDIMS]

          if self.stop_client.value == 0:
             w[nVars] = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32)
             w[nVars+1] = self.r_scl
             np.copyto(self.pi_a_eccf1,
                    ec_jacreduce_pippen_h(
                            np.reshape( w[:nVars+2], -1),
                            ep_vector,
                            0,
                            self.n_cpu,
                            MOD_FP, 1,1, 1, 1, 1))
             tt = time.time()
             self.logger.info(' Process server - Mexp A Done... %s',tt-start2)


          ep_vector = pk_bin[1][:(nVars+2)*NWORDS_256BIT*ECP2_JAC_INDIMS]

          if self.stop_client.value == 0:
             w[nVars+1] = self.s_scl
             np.copyto(self.pi_b_eccf2,
                    ec_jacreduce_pippen_h(
                        np.reshape( w[:nVars+2], -1),
                               ep_vector,
                               1,
                               self.n_cpu,
                               MOD_FP, 1,1, 1, 1, 1)
                       )
   
             tt1 = time.time()
             self.logger.info(' Process server - Mexp B2 Done...%s',tt1-tt)
             tt = time.time()


          if self.zk and self.stop_client.value == 0:
            ep_vector = pk_bin[2][:(nVars+2)*NWORDS_256BIT*ECP_JAC_INDIMS]

            np.copyto(self.pi_b1_eccf1,
                    ec_jacreduce_pippen_h(
                        np.reshape( w[:nVars+2], -1),
                              ep_vector,
                              0,
                              self.n_cpu,
                              MOD_FP, 1,1, 1, 1, 1)
                    )

            tt = time.time()

            self.logger.info(' Process server - Mexp B1  Done...%s',tt-tt1)

          if self.pkbin_mode == 0:
             ep_vector = pk_bin[3][(nPublic+1)*NWORDS_256BIT*ECP_JAC_INDIMS:nVars*NWORDS_256BIT*ECP_JAC_INDIMS]
          else :
            ep_vector = pk_bin[3][:(nVars-nPublic-1)*NWORDS_256BIT*ECP_JAC_INDIMS]

          if self.stop_client.value == 0:
             np.copyto(self.pi_c_eccf1,
                    ec_jacreduce_pippen_h(
                            np.reshape( w[nPublic+1:nVars], -1),
                            ep_vector,
                            0,
                            self.n_cpu,
                            MOD_FP, 1,0, 1, 1, 1)
                    )
             tt1 = time.time()
             self.logger.info(' Process server - Mexp C Done... %s',tt1-tt)

          end2 = time.time()

          self.t_GP['Mexp1'] = (end2 - start2)
          self.logger.info(' Process server - Completed First Mexp...')

    def Mexp2_CPU(self, polH, m):
        tt = time.time()
        pk_bin = pkbinsh_get(self.pk_sh,['delta_1', 'hExps'])
        #m = polH.shape[0]
        self.logger.info(' Process server - Starting Last Mexp...')
        self.logger.info(' Process server - hExps Mexp common part started ...')

        polH[m-1] = self.neg_rs_scl
        scalar_vector = np.reshape( polH, -1)
        self.logger.info(' Process server - scalar copied.')
        pk_bin[1][(m-1)*ECP_JAC_INDIMS*NWORDS_FP:(m)*ECP_JAC_INDIMS*NWORDS_FP] = pk_bin[0]
        EP_vector =  pk_bin[1][:m*NWORDS_256BIT*ECP_JAC_INDIMS]
        self.logger.info(' Process server - vector copied.')

        if self.stop_client.value == 0:
           np.copyto(self.pi_c_eccf1,
                ec_jacreduce_pippen_h(
                       scalar_vector,
                       EP_vector,
                       0,
                       self.n_cpu,
                       MOD_FP, 0,1, 1, 1, 1)
                   )
           self.t_GP['Mexp2']= time.time()-tt

           self.logger.info(' Process server - hExps Mexp common part completed ...%s',self.t_GP['Mexp2'])


    def pysnarkP_CPU(self, conn, wnElems):
        self.logger.info(' Launching Poly Process Client')
        w = np.frombuffer(self.scl_array_sh, dtype=np.uint32).reshape(self.scl_array_shape)

        polH, m = self.FFT_CPU(w, wnElems)
        m=polH.shape[0]

        #write polH once MEXP is done (not before)
        conn.recv()
        self.logger.info(' Process server - Copying polH...')
        np.copyto(w[:m-1], polH[:m-1])

        conn.send([self.t_GP['Eval'], self.t_GP['H']])
        conn.close()

        self.logger.info(' Process server - Completed')

    def startGPServer(self):    
           self.port_first = 8192
           self.port_second = 8193
           self.proof_id = 0
           self.proof_repo = []
           pkbin_vars = pkbinsh_get(self.pk_sh,['nVars'])
           nVars = int(pkbin_vars[0][0])
           self.logger.info('Launching GP Server')
           p = Process(target=self.startServer, args = (self.port_first,0))
           p.start()
           try:
              self.startServer(self.port_second, nVars)
           except Exception as e:
              exc_type, exc_obj, tb = sys.exc_info()
              #traceback.print_stack()
              f = tb.tb_frame
              lineno = tb.tb_lineno
              filename = f.f_code.co_filename
              self.logger.info('Exception occurred. Server stopped  in file %s:%s: %s',filename, lineno,  e)
              run(['killall', '-9', 'python3'])

    def startServer(self, port, nVars):    
           self.logger.info('Launching GP Server')
           jsocket = json_socket.jsonSocket(port=port)

           s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
           s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
           s.bind((jsocket.host,jsocket.port))
           s.listen(1)
           print('Server listening on port ' +str(port) +' ready...')
           self.status_client.value = 1
           while True: # Accept connections from multiple clients
               conn, addr = s.accept()
               msg = jsocket.receive_message(conn)
               if len(msg):
                 # Call some action and return results
                 parsed_dict = ast.literal_eval(json.loads(json.dumps(msg)))
                 if 'stop_server' in parsed_dict and parsed_dict['stop_server'] == 1:
                    self.logger.info('Stopping server')
                    os.kill(os.getppid(), signal.SIGTERM)
                    sys.exit(1)
                 elif 'is_alive' in parsed_dict:
                    new_msg = {}
                    #new_msg['status']=1
                    new_msg['status']=self.status_client.value
                    jsocket.send_message(new_msg, conn)
                    conn.close()
                    continue

                 elif 'stop_client' in parsed_dict:
                     self.stop_client.value = 1
                     self.logger.info('Stopping client...')
                     conn.close()

                 elif 'pid' in parsed_dict:
                     new_msg = {'pid' : os.getpid()}
                     jsocket.send_message(new_msg, conn)
                     conn.close()

                 elif 'status' in parsed_dict:
                     self.proof_repo[-1]['result'] = parsed_dict['status']
                     del parsed_dict['status']
                     self.proof_repo[-1].update(parsed_dict)
                     conn.close()

                 elif 'list' in parsed_dict:
                     if len(self.proof_repo) > int(parsed_dict['list']):
                       if len(self.proof_repo):
                         new_msg = dict(self.proof_repo[int(parsed_dict['list'])])
                       else :
                         new_msg = {
                              'witness_f' : "",
                              'proof_f' : "",
                              'public_data_f' : "",
                              'verify_en' : 0,
                              'proof_id' : -1,
                              'result' : 2,
                              'Init' : [ 0.0, 0.0],
                              'Read_W' : [ 0.0, 0.0],
                              'Eval' : [ 0.0, 0.0],
                              'Mexp' : [ 0.0, 0.0],
                              'Mexp1' : [ 0.0, 0.0],
                              'Mexp2' : [ 0.0, 0.0],
                              'H' : [ 0.0, 0.0],
                              'Proof' : 0.0,
                             }
                       jsocket.send_message(new_msg, conn)
                     elif int(parsed_dict['list']) == -1:
                       new_msg = dict(self.proof_repo[int(parsed_dict['list'][-1])])
                       jsocket.send_message(new_msg, conn)
                     conn.close()
                     
                 elif 'witness_f' in parsed_dict and nVars==0:
                     if self.active_client.value:
                       self.logger.info("Proof ongoing. Discarding new request")
                     else:
                        query = {
                              'witness_f' : parsed_dict['witness_f'],
                              'proof_f' : parsed_dict['proof_f'],
                              'public_data_f' : parsed_dict['public_data_f'],
                              'verify_en' : parsed_dict['verify_en'],
                              'proof_id' : self.proof_id,
                              'result' : -1,
                              'Init' : [ 0.0, 0.0],
                              'Read_W' : [ 0.0, 0.0],
                              'Eval' : [ 0.0, 0.0],
                              'Mexp' : [ 0.0, 0.0],
                              'Mexp1' : [ 0.0, 0.0],
                              'Mexp2' : [ 0.0, 0.0],
                              'H' : [ 0.0, 0.0],
                              'Proof' : 0.0,
                              'zk' : parsed_dict['zk'],
                              'last_mexp_gpu' : parsed_dict['last_mexp_gpu'] if self.n_gpu else 0,
                              'batch_size' : parsed_dict['batch_size'], 
                              'n_streams' : parsed_dict['n_streams'],
                              'cpu' : parsed_dict['cpu'],
                             }
                        self.proof_id+=1
                        self.logger.info('Request for new proof received in primary server')
                        self.proof_repo.append(query)

                        p = json_socket.jsonSocket(port=self.port_second)
                        p.send_message(query)

                     conn.close()

                 elif 'witness_f' in parsed_dict and nVars:
                   self.logger.info('Request for new proof received in secondary server')
                   conn.close()
                   # Initialize CPU Process
                   if self.compute_first_mexp_gpu:
                      self.startProcesses(nVars)

                   self.proof(
                         parsed_dict['witness_f'],
                         parsed_dict['proof_f'], parsed_dict['public_data_f'],
                         verify_en=int(parsed_dict['verify_en']),
                         zk=parsed_dict['zk'],
                         last_mexp_gpu=parsed_dict['last_mexp_gpu'],
                         batch_size=parsed_dict['batch_size'],
                         n_streams=parsed_dict['n_streams'],
                         cpu=parsed_dict['cpu'])

                   new_msg = dict(self.t_GP)
                   new_msg['status'] = self.verify

                   p = json_socket.jsonSocket(port=self.port_first)
                   p.send_message(new_msg)
   

    def startProcesses(self, nVars):
            self.logger.info('Starting Processes...')
            if self.n_gpu > 0:
              self.p_CPU = \
                        Process(
                              target=self.pysnarkP_CPU,
                                 args = (
                                        self.child_conn_CPU,
                                        nVars))
            else:
              self.p_CPU = None

    def initECVal(self):
        np.copyto( 
                  self.pi_a_eccf1,
                  np.reshape(
                        np.concatenate((
                                 ECC.zero[ZUtils.FRDC].as_uint256(),
                                  ECC.one[ZUtils.FRDC].as_uint256())), 
                        (-1,NWORDS_256BIT)) )

        np.copyto(
                self.pi_b_eccf2,
                np.reshape(
                     np.concatenate((
                            ECC.zero[ZUtils.FRDC].as_uint256(),
                            ECC.zero[ZUtils.FRDC].as_uint256(),
                            ECC.one[ZUtils.FRDC].as_uint256(),
                            ECC.zero[ZUtils.FRDC].as_uint256())),
                     (-1,NWORDS_256BIT)) )

        np.copyto(
                  self.pi_c_eccf1,
                      np.reshape(
                           np.concatenate((
                                    ECC.zero[ZUtils.FRDC].as_uint256(),
                                    ECC.one[ZUtils.FRDC].as_uint256())),
                           (-1,NWORDS_256BIT)) )

        np.copyto(
                 self.pi_b1_eccf1,
                       np.reshape(
                          np.concatenate((
                                    ECC.zero[ZUtils.FRDC].as_uint256(),
                                    ECC.one[ZUtils.FRDC].as_uint256())), 
                       (-1,NWORDS_256BIT)) )


        self.init_ec_val = np.tile( 
                             np.asarray([ self.pi_a_eccf1,
                                          self.pi_b_eccf2,
                                          self.pi_b1_eccf1,
                                          self.pi_c_eccf1], dtype=np.object), 
                             (max(self.n_gpu,1), self.n_streams, 1))


    def __del__(self):
       release_h()

    def read_witness_data(self):
       start = time.time()
       ## Open and parse witness data
       if os.path.isfile(self.witness_f):

           pkbin_vars = pkbinsh_get(self.pk_sh,['nVars'])
           nVars = int(pkbin_vars[0][0])

           if self.witness_f.endswith('.json'):
             f = open(self.witness_f,'r')
             np.copyto(
                 self.scl_array[:nVars],
                 np.reshape(
                     np.asarray(
                         [BigInt(c).as_uint256() for c in ast.literal_eval(json.dumps(json.load(f)))],
                                                                       dtype=np.uint32),(-1, NWORDS_256BIT))
                      )
             f.close()

           elif self.witness_f.endswith('.txt'):
             with open(self.witness_f, 'r') as f:
               np.copyto(
                 self.scl_array[:nVars],
                 np.reshape(
                     np.asarray(
                          [BigInt(c).as_uint256() for c in f]),(-1,NWORDS_256BIT))
                     )

           elif self.witness_f.endswith('.bin'):
             np.copyto(
                self.scl_array[:nVars],
                readWitnessFile_h(self.witness_f.encode("UTF-8"),1, nVars ))

           elif self.witness_f.endswith('.dat'):
             np.copyto(
                self.scl_array[:nVars],
                readWitnessFile_h(self.witness_f.encode("UTF-8"),0, nVars ))

           elif self.witness_f.endswith('.wtns'):
             np.copyto(
                self.scl_array[:nVars],
                readWtnsFile_h(self.witness_f.encode("UTF-8"), nVars ))

           elif self.witness_f.endswith('.wshm'):
             readSharedMWtnsFile_h(np.reshape(self.scl_array,-1),self.witness_f.encode("UTF-8"), nVars )

           else:
             np.copyto(
                self.scl_array[:nVars],
                readWitnessShmem_h(nVars) )

       else:
          self.logger.error('Witness file %s doesn\'t exist', self.witness_f)
          return 0

       self.t_GP['Read_W'] = time.time() - start

       return 1

    def logTimeResults(self):
      self.t_GP['Init'] = [round(self.t_GP['Init'],4), round(100*self.t_GP['Init']/self.t_GP['Proof'],2)] 
      self.t_GP['Read_W'] = [round(self.t_GP['Read_W'],4), round(100*self.t_GP['Read_W']/self.t_GP['Proof'],2)] 
      self.t_GP['Mexp'] = self.t_GP['Mexp1'] + self.t_GP['Mexp2']
      self.t_GP['Mexp'] = [round(self.t_GP['Mexp'],4), round(100*self.t_GP['Mexp']/self.t_GP['Proof'],2)] 
      self.t_GP['Mexp1'] = [round(self.t_GP['Mexp1'],4), round(100*self.t_GP['Mexp1']/self.t_GP['Proof'],2)] 
      self.t_GP['Mexp2'] = [round(self.t_GP['Mexp2'],4), round(100*self.t_GP['Mexp2']/self.t_GP['Proof'],2)] 
      self.t_GP['Eval'] = [round(self.t_GP['Eval'],4), round(100*self.t_GP['Eval']/self.t_GP['Proof'],2)] 
      self.t_GP['H'] = [round(self.t_GP['H'],4), round(100*self.t_GP['H']/self.t_GP['Proof'],2)] 
      self.t_GP['Proof'] = round(self.t_GP['Proof'],4)
      self.logger.info('')
      self.logger.info('')
      self.logger.info('#################################### ')
      self.logger.info('Total Time to generate proof : %s seconds', self.t_GP['Proof'])
      self.logger.info('')
      self.logger.info('------ Initialization          : %s ', str(self.t_GP['Init'][0]) + ' sec. (' + str(self.t_GP['Init'][1]) + ' %)')
      self.logger.info('------ Time Read Witness       : %s ', str(self.t_GP['Read_W'][0]) + ' sec. (' + str(self.t_GP['Read_W'][1]) + ' %)')
      self.logger.info('------ Time Multi-exp          : %s ', str(self.t_GP['Mexp'][0]) + ' sec.(' + str(self.t_GP['Mexp'][1]) + ' %)')
      self.logger.info('------ * Time Multi-exp 1      : %s ', str(self.t_GP['Mexp1'][0]) + ' sec.(' + str(self.t_GP['Mexp1'][1]) + ' %)')
      self.logger.info('------ * Time Multi-exp 2      : %s ', str(self.t_GP['Mexp2'][0]) + ' sec.(' + str(self.t_GP['Mexp2'][1]) + ' %)')
      self.logger.info('------ Time Poly Calculation   : %s ', str(self.t_GP['Eval'][0]) + ' sec. (' + str(self.t_GP['Eval'][1]) + ' %)')
      self.logger.info('------ Time Compute Poly H     : %s ', str(self.t_GP['H'][0]) + 'sec. (' + str(self.t_GP['H'][1]) + ' %)')
      self.logger.info('#################################### ')
      self.logger.info('')
      self.logger.info('')

    def proof(self, witness_f, out_proof_f , out_public_f, verify_en=0, zk=1, last_mexp_gpu=1, batch_size=20, n_streams=3, cpu=None):

      # Initaliization
      start = time.time()

      self.out_proof_f = out_proof_f
      self.out_public_f = out_public_f
      self.witness_f = witness_f

      self.verify_en = verify_en
      self.t_GP = {}
      self.t_GP['Init'] = 0
      self.t_GP['Eval'] = 0
      self.t_GP['Mexp'] = 0
      self.t_GP['Mexp1'] = 0
      self.t_GP['Mexp2'] = 0
      self.t_GP['H'] = 0
      self.t_GP['Proof'] = 0
      self.stop_client.value = 0
      self.zk = zk

      if cpu is None or cpu == 0 or cpu > self.max_cpu:
          self.n_cpu = get_nprocs_h()
      else:
          self.n_cpu =  cpu

      if self.n_gpu :
          self.compute_last_mexp_gpu = last_mexp_gpu == 1
          self.n_streams = max(min(n_streams, N_STREAMS_PER_GPU),2)

          if (1 << batch_size)  <=  self.max_batch_size :
             self.batch_size = 1 << batch_size
          else:
             self.batch_size = self.max_batch_size
      else:
          self.batch_size = 0

      if self.active_client.value :
          return

      self.active_client.value = 1
      self.status_client.value = 2

      if (self.verify_en):
        self.verify = 0
      else :
        self.verify = 2

      self.initECVal()

      if self.compute_first_mexp_gpu and  \
         (self.last_batch_size != self.batch_size or self.last_n_streams != self.n_streams) :
          # Init Mexp process
          self.logger.info("Computing batch tables for Mexp...")
          self.init_p_Mexp()
          self.last_batch_size = self.batch_size
          self.last_n_streams = self.n_streams

      self.logger.info('#################################### ')
      self.logger.info("Starting new proof...")
      self.logger.info(' - out_proof_f : %s',out_proof_f)
      self.logger.info(' - out_public_f : %s',out_public_f)
      self.logger.info(' - witness_f : %s',witness_f)
      self.logger.info(' - verify_en : %s', verify_en)
      self.logger.info(' - batch_size : %s', self.batch_size)
      self.logger.info(' - gpus used : %s', self.n_gpu)
      self.logger.info(' - cpus used : %s', self.n_cpu)
      self.logger.info(' - streams used: %s', self.n_streams)
      self.logger.info(' - zero knowledge enabled : %s', self.zk)
      self.logger.info(' - compute last Mexp with GPU : %s', self.compute_last_mexp_gpu)
      self.logger.info('#################################### ')
      self.logger.info('')
      self.logger.info('')

      self.t_GP['Init'] = time.time() - start
      ##### Starting proof

      # Proof fails for internal reasons
      if self.gen_proof() == 0:
         self.verify = -2
         self.t_GP['Init'] = [0,0]
         self.t_GP['Read_W'] = [0,0]
         self.t_GP['Mexp'] = [0,0]
         self.t_GP['Mexp1'] = [0,0]
         self.t_GP['Mexp2'] = [0,0]
         self.t_GP['Eval'] = [0,0]
         self.t_GP['H'] = [0,0]
         self.active_client.value = 0
         self.status_client.value = 1
         return self.verify
      
      if self.stop_client.value:
          self.active_client.value = 0
          self.verify = -2
          self.status_client.value = 1
          self.t_GP['Init'] = [0,0]
          self.t_GP['Read_W'] = [0,0]
          self.t_GP['Mexp'] = [0,0]
          self.t_GP['Mexp1'] = [0,0]
          self.t_GP['Mexp2'] = [0,0]
          self.t_GP['Eval'] = [0,0]
          self.t_GP['H'] = [0,0]
          return self.verify

      self.t_GP['Proof'] = time.time() - start

      self.logger.info(" Proof completed" )
      self.logger.info('#################################### ')
      
      self.logTimeResults()

      self.logger.info(" Writing PData to %s", self.out_public_f)
      self.write_pdata()
      self.logger.info(" Writing Proof to %s", self.out_proof_f)
      self.write_proof()
      self.logger.info(" Testing Proof")
      self.test_results()

      #copy_input_files([self.out_proof_f, self.out_public_f, self.out_proving_key_f, witness_f],self.keep_f)

      self.active_client.value = 0
      self.status_client.value = 1
      return self.verify

    def test_results(self):
      proof_r = True
      snarkjs = {}
      snarkjs['verify'] = 0

      if self.verify_en:
        self.logger.info('#################################### ')
        self.logger.info("Calling snarkjs verify to verify proof ....")
        self.logger.info("")
        snarkjs = self.launch_snarkjs("verify")
        if snarkjs['verify'] == 0:
          self.logger.info("Verification SUCCEDED")
          self.verify = 1
        else:
          self.logger.info("Verification FAILED")
          self.verify = 0
        self.logger.info('#################################### ')

      return

    def launch_snarkjs(self, mode):
        snarkjs = {}
        if mode=="proof" :
          snarkjs['p_f'] = self.keep_f + '/' +  'tmp_p_f.json'
          snarkjs['pd_f'] = self.keep_f + '/' + 'tmp_pd_f.json'
          # snarkjs setup is launched with circuit.json, format extended. Convert input file if necessary
          if self.witness_f.endswith('.json') and self.proving_key_f.endswith('.json') and self.pk['k_binformat']==FMT_EXT :
             witness_file = self.witness_f
             proving_key_file = self.proving_key_f
          elif  self.witness_f.endswith('.json'):
            if self.proving_key_f.endswith('.bin'):
               witness_file = self.witness_f
               proving_key_file = self.proving_key_f.replace('.bin','_cpy.json')
            else: 
               witness_file = self.witness_f
               proving_key_file = self.proving_key_f.replace('.json','_cpy.json')
            proving_key_file = self.keep_f + '/' + proving_key_file.rsplit('/', 1)[1]
            pk_dict = pkvars_to_json(FMT_EXT,EC_T_AFFINE, self.pk)
            pk_json = json.dumps(pk_dict, indent=4, sort_keys=True)
            f = open(proving_key_file, 'w')
            print(pk_json, file=f)
            f.close()
          else:
            self.logger.error(' Witness file %s needs to be .json', self.witness_f)
            sys.exit(1)

          debug_command = ""
          debug_file = ""

          call([self.snarkjs, "proof", "-w", witness_file, "--pk", proving_key_file, "-p", snarkjs['p_f'],"--pub",snarkjs['pd_f'], debug_command, debug_file])

        elif mode == "verify":
          snarkjs['p_f'] = self.out_proof_f
          snarkjs['pd_f'] = self.out_public_f
          if self.verification_key_f is not None and self.verification_key_f.endswith('.json') :
             verification_key_file = self.verification_key_f
          else :
             self.logger.error(' To launch snarkjs, verification file %s needs to be a json file', self.verification_key_f)
             sys.exit(1)
        
          if self.pkbin_mode == 0:
             snarkjs['verify'] = call([self.snarkjs, "verify", "--vk", verification_key_file, "-p", snarkjs['p_f'],"--pub",snarkjs['pd_f']])
          else :
             result = run(["snarkjs", "groth16", "verify", verification_key_file,snarkjs['pd_f'], snarkjs['p_f']], stdout=PIPE)
             snarkjs['verify'] = 1
             if result.stdout.decode('utf-8').split()[2] == "OK!":
                snarkjs['verify'] = 0

        elif mode == "verification_key":
          self.verification_key_f = self.proving_key_f[:-5]
          self.verification_key_f =  self.verification_key_f + "_vk.json"
        
          call(["snarkjs", "zkey", "export", "verificationkey", self.proving_key_f, "verification_key.json"])
          call(["mv", "verification_key.json", self.verification_key_f])
       
        return snarkjs

    def Mexp1_GPU(self):
        start = time.time()
        self.logger.info(' Mexp A started...')

        pk_bin = pkbinsh_get(self.pk_sh,['A','B2','B1','C', 'nPublic'])
        nPublic = pk_bin[4][0]

        # TODO AA
        save_scl = self.scl_array[self.nVars:self.nVars+2]
        self.scl_array[self.nVars] = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32)
        self.scl_array[self.nVars + 1] = self.r_scl
        scl_vector = self.scl_array

        ecp_vector = pk_bin[0][:(self.nVars+2)*ECP_JAC_INDIMS*NWORDS_FP]

        if self.stop_client.value == 0 :
            self.findECPointsDispatch( self.tableA, scl_vector, ecp_vector, ec2=0)
            self.assignECPvalues('A')

        # B2
        self.logger.info(' Mexp B2 started...')

        scl_vector[self.nVars + 1] = self.s_scl

        ecp_vector = pk_bin[1][:(self.nVars+2)*ECP2_JAC_INDIMS*NWORDS_FP]

        if self.stop_client.value == 0 :
          self.findECPointsDispatch( self.tableB2, scl_vector, ecp_vector, ec2=1 )
          self.assignECPvalues('B2')

        # B1
        if self.zk:
           self.logger.info(' Mexp B1 started...')
           
           #scl_vector[-1] = self.s_scl
           ecp_vector = pk_bin[2][:(self.nVars+2)*ECP_JAC_INDIMS*NWORDS_FP]

           if self.stop_client.value == 0 :
              self.findECPointsDispatch( self.tableB1, scl_vector, ecp_vector)
              self.assignECPvalues('B1')

        # C 
        self.logger.info(' Mexp C  started...')

        if self.pkbin_mode == 1:
             ecp_vector = pk_bin[3][:(self.nVars-nPublic-1)*NWORDS_FP*ECP_JAC_INDIMS]
        else:
             ecp_vector = pk_bin[3][(nPublic+1)*ECP_JAC_INDIMS*NWORDS_FP:(self.nVars)*ECP_JAC_INDIMS*NWORDS_FP]

        used_streams = []
        if self.stop_client.value == 0 :
          if self.compute_last_mexp_gpu == False:
              used_streams = self.findECPointsDispatch( self.tableC, scl_vector, ecp_vector, reduce_en = True, scl_offset=nPublic+1)
          else:
              used_streams = self.findECPointsDispatch( self.tableC, scl_vector, ecp_vector, reduce_en = False, scl_offset=nPublic+1)

        # Assign collected values to pi's
        if self.compute_last_mexp_gpu == False and self.stop_client.value == 0:
           self.assignECPvalues('C')
        self.logger.info(' First Mexp completed GPU...')
        
        self.t_GP['Mexp1'] = time.time() - start
        self.scl_array[self.nVars:self.nVars+2] = save_scl 

        return used_streams


    def Mexp2_GPU(self, used_streams):
        start = time.time()
        pk_bin = pkbinsh_get(self.pk_sh,['hExps','delta_1'])
        self.logger.info(' Starting Last Mexp GPU...')

        if self.pkbin_mode:
            m = self.domainSize
        else:
            m = self.domainSize - 1

        if self.stop_client.value == 0:
           #TODO AA
           #self.scl_array[self.nVars:self.nVars+2] = save_scl 
           self.scl_array[m] = self.s_scl
           self.scl_array[m+1] = self.r_scl
           self.scl_array[m+2] = self.neg_rs_scl
           self.scl_array[m+3] = np.asarray([1,0,0,0,0,0,0,0],dtype=np.uint32)
   
           scl_vector = self.scl_array[:m+4]
   
           pk_bin[0][m*ECP_JAC_INDIMS*NWORDS_FP:(m+1)*ECP_JAC_INDIMS*NWORDS_FP] = np.reshape(self.pi_a_eccf1,-1)
           pk_bin[0][(m+1)*ECP_JAC_INDIMS*NWORDS_FP:(m+2)*ECP_JAC_INDIMS*NWORDS_FP] = np.reshape(self.pi_b1_eccf1,-1)
           pk_bin[0][(m+2)*ECP_JAC_INDIMS*NWORDS_FP:(m+3)*ECP_JAC_INDIMS*NWORDS_FP] = pk_bin[1]
           pk_bin[0][(m+3)*ECP_JAC_INDIMS*NWORDS_FP:(m+4)*ECP_JAC_INDIMS*NWORDS_FP] = np.reshape(self.pi_c_eccf1,-1)
   
           ecp_vector = pk_bin[0][:(m+4)*ECP_JAC_INDIMS*NWORDS_FP]
   
           self.logger.info(' Starting Dispatch...')
           self.findECPointsDispatch( self.tableH, scl_vector, ecp_vector, ec2=0, used_streams=used_streams)
           self.logger.info(' Collecting Results...')
        
           self.assignECPvalues('C')
   
           end = time.time()
           self.t_GP['Mexp2'] = (end - start)
           self.logger.info(' Last Mexp completed')

    def Mexp2_GPU_CPU(self):
        start = time.time()
        scalar_v =  np.reshape(
                         np.concatenate((
                                  np.asarray([[1,0,0,0,0,0,0,0]], dtype=np.uint32),
                                  [self.s_scl],
                                  [self.r_scl] )),
                         -1)
        ep_v = np.reshape(
                         np.concatenate((
                                  self.pi_c_eccf1,
                                  self.pi_a_eccf1,
                                  self.pi_b1_eccf1 )),
                            -1)
        self.logger.info(' Process server - hExps Mexp ZK part started ...')
        np.copyto(self.pi_c_eccf1,
           ec_jacreduce_h(
                       scalar_v,
                       ep_v,
                       "".encode("UTF-8"),
                       0,
                       0,
                       2,
                       0,
                       self.n_cpu,
                       MOD_FP, 1, 1, 1))

        self.logger.info(' Process server - hExps Mexp ZK part completed ...')
        self.t_GP['Mexp2'] += time.time() - start

    def initRS(self):
        if self.zk:
          np.copyto(
                self.r_scl,
                BigInt(
                    random.randint(
                         1,
                         ZField.get_extended_p().as_long()-1)).as_uint256())
  
          np.copyto(
                self.s_scl,
                BigInt(
                      random.randint(
                             1,
                             ZField.get_extended_p().as_long()-1)).as_uint256())
        else:
          np.copyto(
                self.r_scl,
                np.asarray([0,0,0,0,0,0,0,0],dtype=np.uint32) )
  
          np.copyto(
                self.s_scl,
                np.asarray([0,0,0,0,0,0,0,0],dtype=np.uint32) )

        r_mont = to_montgomeryN_h(np.reshape(self.r_scl, -1), MOD_FR)
        np.copyto(
                 self.neg_rs_scl,
                  montmult_neg_h(
                             np.reshape(r_mont, -1),
                             np.reshape(self.s_scl, -1), MOD_FR))

        self.logger.info('#################################### ')
        self.logger.info(' Random numbers :')
        self.logger.info(' - r : %s',str(BigInt.from_uint256(self.r_scl).as_long()) )
        self.logger.info(' - s : %s',str(BigInt.from_uint256(self.s_scl).as_long()) )
        self.logger.info('#################################### ')


    def gen_proof(self ):

        pk_bin = pkbinsh_get(self.pk_sh,['nPublic'])
        nPublic = pk_bin[0][0]

        # Intialize r & s
        self.initRS()

        # Read Witness
        self.logger.info(' Reading Witness...')
        if self.read_witness_data() == 0:
           return 0

        self.public_signals = np.copy(self.scl_array[1:nPublic+1])
       
        # FFT
        if self.compute_first_mexp_gpu == False:
           polH, m = self.FFT_CPU(self.scl_array, self.nVars)

        else :
           self.p_CPU.start()

        # Mexp 1
        if self.compute_first_mexp_gpu:
          used_streams = self.Mexp1_GPU()

          # Synchronize CPU/GPU
          if self.compute_last_mexp_gpu:
            self.parent_conn_CPU.send([])

          [self.t_GP['Eval'], self.t_GP['H']] = self.parent_conn_CPU.recv()
          self.p_CPU.terminate()
          self.p_CPU.join()

        else:
          self.Mexp1_CPU(self.scl_array)

        # Mexp 2
        if self.compute_last_mexp_gpu:
          self.Mexp2_GPU(used_streams)
        else :
          self.Mexp2_CPU(polH, m)
          if self.zk == 1:
             self.Mexp2_GPU_CPU()
     
        return 1
 

    def assignECPvalues(self, label):
        """
        Labels : 'A', 'B2', 'B1', 'C' . hExps = 'C'
        """
        EC_idx = self.ec_type_dict[label][4]


        if label == 'B2' :
             ecp = ec2_jacaddreduce_h(
                                np.reshape(np.concatenate(np.reshape(self.init_ec_val[:,:,EC_idx],-1)),-1),
                                MOD_FP,
                                1,   # to affine
                                1,   # Add z coordinate to inout
                                1)   # strip z coordinate from affine result
        else:
             ecp = ec_jacaddreduce_h(
                                np.reshape(np.concatenate(np.reshape(self.init_ec_val[:,:,EC_idx],-1)),-1),
                                MOD_FP,
                                1,   # to affine
                                1,   # Add z coordinate to inout
                                1)   # strip z coordinate from affine result

        if label == 'A':
            np.copyto(self.pi_a_eccf1, ecp)
        elif label == 'B2' :
            np.copyto(self.pi_b_eccf2, ecp)
        elif label == 'B1' :
            np.copyto(self.pi_b1_eccf1, ecp)
        else :
            np.copyto(self.pi_c_eccf1, ecp)

        for idx,v in enumerate(self.init_ec_val[:,:,EC_idx][0]):
            self.init_ec_val[:,:,EC_idx][0][idx]  = np.zeros(v.shape, dtype=np.uint32)


    def streamsDel(self, dispatch_table):
       for bidx,p in enumerate(dispatch_table):
          P = p[0]
          cuda_ec128 = self.ec_type_dict[P][0]
          gpu_id = p[3]
          stream_id = p[4]
          #cuda_ec128.streamSync(gpu_id,stream_id)
          cuda_ec128.streamDel(gpu_id,stream_id)

    def findECPointsDispatch(self, dispatch_table, scl_vector, ecp_vector, ec2=0, reduce_en=True, used_streams=None, scl_offset=0):

       ZField.set_field(MOD_FP)
       n_par_batches = self.n_gpu * max((self.n_streams - 1),1)
       pending_dispatch_table = []
       n_dispatch=len(pending_dispatch_table)
       indims = ECP_JAC_INDIMS
       outdims = ECP_JAC_OUTDIMS
       edims = ECP_JAC_OUTDIMS
       if ec2 == 1:
           indims = ECP2_JAC_INDIMS
           outdims = ECP2_JAC_OUTDIMS
           edims = ECP2_JAC_INDIMS + 1
    
       if used_streams is None:
           used_streams = [] 
           for i in range(self.n_gpu):
             used_streams.append(set([]))

       for bidx, p in enumerate(dispatch_table):
          #Retrieve point name : A,B1,B2,..
          P = p[0]    
          # Retrieve cuda pointer
          cuda_ec128 = self.ec_type_dict[P][0]
          # Retrieve EC type : EC -> 0, EC2 -> 1
          start_idx = p[1]
          end_idx   = p[2]
          gpu_id    = p[3]
          stream_id = p[4]
          pidx = self.ec_type_dict[P][4]

          #nsamples needs to be multiple of 128
          start1=time.time()
          nsamples = int((end_idx - start_idx+128-1)/(128))*128
          offset = nsamples - (end_idx - start_idx)
          if bidx != len(dispatch_table) -1:
              batch = self.mexp1Batch
              if ec2:
                batch = self.mexp2Batch
          else:
            batch = np.zeros((nsamples*edims, NWORDS_FR),dtype=np.uint32)

          batch[offset:nsamples] = scl_vector[start_idx+scl_offset:end_idx+scl_offset]
          batch[nsamples+indims*offset:] = np.reshape(
                                            ecp_vector[start_idx*NWORDS_FP*indims:end_idx*NWORDS_FP*indims],
                                            (-1,NWORDS_FP))
          first_time = 1
          if stream_id in used_streams[gpu_id] :
            first_time = 0

          ec_pippen_mul(cuda_ec128, batch ,MOD_FP, ec2=ec2, gpu_id=gpu_id, stream_id=stream_id, first_time=first_time)
          used_streams[gpu_id].add(stream_id)

          if stream_id == 0:
             try:
               cuda_ec128.streamDel(gpu_id, stream_id)
             except ValueError:
               self.logger.error('Exception occurred when getting EC results. Exiting program...')
               sys.exit(1)
               
          else :
             pending_dispatch_table.append(p)
             n_dispatch +=1


             # Collect results. Leave last batch uncollected to maximize parallelization
             if n_dispatch == n_par_batches:
                 n_dispatch=0

                 try:
                    self.streamsDel(pending_dispatch_table)
                 except ValueError:
                    self.logger.error('Exception occurred when getting EC results. Exiting program...')
                    sys.exit(1)
                 pending_dispatch_table = []

          if self.stop_client.value:
             # Collect final results
             self.streamsDel(pending_dispatch_table)
             return

       # Collect final results
       self.streamsDel(pending_dispatch_table)

       if reduce_en :
          for gpu_id, streams  in enumerate(used_streams):
              for stream_id in streams:
                 ec_pippen_reduce(cuda_ec128, batch, MOD_FP, ec2=ec2, gpu_id=gpu_id, stream_id=stream_id)
  
          for gpu_id, streams  in enumerate(used_streams):
              for stream_id in streams:
                result, t = cuda_ec128.streamSync(gpu_id,stream_id)
                if len(result) == ECP_JAC_OUTDIMS:
                    self.init_ec_val[gpu_id][stream_id][pidx] =\
                               ec_jac2aff_h(
                                result.reshape(-1),
                                MOD_FP,
                                strip_last=1) 
                else:
                   self.init_ec_val[gpu_id][stream_id][pidx] =\
                            ec2_jac2aff_h(
                                 result.reshape(-1),
                                 MOD_FP,
                                 strip_last=1) 
          return None                  
       else:
          return used_streams

    def write_pdata(self):
        if self.public_signals is None :
            return
        if self.out_public_f.endswith('.json'):
           # Write public file
           ZField.set_field(MOD_FR)
           ps = [str(BigInt.from_uint256(el).as_long()) for el in self.public_signals]
           j = json.dumps(ps, indent=4)
           f = open(self.out_public_f, 'w')
           print(j,file=f)
           f.close()
        elif self.out_public_f.endswith('bin') :
           public_bin = np.concatenate((
                   np.asarray([self.public_signals.shape[0]], dtype=np.uint32),
                   np.reshape(self.public_signals,(-1))))
           writeU256DataFile_h(public_bin, self.out_public_f.encode("UTF-8"))

    def write_proof(self):
        if self.out_proof_f.endswith('.json'):
           ZField.set_field(MOD_FP)
           # write proof file
           P = ECC.from_uint256(
                            np.concatenate((
                                       self.pi_a_eccf1,
                                       [ECC.one[ZUtils.FRDC].as_uint256()])),
                       in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)

           pi_a = [el.extend().as_str() for el in P][0]
   
           P = ECC.from_uint256(
                            np.concatenate((
                                     self.pi_c_eccf1,
                                     [ECC.one[ZUtils.FRDC].as_uint256()])),
                       in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True)
           pi_c = [el.extend().as_str() for el in P][0]
   
           P = ECC.from_uint256(
                          np.reshape(
                               np.concatenate((
                                   self.pi_b_eccf2,
                                    [ECC.one[ZUtils.FRDC].as_uint256(),
                                     ECC.zero[ZUtils.FRDC].as_uint256()])),
                              (-1,2,NWORDS_256BIT)),
                            in_ectype=EC_T_AFFINE, out_ectype=EC_T_AFFINE, reduced=True, ec2=True)

           pi_b = [el.extend().as_str() for el in P][0]
           proof = {"pi_a" : pi_a, "pi_b" : pi_b, "pi_c" : pi_c, "protocol" : "groth"}
           j = json.dumps(proof, indent=4, sort_keys=True)
           f = open(self.out_proof_f, 'w')
           print(j, file=f)
           f.close()
        elif self.out_proof_f.endswith('.bin'):
           pi_a_eccf1 = from_montgomeryN_h(np.reshape(self.pi_a_eccf1,(-1)), MOD_FP)
           pi_b_eccf2 = from_montgomeryN_h(np.reshape(self.pi_b_eccf2,(-1)), MOD_FP)
           pi_c_eccf1 = from_montgomeryN_h(np.reshape(self.pi_c_eccf1,(-1)), MOD_FP)
           proof_bin = np.concatenate((
                    pi_a_eccf1, 
                    pi_b_eccf2,
                    pi_c_eccf1))
           writeU256DataFile_h(proof_bin, self.out_public_f.encode("UTF-8"))
               
    def evalPoly(self,pA_T,w, pX, nVars, m, pidx):
        # Convert witness to montgomery in zpoly_maddm_h
        reduce_coeff = 0
        mpoly_eval_h(np.reshape(pA_T,-1),w[:nVars],pX, reduce_coeff, m, 0, nVars, 1 , pidx)
    
