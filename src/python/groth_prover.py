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
from subprocess import call, run
import logging
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
    
    def __init__(self, proving_key_f, verification_key_f=None,curve='BN128',
                 out_pk_f=None, out_pk_format=FMT_MONT, test_f=None, n_streams=N_STREAMS_PER_GPU, n_gpus=1,start_server=1,
                 benchmark_f=None, seed=None, snarkjs=None, verify_en=0, keep_f=None, reserved_cpus=0, batch_size=20, write_table_f=None, zk=1):

        # Check valid folder exists
        if keep_f is None:
            print ("Repo directory needs to be provided\n")
            sys.exit(1)

        self.keep_f = gen_reponame(keep_f, sufix="_PROVER")

        logging.basicConfig(filename=self.keep_f + '/log', filemode='w',
                format='%(asctime)s - %(levelname)s - %(message)s',
                level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')

        if not use_pycusnarks :
          logging.error('PyCUSnarks shared library not found. Exiting...')
          sys.exit(1)

        if seed is not None:
          self.seed = seed
        else:
          x = os.urandom(4)
          self.seed = int(x.hex(),16)

        random.seed(self.seed) 

        self.sort_en = 0
        self.compute_ntt_gpu = False
        self.compute_last_mexp_gpu = True

        self.write_table_en = False
        self.write_table_f = write_table_f
        if write_table_f is not None:
          self.write_table_en = True

        self.roots_f = cfg.get_roots_file()
        self.n_bits_roots = cfg.get_n_roots()

        self.batch_size = None
        if batch_size > 20:
            batch_size = 20

        self.batch_size = 1<<batch_size  # include roots. Max is 1<<20

        self.ecbn128  = ECBN128(self.batch_size,   seed=self.seed)
        self.ec2bn128 =\
                EC2BN128(int((ECP2_JAC_OUTDIMS/ECP2_JAC_INDIMS) * self.batch_size), seed=self.seed)
        if self.compute_ntt_gpu:
            self.cuzpoly = ZCUPoly(5*self.batch_size  + 2, seed=self.seed)
    
        self.out_proving_key_f = out_pk_f
        self.out_proving_key_format = out_pk_format
        self.proving_key_f = proving_key_f
        self.verification_key_f = verification_key_f
        self.out_proof_f = None
        self.out_public_f = None
        self.curve_data = ZUtils.CURVE_DATA[curve]
        # Initialize Group 
        ZField(self.curve_data['prime'])
        # Initialize Field 
        ZField.add_field(self.curve_data['prime_r'],self.curve_data['factor_data'])
        ECC.init(self.curve_data)
        ZPoly.init(MOD_FIELD)

        ZField.set_field(MOD_GROUP)

        self.pk = getPK()
        self.verify = 2
        self.stop_client = mp.Value('i',0)

        self.n_gpu = min(get_ngpu(max_used_percent=99.),n_gpus)
        if 'CUDA_VISIBLE_DEVICES' in os.environ and len(os.environ['CUDA_VISIBLE_DEVICES']) > 0:
           self.n_gpu = min(self.n_gpu,len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))

        if self.n_gpu == 0:
          logging.error('No available GPUs')
          sys.exit(1)

        self.public_signals = None
        self.witness_f = None
        self.snarkjs = snarkjs
        self.verify_en = None

        #self.affinity = get_affinity(r_cpu=reserved_cpus)
        #print(self.affinity)

        ZField.set_field(MOD_FIELD)
        self.t_GP = {}

        init_h()
        
        #scalar array : extended witness / polH
        self.scl_array = None
        self.sorted_scl_array = None
        self.sorted_scl_array_idx = None


        if n_streams > get_nstreams():
          self.n_streams = get_nstreams()
        else :
          self.n_streams = n_streams

        #copy_input_files([proving_key_f, verification_key_f], self.keep_f)
        if test_f is None:
           self.test_f = test_f
        else:
           self.test_f= self.keep_f + '/' + test_f

        # convert data to array of bytes so that it can be easily transfered to shared mem
        if self.test_f :
           self.load_pkdata()
           # if snarkjs is to be launched to compare results,
           #  I am assuming circuit is small, so i keep 
           #   a version to be able to generate json. Else, results will overwrite input data
           self.pk_short = pkvars_to_bin(FMT_MONT, EC_T_AFFINE, self.pk, ext=False)

        # Initialize shared memory
        logging.info('Initializing memories...')

        # PK_BIN
        pkbin_nWords = int(os.path.getsize(self.proving_key_f)/4)
        self.pk_sh = RawArray(c_uint32, pkbin_nWords)
        self.pk = np.frombuffer(self.pk_sh, dtype=np.uint32)
        logging.info('Reading Proving Key...')
        readU256PKFileTo_h(self.proving_key_f.encode("UTF-8"), self.pk)
             
        pkbin_vars = pkbin_get(self.pk,['nVars','domainSize', 'delta_1', 'hExps', 'nPublic'])
        nVars = int(pkbin_vars[0][0])
        domainSize = int(pkbin_vars[1][0])
        delta_1 = pkbin_vars[2]
        hExps = pkbin_vars[3]
        hExps[2*(domainSize+1)*NWORDS_256BIT:2*(domainSize+2)*NWORDS_256BIT] = delta_1
        nPublic = int(pkbin_vars[4][0])

        # scl_array
        self.scl_array_sh = RawArray(c_uint32, domainSize * NWORDS_256BIT)     
        self.scl_array = np.frombuffer(
                     self.scl_array_sh, dtype=np.uint32).reshape((domainSize, NWORDS_256BIT))
        # Size is domainSize To store polH + three additional coeffs
        self.sorted_scl_array_idx_sh = RawArray(c_uint32, domainSize + 4)
        self.sorted_scl_array_idx =\
                np.frombuffer(self.sorted_scl_array_idx_sh, dtype=np.uint32)

        # sorted scl_array
        # sorted witness + [one] + r/s or sorted polH
        self.sorted_scl_array_sh = RawArray(c_uint32, (domainSize + 4) * NWORDS_256BIT)  
        self.sorted_scl_array    =\
                np.frombuffer(
                             self.sorted_scl_array_sh,
                             dtype=np.uint32).reshape((domainSize+4, NWORDS_256BIT))

        # pA_T
        self.pA_T_sh = RawArray(c_uint32, domainSize * NWORDS_256BIT)
        self.pA_T = np.frombuffer(
                     self.pA_T_sh, dtype=np.uint32).reshape((domainSize, NWORDS_256BIT))
        np.copyto(
                self.pA_T,
                np.zeros((domainSize, NWORDS_256BIT), dtype=np.uint32))

        # pB_T
        self.pB_T_sh = RawArray(c_uint32, domainSize * NWORDS_256BIT)
        self.pB_T = np.frombuffer(
                     self.pB_T_sh, dtype=np.uint32).reshape((domainSize, NWORDS_256BIT))
        np.copyto(
                self.pB_T,
                np.zeros((domainSize, NWORDS_256BIT), dtype=np.uint32))

        # Roots
        self.roots_rdc_u256_sh = RawArray(c_uint32,  (1 << self.n_bits_roots) * NWORDS_256BIT)
        self.roots_rdc_u256 =\
                np.frombuffer(
                        self.roots_rdc_u256_sh,
                        dtype=np.uint32).reshape((1<<self.n_bits_roots, NWORDS_256BIT))
        np.copyto(
                self.roots_rdc_u256,
                readU256DataFile_h(
                    self.roots_f.encode("UTF-8"),
                    1<<self.n_bits_roots, 1<<self.n_bits_roots) )

        # pis
        self.pi_a_eccf1_sh = RawArray(c_uint32, ECP_JAC_INDIMS * NWORDS_256BIT)
        self.pi_a_eccf1 = \
                 np.frombuffer( self.pi_a_eccf1_sh,
                                dtype=np.uint32).reshape(ECP_JAC_INDIMS, NWORDS_256BIT)  

        self.pi_b_eccf2_sh = RawArray(c_uint32, ECP2_JAC_INDIMS * NWORDS_256BIT)
        self.pi_b_eccf2 = \
                 np.frombuffer( self.pi_b_eccf2_sh,
                                dtype=np.uint32).reshape(ECP2_JAC_INDIMS, NWORDS_256BIT)  

        self.pi_c_eccf1_sh = RawArray(c_uint32, ECP_JAC_INDIMS * NWORDS_256BIT)
        self.pi_c_eccf1 = \
                 np.frombuffer( self.pi_c_eccf1_sh,
                                dtype=np.uint32).reshape(ECP_JAC_INDIMS, NWORDS_256BIT)  

        self.pi_c2_eccf1_sh = RawArray(c_uint32, ECP_JAC_INDIMS * NWORDS_256BIT)
        self.pi_c2_eccf1 = \
                 np.frombuffer( self.pi_c2_eccf1_sh,
                                dtype=np.uint32).reshape(ECP_JAC_INDIMS, NWORDS_256BIT)  

        self.pi_b1_eccf1_sh = RawArray(c_uint32, ECP_JAC_INDIMS * NWORDS_256BIT)
        self.pi_b1_eccf1 = \
                 np.frombuffer( self.pi_b1_eccf1_sh,
                                dtype=np.uint32).reshape(ECP_JAC_INDIMS, NWORDS_256BIT)  

        self.init_ec_val_sh = RawArray(c_uint32, (3*ECP_JAC_INDIMS+ECP2_JAC_INDIMS)*self.n_gpu*self.n_streams*NWORDS_256BIT)
        self.init_ec_val = \
                 np.frombuffer( self.init_ec_val_sh,
                                dtype=np.uint32)

        #scl r,s, rs
        self.r_scl_sh = RawArray(c_uint32, NWORDS_256BIT)
        self.r_scl = np.frombuffer(self.r_scl_sh, dtype=np.uint32)
         
        self.s_scl_sh = RawArray(c_uint32, NWORDS_256BIT)
        self.s_scl = np.frombuffer(self.s_scl_sh, dtype=np.uint32)

        self.neg_rs_scl_sh = RawArray(c_uint32, NWORDS_256BIT)
        self.neg_rs_scl = np.frombuffer(self.neg_rs_scl_sh, dtype=np.uint32)

        self.zk = zk

        logging.info('#################################### ')
        logging.info('Initializing Groth prover with the following parameters :')
        logging.info(' - curve : %s',curve)
        logging.info(' - proving_key_f : %s', proving_key_f)
        logging.info(' - verification_key_f : %s',verification_key_f)
        logging.info(' - out_pk_f : %s',out_pk_f)
        logging.info(' - out_pk_format : %s',out_pk_format) 
        logging.info(' - test_f : %s',self.test_f)
        logging.info(' - benchmark_f : %s', benchmark_f)
        logging.info(' - seed : %s', self.seed)
        logging.info(' - snarkjs : %s', snarkjs)
        logging.info(' - keep_f : %s', keep_f)
        logging.info(' - n available GPUs : %s', self.n_gpu)
        logging.info(' - n available CPUs : %s', get_nprocs_h())
        logging.info(' - sort enable : %s', self.sort_en)
        logging.info(' - write_table_en : %s', self.write_table_en)
        logging.info(' - table_f : %s', self.write_table_f)
        logging.info(' - compute NTT in GPU : %s', self.compute_ntt_gpu)
        logging.info(' - compute last Mexp in GPU : %s', self.compute_last_mexp_gpu)
        logging.info(' - zero knowledge enabled : %s', self.zk)
        logging.info(' - N Constraints : %s', nVars)
        logging.info(' - Domain Size : %s', domainSize)
        logging.info(' - N Public : %s', nPublic)
        logging.info('#################################### ')
 
        if self.write_table_f:
          logging.info('#################################### ')
          logging.info('..')
          Tables = readU256DataFile_h(self.write_table_f.encode("UTF-8"), 
                  int(nVars*2*6*(1<<U256_BSELM)/U256_BSELM) + int(domainSize*2*(1<<U256_BSELM)/U256_BSELM), 
                  int(nVars*2*5*(1<<U256_BSELM)/U256_BSELM) + int(domainSize*2*(1<<U256_BSELM)/U256_BSELM))
          logging.info('# Reading Tables (%s Bytes)...', Tables.shape)
          logging.info('..')
          logging.info('#################################### ')
           
        if self.out_proving_key_f is not None:
             if self.out_proving_key_f.endswith('.json'):
               pk_dict =pkvars_to_json(self.out_proving_key_format, EC_T_AFFINE, self.pk)
               pk_json = json.dumps(pk_dict, indent=4, sort_keys=True)
               f = open(self.out_proving_key_f, 'w')
               print(pk_json, file=f)
               f.close()
               del pk_dict
               del pk_json
             elif self.out_proving_key_f.endswith('.bin'):
               pk_bin = pkvars_to_bin(self.out_proving_key_format, EC_T_AFFINE, self.pk, ext=False)
               writeU256DataFile_h(pk_bin, self.out_proving_key_f.encode("UTF-8"))
               del pk_bin
             elif self.out_proving_key_f.endswith('.npz'):
                np.savez_compressed(proving_key_fnpz, alfa_1_u256 =\
                        self.pk['alfa_1'],
                        beta_1_u256 = self.pk['beta_1'], delta_1_u256 = self.pk['delta_1'],
                        beta_2_u256 = self.pk['beta_2'], delta_2_u256 = self.pk['delta_2'],
                        A_u256 = self.pk['A'], B1_u256=self.pk['B1'], B2_u256 = self.pk['B2'],
                        C_u256 = self.pk['C'], hExps_u256 =self.pk['hExps'],
                        polsA_u256 = self.pk['polsA'],
                        polsB_u256 = self.pk['polsB'], polsC_u256 = self.pk['polsC'],
                        nvars = self.pk['nVars'], npublic=self.pk['nPublic'],
                        domain_bits=self.pk['domainBits'],
                        domain_size = self.pk['domainSize'])


        self.ec_lable = np.asarray(['A', 'B2', 'B1', 'C','hExps'])
                             # Point Name, cuda pointer, step, idx, ec2, pi
        self.ec_type_dict = {'A'     : [self.ecbn128,  2, 0, 0, 0],
                             'B2'    : [self.ec2bn128, 4, 1, 1, 1],
                             'B1'    : [self.ecbn128,  2, 2, 0, 2 ],
                             'C'     : [self.ecbn128,  2, 3, 0, 3 ],
                             'hExps' : [self.ecbn128,  2, 4, 0, 3 ] }

        # Init CPU process
        self.init_p_CPU()

        # Init Mexp process
        self.init_p_Mexp()

        if not start_server:
           self.startProcesses(nVars)

    def init_p_CPU(self):
        self.parent_conn_CPU, self.child_conn_CPU = Pipe()

    def init_p_Mexp(self):
        self.parent_conn_Mexp = []
        self.child_conn_Mexp = []
        self.p_Mexp = []
        for gpu_idx in range(self.n_gpu+1):
           p_conn, c_conn = Pipe()
           self.parent_conn_Mexp.append(p_conn)
           self.child_conn_Mexp.append(c_conn)


        pk_bin = pkbin_get(self.pk,['nVars', 'nPublic', 'domainSize'])

        nVars = pk_bin[0][0]
        nPublic = pk_bin[1][0]
        domainSize = pk_bin[2][0]

        nbatches1 = math.ceil((nVars - (nPublic+1))/(self.batch_size-1)) + 1
        nbatches2 = math.ceil((nVars+2 - (nPublic+1))/(self.batch_size-1)) + 1

        if nbatches1 != nbatches2:
          nbatches = math.ceil((nVars+2 - (nPublic+1))/(self.batch_size-2)) + 1
          self.batch_size_mexp_phase1 = self.batch_size-1
        else:
          nbatches = nbatches1
          self.batch_size_mexp_phase1 = self.batch_size

        next_gpu_idx = 0
        first_stream_idx = min(self.n_streams-1,1)

        if self.zk:
          ec_lable_ph1 = np.asarray(['A', 'B1', 'C'])
          ec_lable_ph2 = np.asarray(['A', 'B2', 'B1'])
          n_els_ph1 = len(ec_lable_ph1)
          n_els_ph2 = len(ec_lable_ph2)
        else:
          ec_lable_ph1 = np.asarray(['A', 'C'])
          n_els_ph1 = len(ec_lable_ph1)
          ec_lable_ph2 = np.asarray(['A', 'B2'])
          n_els_ph2 = len(ec_lable_ph2)

        # EC reduce A, B2, B1 and C from nPublic+1 to end	
        self.dispatch_table_phase1 = buildDispatchTable( nbatches-1,
                                         #self.ec_type_dict['C'][2]+1,
                                         n_els_ph1,
                                         self.n_gpu, self.n_streams, self.batch_size_mexp_phase1-1,
                                         nPublic+1, nVars,
                                         start_pidx=self.ec_type_dict['A'][2],
                                         start_gpu_idx=next_gpu_idx,
                                         start_stream_idx=first_stream_idx,
                                         #ec_lable = self.ec_lable)
                                         ec_lable = ec_lable_ph1)

        #TODO : trying to separate  G2 MExp from G1 Mexp in phase 1 to check if some time is gained
        self.dispatch_table_phase1b = buildDispatchTable( nbatches-1,
                                         #self.ec_type_dict['C'][2]+1,
                                         1,
                                         self.n_gpu, self.n_streams, self.batch_size_mexp_phase1-1,
                                         nPublic+1, nVars,
                                         start_pidx=0,
                                         start_gpu_idx=next_gpu_idx,
                                         start_stream_idx=first_stream_idx,
                                         #ec_lable = self.ec_lable)
                                         ec_lable = np.asarray(['B2']))

        # EC reduce A, B2 and B1 from 0 to nPublic+1
        self.dispatch_table_phase2 = buildDispatchTable(1, 
                                #self.ec_type_dict['B1'][2]+1,
                                n_els_ph2,
                                self.n_gpu, self.n_streams, nPublic+1,
                                0, nPublic+1,
                                start_pidx=self.ec_type_dict['A'][2],
                                start_gpu_idx=next_gpu_idx,
                                start_stream_idx=first_stream_idx,
                                #ec_lable = self.ec_lable)
                                ec_lable = ec_lable_ph2)

        self.batch_size_mexp_phase3 = self.batch_size
        while True:
           nbatches = math.ceil((domainSize - 1)/(self.batch_size_mexp_phase3 - 1))
           if domainSize -1 - max(1,(nbatches-1))*(self.batch_size_mexp_phase3 - 1)  < 3:
              self.batch_size_mexp_phase3 -= 3
           else :
             break

       
        # EC reduce hExps
        self.dispatch_table_phase3 = buildDispatchTable( nbatches, 1,
                                         self.n_gpu, self.n_streams, self.batch_size_mexp_phase3-1,
                                         0, domainSize-1,
                                         start_pidx=self.ec_type_dict['hExps'][2],
                                         ec_lable = self.ec_lable)

    def pysnarkP_CPU(self, conn, wnElems, w_sh, w_shape, pA_T_sh, pA_T_shape, pB_T_sh, pB_T_shape, pi_c2_eccf1_sh):
        logging.info(' Launching Poly Process Client')
        logging.info(' Evaluating QAP')
        pk = self.pk
        #w = self.scl_array[:wnElems]
        w = np.frombuffer(w_sh, dtype=np.uint32).reshape(w_shape)
        pA_T = np.frombuffer(pA_T_sh, dtype=np.uint32).reshape(pA_T_shape)
        pB_T = np.frombuffer(pB_T_sh, dtype=np.uint32).reshape(pB_T_shape)
        pi_c2_eccf1 = np.frombuffer(pi_c2_eccf1_sh, dtype=np.uint32).reshape(-1,NWORDS_256BIT)
        #print("ProcessServer : "+str(pk.shape[0]))
        start = time.time()

        #ZField.set_field(MOD_FIELD)
        pk_bin = pkbin_get(pk,['nVars', 'domainSize', 'polsA', 'polsB','hExps'])
        nVars = pk_bin[0][0]
        m = pk_bin[1][0]
        pA = np.reshape(pk_bin[2][:m*NWORDS_256BIT],(m,NWORDS_256BIT))
        pB = np.reshape(pk_bin[3][:m*NWORDS_256BIT],(m,NWORDS_256BIT))

        logging.info(' Process server - Evaluating Poly A...')
        np.copyto(pA_T,self.evalPoly(w[:wnElems], pA, nVars, m, MOD_FIELD))
        logging.info(' Process server - Evaluating Poly B...')
        np.copyto(pB_T,self.evalPoly(w[:wnElems], pB, nVars, m, MOD_FIELD))
        logging.info(' Process server - Completed Evaluating Poly B...')
        end = time.time()

        start1 = time.time()
        if self.compute_ntt_gpu is False:
          logging.info(' Process server - Calculate H...')
          ifft_params = ntt_build_h(self.pA_T.shape[0])
          polH = ntt_interpolandmul_h(np.reshape(self.pA_T,-1),
                                     np.reshape(self.pB_T,-1),
                                     np.reshape(self.roots_rdc_u256[::1<<(self.n_bits_roots - ifft_params['levels']-1)] ,-1), 2, MOD_FIELD)

        end1 = time.time()

        start2 = time.time()

        if self.compute_last_mexp_gpu is False:
          logging.info(' Process server - Starting Last Mexp...')
          # vector needs to be power of two, so i add one last element
          polH[-1] = np.asarray([0,0,0,0,0,0,0,0],dtype=np.uint32)
          np.copyto(pi_c2_eccf1,
                    ec_jacreduce_h(
                         np.reshape(polH[m:],-1),
                         pk_bin[4][:m*NWORDS_256BIT*ECP_JAC_INDIMS],
                         MOD_GROUP, 1, 1, 1)
                    )
          logging.info(' Process server - Completed Last Mexp...')
        else:
          logging.info(' Process server - Waiting for Mexp to be completed...')

          #write polH once MEXP is done (not before)
          conn.recv()
          if self.compute_ntt_gpu is False:
            np.copyto(w[:m-1], polH[m:-1])
            logging.info(' Process server - Copying polH...')
  
        end2 = time.time()
        # t poly prep, t1, NTT t2 : last Mexp
        conn.send([end-start, end1-start1, end2-start2])
        logging.info(' Process server - Completed')
        conn.close()

    def startGPServer(self):    
           self.port_first = 8192
           self.port_second = 8193
           self.proof_id = 0
           self.proof_repo = []
           pkbin_vars = pkbin_get(self.pk,['nVars','domainSize'])
           nVars = int(pkbin_vars[0][0])
           logging.info('Launching GP Server')
           p = Process(target=self.startServer, args = (self.port_first,0))
           p.start()
           try:
              self.startServer(self.port_second, nVars)
           except:
              logging.info('Exception occurred. Server stopped')
              run(['killall', '-9', 'python3'])

    def startServer(self, port, nVars):    
           logging.info('Launching GP Server')
           jsocket = json_socket.jsonSocket(port=port)

           s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
           s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
           s.bind((jsocket.host,jsocket.port))
           s.listen(1)
           print('Server listening on port ' +str(port) +' ready...')
           while True: # Accept connections from multiple clients
               #logging.info('Waiting for client at port %s...', port)
               conn, addr = s.accept()
               #logging.info('New connecton from address %s port %s', addr, port)
               msg = jsocket.receive_message(conn)
               if len(msg):
                 # Call some action and return results
                 parsed_dict = ast.literal_eval(json.loads(json.dumps(msg)))
                 if 'stop_server' in parsed_dict and parsed_dict['stop_server'] == 1:
                    logging.info('Stopping server')
                    os.kill(os.getppid(), signal.SIGTERM)
                    sys.exit(1)
                 elif 'is_alive' in parsed_dict:
                    new_msg = {}
                    new_msg['status']=1
                    jsocket.send_message(new_msg, conn)
                    conn.close()
                    continue

                 elif 'stop_client' in parsed_dict:
                     self.stop_client.value = 1
                     logging.info('Stopping client...')
                     conn.close()

                 elif 'status' in parsed_dict:
                     self.proof_repo[-1]['result'] = parsed_dict['status']
                     del parsed_dict['status']
                     self.proof_repo[-1].update(parsed_dict)
                     conn.close()

                 elif 'list' in parsed_dict:
                     if len(self.proof_repo) > int(parsed_dict['list']):
                       new_msg = dict(self.proof_repo[int(parsed_dict['list'])])
                       jsocket.send_message(new_msg, conn)
                     elif int(parsed_dict['list']) == -1:
                       new_msg = dict(self.proof_repo[int(parsed_dict['list'][-1])])
                       jsocket.send_message(new_msg, conn)
                     conn.close()
                     
                 elif 'witness_f' in parsed_dict and nVars==0:
                     query = { 'witness_f' : parsed_dict['witness_f'], 'proof_f' : parsed_dict['proof_f'],
                             'public_data_f' : parsed_dict['public_data_f'], 'verify_en' : parsed_dict['verify_en'], 'proof_id' : self.proof_id, 'result' : -1 }
                     self.proof_id+=1
                     logging.info('Request for new proof received in primary server')
                     self.proof_repo.append(query)

                     p = json_socket.jsonSocket(port=self.port_second)
                     p.send_message(query)
                     conn.close()

                 elif 'witness_f' in parsed_dict and nVars:
                   logging.info('Request for new proof received in secondary server')
                   conn.close()
                   # Initialize CPU Process
                   self.startProcesses(nVars)

                   self.proof(
                         parsed_dict['witness_f'],
                         parsed_dict['proof_f'], parsed_dict['public_data_f'],
                         verify_en=int(parsed_dict['verify_en']))

                   new_msg = dict(self.t_GP)
                   new_msg['status'] = self.verify

                   p = json_socket.jsonSocket(port=self.port_first)
                   p.send_message(new_msg)
   

    def startProcesses(self, nVars):
          logging.info('Starting Processes...')
          self.p_CPU = \
                      Process(
                            target=self.pysnarkP_CPU,
                               args = (
                                      self.child_conn_CPU,
                                      nVars,
                                      self.scl_array_sh, self.scl_array.shape,
                                      self.pA_T_sh,self.pA_T.shape,
                                      self.pB_T_sh, self.pB_T.shape,
                                      self.pi_c2_eccf1))

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
                  self.pi_c2_eccf1,
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


        self.init_ec_val = np.tile( np.asarray(
                                                  [self.pi_a_eccf1, self.pi_b_eccf2, self.pi_b1_eccf1, self.pi_c_eccf1], 
                                                  dtype=np.object), 
                                  (self.n_gpu, self.n_streams, 1))


    def __del__(self):
       release_h()
       logging.shutdown()

    def read_witness_data(self):
       ## Open and parse witness data
       if os.path.isfile(self.witness_f):

           pkbin_vars = pkbin_get(self.pk,['nVars','domainSize'])
           nVars = int(pkbin_vars[0][0])
           domainSize = int(pkbin_vars[1][0])

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

             #from txt to json
             """
             w_arr = []
             with open(self.witness_f, 'r') as f:
               for w in f:
                  w_arr.append(w.rstrip())
               w_json = json.dumps(w_arr, indent=4)
             w_file = self.witness_f.replace('txt','json')
             f = open(w_file, 'w')
             print(w_json, file=f)
             f.close()
             del w_json
             del w_arr
             """

           elif self.witness_f.endswith('.bin'):
             np.copyto(
                self.scl_array[:nVars],
                readWitnessFile_h(self.witness_f.encode("UTF-8"),1, nVars ))

           elif self.witness_f.endswith('.dat'):
             np.copyto(
                self.scl_array[:nVars],
                readWitnessFile_h(self.witness_f.encode("UTF-8"),0, nVars ))
           else:
             np.copyto(
                self.scl_array[:nVars],
                readWitnessShmem_h(nVars) )
             """
             if self.witness_f.endswith('.txt'):
               w_file = self.witness_f.replace('txt','bin')
             elif self.witness_f.endswith('.json'):
               w_file = self.witness_f.replace('json','bin')

             writeWitnessFile_h(np.reshape(self.scl_array,-1),self.w_file.encode("UTF-8"))
             """

       else:
          logging.error('Witness file %s doesn\'t exist', self.witness_f)
          sys.exit(1)

       #self.sorted_scl_array_idx = \
              #sortu256_idx_h(self.scl_array, sort_en)
       #self.sorted_scl_array = \
              #self.scl_array[self.sorted_scl_array_idx]
       
    def load_pkdata(self):
       if self.proving_key_f.endswith('npz'):
          npzfile = np.load(self.proving_key_f)
          self.pk['protocol'] = np.uint32(PROTOCOL_T_GROTH)
          self.pk['Rbitlen']  = np.asarray(ZField.get_reduction_data()['Rbitlen'],dtype=np.uint32)
          self.pk['k_binformat'] = np.uint32(FMT_MONT)
          self.pk['k_ecformat'] = np.uint32(EC_T_AFFINE)
          self.pk['alfa_1'] = npzfile['alfa_1_u256']
          self.pk['beta_1'] = npzfile['beta_1_u256']
          self.pk['delta_1']= npzfile['delta_1_u256']
          self.pk['beta_2'] = npzfile['beta_2_u256']
          self.pk['delta_2'] = npzfile['delta_2_u256']

          self.pk['A'] = npzfile['A_u256']
          self.pk['B1'] = npzfile['B1_u256']
          self.pk['B2'] = npzfile['B2_u256']
          self.pk['C']  = npzfile['C_u256']
          self.pk['hExps'] = npzfile['hExps_u256']
          self.pk['polsA'] = npzfile['polsA_u256']
          self.pk['polsB'] = npzfile['polsB_u256']
          self.pk['polsC'] = npzfile['polsC_u256']
          self.pk['nVars'] = npzfile['nvars']
          self.pk['nPublic'] = npzfile['npublic']
          self.pk['domainSize'] = npzfile['domain_size']

       ## Open and parse proving key data
       elif self.proving_key_f.endswith('.json'):
           f = open(self.proving_key_f,'r')
           tmp_data = json.load(f)
           vk_proof = json_to_dict(tmp_data)
           f.close()
           self.pk = pkjson_to_vars(vk_proof, self.proving_key_f)  
           del vk_proof
           del tmp_data

       elif self.proving_key_f.endswith('.bin'):
          pk_bin = readU256PKFile_h(self.proving_key_f.encode("UTF-8"))
          self.pk = pkbin_to_vars(pk_bin)
          del pk_bin

       logging.info('')
       logging.info('')
       logging.info('#################################### ')
       logging.info(' - nVars      : %s', self.pk['nVars'])
       logging.info(' - nPublic    : %s', self.pk['nPublic'])
       logging.info(' - domainBits : %s', self.pk['domainBits'])
       logging.info('#################################### ')
       logging.info('')
       logging.info('')

    def logTimeResults(self):
    
      self.t_GP['init'] = [round(self.t_GP['init'],4), round(100*self.t_GP['init']/self.t_GP['Proof'],2)] 
      self.t_GP['Read_W'] = [round(self.t_GP['Read_W'],4), round(100*self.t_GP['Read_W']/self.t_GP['Proof'],2)] 
      self.t_GP['Mexp'] = self.t_GP['Mexp1'] + self.t_GP['Mexp2']
      self.t_GP['Mexp'] = [round(self.t_GP['Mexp'],4), round(100*self.t_GP['Mexp']/self.t_GP['Proof'],2)] 
      self.t_GP['Mexp1'] = [round(self.t_GP['Mexp1'],4), round(100*self.t_GP['Mexp1']/self.t_GP['Proof'],2)] 
      self.t_GP['Mexp2'] = [round(self.t_GP['Mexp2'],4), round(100*self.t_GP['Mexp2']/self.t_GP['Proof'],2)] 
      self.t_GP['Eval'] = [round(self.t_GP['Eval'],4), round(100*self.t_GP['Eval']/self.t_GP['Proof'],2)] 
      self.t_GP['H'] = [round(self.t_GP['H'],4), round(100*self.t_GP['H']/self.t_GP['Proof'],2)] 
      self.t_GP['Proof'] = round(self.t_GP['Proof'],4)
      logging.info('')
      logging.info('')
      logging.info('#################################### ')
      logging.info('Total Time to generate proof : %s seconds', self.t_GP['Proof'])
      logging.info('')
      logging.info('------ Initialization          : %s ', str(self.t_GP['init'][0]) + ' sec. (' + str(self.t_GP['init'][1]) + ' %)')
      logging.info('------ Time Read Witness       : %s ', str(self.t_GP['Read_W'][0]) + ' sec. (' + str(self.t_GP['Read_W'][1]) + ' %)')
      logging.info('------ Time Multi-exp.         : %s ', str(self.t_GP['Mexp'][0]) + ' sec.(' + str(self.t_GP['Mexp'][1]) + ' %)')
      logging.info('------ Time Multi-exp 1.         : %s ', str(self.t_GP['Mexp1'][0]) + ' sec.(' + str(self.t_GP['Mexp1'][1]) + ' %)')
      logging.info('------ Time Multi-exp 2.         : %s ', str(self.t_GP['Mexp2'][0]) + ' sec.(' + str(self.t_GP['Mexp2'][1]) + ' %)')
      logging.info('------ Time Poly Calculation   : %s ', str(self.t_GP['Eval'][0]) + ' sec. (' + str(self.t_GP['Eval'][1]) + ' %)')
      logging.info('------ Time Compute Poly H     : %s ', str(self.t_GP['H'][0]) + 'sec. (' + str(self.t_GP['H'][1]) + ' %)')
      logging.info('#################################### ')
      logging.info('')
      logging.info('')

    def proof(self, witness_f, out_proof_f , out_public_f, verify_en=0):

      # Initaliization
      start = time.time()

      self.out_proof_f = out_proof_f
      self.out_public_f = out_public_f
      self.witness_f = witness_f

      self.verify_en = verify_en
      self.t_GP = {}
      self.stop_client.value = 0

      if (self.verify_en):
        self.verify = 0
      else :
        self.verify = 2

      self.initECVal()

      logging.info('#################################### ')
      logging.info("Starting new proof...")
      logging.info(' - out_proof_f : %s',out_proof_f)
      logging.info(' - out_public_f : %s',out_public_f)
      logging.info(' - witness_f : %s',witness_f)
      logging.info(' - verify_en : %s', verify_en)
      logging.info(' - batch_size : %s', self.batch_size)
      logging.info(' - gpus used : %s', self.n_gpu)
      logging.info(' - streams used: %s', self.n_streams)
      logging.info('#################################### ')
      logging.info('')
      logging.info('')

      self.t_GP['init'] = time.time() - start
      ##### Starting proof

      self.gen_proof()
      
      if self.stop_client.value:
          self.verify = -2
          return self.verify

      self.t_GP['Proof'] = time.time() - start

      logging.info(" Proof completed" )
      logging.info('#################################### ')
      
      self.logTimeResults()

      # convert data to pkvars if necessary (only if test_f is set)
      if self.test_f is not None:
        self.pk =  pkbin_to_vars(self.pk_short)
        del self.pk_short

      self.write_pdata()
      self.write_proof()
      self.test_results()

      #copy_input_files([self.out_proof_f, self.out_public_f, self.out_proving_key_f, witness_f],self.keep_f)

      return self.verify

    def test_results(self):
      proof_r = True
      snarkjs = {}
      snarkjs['verify'] = 0
      if self.test_f is not None:
        logging.info("Calling snarkjs proof to compare computed proving and verification keys")
        # Write rand json
        randout_dict={}
        randout_dict['r'] = str(BigInt.from_uint256(self.r_scl).as_long())
        randout_dict['s'] = str(BigInt.from_uint256(self.s_scl).as_long())
        randout_json = json.dumps(randout_dict, indent=4, sort_keys=True)
        f = open(self.test_f, 'w')
        print(randout_json, file=f)
        f.close()
        snarkjs = self.launch_snarkjs("proof")
        p_r = pysnarks_compare(self.out_proof_f, snarkjs['p_f'], ['pi_a', 'pi_b', 'pi_c'],0)
        pd_r = pysnarks_compare(self.out_public_f, snarkjs['pd_f'], None, 0)
        proof_r = p_r and pd_r
        if proof_r:
          logging.info("Compared keys are equal")
        elif p_r:
          logging.info("Verification keys are different")
        elif pd_r:
          logging.info("Proving keys are different")
        else:
          logging.info("Verification and Proving keys are different")

      if self.verify_en:
        logging.info('#################################### ')
        logging.info("Calling snarkjs verify to verify proof ....")
        logging.info("")
        snarkjs = self.launch_snarkjs("verify")
        if snarkjs['verify'] == 0:
          logging.info("Verification SUCCEDED")
          self.verify = 1
        else:
          logging.info("Verification FAILED")
          self.verify = 0
        logging.info('#################################### ')

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
            #proving_key_file = self.keep_f + proving_key_file
            pk_dict = pkvars_to_json(FMT_EXT,EC_T_AFFINE, self.pk)
            pk_json = json.dumps(pk_dict, indent=4, sort_keys=True)
            f = open(proving_key_file, 'w')
            print(pk_json, file=f)
            f.close()
          else:
            logging.error(' Witness file %s needs to be .json', self.witness_f)
            sys.exit(1)

          if self.test_f is not None:
             debug_command = "--d"
             debug_file = self.test_f
          else:
             debug_command = ""
             debug_file = ""

          call([self.snarkjs, "proof", "-w", witness_file, "--pk", proving_key_file, "-p", snarkjs['p_f'],"--pub",snarkjs['pd_f'], debug_command, debug_file])

        elif mode == "verify":
          snarkjs['p_f'] = self.out_proof_f
          snarkjs['pd_f'] = self.out_public_f
          if self.verification_key_f is not None and self.verification_key_f.endswith('.json') :
             verification_key_file = self.verification_key_f
          else :
             logging.error(' To launch snarkjs, verification file %s needs to be a json file', self.verification_key_f)
             sys.exit(1)
        
          snarkjs['verify'] = call([self.snarkjs, "verify", "--vk", verification_key_file, "-p", snarkjs['p_f'],"--pub",snarkjs['pd_f']])
       
        return snarkjs

    def gen_proof(self ):
        """
          public_signals, pi_a_eccf1, pi_b_eccf2, pi_c_eccf1 

            inputs  : Witness from file 
            Outputs : Bin witnes (2)

          1 - Read all witness data -> CPU. When batch samples available,
                  start process 3

          ------------------------
            inputs  : Witness (1)
            Outputs : pi's (final result),  pi_c (4).

          2.1 - get witness batch  
          2.2 - sort witness1 batch [0-nVars] -> CPU
          2.3 - EC multi expo with batch A/B1/B2 and witnesss1 -> pi_a, pi_b, pib -> GPU
          2.4 - sort witness2 batch [nPublic+1,nVars]
          2.5 - EC multi expo with batch C and witnesss2 -> pi_c
          ------------------------
       
            inputs  : pols from file
                      witness (1). But reqiures all witness read
            Outputs : pX_T (I can't start 4 until all poool is availlable)

          3.1  Evaluate pols pA, pB and pC. Complete poly needs to be evaluated
                 before starting FFTs. CPU cores can work in this. Ideally,
                 multiple cores help evaluate pA. When it is done, repeat for pB,...
                 polsA   ----> polA_T  ----> delete polsA. I need mutexes. Modify 
                 C routine to do this
          ------------------------

            inputs  : pX_T (3)A
            output  : pH at the end of process (5). 

          4.1   Get polA_T, polB_T and polB_T batch
          4.2   perform 3xIFFT, 1 poly MUL and 1 poly SUB. Dedicate GPUs to do partial IFFT
                and CPU to combine results to do 3-step fft. Complete  IFFT-A, IFFT-B, IFFT-C,
                 FFT-A, FFT-B, FFT-AxFFT-B, IFFTAxB, SUB and finaly get pH
          -----------------------
            
            Input : pH (4), hExps (file), pi_c (2)
            Output : pi_c

          5.1 - EC multi expo with batch hExps and shorted pH -
 
         
         ------
          
            1 1 1 1 1 1 1 1 1  (CPU)
                  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 (CPU/GPU ecbn/ec2bn)
                            3 3 3 3 3 3 pA done 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 (CPU)
                                               StartIFFT PA 4 4 4 4 4 4 4 4 4 4 4 4 PH Done (GPU Zpoly)
                                                                                     5 5 5 5 5 5 5 5 5 5 5 5 5 5 (CPU/GPU ecbn/ec2bn) 



        """
        ######################
        # Beginning of P1 - Read Witness
        ######################
        start = time.time()
        # Read witness info
        logging.info(' Reading Witness...')
        self.read_witness_data()

        end = time.time()
        self.t_GP['Read_W'] = end - start
       
        start = time.time()
        self.p_CPU.start()
        self.t_GP['Eval'] = time.time()-start
 
        ######################
        # Beginning of P2 
        #   - Get witness batch, sort and EC Multiexp
        ######################
        start = time.time()

        ZField.set_field(MOD_FIELD)

        if self.zk:
          # Init r and s scalars
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

        r_mont = to_montgomeryN_h(np.reshape(self.r_scl, -1), MOD_FIELD)
        np.copyto(
                 self.neg_rs_scl,
                  montmult_neg_h(
                             np.reshape(r_mont, -1),
                             np.reshape(self.s_scl, -1), MOD_FIELD))

        logging.info('#################################### ')
        logging.info(' Random numbers :')
        logging.info(' - r : %s',str(BigInt.from_uint256(self.r_scl).as_long()) )
        logging.info(' - s : %s',str(BigInt.from_uint256(self.s_scl).as_long()) )
        logging.info('#################################### ')

        logging.info(' Starting First Mexp...')

        pk_bin = pkbin_get(self.pk,['nPublic', 'domainSize'])
        nPublic = pk_bin[0][0]
        domainSize = pk_bin[1][0]
        self.public_signals = np.copy(self.scl_array[1:nPublic+1])

        pk_bin = pkbin_get(self.pk,['A','B2','B1','C', 'hExps','delta_1'])

 
        #TODO : trying to separate  G2 MExp from G1 Mexp in phase 1 to check if some time is gained
        self.findECPointsDispatch(
                                  self.dispatch_table_phase1b,
                                  self.batch_size_mexp_phase1,
                                  self.dispatch_table_phase1b.shape[0]-1, pk_bin,
                                  sort_idx = self.ec_type_dict['B2'][2],
                                  change_s_scl_idx = [self.ec_type_dict['B2'][2]],
                                  sort_en=self.sort_en)

        if self.zk:
              n_els = self.dispatch_table_phase1.shape[0]-(self.ec_type_dict['C'][2])
        else:
              n_els = self.dispatch_table_phase1.shape[0]-2

        self.findECPointsDispatch(
                                  self.dispatch_table_phase1,
                                  self.batch_size_mexp_phase1,
                                  #self.dispatch_table_phase1.shape[0]-(self.ec_type_dict['C'][2]+1), pk_bin,
                                  #self.dispatch_table_phase1.shape[0]-(self.ec_type_dict['C'][2]), pk_bin,
                                  n_els, pk_bin,
                                  #change_s_scl_idx = [self.ec_type_dict['B2'][2], self.ec_type_dict['B1'][2]],
                                  change_s_scl_idx = [self.ec_type_dict['B1'][2]],
                                  change_r_scl_idx = [self.ec_type_dict['A'][2]],
                                  sort_en=self.sort_en)

        if self.stop_client.value:
            self.p_CPU.terminate()
            self.p_CPU.join()
            logging.info('Client stopped...')
            return

        #print(self.dispatch_table_phase2, self.dispatch_table_phase2.shape[0]+1)
        self.findECPointsDispatch(
                             self.dispatch_table_phase2,
                             nPublic+2,
                             self.dispatch_table_phase2.shape[0]+1, pk_bin,
                             sort_en = self.sort_en)

        if self.stop_client.value:
            self.p_CPU.terminate()
            self.p_CPU.join()
            logging.info('Client stopped...')
            return

        # Assign collected values to pi's
        self.assignECPvalues(compute_ECP=False)
        logging.info(' First Mexp completed...')

        end = time.time()
        self.t_GP['Mexp1'] = (end - start)

        ######################
        # Beginning of P3 and P4
        #  P3 - Poly Eval
        #  P4 - Poly Operations
        ######################

        # Retrieve Poly Eval Results
        if self.compute_last_mexp_gpu:
          self.parent_conn_CPU.send([])

        t = self.parent_conn_CPU.recv()
        self.t_GP['Eval'] += t[0]
        self.p_CPU.terminate()
        self.p_CPU.join()

        if self.compute_ntt_gpu:
          self.t_GP['H'] = self.calculateH()
        else:
          self.t_GP['H'] = t[1]

        ######################
        # Beginning of P5
        #   - Final EC MultiExp
        ######################
        start = time.time()

        if self.compute_last_mexp_gpu:
           logging.info(' Starting Last Mexp...')
           self.findECPointsDispatch(
                                  self.dispatch_table_phase3,
                                  self.batch_size_mexp_phase3,
                                  self.dispatch_table_phase3.shape[0]-1, pk_bin,
                                  sort_idx = self.ec_type_dict['hExps'][2],
                                  change_rs_scl_idx = [self.ec_type_dict['hExps'][2]],
                                  sort_en=self.sort_en)
        
           self.assignECPvalues(compute_ECP=True)
        else:
           self.t_GP['Mexp2'] = t[2]
           self.reduceLastMexp(pk_bin[5])

        end = time.time()
        self.t_GP['Mexp2'] = (end - start)
     
        return 
 
    def reduceLastMexp(self, delta):
       EC_C_idx = self.ec_type_dict['C'][4]

       a5 = ec_jacaddreduce_h(
                           np.reshape(np.concatenate(np.reshape(self.init_ec_val[:,:,EC_C_idx],-1)),-1),
                           MOD_GROUP,
                           1,   # to affine
                           1,   # Add z coordinate to inout
                           0)   # strip z coordinate from affine result
       if self.zk:
           a1 = ec_jacscmul_h(
                 np.reshape(self.s_scl,-1),
                 np.reshape(self.pi_a_eccf1,-1),
                 MOD_GROUP,1)

           a2 = ec_jacscmul_h(
                 np.reshape(self.r_scl,-1),
                 np.reshape(self.pi_b1_eccf1,-1),
                 MOD_GROUP,1)

           a3 = ec_jacscmul_h(
                 np.reshape(self.neg_rs_scl,-1),
                 np.reshape(delta,-1),
                 MOD_GROUP,1)

           a4 = np.concatenate((self.pi_c2_eccf1,
                                [ECC.one[ZUtils.FRDC].as_uint256()]))


           a = np.concatenate((a1,a2,a3,a4,a5))

           np.copyto(
                    self.pi_c_eccf1,
                    ec_jacaddreduce_h(
                                      np.reshape(a,-1),
                                      MOD_GROUP,
                                      1,   # to affine
                                      0,   # Add z coordinate to input
                                      1)   # strip z coordinate from affine result
                    )
       else:
           np.copyto(
                   self.pi_c_eccf1, a5[:2])

    def assignECPvalues(self, compute_ECP=False):

        EC_A_idx = self.ec_type_dict['A'][4]
        EC_B2_idx = self.ec_type_dict['B2'][4]
        EC_B1_idx = self.ec_type_dict['B1'][4]
        EC_C_idx = self.ec_type_dict['C'][4]

        if compute_ECP is False:
            np.copyto(
                       self.pi_a_eccf1,
                       ec_jacaddreduce_h(
                                       np.reshape(np.concatenate(np.reshape(self.init_ec_val[:,:,EC_A_idx],-1)),-1),
                                       MOD_GROUP,
                                       1,   # to affine
                                       1,   # Add z coordinate to inout
                                       1)   # strip z coordinate from affine result
                                   )

            np.copyto(
                      self.pi_b_eccf2,
                      ec2_jacaddreduce_h(
                                      np.reshape(np.concatenate(np.reshape(self.init_ec_val[:,:,EC_B2_idx],-1)),-1),
                                      MOD_GROUP,
                                      1,   # to affine
                                      1,   # Add z coordinate to inout
                                      1)   # strip z coordinate from affine result
                                   )

            np.copyto(
                      self.pi_b1_eccf1,
                      ec_jacaddreduce_h(
                                    np.reshape(np.concatenate(np.reshape(self.init_ec_val[:,:,EC_B1_idx],-1)),-1),
                                    MOD_GROUP,
                                    1,   # to affine
                                    1,   # Add z coordinate to inout
                                    1)   # strip z coordinate from affine result
                                  )
        else:
            np.copyto(
                      self.pi_c_eccf1,
                      ec_jacaddreduce_h(
                                      np.reshape(np.concatenate(np.reshape(self.init_ec_val[:,:,EC_C_idx],-1)),-1),
                                      MOD_GROUP,
                                      1,   # to affine
                                      1,   # Add z coordinate to inout
                                      1)   # strip z coordinate from affine result
                                   )

    def compute_proof_ecp(self, pyCuOjb, ecbn128_samples, ec2, shamir_en=0, gpu_id=0, stream_id=0):
            ZField.set_field(MOD_GROUP)
            #TODO : remove in_v
            #print(ecbn128_samples.shape, ec2, shamir_en, gpu_id, stream_id)
            in_v, ecp,t = ec_mad_cuda2(pyCuOjb, ecbn128_samples, MOD_GROUP, ec2, shamir_en, gpu_id, stream_id,True)
            if ec2 and stream_id == 0:
              ecp = ec2_jac2aff_h(ecp.reshape(-1),MOD_GROUP)
            elif stream_id == 0:
              ecp = ec_jac2aff_h(ecp.reshape(-1),MOD_GROUP)

            """
            # Debug code to test result
            enable=0
            if enable:
              total_samples = int(ecp.shape[0]/3)
              offset = int((int(in_v.shape[0]/3) - K.shape[0] + 1)/128)
              first_v_sample = int((int(in_v.shape[0]/3) - K.shape[0] + 1)%128)
              
              start_idx = offset
              n_samples = total_samples - start_idx
              #p1 = ec_jacscmul_h(in_v[start_idx:start_idx+n_samples].reshape(-1), in_v[total_samples+2*start_idx:total_samples+2*(start_idx+n_samples)].reshape(-1),MOD_GROUP, add_last=1)
              #p1 = ec_jac2aff_h(p1.reshape(-1),MOD_GROUP)
              #r1 = all(np.concatenate(p1 == ecp[3*start_idx:3*(start_idx+n_samples)]))
              p = []
              p.append(ecp[start_idx*3:start_idx*3+3])
              #p_ec = ECC.from_uint256(p[-1], in_ectype=2,out_ectype=2,reduced=True)[0]
              for idx in range(start_idx+1,total_samples):
                 p.append(ec_jacadd_h(p[-1].reshape(-1), ecp[3*idx:3*idx+3].reshape(-1), MOD_GROUP))
                 #p1_ec = ECC.from_uint256(p1[3*idx:3*idx+3], in_ectype=2,out_ectype=2,reduced=True)[0]
                 #p_ec = p_ec + p1_ec
                 #pr_ec = ECC.from_uint256(p[-1], in_ectype=1,out_ectype=2,reduced=True)[0]
                 #if p_ec != pr_ec:
                     #printf(idx)
                     #break
              p2 = ec_jac2aff_h(np.reshape(np.asarray(p[-1]),-1),MOD_GROUP)
              #a = ECC.from_uint256(p2,in_ectype=2, out_ectype=2, reduced=True)
              #b = [x.extend().as_str() for x in a]
              #j = json.dumps(b)
              #f = open('../../circuits/pib1_t2.json', 'w')
              #print(j,file=f)
              #f.close()
            """
                

            if ec2:
              return ecp[0:6], t
            else:
              #return ecp, t
              return ecp[0:3], t

    def getECResults(self, dispatch_table):
       for bidx,p in enumerate(dispatch_table):
          P = p[0]
          cuda_ec128 = self.ec_type_dict[P][0]
          step = self.ec_type_dict[P][1]
          pidx = self.ec_type_dict[P][4]
          gpu_id = p[3]
          stream_id = p[4]
          result, t = cuda_ec128.streamSync(gpu_id,stream_id)
          """
          print("Results : gpu_id : "+str(gpu_id) + " stream_id : "+ str(stream_id) + " ec2 : " + str(step==4))
          print("Results prev")
          print(self.init_ec_val[gpu_id][stream_id][pidx])
          """
          if step==4:
             self.init_ec_val[gpu_id][stream_id][pidx] =\
                            ec2_jac2aff_h(
                             result.reshape(-1),
                             MOD_GROUP,
                             strip_last=1) 
          else:
             self.init_ec_val[gpu_id][stream_id][pidx] =\
                        ec_jac2aff_h(
                              result.reshape(-1),
                              MOD_GROUP,
                              strip_last=1) 
          #self.init_ec_val[gpu_id][stream_id][pidx] = result[:step]
          """
          print("Results after")
          print(self.init_ec_val[gpu_id][stream_id][pidx])
          print("\n\n\n")
          """

    def init_EC_P(self, batch_size):
       nsamples = np.product(get_shfl_blockD(batch_size))
       EC_P1 = np.zeros((nsamples*(ECP_JAC_INDIMS  + U256_NDIMS),NWORDS_256BIT), dtype=np.uint32)
       EC_P2 = np.zeros((nsamples*(ECP2_JAC_INDIMS + U256_NDIMS),NWORDS_256BIT), dtype=np.uint32)
       EC_P = [EC_P1, EC_P2]
       scl_start_idx = nsamples - batch_size
       ec_start_idx = [nsamples+ECP_JAC_INDIMS*(nsamples-batch_size), 
                       nsamples+ECP2_JAC_INDIMS*(nsamples-batch_size)]

       ## add scl to multiply previous EC_P
       EC_P[0][nsamples-1] = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32)
       EC_P[1][nsamples-1] = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32)

       return nsamples, EC_P, scl_start_idx, ec_start_idx

    def findECPointsDispatch(self, dispatch_table, batch_size, last_batch_idx, pk_bin,
                             sort_idx=0, change_s_scl_idx=[-1], change_r_scl_idx=[-1], change_rs_scl_idx=[-1], sort_en=0):

       ZField.set_field(MOD_GROUP)
       nsamples, EC_P, scl_start_idx, ec_start_idx = self.init_EC_P(batch_size)
       extra_samples = 0

       n_par_batches = self.n_gpu * max((self.n_streams - 1),1)
       pending_dispatch_table = []
       n_dispatch=len(pending_dispatch_table)

       for bidx, p in enumerate(dispatch_table):
          #Retrieve point name : A,B1,B2,..
          P = p[0]    
          # Retrieve cuda pointer
          cuda_ec128 = self.ec_type_dict[P][0]
          step = self.ec_type_dict[P][1]
          # Retrieve point idx : A->0, B2->1, B1->2, C->3
          EPidx = self.ec_type_dict[P][2]
          # Retrieve pis 
          pidx = self.ec_type_dict[P][4]
          # Retrieve EC type : EC -> 0, EC2 -> 1
          ecidx = self.ec_type_dict[P][3]
          start_idx = p[1]
          end_idx   = p[2]
          gpu_id    = p[3]
          stream_id = p[4]
          init_ec_val = self.init_ec_val[gpu_id][stream_id][pidx]

          if bidx >= last_batch_idx:
            if bidx == last_batch_idx:
              # End point is end point + 2 additional scalars + previous point
              extra_samples = 2
              if EPidx in change_rs_scl_idx:
                  extra_samples +=1
              batch_size = end_idx+extra_samples+1-start_idx
              nsamples, EC_P, scl_start_idx, ec_start_idx = self.init_EC_P(batch_size)
              # 1 * EC_P
              self.sorted_scl_array[end_idx:end_idx+1]   = np.asarray([1,0,0,0,0,0,0,0], dtype=np.uint32)
              self.sorted_scl_array_idx[end_idx] = end_idx - start_idx
              self.sorted_scl_array_idx[end_idx+1] = end_idx+1 - start_idx

            # r/s * EC_P
            if EPidx in change_r_scl_idx :
                self.sorted_scl_array[end_idx+1:end_idx+2] = self.r_scl
            elif EPidx in change_s_scl_idx:
                self.sorted_scl_array[end_idx+1:end_idx+2] = self.s_scl
            elif EPidx in change_rs_scl_idx:
                self.sorted_scl_array[end_idx:end_idx+1] = self.s_scl
                self.sorted_scl_array[end_idx+1:end_idx+2] = self.r_scl
                self.sorted_scl_array[end_idx+2:end_idx+3] = self.neg_rs_scl

                self.sorted_scl_array_idx[end_idx+2] = end_idx+2 - start_idx

                pk_bin[EPidx][step*end_idx*NWORDS_256BIT:step*(end_idx+1)*NWORDS_256BIT] =\
                        np.reshape(self.pi_a_eccf1,-1)[:step*NWORDS_256BIT]
                pk_bin[EPidx][step*(end_idx+1)*NWORDS_256BIT:step*(end_idx+2)*NWORDS_256BIT] =\
                       np.reshape(self.pi_b1_eccf1,-1)[:step*NWORDS_256BIT]

            else:
                self.sorted_scl_array[end_idx:end_idx+1]  = np.asarray([0,0,0,0,0,0,0,0], dtype=np.uint32)
                self.sorted_scl_array[end_idx:end_idx+2]  = np.asarray([0,0,0,0,0,0,0,0], dtype=np.uint32)
                self.sorted_scl_array_idx[end_idx] = 0
                self.sorted_scl_array_idx[end_idx+1] = 0

            EC_P[ecidx][nsamples-1-extra_samples:nsamples-1] = self.sorted_scl_array[end_idx:end_idx+extra_samples]

          if EPidx==sort_idx:
              # Sort scl batch
              self.sorted_scl_array_idx[start_idx:end_idx] = \
                   sortu256_idx_h(self.scl_array[start_idx:end_idx], sort_en)
              self.sorted_scl_array[start_idx:end_idx] = \
                   self.scl_array[start_idx:end_idx][self.sorted_scl_array_idx[start_idx:end_idx]]
              # Copy sorted scl batch
              EC_P[0][scl_start_idx:nsamples-1] = self.sorted_scl_array[start_idx:end_idx+extra_samples]
              EC_P[1][scl_start_idx:nsamples-1] = self.sorted_scl_array[start_idx:end_idx+extra_samples]

          # Copy sorted EC points batch
          EC_P[ecidx][ec_start_idx[ecidx]:-step] = \
                     np.reshape(
                        np.reshape(
                           pk_bin[EPidx][step*start_idx*NWORDS_256BIT:step*(end_idx+extra_samples)*NWORDS_256BIT],
                           (-1,step,NWORDS_256BIT))[self.sorted_scl_array_idx[start_idx:end_idx+extra_samples]],
                        (-1, NWORDS_256BIT)
                    )

          # Copy last batch EC point sum
          EC_P[ecidx][-step:] = init_ec_val

          result, t = self.compute_proof_ecp(
              cuda_ec128,
              EC_P[ecidx],
              step == 4, shamir_en=1, gpu_id=gpu_id, stream_id=stream_id)


          """
          result = ec_jacreduce_h(np.reshape(EC_P[ecidx][:nsamples],-1),
                  np.reshape(EC_P[ecidx][nsamples:],-1), MOD_GROUP, 1, 1, 1)
          """
          if stream_id == 0:
             self.init_ec_val[gpu_id][stream_id][pidx] = result[:step]
             try:
               cuda_ec128.streamDel(gpu_id, stream_id)
             except ValueError:
               logging.error('Exception occurred when getting EC results. Exiting program...')
               sys.exit(1)
               
          else :
             pending_dispatch_table.append(p)
             n_dispatch +=1


             # Collect results. Leave last batch uncollected to maximize parallelization
             if n_dispatch == n_par_batches:
                 n_dispatch=0

                 try:
                    self.getECResults(pending_dispatch_table)
                 except ValueError:
                    logging.error('Exception occurred when getting EC results. Exiting program...')
                    sys.exit(1)
                 pending_dispatch_table = []

          if self.stop_client.value:
              return

       # Collect final results
       self.getECResults(pending_dispatch_table)

    def write_pdata(self):
        if self.out_public_f.endswith('.json'):
           # Write public file
           ZField.set_field(MOD_FIELD)
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
           ZField.set_field(MOD_GROUP)
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
           pi_a_eccf1 = from_montgomeryN_h(np.reshape(self.pi_a_eccf1,(-1)), MOD_GROUP)
           pi_b_eccf2 = from_montgomeryN_h(np.reshape(self.pi_b_eccf2,(-1)), MOD_GROUP)
           pi_c_eccf1 = from_montgomeryN_h(np.reshape(self.pi_c_eccf1,(-1)), MOD_GROUP)
           proof_bin = np.concatenate((
                    pi_a_eccf1, 
                    pi_b_eccf2,
                    pi_c_eccf1))
           writeU256DataFile_h(proof_bin, self.out_public_f.encode("UTF-8"))
               
    def evalPoly(self,w, pX, nVars, m, pidx):
        # Convert witness to montgomery in zpoly_maddm_h
        #polA_T, polB_T, polC_T are montgomery -> polsA_sps_u256, polsB_sps_u256, polsC_sps_u256 are montgomery
        reduce_coeff = 0  
        polX_T = mpoly_eval_h(w[:nVars],np.reshape(pX,-1), reduce_coeff, m, 0, nVars, mp.cpu_count(), pidx)
        return polX_T

    def calculateH(self):

        ZField.set_field(MOD_FIELD)
        pk_bin = pkbin_get(self.pk,['nVars', 'domainSize', 'polsA', 'polsB'])
        nVars = pk_bin[0][0]
        m = pk_bin[1][0]
        """
        pA = np.reshape(pk_bin[2][:m*NWORDS_256BIT],(m,NWORDS_256BIT))
        pB = np.reshape(pk_bin[3][:m*NWORDS_256BIT],(m,NWORDS_256BIT))
        """
        # Convert witness to montgomery in zpoly_maddm_h
        #polA_T, polB_T, polC_T are montgomery -> polsA_sps_u256, polsB_sps_u256, polsC_sps_u256 are montgomery

        start = time.time()
        logging.info(' Calculating H...')

        ifft_params = ntt_build_h(self.pA_T.shape[0])

        if self.n_bits_roots < ifft_params['levels']:
          logging.error('Insufficient number of roots in ' + self.roots_f + 'Required number of roots is '+ str(1<< ifft_params['levels']))
          sys.exit(1)
    
        # TEST Vectors
        #pA_T = readU256DataFile_h("../../test/c/aux_data/zpoly_samples_tmp2.bin".encode("UTF-8"), 1<<17 , 1<<17 )
        #pB_T = readU256DataFile_h("../../test/c/aux_data/zpoly_samples_tmp2.bin".encode("UTF-8"), 1<<17 , 1<<17 )
        pH,t1 = zpoly_interp_and_mul_cuda(
                                              self.cuzpoly,
                                              np.concatenate((self.pA_T,self.pB_T)),
                                              ifft_params, 
                                              ZField.get_field(), 
                                              self.roots_rdc_u256,
                                              self.batch_size, n_gpu=self.n_gpu)


        np.copyto(self.scl_array[:m-1], pH[m:-1])

        return time.time()-start
