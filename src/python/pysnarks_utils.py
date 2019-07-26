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
// File name  : pysnarks_utils.py
//
// Date       : 13/05/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Python Snarks services utils implementation 
# 


// Description:
//    
//   TODO
//    
// ------------------------------------------------------------------

"""
import json,ast
import os.path
import os
import numpy as np
import time
import sys
import multiprocessing as mp
from datetime import datetime

from random import randint


from zutils import ZUtils
from zfield import *
from ecc import *
from zpoly import *
from constants import *

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
try:
  from pycusnarks import *
  use_pycusnarks = True
except ImportError:
    use_pycusnarks = False


def json_to_dict(data, labels=None ):
    if labels is None:
       data = {str(k) : data[k] for k in data.keys()}
    else:
        data = {str(k) : data[k] for k in data.keys() if k in labels}
    for k, v in data.items():
        if type(data[k]) is list:
          json_to_list(data[k])
        elif type(data[k]) is dict:
          json_to_dict(data[k])
        elif is_long(v):
          data[k] = int(v)

    return data

def json_to_list(data):
     for idx,el in enumerate(data):
         if type(el) is list:
             json_to_list(el)
         elif type(el) is dict:
             data[idx] = json_to_dict(el)
         elif sys.version_info[0] == 2:
           if type(el) is unicode or type(el) is str:
              if el.isdigit():
                data[idx] = int(el)
              else:
                 data[idx] = el
         elif sys.version_info[0] >= 3:
           if  type(el) is str:
              if el.isdigit():
                data[idx] = int(el)
              else:
                 data[idx] = el
     return

def pysnarks_compare(f1_str, f2_str, labels, npublic):
     try:
       f1 = open(f1_str,'r')
     except Exception as e:
       return False
      
     djson = json.load(f1)
     if isinstance(djson,dict):
       d1 = json_to_dict(djson, labels)
     else:
       d1 = json_to_list(djson)
     f1.close()

     try:
       f2 = open(f2_str,'r')
     except Exception as e:
       return False
     djson = json.load(f2)
     if isinstance(djson,dict):
       d2 = json_to_dict(djson, labels)
     else:
       d2 = json_to_list(djson)
     f2.close()
     
     djson = None

     if labels is not None and 'C' in labels:
        d1['C'] = d1['C'][npublic+1:]
        d2['C'] = d2['C'][npublic+1:]

     return d1 == d2

   

def is_long(input):
        try:
            num = int(input)
        except ValueError:
            return False
        return True

def getCircuit():
    cir = {}
    cir['nWords']   = None
    cir['ftype'] = np.uint32(SNARKSFILE_T_PK)
    cir['protocol'] = None
    cir['nPubInputs'] = None
    cir['Rbitlen'] = None
    cir['cirformat']    = None
    cir['nVars']        = None
    cir['nOutputs']     = None
    cir['nConstraints'] = None
    cir['field_r'] = None
    cir['group_q'] = None
    cir['R1CSA_nWords'] = None
    cir['R1CSB_nWords'] = None
    cir['R1CSC_nWords'] = None
    cir['R1CSA']        = None
    cir['R1CSB']       = None
    cir['R1CSC']        = None
    
    return cir


def cirjson_to_vars(in_circuit_f, in_circuit_format, out_circuit_format):
        """
          Converts from circom .json output file to binary format required to 
            calculate snarks setup. Only the following entries are used:
             - constraints -> R1CS a,b,c
             - nPubInputs  -> k
             - nVars       -> N
             - nOutputs    ->
             - cirformat      -> EXT[0]/MONT[1]

          R1CS binary format:
            N constraints  -------------------------------- 32 bits  
            cumsum(  -> cumulative
              N coeff constraints[0] ---------------------- 32 bits
              N coeff constraints[1] ---------------------- 32 bits : N constraints[0] + N constraints[1]
              ----
              N coeff constraints[N-1] -------------------- 32 bits : N contraints[0] + N constraints[1] +
                                                                      N constraints[2] +...+ Nconstraints[N-1]
            )
            Coeff[0,0] constraint 0, coeff 0 -------------- 32 bits
            Coeff[0,1] constraint 0, coeff 1 -------------- 32 bits
            ----
            Coeff[0,C0-1] constraint 0, coeff C0-1 -------- 32 bits
            Val[0,0] constraint 0, value 0 ---------------- 256 bits (8 words) : word 0 is LSW
            Val[0,1] constraint 0, value 1 ---------------- 256 bits 
            ----
            Val[0,C0-1] constraint 0, value C0-1 - -------- 256 bits 
            Coeff[1,0] constraint 1, coeff 0 -------------- 32 bits
            Coeff[1,1] constraint 1, coeff 1 -------------- 32 bits
            ----
            Coeff[1,C1-1] constraint 1, coeff C1-1 -------- 32 bits
            Val[1,0] constraint 1, value 0 ---------------- 256 bits 
            Val[1,1] constraint 1, value 1 ---------------- 256 bits 
            ----
            Val[1,C1-1] constraint 1, value C1-1 -- ------- 256 bits 
            ----
            ----
            Coeff[N-1,0] constraint N-1, coeff 0 ---------- 32 bits
            Coeff[N-1,1] constraint N-1, coeff 1 ---------- 32 bits
            ----
            Coeff[N-1,CN_1-1] constraint N-1, coeff CN_1-1  32 bits
            Val[N-1,0] constraint 1, value 0 -------------- 256 bits 
            Val[N-1,1] constraint 1, value 1 -------------- 256 bits 
            ----
            Val[N-1,CN_1-1] constraint 1, value CN_1-1 ---- 256 bits 

          Binary file format
            nWords : File size in 32 bit workds --------------- 32 bits
            nPubInputs : -------------------------------------- 32 bits
            nOutputs   : -------------------------------------- 32 bits
            nVars      : -------------------------------------- 32 bits
            nConstraints : Number of constraints--------------- 32 bits
            cirformat : Extended[0]/Montgomery[1]----------------- 32 bits
            R1CSA_nWords : R1CSA size in 32 bit words --------- 32 bits
            R1CSB_nWords : R1CSB size in 32 bit words --------- 32 bits
            R1CSC_nWords : R1CSC size in 32 bit words --------- 32 bits
            R1CSA        :  R1CS  format 
            R1CSB        :  R1CS format
            R1CSC        : R1Cs format
 
            
        """
        labels = ['constraints', 'nPubInputs','nOutputs','nVars']
        f = open(in_circuit_f,'r')
        cir_json_data = json.load(f)
        cir_data = json_to_dict(cir_json_data, labels)
        f.close()
        tmp_in_circuit_format = in_circuit_format

        if 'cirformat' in cir_data:
            tmp_in_circuit_format = cir_data['cirformat']

        worker = mp.Pool(processes=min(3,mp.cpu_count()-1))

        r1 = worker.apply_async(cirjson_to_r1cs, args = (0,tmp_in_circuit_format, out_circuit_format, cir_data))
        r2 = worker.apply_async(cirjson_to_r1cs, args=(1,tmp_in_circuit_format, out_circuit_format, cir_data))
        r3 = worker.apply_async(cirjson_to_r1cs, args = (2,tmp_in_circuit_format, out_circuit_format, cir_data))

        R1CSA_len, R1CSA_u256 = r1.get()
        R1CSB_len, R1CSB_u256 = r2.get()
        R1CSC_len, R1CSC_u256 = r3.get()

        worker.terminate()

        fsize = CIRBIN_H_N_OFFSET + R1CSA_len + R1CSB_len + R1CSC_len

        # Init circuit fields
        cir = getCircuit()

        cir['nWords']       =  np.uint32(fsize)
        cir['protocol']     = PROTOCOL_T_GROTH  # Groth
        cir['nPubInputs']   =  np.uint32(cir_data['nPubInputs'])
        cir['Rbitlen']        = np.asarray(ZField.get_reduction_data()['Rbitlen'],dtype=np.uint32)
        cir['cirformat']       =  np.uint32(out_circuit_format)
        cir['nVars']        =  np.uint32(cir_data['nVars'])
        cir['nOutputs']     =  np.uint32(cir_data['nOutputs'])
        cir['nConstraints'] =  np.uint32(len(cir_data['constraints']))
        pidx = ZField.get_field()
        ZField.set_field(MOD_FIELD)
        cir['field_r']        = ZField.get_extended_p().as_uint256()
        ZField.set_field(MOD_GROUP)
        cir['group_q']       = ZField.get_extended_p().as_uint256()
        ZField.set_field(pidx)
        cir['R1CSA_nWords'] =  np.uint32(R1CSA_len)
        cir['R1CSB_nWords'] =  np.uint32(R1CSB_len)
        cir['R1CSC_nWords'] =  np.uint32(R1CSC_len)
        cir['R1CSA']        =  R1CSA_u256
        cir['R1CSB']        =  R1CSB_u256
        cir['R1CSC']        =  R1CSC_u256 
    
        return cir
 
def cirjson_to_r1cs(idx, in_circuit_format, out_circuit_format, cir_data):
        if in_circuit_format == out_circuit_format:
          R1CS_u256 = [ZPolySparse(coeff[idx]).as_uint256() for coeff in cir_data['constraints']]
        elif in_circuit_format == FMT_EXT:
          R1CS_u256 = [ZPolySparse(coeff[idx]).reduce().as_uint256() for coeff in cir_data['constraints']]
        else :
          R1CS_u256 = [ZPolySparse(coeff[idx]).extend().as_uint256() for coeff in cir_data['constraints']]

        R1CS_l = []
        R1CS_p = []
        for l,p in R1CS_u256:
            R1CS_l.append(l)
            R1CS_p.append(p)
        R1CS_u256 = np.asarray(np.concatenate((np.asarray([len(R1CS_l)]),
                                              np.concatenate(
                                                 (np.cumsum(R1CS_l), 
                                                  np.concatenate(R1CS_p))))),
                                                  dtype=np.uint32)
        R1CS_len = R1CS_u256.shape[0]

        return  R1CS_len, R1CS_u256 


def cirvars_to_bin(cir):
        return  np.concatenate((
                       [cir['nWords'],
                        cir['nPubInputs'],
                        cir['nOutputs'],
                        cir['nVars'],
                        cir['nConstraints'],
                        cir['cirformat'],
                        cir['R1CSA_nWords'],
                        cir['R1CSB_nWords'],
                        cir['R1CSC_nWords']],
                        cir['R1CSA'],
                        cir['R1CSB'],
                        cir['R1CSC']))


def cirbin_to_vars(self, ciru256_data):
        R1CSA_offset = CIRBIN_H_N_OFFSET
        R1CSB_offset = CIRBIN_H_N_OFFSET +  \
                       np.uint32(ciru256_data[CIRBIN_H_CONSTA_NWORDS_OFFSET])
        R1CSC_offset = CIRBIN_H_N_OFFSET + \
                       np.uint32(ciru256_data[CIRBIN_H_CONSTA_NWORDS_OFFSET]) + \
                       np.uint32(ciru256_data[CIRBIN_H_CONSTB_NWORDS_OFFSET])

        cir = getCircuit()
        cir['nWords']        =  np.uint32(ciru256_data[CIRBIN_H_NWORDS_OFFSET])
        cir['nPubInputs']    =  np.uint32(ciru256_data[CIRBIN_H_NPUBINPUTS_OFFSET])
        cir['nOutputs']      =  np.uint32(ciru256_data[CIRBIN_H_NOUTPUTS_OFFSET])
        cir['nVars']         =  np.uint32(ciru256_data[CIRBIN_H_NVARS_OFFSET])
        cir['nConstraints']  =  np.uint32(ciru256_data[CIRBIN_H_NCONSTRAINTS_OFFSET])
        cir['cirformat']       =  np.uint32(ciru256_data[CIRBIN_H_FORMAT_OFFSET])
        cir['R1CSA_nWords'] =  np.uint32(ciru256_data[CIRBIN_H_CONSTA_NWORDS_OFFSET])
        cir['R1CSB_nWords'] =  np.uint32(ciru256_data[CIRBIN_H_CONSTB_NWORDS_OFFSET])
        cir['R1CSC_nWords'] =  np.uint32(ciru256_data[CIRBIN_H_CONSTC_NWORDS_OFFSET])
        cir['R1CSA']        =  ciru256_data[R1CSA_offset:R1CSB_offset] 
        cir['R1CSB']       =  ciru256_data[R1CSB_offset:R1CSC_offset]
        cir['R1CSC']        =  ciru256_data[R1CSC_offset:] 

        return cir

    
    
def cirr1cs_to_mpoly(r1cs, cir_header, fmat, extend):
        to_mont = 0
        ZField.set_field(MOD_FIELD)
        pidx = ZField.get_field()
        if fmat == ZUtils.FEXT:
           to_mont = 1

        poly_len = r1cs_to_mpoly_len_h(r1cs,cir_header, extend)
        pols = r1cs_to_mpoly_h(poly_len, r1cs, cir_header, to_mont, pidx, extend)
        
        return pols

def cirvars_to_pkvars(pk, cir):

        pk['protocol'] = cir['protocol']
        pk['Rbitlen'] = cir['Rbitlen']

        pk['nVars'] = cir['nVars']
        pk['nPublic']    = cir['nPubInputs'] + cir['nOutputs']
        pk['domainBits'] =  np.uint32(math.ceil(math.log(cir['nConstraints']+ 
                                           cir['nPubInputs'] + 
                                           cir['nOutputs'],2)))
        pk['domainSize'] = 1 << pk['domainBits']
        pk['field_r'] = np.copy(cir['field_r'])
        pk['group_q'] = np.copy(cir['group_q'])

def getPK():
      pk = {}

      pk['nWords'] = 0
      pk['ftype'] = np.uint32(SNARKSFILE_T_PK)
      pk['protocol'] = None
      pk['Rbitlen'] = None
      pk['k_binformat'] = None
      pk['k_ecformat'] = None
      pk['nVars'] = None
      pk['nPublic'] = None
      pk['domainBits'] = None
      pk['domainSize'] = None
      pk['field_r'] = None
      pk['group_q'] = None
      pk['polsA_nWords'] = None
      pk['polsB_nWords'] = None
      pk['polsC_nWords'] = None
      pk['A_nWords'] = None
      pk['B1_nWords'] = None
      pk['B2_nWords'] = None
      pk['C_nWords'] = None
      pk['hExps_nWords'] = None
      pk['polsA'] = None
      pk['polsB'] = None
      pk['polsC'] = None
      pk['alfa_1'] = None
      pk['beta_1'] = None
      pk['delta_1'] = None
      pk['beta_2'] = None
      pk['delta_2'] = None
      pk['A'] = None
      pk['B1'] = None
      pk['B2'] = None
      pk['C'] = None
      pk['hExps'] = None
      pk['IC'] = None

      return pk

def mpoly_to_json(mpoly, reduced):
     ZField.set_field(MOD_FIELD)
     spoly = mpoly_to_sparseu256_h(mpoly)
     if reduced:
          P = [{k : str(BigInt.from_uint256(p[k]).as_long()) for  k in p.keys()} for p in spoly]
     else:
          P = [{k : str(ZFieldElRedc(BigInt.from_uint256(p[k])).extend().as_long()) for  k in p.keys()} for p in spoly]
     return P

def ecp_to_json(ecp, out_ec, b_reduce, ec2):
        ZField.set_field(MOD_GROUP)
        if ec2:
           P = ECC.from_uint256(ecp.reshape((-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, ec2=True, remove_last=True)
        else:
           P = ECC.from_uint256(np.reshape(ecp,(-1,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, remove_last=True)

        if not b_reduce:
           p = [x.extend().as_str() for x in P]
        else:
           p = [x.as_str() for x in P]

        return p


def pkvars_to_json(out_bin, out_ec, pk):
        pk_dict= {}
        pk_dict['ftype'] = "PK_FILE"
        pk_dict['protocol'] = "groth"
        pk_dict['field_r'] = str(ZFieldElExt.from_uint256(pk['field_r']).as_long())
        pk_dict['group_q'] = str(ZFieldElExt.from_uint256(pk['group_q']).as_long())
        if out_bin == FMT_EXT:
           pk_dict['k_binformat'] = "normal"
           b_reduce = False
        else:
           pk_dict['k_binformat'] = "montgomery"
           b_reduce=True

        pk_dict['Rbitlen'] = int(pk['Rbitlen'])

        if out_ec == EC_T_AFFINE:
           pk_dict['k_ecformat'] = "affine"
        elif out_ec == EC_T_JACOBIAN: 
           pk_dict['k_ecformat'] = "jacobian"
        else :
           pk_dict['k_ecformat'] = "projective"

           
        pk_dict['nVars'] = int(pk['nVars'])
        pk_dict['nPublic'] = int(pk['nPublic'])
        pk_dict['domainBits'] = int(pk['domainBits'])
        pk_dict['domainSize'] = int(pk['domainSize'])

        ZField.set_field(MOD_FIELD)

        worker = mp.Pool(processes=min(5,mp.cpu_count()-1))

        if out_bin == FMT_EXT:
          r1 = worker.apply_async(mpoly_to_json, args=(pk['polsA'], False))
          r2 = worker.apply_async(mpoly_to_json, args=(pk['polsB'], False))
          r3 = worker.apply_async(mpoly_to_json, args=(pk['polsC'], False))
        else:
          r1 = worker.apply_async(mpoly_to_json, args=(pk['polsA'], True))
          r2 = worker.apply_async(mpoly_to_json, args=(pk['polsB'], True))
          r3 = worker.apply_async(mpoly_to_json, args=(pk['polsC'], True))
       
        pk_dict['polsA'] = r1.get()
        pk_dict['polsB'] = r2.get()
        pk_dict['polsC'] = r3.get()

        ZField.set_field(MOD_GROUP)

        r1 = worker.apply_async(ecp_to_json, args=(pk['A'], out_ec, b_reduce, False))
        r2 = worker.apply_async(ecp_to_json, args=(pk['B1'], out_ec, b_reduce, False))
        r3 = worker.apply_async(ecp_to_json, args=(pk['B2'], out_ec, b_reduce, True))
        r4 = worker.apply_async(ecp_to_json, args=(pk['C'], out_ec, b_reduce, False))
        r5 = worker.apply_async(ecp_to_json, args=(pk['hExps'], out_ec, b_reduce, False))

        pk_dict['A'] = r1.get()
        pk_dict['B1'] = r2.get()
        pk_dict['B2'] = r3.get()
        pk_dict['C'] = r4.get()
        pk_dict['hExps'] = r5.get()

        worker.terminate()

        if not b_reduce:
          pk_dict['vk_alfa_1'] = ECC.from_uint256(np.reshape(pk['alfa_1'],(-1,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, remove_last=True)[0].extend().as_str()
          pk_dict['vk_beta_1'] = ECC.from_uint256(np.reshape(pk['beta_1'],(-1,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, remove_last=True)[0].extend().as_str()
          pk_dict['vk_delta_1'] = ECC.from_uint256(np.reshape(pk['delta_1'],(-1,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, remove_last=True)[0].extend().as_str()
          pk_dict['vk_beta_2'] = ECC.from_uint256(np.reshape(pk['beta_2'],(-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, ec2=True, remove_last=True)[0].extend().as_str()
          pk_dict['vk_delta_2'] = ECC.from_uint256(np.reshape(pk['delta_2'],(-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, ec2=True, remove_last=True)[0].extend().as_str()

        else:
          pk_dict['vk_alfa_1'] = ECC.from_uint256(np.reshape(pk['alfa_1'],(-1,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, remove_last=True)[0].as_str()
          pk_dict['vk_beta_1'] = ECC.from_uint256(np.reshape(pk['beta_1'],(-1,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, remove_last=True)[0].as_str()
          pk_dict['vk_delta_1'] = ECC.from_uint256(np.reshape(pk['delta_1'],(-1,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, remove_last=True)[0].as_str()
          pk_dict['vk_beta_2'] = ECC.from_uint256(np.reshape(pk['beta_2'],(-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, ec2=True, remove_last=True)[0].as_str()
          pk_dict['vk_delta_2'] = ECC.from_uint256(np.reshape(pk['delta_2'],(-1,2,NWORDS_256BIT)), in_ectype=EC_T_AFFINE, out_ectype=out_ec, reduced=True, ec2=True, remove_last=True)[0].as_str()


        return pk_dict

def pkbin_get(pk_bin, labels):
    #Header + field p + group g + X_nWords
    offset_data = PKBIN_H_N_OFFSET + 2 * NWORDS_256BIT + 8
    polsA_nWords = pk_bin[PKBIN_H_N_OFFSET + 2 * NWORDS_256BIT]
    polsB_nWords = pk_bin[PKBIN_H_N_OFFSET + 2 * NWORDS_256BIT + 1]
    polsC_nWords = pk_bin[PKBIN_H_N_OFFSET + 2 * NWORDS_256BIT + 2]
    polsH_nWords = (pk_bin[PKBIN_H_DOMAINSIZE_OFFSET]*2-1)*NWORDS_256BIT 
    A_nWords = pk_bin[PKBIN_H_N_OFFSET + 2 * NWORDS_256BIT + 3]
    B1_nWords = pk_bin[PKBIN_H_N_OFFSET + 2 * NWORDS_256BIT + 4]
    B2_nWords = pk_bin[PKBIN_H_N_OFFSET + 2 * NWORDS_256BIT + 5]
    C_nWords = pk_bin[PKBIN_H_N_OFFSET + 2 * NWORDS_256BIT + 6]
    hExps_nWords = pk_bin[PKBIN_H_N_OFFSET + 2 * NWORDS_256BIT + 7]

    offset_ec_data = offset_data + polsA_nWords + polsB_nWords + polsC_nWords
    
    ret_val = []
    for label in labels:
       if label=='nVars':
         ret_val.append(pk_bin[PKBIN_H_NVARS_OFFSET:PKBIN_H_NVARS_OFFSET+1])
       elif label== 'nPublic':
         ret_val.append(pk_bin[PKBIN_H_NPUBLIC_OFFSET:PKBIN_H_NPUBLIC_OFFSET+1])
       elif label== 'domainSize':
         ret_val.append(pk_bin[PKBIN_H_DOMAINSIZE_OFFSET:PKBIN_H_DOMAINSIZE_OFFSET+1])
       elif label== 'polsA':
         polsA_offset = offset_data
         ret_val.append(pk_bin[polsA_offset:polsA_offset+polsA_nWords])
       elif label== 'polsH':
         polsH_offset = offset_data
         ret_val.append(pk_bin[polsH_offset:polsH_offset+polsH_nWords])
       elif label== 'polsB':
         polsB_offset = offset_data + polsA_nWords
         ret_val.append(pk_bin[polsB_offset:polsB_offset+polsB_nWords])
       elif label== 'polsC':
         polsC_offset = offset_data + polsA_nWords + polsB_nWords
         ret_val.append(pk_bin[polsC_offset:polsC_offset+polsC_nWords])
       elif label== 'alfa_1':
         alfa_1_offset = offset_ec_data
         ret_val.append(pk_bin[alfa_1_offset:alfa_1_offset+2*NWORDS_256BIT])
       elif label== 'beta_1':
         beta_1_offset = offset_ec_data + 2*NWORDS_256BIT
         ret_val.append(pk_bin[beta_1_offset:beta_1_offset+2*NWORDS_256BIT])
       elif label== 'delta_1':
         delta_1_offset = offset_ec_data + 4*NWORDS_256BIT
         ret_val.append(pk_bin[delta_1_offset:delta_1_offset+2*NWORDS_256BIT])
       elif label== 'beta_2':
         beta_2_offset = offset_ec_data + 6*NWORDS_256BIT
         ret_val.append(pk_bin[beta_2_offset:beta_2_offset+4*NWORDS_256BIT])
       elif label== 'delta_2':
         delta_2_offset = offset_ec_data + 10*NWORDS_256BIT
         ret_val.append(pk_bin[delta_2_offset:delta_2_offset+4*NWORDS_256BIT])
       elif label == 'A':
         A_offset = offset_ec_data + 14*NWORDS_256BIT
         ret_val.append(pk_bin[A_offset:A_offset+A_nWords])
       elif label == 'B1':
         B1_offset = offset_ec_data + 14*NWORDS_256BIT + A_nWords
         ret_val.append(pk_bin[B1_offset:B1_offset+B1_nWords])
       elif label == 'B2':
         B2_offset = offset_ec_data + 14*NWORDS_256BIT + A_nWords + \
                     B1_nWords
         ret_val.append(pk_bin[B2_offset:B2_offset+B2_nWords])
       elif label == 'C':
         C_offset = offset_ec_data + 14*NWORDS_256BIT + A_nWords + \
                     B1_nWords + B2_nWords
         ret_val.append(pk_bin[C_offset:C_offset+C_nWords])
       elif label == 'hExps':
         hExps_offset = offset_ec_data + 14*NWORDS_256BIT + A_nWords + \
                     B1_nWords + B2_nWords + C_nWords
         ret_val.append(pk_bin[hExps_offset:hExps_offset+hExps_nWords])
       else: 
         sys.exit(1)

    return ret_val
    
def pkvars_to_bin(out_bin, out_ec, pk, ext=False): 
        pk_bin = np.concatenate( (
                   np.asarray([pk['nWords']], dtype=np.uint32),
                   np.asarray([SNARKSFILE_T_PK], dtype=np.uint32),
                   np.asarray([pk['protocol']], dtype=np.uint32),
                   np.asarray([pk['Rbitlen']], dtype=np.uint32),
                   np.asarray([out_bin], dtype=np.uint32),
                   np.asarray([pk['k_ecformat']], dtype=np.uint32),
                   np.asarray([pk['nVars']], dtype=np.uint32),
                   np.asarray([pk['nPublic']], dtype=np.uint32),
                   np.asarray([pk['domainBits']], dtype=np.uint32),
                   np.asarray([pk['domainSize']], dtype=np.uint32),
                   np.asarray(pk['field_r'], dtype=np.uint32),
                   np.asarray(pk['group_q'], dtype=np.uint32)) )

        polsA_extnWords = pk['polsA_nWords']
        polsB_extnWords = pk['polsB_nWords']
        polsC_extnWords = pk['polsC_nWords']

        A_extnWords = pk['A_nWords']
        B1_extnWords = pk['B1_nWords']
        B2_extnWords = pk['B2_nWords']
        C_extnWords = pk['C_nWords']
        hExps_extnWords = pk['hExps_nWords']

        A_ext = np.zeros(0,dtype=np.uint32)
        B1_ext = np.zeros(0,dtype=np.uint32)
        B2_ext = np.zeros(0,dtype=np.uint32)
        C_ext = np.zeros(0,dtype=np.uint32)
        hExps_ext = np.zeros(0,dtype=np.uint32)

        if ext:
          polsA_extnWords = max(pk['polsA_nWords'], pk['domainSize']*NWORDS_256BIT)
          polsB_extnWords = max(pk['polsB_nWords'], pk['domainSize']*NWORDS_256BIT)
          polsC_extnWords = max(pk['polsC_nWords'], pk['domainSize']*NWORDS_256BIT)

          A_extnWords = pk['A_nWords'] + 4 * NWORDS_256BIT  # A + alfa_1 + delta_1
          B1_extnWords = pk['B1_nWords'] + 4 * NWORDS_256BIT  # B1 + beta_1 + delta_1
          B2_extnWords = pk['B2_nWords'] + 8 * NWORDS_256BIT  # B2 + beta_2 + delta_2
          C_extnWords = pk['C_nWords'] 
          hExps_extnWords = pk['hExps_nWords']  + 12 * NWORDS_256BIT # hExps + pi_c + pi_a + pib1 + delta_1 + pi_b
         
          if out_bin == FMT_EXT:
            A_ext = np.concatenate(
                      np.reshape(from_montgomeryN_h(pk['alfa_1'], MOD_GROUP,1),-1),
                      np.reshape(from_montgomeryN_h(pk['delta_1'], MOD_GROUP,1),-1)
                                  )
            B1_ext = np.concatenate(
                      np.reshape(from_montgomeryN_h(pk['beta_1'], MOD_GROUP,1),-1),
                      np.reshape(from_montgomeryN_h(pk['delta_1'], MOD_GROUP,1),-1)
                                  )
            B2_ext = np.concatenate(
                      np.reshape(from_montgomeryN_h(pk['beta_2'], MOD_GROUP,1),-1),
                      np.reshape(from_montgomeryN_h(pk['delta_2'], MOD_GROUP,1),-1)
                                   )
            C_ext = np.zeros(0,dtype=np.uint32)
            hExps_ext = np.zeros(pk['hExps_nWords'],dtype=np.uint32)
            hExps_ext[6*NWORDS_256BIT:8*NWORDS_256BIT] = np.reshape(from_montgomeryN_h(pk['delta_1'], MOD_GROUP,1),-1)

          else:
            A_ext = np.concatenate( (pk['alfa_1'], pk['delta_1']) )
            B1_ext = np.concatenate( (pk['beta_1'], pk['delta_1']) )
            B2_ext = np.concatenate( (pk['beta_2'], pk['delta_2']) )
            C_ext = np.zeros(0,dtype=np.uint32)
            hExps_ext = np.zeros(pk['hExps_nWords'],dtype=np.uint32)
            hExps_ext[6*NWORDS_256BIT:8*NWORDS_256BIT] = pk['delta_1']


        polsA_ext = np.zeros(polsA_extnWords - pk['polsA_nWords'],dtype=np.uint32)
        polsB_ext = np.zeros(polsB_extnWords - pk['polsB_nWords'],dtype=np.uint32)
        polsC_ext = np.zeros(polsC_extnWords - pk['polsC_nWords'],dtype=np.uint32)


        pk_bin = np.concatenate( (
                      pk_bin,
                      np.asarray([polsA_extnWords],dtype=np.uint32),
                      np.asarray([polsB_extnWords],dtype=np.uint32),
                      np.asarray([polsC_extnWords],dtype=np.uint32),
                      np.asarray([A_extnWords],dtype=np.uint32),
                      np.asarray([B1_extnWords],dtype=np.uint32),
                      np.asarray([B2_extnWords],dtype=np.uint32),
                      np.asarray([C_extnWords],dtype=np.uint32),
                      np.asarray([hExps_extnWords],dtype=np.uint32)) )

        if out_bin == FMT_EXT:
           mpoly_from_montgomery_h(pk['polsA'], MOD_FIELD)
           mpoly_from_montgomery_h(pk['polsB'], MOD_FIELD)
           mpoly_from_montgomery_h(pk['polsC'], MOD_FIELD)
           pk_bin = np.concatenate( (
                      pk_bin,
                      pk['polsA'],
                      polsA_ext,
                      pk['polsB'],
                      polsB_ext,
                      pk['polsC'], 
                      polsC_ext,
                      np.reshape(from_montgomeryN_h(pk['alfa_1'], MOD_GROUP,1),-1),
                      np.reshape(from_montgomeryN_h(pk['beta_1'], MOD_GROUP,1),-1),
                      np.reshape(from_montgomeryN_h(pk['delta_1'], MOD_GROUP,1),-1),
                      np.reshape(from_montgomeryN_h(pk['beta_2'], MOD_GROUP,1),-1),
                      np.reshape(from_montgomeryN_h(pk['delta_2'], MOD_GROUP,1),-1),
                      np.reshape(from_montgomeryN_h(pk['A'], MOD_GROUP,1),-1),
                      A_ext,
                      np.reshape(from_montgomeryN_h(np.reshape(pk['B1'],-1), MOD_GROUP,1),-1),
                      B1_ext,
                      np.reshape(from_montgomeryN_h(np.reshape(pk['B2'],-1), MOD_GROUP,2),-1),
                      B2_ext,
                      np.reshape(from_montgomeryN_h(np.reshape(pk['C'],-1), MOD_GROUP,1),-1),
                      C_ext,
                      np.reshape(from_montgomeryN_h(np.reshape(pk['hExps'],-1), MOD_GROUP,1),-1),
                      hExps_ext))
           mpoly_to_montgomery_h(pk['polsA'], MOD_FIELD)
           mpoly_to_montgomery_h(pk['polsB'], MOD_FIELD)
           mpoly_to_montgomery_h(pk['polsC'], MOD_FIELD)
        else:
           pk_bin = np.concatenate((
                  pk_bin,
                  pk['polsA'],
                  polsA_ext,
                  pk['polsB'],
                  polsB_ext,
                  pk['polsC'],
                  polsC_ext,
                  pk['alfa_1'],    
                  pk['beta_1'],
                  pk['delta_1'],
                  pk['beta_2'],
                  pk['delta_2'],
                  np.reshape(pk['A'],-1),
                  A_ext,
                  np.reshape(pk['B1'],-1),
                  B1_ext,
                  np.reshape(pk['B2'],-1),
                  B2_ext,
                  np.reshape(pk['C'],-1),
                  C_ext,
                  np.reshape(pk['hExps'],-1), 
                  hExps_ext) )
       
        pk_bin[0] = pk_bin.shape[0]
        return pk_bin

def pkjson_to_pyec(inv, ec2):
  if ec2:
    P = [ECC_F2(p) for p in inv]
  else:
    P = [ECC_F1(p) for p in inv]

  return P

def pkjson_to_pyspol(inp):
   P = [ZPolySparse(el) if el is not {} else ZPolySparse({'0':0}) for el in inp]

   return P

def pkjson_to_pyvars(pk_proof):
        # Init witness to Field El.
        # TODO :  I am assuming that all field el are FielElExt (witness_scl, polsA_sps, polsB_sps, polsC_sps, alfa1...
        # Witness is initialized a BitInt as it will operate on different fields
        #self.witness_scl = [BigInt(el) for el in self.witness_scl]

        pk = getPK()
        pk['alfa_1'] = ECC_F1(p=pk_proof['vk_alfa_1'])
        pk['beta_1'] = ECC_F1(p=pk_proof['vk_beta_1'])
        pk['delta_1'] = ECC_F1(p=pk_proof['vk_delta_1'])

        beta2 = [Z2FieldEl(el) for el in pk_proof['vk_beta_2']]
        pk['beta_2'] = ECC_F2(beta2)
        delta2 = [Z2FieldEl(el) for el in pk_proof['vk_delta_2']]
        pk['delta_2'] = ECC_F2(delta2)

        worker = mp.Pool(processes=min(5,mp.cpu_count()-1))

        r1     = worker.apply_async(pkjson_to_pyec, args=(pk_proof['A'], False))
        r2     = worker.apply_async(pkjson_to_pyec, args=(pk_proof['B1'], False))
        r3     = worker.apply_async(pkjson_to_pyec, args=(pk_proof['B2'], True))
        r4     = worker.apply_async(pkjson_to_pyec, args=(pk_proof['C'], False))
        r5     = worker.apply_async(pkjson_to_pyec, args=(pk_proof['hExps'], False))

        pk['A']     = r1.get()
        pk['B1']    = r2.get()
        pk['B2']    = r3.get()
        pk['C']     = r4.get()
        pk['hExps'] = r5.get()

        worker.terminate()

        ZField.set_field(MOD_FIELD)

        # TODO : This representation may not be optimum. I only have good representation of sparse polynomial,
        #  but not of array of sparse poly (it is also sparse). I should encode it as a dictionary as wekk

        pk['polsA'] = pkjson_to_pyspol(pk_proof['polsA'])
        pk['polsB'] = pkjson_to_pyspol(pk_proof['polsB'])
        pk['polsC'] = pkjson_to_pyspol(pk_proof['polsC'])

        return pk


def pkpyec_to_vars(ecp, remove_last, as_reduced):
     ZField.set_field(MOD_GROUP)
     P = ECC.as_uint256(ecp, remove_last, as_reduced)

     return P

def pkpyspol_to_vars(spolp):
       ZField.set_field(MOD_FIELD)
       pols_l = []
       pols_p = []
       for pol in spolp:
         l,p = pol.reduce().as_uint256() 
         pols_l.append(l)
         pols_p.append(p)
       P = np.asarray(np.concatenate((np.asarray([len(pols_l)]),
                                    #np.concatenate((np.cumsum(pols_l),np.concatenate(pols_p))))),dtype=np.uint32)
                                    np.concatenate((pols_l,np.concatenate(pols_p))))),dtype=np.uint32)

       return P


def pkjson_to_vars(pk_proof, proving_key_f):
       pk = pkjson_to_pyvars(pk_proof)

       ZField.set_field(MOD_GROUP)
       pk['alfa_1'] = np.reshape(pkpyec_to_vars(pk['alfa_1'],True, True),-1)
       pk['beta_1'] = np.reshape(pkpyec_to_vars(pk['beta_1'],True, True),-1)
       pk['delta_1'] = np.reshape(pkpyec_to_vars(pk['delta_1'],True, True),-1)
       pk['beta_2'] = np.reshape(pkpyec_to_vars(pk['beta_2'],True, True),-1)
       pk['delta_2'] = np.reshape(pkpyec_to_vars(pk['delta_2'],True, True),-1)

       worker = mp.Pool(processes=min(5,mp.cpu_count()-1))

       r1 = worker.apply_async(pkpyec_to_vars, args=(pk['A'],True, True))
       r2 = worker.apply_async(pkpyec_to_vars, args=(pk['B1'],True, True))
       r3 = worker.apply_async(pkpyec_to_vars, args=(pk['B2'],True, True))
       r4 = worker.apply_async(pkpyec_to_vars, args=(pk['C'],True, True))
       r5 = worker.apply_async(pkpyec_to_vars, args=(pk['hExps'],True, True))
       
       pk['A'] = np.reshape(r1.get(),-1)
       pk['B1'] = np.reshape(r2.get(),-1)
       pk['B2'] = np.reshape(r3.get(),-1)
       pk['C']  = np.reshape(r4.get(),-1)
       pk['hExps'] = np.reshape(r5.get(),-1)

       worker.terminate()

       ZField.set_field(MOD_FIELD)
       
       pk['polsA'] = pkpyspol_to_vars(pk['polsA'])
       pk['polsB'] = pkpyspol_to_vars(pk['polsB'])
       pk['polsC'] = pkpyspol_to_vars(pk['polsC'])

                    
       pk['ftype'] = np.uint32(SNARKSFILE_T_PK)
       pk['protocol'] = np.uint32(PROTOCOL_T_GROTH)
       pk['Rbitlen'] =  np.asarray(ZField.get_reduction_data()['Rbitlen'],dtype=np.uint32)
       if 'k_binformat' in pk_proof:
           if pk['k_binformat'] == "normal":
              pk['k_binformat'] == np.uint32(FMT_EXT)
           else:
              pk['k_binformat'] == np.uint32(FMT_MONT)
       else :
          pk['k_binformat'] == np.uint32(FMT_EXT)
                
       pk['k_ecformat'] = np.uint32(EC_T_AFFINE)
       pk['nVars'] = np.uint32(pk_proof['nVars'])
       pk['nPublic'] = np.uint32(pk_proof['nPublic'])
       pk['domainBits'] = np.uint32(pk_proof['domainBits'])
       pk['domainSize'] = np.uint32(pk_proof['domainSize'])
       pidx = ZField.get_field()
       ZField.set_field(MOD_GROUP)
       pk['group_q'] = ZField.get_extended_p().as_uint256()
       ZField.set_field(MOD_FIELD)
       pk['field_r']        = ZField.get_extended_p().as_uint256()
       ZField.set_field(pidx)

       pk['polsA_nWords'] = np.uint32(pk['polsA'].shape[0])
       pk['polsB_nWords'] = np.uint32(pk['polsB'].shape[0])
       pk['polsC_nWords'] = np.uint32(pk['polsC'].shape[0])
       pk['A_nWords'] =  np.uint32(pk['A'].shape[0] )
       pk['B1_nWords'] =  np.uint32(pk['B1'].shape[0] )
       pk['B2_nWords'] =  np.uint32(pk['B2'].shape[0] )
       pk['C_nWords'] =  np.uint32(pk['C'].shape[0] )
       pk['hExps_nWords'] =  np.uint32(pk['hExps'].shape[0] )
       pk['nWords'] =  np.uint32(PKBIN_H_N_OFFSET + 2*NWORDS_256BIT + 8 + \
                       pk['polsA_nWords'] + pk['polsB_nWords'] + \
                       pk['polsC_nWords'] + pk['A_nWords'] + \
                       pk['B1_nWords'] + pk['B2_nWords'] + \
                       pk['C_nWords'] + pk['hExps_nWords'] + \
                       3 * 2 * NWORDS_256BIT + 2 * 4 * NWORDS_256BIT)

       return pk

def pkbin_to_vars(pk_bin):
          pk = getPK()

          pk['nWords'] = pk_bin[PKBIN_H_NWORDS_OFFSET]
          pk['ftype'] = pk_bin[PKBIN_H_FTYPE_OFFSET]
          pk['protocol'] = pk_bin[PKBIN_H_PROTOCOL_OFFSET]
          pk['Rbitlen']  = pk_bin[PKBIN_H_RBITLEN_OFFSET]
          pk['k_binformat'] = pk_bin[PKBIN_H_BINFORMAT_OFFSET]
          pk['k_ecformat'] =  pk_bin[PKBIN_H_ECFORMAT_OFFSET]
          pk['nVars'] =  pk_bin[PKBIN_H_NVARS_OFFSET]
          pk['nPublic'] = pk_bin[PKBIN_H_NPUBLIC_OFFSET]
          pk['domainBits'] = pk_bin[PKBIN_H_DOMAINBITS_OFFSET]
          pk['domainSize'] = pk_bin[PKBIN_H_DOMAINSIZE_OFFSET]
          offset_data = PKBIN_H_N_OFFSET
          pk['field_r'] = pk_bin[offset_data:offset_data+NWORDS_256BIT]
          offset_data += NWORDS_256BIT
          pk['group_q'] = pk_bin[offset_data:offset_data+NWORDS_256BIT]
          offset_data += NWORDS_256BIT

          pk['polsA_nWords'] = pk_bin[offset_data]
          offset_data += 1
          pk['polsB_nWords'] = pk_bin[offset_data]
          offset_data += 1
          pk['polsC_nWords'] = pk_bin[offset_data]
          offset_data += 1
          pk['A_nWords'] =  pk_bin[offset_data]
          offset_data += 1
          pk['B1_nWords'] = pk_bin[offset_data]
          offset_data += 1
          pk['B2_nWords'] = pk_bin[offset_data]
          offset_data += 1
          pk['C_nWords']  = pk_bin[offset_data]
          offset_data += 1
          pk['hExps_nWords'] = pk_bin[offset_data]
          offset_data += 1


          offset_ec_data = offset_data + \
                           pk['polsA_nWords'] + \
                           pk['polsB_nWords'] + \
                           pk['polsC_nWords'] 
                          
          if pk['k_binformat'] == FMT_EXT:
             to_montgomeryN_h(pk_bin[offset_data:offset_ec_data], MOD_FIELD)

          pk['polsA'] = pk_bin[offset_data:offset_data+pk['polsA_nWords']]
          offset_data += pk['polsA_nWords']
          pk['polsB'] = pk_bin[offset_data:offset_data+pk['polsB_nWords']]
          offset_data += pk['polsB_nWords']
          pk['polsC'] =  pk_bin[offset_data:offset_data+pk['polsC_nWords']]
          offset_data += pk['polsC_nWords']
         
          if pk['k_binformat'] == FMT_EXT:
             to_montgomeryN_h(pk_bin[offset_ec_data:], MOD_GROUP)
            
          pk['alfa_1'] = pk_bin[offset_data:offset_data+2*NWORDS_256BIT]
          offset_data += 2*NWORDS_256BIT
          pk['beta_1'] = pk_bin[offset_data:offset_data+2*NWORDS_256BIT]
          offset_data += 2*NWORDS_256BIT
          pk['delta_1']= pk_bin[offset_data:offset_data+2*NWORDS_256BIT]
          offset_data += 2*NWORDS_256BIT
          pk['beta_2'] = pk_bin[offset_data:offset_data+4*NWORDS_256BIT]
          offset_data += 4*NWORDS_256BIT
          pk['delta_2'] = pk_bin[offset_data:offset_data+4*NWORDS_256BIT]
          offset_data += 4*NWORDS_256BIT


          pk['A'] =  pk_bin[offset_data:offset_data+pk['A_nWords']]
          offset_data += pk['A_nWords']
          pk['B1'] = pk_bin[offset_data:offset_data+pk['B1_nWords']]
          offset_data += pk['B1_nWords']
          pk['B2'] =  pk_bin[offset_data:offset_data+pk['B2_nWords']]
          offset_data +=  pk['B2_nWords']
          pk['C']  = pk_bin[offset_data:offset_data+pk['C_nWords']]
          offset_data +=  pk['C_nWords']
          pk['hExps'] = pk_bin[offset_data:offset_data+pk['hExps_nWords']]
          offset_data +=  pk['hExps_nWords']

          return pk

def gen_reponame(repo_f,sufix=None):
    now = datetime.now()
    #reponame = repo_f + now.strftime("%Y%m%d__%H%M%S")
    reponame = repo_f 
    if sufix is not None:
       reponame = reponame + sufix

    if not os.path.exists(reponame):
        os.makedirs(reponame)
    return reponame

def copy_input_files(file_list, dest):
    for f in file_list:
        if f is not None:
           os.system('cp -f ' + f + ' '  +dest)

