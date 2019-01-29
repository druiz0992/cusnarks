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
// File name  : groth_protocol
//
// Date       : 27/01/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Zero Kowledge Groth protocol implementation
# 


// Description:
//    
//   TODO
//    
// ------------------------------------------------------------------

"""
import json,ast
import os.path
import numpy as np

from zutils import ZUtils
from random import randint
from zfield import ZField, ZFieldEl, ZFieldElExt, ZFieldElRedc
from ecc import ECC, ECCAffine, ECCProjective, ECCJacobian, ECC_F1
from eccf2 import ECC_F2
from zpoly import ZPoly, ZPolySparse

DEFAULT_WITNESS_LOC = "../../data/witness.json"
DEFAULT_PROVING_KEY_LOC = "../../data/proving_key.json"
DEFAULT_PROOF_LOC =  "../../data/proof.json"
DEFAULT_PUBLIC_LOC =  "../../data/public.json"

class GrothSnarks(object):

    def __init__(self, curve='BN128', witness_f=DEFAULT_WITNESS_LOC, proving_key_f = DEFAULT_PROVING_KEY_LOC, proof_f = DEFAULT_PROOF_LOC, public_f = DEFAULT_PUBLIC_LOC, in_point_fmt=ZUtils.FEXT, work_point_fmt=ZUtils.FEXT, in_coor_fmt = ZUtils.JACOBIAN, work_coor_fmt= ZUtils.JACOBIAN):

        ## Open and parse witness data
        if os.path.isfile(witness_f):
            f = open(witness_f,'r')
            self.witness_scl = [long(c) for c in ast.literal_eval(json.dumps(json.load(f)))]
            f.close()

        ## Open and parse proving key data
        if os.path.isfile(proving_key_f):
            f = open(proving_key_f,'r')
            tmp_data = json.load(f)
            self.vk_proof = GrothSnarks.json_to_dict(tmp_data)
            f.close()

        self.proof_f = proof_f
        self.public_f = public_f
        self.curve_data = ZUtils.CURVE_DATA[curve]
        ZField(self.curve_data['prime'], self.curve_data['factor_data'])
        ECCAffine(curve=self.curve_data['curve_params'])
        ZUtils.set_default_in_p_format(in_point_fmt)  # FEXT
        ZUtils.set_default_in_rep_format(in_coor_fmt) # AFFINE/PROJECTIVE/JACOBIAN
        self.prime = ZField.get_extended_p()

        #init class variables
        self.init_vars()

        """
          TODO : pending capability to switch from FEXt to REDC and from/to affine/projective/affine
            right now, everything is ext and jacobian. Also witness_scl needs to be xformed

        ZUtils.set_default_in_p_format(work_point_fmt)  # FEXT
        ZUtils.set_default_in_rep_format(work_coor_fmt) # AFFINE/PROJECTIVE/JACOBIAN

        # xfrom vars Representation format, point format
        coor_cb, point_cb = self.get_cb(in_coor_fmt, work_coor_fmt,in_point_fmt, work_point_fmt)  
        self.xform_vars(coor_cb, point_cb)
        """

    """
    def get_cb(self, in_coor_fmt, work_coor_fmt, in_point_fmt, work_point_fmt):
        coor_cb, point_cb = None
        if in_coor_fmt != work_point_fmt:
            if work_coor_fmt == ZUtils.AFFINE:
                coor_cb = to_affine
            elif work_coor_fmt == ZUtils.JACOBIAN:
                coor_cb = to_jacobian
            else:
                coor_cb = to_projective

        if in_point_fmt != work_point_fmt:
            if work_point_fmt == ZUtils.FEX:
                point_cb = extend
            else:
                point_cb = reduce

        return coor_cb, point_cb

 
    def xform_vars(self, coord_cb, point_cb):
        self.pi_a_eccf1 = self.pi_a_eccf1.coord_cb()
        self.pi_b_eccf2 = ECC_F2()
        self.pi_c_eccf1 = ECC_F1()
  
        self.polsA = ZPolySparse(self.vk_proof['polsA'])
        self.polsB = ZPolySparse(self.vk_proof['polsB'])
        self.polsC = ZPolySparse(self.vk_proof['polsC'])

        self.vk_alfa_1_eccf1 = ECC_F1(p=self.vk_proof['vk_alfa_1'])
        self.vk_beta_1_eccf1 = ECC_F1(p=self.vk_proof['vk_beta_1'])
        self.vk_delta_1_eccf1 = ECC_F1(p=self.vk_proof['vk_delta_1'])

        self.vk_beta_2_eccf2 = ECC_F2(p1=beta2[0], p2=beta[1])
        self.vk_delta_2_eccf2 = ECC_F2(p1=delta2[0], p2=delta2[1])

        
        self.A_eccf1 = 
        self.B1_eccf1 = 
        self.B2_eccf2 = 
        self.C_eccf1 = []
        self.hExps_eccf1 = []

        B2 = np.asarray(self.vk_prook['B2'])
        for s in xrange(self.vk_proof['nVars']):
            self.A_eccf1.append(ECC_F1(p=self.vk_proof['A'][s]))
            self.B1_eccf1.append(ECC_F1(p=self.vk_proof['B1'][s]))
            #TODO -> Fix B2
            B2T = np.transpose(B2[s]).tolist()
            self.B2_eccf2.append(ECC_F2(p1=B2T[0], p2=B2T[1]))

         for s in xrange(self.vk_proof.nPublic+1,self.vk_proof['nVars']):
             #TODO : Check values of C
             self.C_eccf1.append(ECC_F1(p=self.vk_proof['C'][s]))
    
         for s in xrange(len(self.vk_proof['hExps'])):
             self.hExps_eccf1.append(ECC_F1(p=self.vk_proof['hExps'][s]))
    """
    def init_vars(self):
        # Init witness to Field El.
        # TODO :  I am assuming that all field el are FielElExt (witness_scl, polsA_sps, polsB_sps, polsC_sps, alfa1...
        self.witness_scl = [ZFieldElExt(el) for el in self.witness_scl]
     
        # Init pi's  
        self.pi_a_eccf1 = ECC_F1()
        self.pi_b_eccf2 = ECC_F2()
        self.pi_c_eccf1 = ECC_F1()

        # TODO : This representation may not be optimum. I only have good representation of sparse polynomial,
        #  but not of array of sparse poly (it is also sparse). I should encode it as a dictionary as wekk
        self.polsA_sps = [ZPolySparse(el) if el is not {} else ZPolySparse({'0':0}) for el in self.vk_proof['polsA']]
        self.polsB_sps = [ZPolySparse(el) if el is not {} else ZPolySparse({'0':0}) for el in self.vk_proof['polsB']]
        self.polsC_sps = [ZPolySparse(el) if el is not {} else ZPolySparse({'0':0}) for el in self.vk_proof['polsC']]

        self.vk_alfa_1_eccf1 = ECC_F1(p=self.vk_proof['vk_alfa_1'])
        self.vk_beta_1_eccf1 = ECC_F1(p=self.vk_proof['vk_beta_1'])
        self.vk_delta_1_eccf1 = ECC_F1(p=self.vk_proof['vk_delta_1'])

        #TODO -> check beta2
        beta2 = np.transpose(self.vk_proof['vk_beta_2']).tolist()
        self.vk_beta_2_eccf2 = ECC_F2(p1=beta2[0], p2=beta2[1])
        #TODO -> check delta 2
        delta2 = np.transpose(self.vk_proof['vk_delta_2']).tolist()
        self.vk_delta_2_eccf2 = ECC_F2(p1=delta2[0], p2=delta2[1])

        
        self.A_eccf1 = []
        self.B1_eccf1 = []
        self.B2_eccf2 = []
        self.C_eccf1 = []
        self.hExps_eccf1 = []
        self.public_signals = None

        B2 = np.asarray(self.vk_proof['B2'])
        for s in xrange(self.vk_proof['nVars']):
            self.A_eccf1.append(ECC_F1(p=self.vk_proof['A'][s]))
            self.B1_eccf1.append(ECC_F1(p=self.vk_proof['B1'][s]))
            #TODO -> Fix B2
            B2T = np.transpose(B2[s]).tolist()
            self.B2_eccf2.append(ECC_F2(p1=B2T[0], p2=B2T[1]))

        for s in xrange(self.vk_proof['nPublic']+1,self.vk_proof['nVars']):
             #TODO : Check values of C
             self.C_eccf1.append(ECC_F1(p=self.vk_proof['C'][s]))
    
        for s in xrange(len(self.vk_proof['hExps'])):
             self.hExps_eccf1.append(ECC_F1(p=self.vk_proof['hExps'][s]))


    @classmethod
    def json_to_dict(cls, data):
       data = {str(k) : data[k] for k in data.keys()}
       for k, v in data.items():
           if type(data[k]) is list:
               GrothSnarks.json_to_list(data[k])
           elif type(data[k]) is dict:
               GrothSnarks.json_to_dict(data[k])
           elif GrothSnarks.is_long(v):
               data[k] = long(v)

       return data

    @classmethod
    def json_to_list(cls,data):
        for idx,el in enumerate(data):
            if type(el) is list:
                GrothSnarks.json_to_list(el)
            elif type(el) is dict:
                data[idx] = GrothSnarks.json_to_dict(el)
            elif type(el) is unicode or type(el) is str:
                data[idx] = long(el)
        return

    @classmethod
    def is_long(cls, input):
        try:
            num = long(input)
        except ValueError:
            return False
        return True

    def gen_proof(self):
        """
          public_signals, pi_a_eccf1, pi_b_eccf2, pi_c_eccf1 
        """
        # Init r and s scalars
        #TODO . BigInt mod polFprime
        r_scl = ZFieldElExt(randint(1,self.prime.as_long()-1))
        s_scl = ZFieldElExt(randint(1,self.prime.as_long()-1))

        #TODO
        """"
        if self.in_point_fmt == FRDC:
            r_scl = r_scl.reduce()
            s_scl = s_scl.reduce()
        """

        # init Pi B1 ECC point
        pib1_eccf1 = ECC_F1()

        # Accumulate multiplication of S EC points and S scalar. Parallelization
        # can be accomplised by each thread performing a multiplication and storing
        # values in same input EC point
        # Second step is to add all ECC points
        for s in xrange(self.vk_proof['nVars']):
            # pi_a = pi_a + A[s] * witness[s];
            self.pi_a_eccf1 = self.pi_a_eccf1 + (self.A_eccf1[s] * self.witness_scl[s]) 

            #pi_b = pi_b + B[s] * witness[s];
            #self.pi_b_eccf2 = self.pi_b_eccf2 + (self.B2_eccf2[s] * self.witness_scl[s])
            k1 = (self.B2_eccf2[s] * self.witness_scl[s])
            self.pi_b_eccf2 = self.pi_b_eccf2 + k1
            print s
            print self.B2_eccf2[s].ecc1.as_list()
            print self.B2_eccf2[s].ecc2.as_list()
            print k1.ecc1.as_list()
            print k1.ecc2.as_list()
            print self.pi_b_eccf2.ecc1.as_list()
            print self.pi_b_eccf2.ecc2.as_list()

            pib1_eccf1 = pib1_eccf1 + (self.B1_eccf1[s] * self.witness_scl[s]) 

        # pi_a = pi_a + alfa1 + delta1 * r
        self.pi_a_eccf1  = self.pi_a_eccf1 + self.vk_alfa_1_eccf1
        self.pi_a_eccf1  = self.pi_a_eccf1 + (self.vk_delta_1_eccf1 * r_scl)

        # pi_b = pi_b + beta2 + delta2 * s
        self.pi_b_eccf2  = self.pi_b_eccf2 + self.vk_beta_2_eccf2 
        self.pi_b_eccf2  = self.pi_b_eccf2 + (self.vk_delta_2_eccf2 * s_scl)

        # pib1 = pib1 + beta1 + delta1 * s
        pib1_eccf1 = pib1_eccf1 + self.vk_beta_1_eccf1
        pib1_eccf1 = pib1_eccf1 + (self.vk_delta_1_eccf1 * s_scl)

        # TODO
        d1 = d2 = d3 = 0
        polH = self.calculateH(d1, d2, d3)
        #const h = self.calculateH(vk_proof, witness, PolF.F.zero, PolF.F.zero, PolF.F.zero);
        coeffH = polH.get_coeff()

        # Accumulate products of S ecc points and S scalars (same as at the beginning)
        for i in range(len(coeffH)):
           self.pi_c_eccf1 = self.pi_c_eccf1 + (self.hExps_eccf1[i] * coeffH[i])

        
        # pi_c = pi_c  + pi_a * s + pib1 * r + delta1  * - (r * s)
        self.pi_c_eccf1  = self.pi_c_eccf1 + (self.pi_a_eccf1 * s_scl)
        self.pi_c_eccf1  = self.pi_c_eccf1 + (pib1_eccf1 * r_scl)
        ## TODO 
        #proof.pi_c  = G1.add( proof.pi_c, G1.mulScalar( vk_proof.vk_delta_1, PolF.F.affine(PolF.F.neg(PolF.F.mul(r,s) ))));
        self.pi_c_eccf1  = self.pi_c_eccf1 + (self.vk_delta_1_eccf1 * -(r_scl * s_scl))

        self.public_signals = self.witness_scl[1:self.vk_proof['nPublic']+1]


    def write_json(self):
        # Write public file
        ps = [str(el.as_long()) for el in self.public_signals]
        j = json.dumps(ps, indent=4).encode('utf8')
        f = open(self.public_f, 'w')
        print >> f, j
        f.close()

        # write proof file
        pi_a = [str(el) for el in self.pi_a_eccf1.to_affine().as_list() + [1]]
        pi_c = [str(el) for el in self.pi_a_eccf1.to_affine().as_list() + [1]]
        pi_b = []
        pi_b.append([str(el) for el in self.pi_b_eccf2.ecc1.to_affine().as_list() + [1]])
        pi_b.append([str(el) for el in self.pi_b_eccf2.ecc2.to_affine().as_list() + [1]])
        pi_b[1]=pi_b[0]
        proof = {"pi_a" : pi_a, "pi_b" : np.transpose(pi_b).tolist(), "pi_c" : pi_c, "protocol" : "groth"}
        j = json.dumps(proof, indent=4).encode('utf8')
        f = open(self.proof_f, 'w')
        print >> f, j
        f.close()

    def calculateH(self, d1, d2, d3):
        #d1 = PolF.F.zero, d2 = PolF.F.zero, d3 = PolF.F.zero);

        m = self.vk_proof['domainSize']

        # Init dense poly of degree m-1 (all zero)
        polA_T = ZPoly([ZFieldElExt(0) for i in xrange(m)])
        polB_T = ZPoly([ZFieldElExt(0) for i in xrange(m)])
        polC_T = ZPoly([ZFieldElExt(0) for i in xrange(m)])

        # interpretation of polsA is that there are up to S sparse polynomials, and each sparse poly
        # has C sparse coeffs
        # polA_T = polA_T + witness[s] * polsA[s]
        #
        #  for (let s=0; s<vk_proof.nVars; s++) {
        #     for (let c in vk_proof.polsA[s]) {
        #           polA_T[c] = F.add(polA_T[c], F.mul(witness[s], vk_proof.polsA[s][c]));
        #     }
        #
        # Ex:
        # s iterates from 0 to 4
        # polsA = [{'1': 1}, {'2' : 1}, {'0': 3924283749832748327}, {}]
        # polsA[0] = {'1' : 1}, polsA[1] = {'2':1}, polsA[2] = {'0',3243243284}, polsA[3]={}
        # c in polsA[0] : '1', c in polsA[1] : '2', c in polsA[2] : '0', c in polsA[3] : {}
        # polA_T[1] = polA_t[1] + (witness[0] * polsA[0]['1'])
        # polA_T[2] = polA_t[2] + (witness[1] * polsA[1]['2'])
        # polA_T[0] = polA_t[0] + (witness[2] * polsA[2]['0'])
        

        for s in xrange(self.vk_proof['nVars']):
            polA_T = polA_T + self.witness_scl[s] * self.polsA_sps[s]
            polB_T = polB_T + self.witness_scl[s] * self.polsB_sps[s]
            polC_T = polC_T + self.witness_scl[s] * self.polsC_sps[s]

        ZField.find_roots(ZUtils.NROOTS, rformat_ext = True)

        # in : poly deg nVars-1.  out : poly deg nVars - 1
        polA_S = ZPoly(polA_T)
        polA_S.intt()
        # in : poly deg nVars-1.  out : poly deg nVars - 1
        polB_S = ZPoly(polB_T)
        polB_S.intt()

        polC_S = ZPoly(polC_T)
        polC_S.intt()

        # in : 2xpoly deg nVars-1.  out : poly deg nVars - 1
        polAB_S = ZPoly(polA_S)
        polAB_S.poly_mul_fft(ZPoly(polB_S))
        polAB_S2 = ZPoly(polA_S)
        polAB_S2.poly_mul_normal(ZPoly(polB_S))

        polABC_S = polAB_S - polC_S
    
        polZ_S = ZPoly([ZFieldElExt(-1)] + [ZFieldElExt(0) for i in range(m-1)] + [ZFieldElExt(1)])
    
        polH_S = polABC_S.poly_div(polZ_S)

        """
        H_S_copy = ZPoly(polH_S)
        H_S_copy.poly_mul(polZ_S)

        if H_S_copy == polABC_S:
            print "OK"
        else:
            print "KO"
        """

        # add coefficients of the polynomial (d2*A + d1*B - d3) + d1*d2*Z 
    
        polH_S = polH_S.expand_to_degree(m)
  
        for idx in xrange(m):
           d2A = d2 * polA_S.zcoeff[idx]
           d1B = d1 * polB_S.zcoeff[idx]
           polH_S.zcoeff[idx] += d2A + d1B

        polH_S.zcoeff[0] -= d3
    
        # Z = x^m -1
        d1d2 = d1 * d2
        polH_S.zcoeff[m] += d1d2
        polH_S.zcoeff[0] -= d1d2
    
        polH_S = polH_S.norm()
    
        return polH_S

if __name__ == "__main__":
    G = GrothSnarks()
    G.gen_proof()
    G.write_json()
