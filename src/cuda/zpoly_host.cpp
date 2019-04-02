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

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : zpoly_host.cpp
//
// Date       : 27/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//   Implementation of zpoly class
*/

#include <assert.h>
#include <iostream>
#include <stdio.h>

#include "types.h"
#include "rng.h"
#include "log.h"
#include "utils_host.h"
#include "zpoly_host.h"

/*
   pout [0] : N_poly
        [1] : M[0] = N coeff poly 0
        [2.. M[0]+1] : 
        [M[0]+2 .. (2M[0]+1)*NWORDS_256BIT] : Coeffs
*/
void maddm_h(uint32_t *pout, uint32_t *scalar, uint32_t *pin, uint32_t last_idx, uint32_t ncoeff, uint32_t pidx)
{
  uint32_t n_zpoly = pin[0];
  uint32_t zcoeff_d_offset = 1 + n_zpoly;
  uint32_t zcoeff_v_offset;
  uint32_t n_zcoeff;
  uint32_t scl[NWORDS_256BIT];
  uint32_t i,j;
  uint32_t *zcoeff_v_in, *zcoeff_v_out, zcoeff_d;

  for (i=0; i<last_idx; i++){
    to_montgomery_h(scl, &scalar[i*NWORDS_256BIT], pidx);
    n_zcoeff = pin[1+i];
    zcoeff_v_offset = zcoeff_d_offset + n_zcoeff;
    
    for (j=0; j< n_zcoeff; j++){
       zcoeff_d = pin[zcoeff_d_offset+j];
       zcoeff_v_in = &pin[zcoeff_v_offset+j*NWORDS_256BIT];
       zcoeff_v_out = &pout[zcoeff_d*NWORDS_256BIT];
       montmult_h(zcoeff_v_in, zcoeff_v_in, scl, pidx);
       addm_h(zcoeff_v_out, zcoeff_v_out, zcoeff_v_in, pidx);
    }
    zcoeff_d_offset = n_zcoeff*(NWORDS_256BIT+1) +1 + n_zpoly;
  }
}

#if 0
CUZPolySps::CUZPolySps(uint32_t n_coeff) 
{
   coeff_d = NULL;
   coeff_d = new uint32_t[n_coeff]; 
   coeff_v = new uint32_t[n_coeff * NWORDS_256BIT + 1];
   coeff_v[0] = n_coeff;
}

CUZPolySps::CUZPolySps(uint32_t n_coeff, uint32_t create_coeff) 
{
   coeff_d = NULL;
   if (create_coeff) {
      coeff_d = new uint32_t[n_coeff]; 
   } 
   coeff_v = new uint32_t[n_coeff * NWORDS_256BIT + 1];
   coeff_v[0] = n_coeff;
}

CUZPolySps::~CUZPolySps()
{
  if (coeff_d != NULL){
    delete[] coeff_d;
  }
  delete[] coeff_v;
}

uint32_t CUZPolySps::getNCoeff()
{
  return coeff_v[CUZPOLY_DEG_OFFSET];
}
uint32_t * CUZPolySps::getDCoeff()
{
  return coeff_d;
}
uint32_t * CUZPolySps::getVCoeff()
{
  return &coeff_v[CUZPOLY_COEFF_OFFSET];
}
uint32_t * CUZPolySps::getZPolySpsRep()
{
  return coeff_v;
}
void CUZPolySps::show()
{
   uint32_t ncoeff = getNCoeff();
   uint32_t *coeff_d = getDCoeff();
   uint32_t *coeff_v = getVCoeff();
   uint32_t i,j;

   printf("N Coeff: %d\n",ncoeff);
   for (i=0; i< ncoeff; i++){
      if (coeff_d != NULL){
         printf("C[%u] : ", coeff_d[i]);
      } else {
         printf("C[%u] : ", i);
      }
      for (j=0; j < NWORDS_256BIT; j++){
        printf("%u ", coeff_v[i*NWORDS_256BIT+j]);
      }
      printf("\n");
   }
}
void CUZPolySps::rand(uint32_t seed)
{
   _RNG *rng = _RNG::get_instance(seed);
   uint32_t i, n_coeff;
 
   n_coeff = getNCoeff();
   rng->randu256(getVCoeff(), n_coeff, NULL);
   rng->randu32(getDCoeff(),n_coeff);
}

void CUZPolySps::mulmScalar(uint32_t *scalar, uint32_t pidx, uint32_t convert=0)
{
  uint32_t i;
  uint32_t n_coeff = getNCoeff();
  uint32_t *val = getVCoeff();
  uint32_t scalar2[NWORDS_256BIT];

  if (convert){
     to_montgomery_h(scalar2, scalar, pidx);
  } else {
     memcpy(scalar2,scalar,sizeof(uint32_t)*NWORDS_256BIT);
  }
 
  for (i=0; i < n_coeff; i++){
     montmult_h(&val[i*NWORDS_256BIT],&val[i*NWORDS_256BIT],scalar2,pidx);
  }
}

void CUZPolySps::addm(CUZPolySps **p, uint32_t n_zpoly, uint32_t pidx)
{
  uint32_t n_coeff = getNCoeff();
  uint32_t i,j;
  uint32_t pn_coeff;
  uint32_t *p_coeff, *p_val;
 
  for (i=0; i < n_zpoly; i++){
    pn_coeff = p[i]->getNCoeff();
    p_coeff = p[i]->getDCoeff();
    p_val = p[i]->getVCoeff();
    if (p_coeff == NULL){
      for (j=0; j < pn_coeff; j++){
        addm_h(&coeff_v[1+j*NWORDS_256BIT], &coeff_v[1+j*NWORDS_256BIT],&p_val[j*NWORDS_256BIT],pidx);
      }
    } else {
      for (j=0; j < pn_coeff; j++){
        addm_h(&coeff_v[1+p_coeff[j]*NWORDS_256BIT], &coeff_v[1+p_coeff[j]*NWORDS_256BIT],&p_val[j*NWORDS_256BIT],pidx);
      }
    }
  }
}
void CUZPolySps::maddm(uint32_t *scalar, uint32_t *p, uint32_t last_idx, uint32_t pidx, uint32_t convert)
{
  uint32_t i,j;
  uint32_t scalar2[NWORDS_256BIT];
  uint32_t pn_coeff;
  uint32_t *p_coeff, *p_val;
  uint32_t coeff_offset=1;

  for (i=1; i < coeff_v[0]; i++){
    coeff_offset+= coeff_v[i];
  }

  for (i=0; i < last_idx; i++){
    p_val = &coeff_v[coeff_offset];
    pn_coeff = coeff_v[i+1];
    coeff_offset +=pn_coeff;

    if (convert){
      to_montgomery_h(scalar2, &scalar[i*NWORDS_256BIT], pidx);
    } else {
       memcpy(scalar2,&scalar[i*NWORDS_256BIT],sizeof(uint32_t)*NWORDS_256BIT);
    }

    for (j=0; j < pn_coeff; j++){
      montmult_h(&p_val[j*NWORDS_256BIT],&p_val[j*NWORDS_256BIT],scalar2,pidx);
      addm_h(&coeff_v[1+j*NWORDS_256BIT], &coeff_v[1+j*NWORDS_256BIT],&p_val[j*NWORDS_256BIT],pidx);
    }
  }
}

CUZPolySpsArray::CUZPolySpsArray(CUZPolySps **zpoly, uint32_t n_zpoly)
{
   uint32_t i;
   uint32_t n_coeff = 0, prev_coeff = 0, idx;

   for (i=0; i< n_zpoly; i++){
      n_coeff += zpoly[i]->getNCoeff();
   }

   coeff_d = new uint32_t[n_coeff];
   coeff_v = new uint32_t[n_coeff * NWORDS_256BIT + 1 + n_zpoly];

   // initialize n poly
   coeff_v[0] = n_zpoly;
   idx=1 + n_zpoly;
   n_coeff = zpoly[0]->getNCoeff();
   memcpy(&coeff_v[idx], zpoly[0]->getVCoeff(), sizeof(uint32_t)*NWORDS_256BIT*n_coeff);
   memcpy(&coeff_d[idx], zpoly[0]->getDCoeff(), sizeof(uint32_t)*n_coeff);

   for (i=1; i< n_zpoly; i++){
     coeff_v[i] = n_coeff + prev_coeff;
     prev_coeff = coeff_v[i];
     idx+=coeff_v[i];
     n_coeff = zpoly[i]->getNCoeff();
     memcpy(&coeff_v[idx], zpoly[i]->getVCoeff(), sizeof(uint32_t)*NWORDS_256BIT*n_coeff);
     memcpy(&coeff_d[idx], zpoly[i]->getDCoeff(), sizeof(uint32_t)*n_coeff);
   }
   coeff_v[i] = n_coeff + prev_coeff;

}

CUZPolySpsArray::~CUZPolySpsArray()
{
  delete coeff_d;
  delete coeff_v;
}

uint32_t CUZPolySpsArray::getNZPolySps()
{
  return coeff_v[0];
}

CUZPolySps * CUZPolySpsArray::getZPolySps(uint32_t index)
{
   uint32_t n_coeff = coeff_v[index+1];
   CUZPolySps *p = new CUZPolySps(n_coeff);
   
   return p;
}
uint32_t *CUZPolySpsArray::getZPolySpsRep()
{
  return coeff_v;
}

void CUZPolySpsArray::show(uint32_t i)
{
  CUZPolySps *p = getZPolySps(i);
  p->show();
  delete p;
}
#endif

