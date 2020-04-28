
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
// File name  : ec.cpp
//
// Date       : 6/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//
// ------------------------------------------------------------------

#include <stdio.h>
#include <omp.h>

#include "types.h"
#include "constants.h"
#include "rng.h"
#include "log.h"
#include "bigint.h"
#include "ff.h"
#include "file_utils.h"


/*
  Read header circuit binary file

  char * filename      : location of file to be written

  circuit bin header file format:
*/
void readU256CircuitFileHeader_h(cirbin_hfile_t *hfile, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");
  fread(&hfile->nWords, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->nPubInputs, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->nOutputs, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->nVars, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->nConstraints, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->cirformat, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->R1CSA_nWords, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->R1CSB_nWords, sizeof(unsigned long long), 1, ifp); 
  fread(&hfile->R1CSC_nWords, sizeof(unsigned long long), 1, ifp); 
  fclose(ifp);

}
/*
  Read circuit binary file
       
*/
void readU256CircuitFile_h(uint32_t *samples, const char *filename, unsigned long long nwords=0)
{
  FILE *ifp = fopen(filename,"rb");
  unsigned long long i=0;
  if (!nwords){
    while (!feof(ifp)){
      fread(&samples[i++], sizeof(uint32_t), 1, ifp); 
    }
  } else {
      fread(samples, sizeof(uint32_t), nwords, ifp); 
  }
  fclose(ifp);

}

#ifdef ROLLUP
void readR1CSFileHeader_h(r1csv1_t *r1cs_hdr, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");
  uint32_t k=0,i;
  uint32_t tmp_word, n_coeff;
  t_int64 offset;

  r1cs_hdr->R1CSA_nCoeff=0;
  r1cs_hdr->R1CSB_nCoeff=0;
  r1cs_hdr->R1CSC_nCoeff=0;

  fread(&r1cs_hdr->magic_number, sizeof(uint32_t), 1, ifp); 
  if (r1cs_hdr->magic_number != R1CS_HDR_MAGIC_NUMBER){
    printf("Unexpected R1CS header format\n");
    fclose(ifp);
    exit(1);
  }

  fread(&r1cs_hdr->version, sizeof(uint32_t), 1, ifp); 
  if (r1cs_hdr->version != R1CS_HDR_V01){
    printf("Unexpected R1CS version\n");
    fclose(ifp);
    exit(1);
  }

  fseek(ifp, R1CS_HDR_FIELDDEFSIZE_OFFSET_NBYTES * sizeof(char), SEEK_SET);
  fread(&offset, sizeof(uint32_t), 1, ifp); 
  offset &= 0xFFFF;
  //printf("offset : %d\n", offset);
  fseek(ifp, (R1CS_HDR_FIELDDEFSIZE_OFFSET_NBYTES + 8 + offset) * sizeof(char), SEEK_SET);

  fread(&r1cs_hdr->word_width_bytes, sizeof(uint32_t), 1, ifp); 
  if (r1cs_hdr->word_width_bytes != 4){
     printf("Unexpected R1CS word width\n");
     fclose(ifp);
     exit(1);
  }

  fread(&r1cs_hdr->nVars, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPubOutputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPubInputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPrivInputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nLabels, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nConstraints, sizeof(uint32_t), 1, ifp); 

  fread(&offset, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->constraintLen, sizeof(uint32_t), 2, ifp); 

  /*
  printf("word_width_bytes : %d\n", r1cs_hdr->word_width_bytes);
  printf("nVars : %d\n", r1cs_hdr->nVars);
  printf("nPubOutputs : %d\n", r1cs_hdr->nPubOutputs);
  printf("nPubInputs : %d\n", r1cs_hdr->nPubInputs);
  printf("nPrivInputs : %d\n", r1cs_hdr->nPrivInputs);
  printf("nLabels : %d\n", r1cs_hdr->nLabels);
  printf("nConstraints : %d\n", r1cs_hdr->nConstraints);
  printf("Const section len : %lld\n",r1cs_hdr->constraintLen);
  */

  r1cs_hdr->constraintOffset = ftell(ifp);

  r1cs_hdr->R1CSA_nCoeff = 0;
  r1cs_hdr->R1CSB_nCoeff = 0;
  r1cs_hdr->R1CSC_nCoeff = 0;

  offset = r1cs_hdr->constraintLen;

  while (offset > 0){
    fread(&n_coeff, sizeof(uint32_t), 1, ifp); 
    offset-=sizeof(uint32_t);
    if (k%3 == R1CSA_IDX){
      r1cs_hdr->R1CSA_nCoeff+= (n_coeff);
    } else if (k%3 == R1CSB_IDX){
      r1cs_hdr->R1CSB_nCoeff+= (n_coeff);
    } else {
      r1cs_hdr->R1CSC_nCoeff+= (n_coeff);
    }
    for (i=0; i< n_coeff; i++){
      fseek(ifp, 4, SEEK_CUR);
      fread(&tmp_word, sizeof(char), 1, ifp); 
      tmp_word &= 0xFF;
      fseek(ifp, tmp_word, SEEK_CUR);
      offset-=(tmp_word + sizeof(char) + sizeof(uint32_t));
    }
    k++;
  }

  /*
  printf("N coeff R1CSA : %d\n", r1cs_hdr->R1CSA_nCoeff);
  printf("N coeff R1CSB : %d\n", r1cs_hdr->R1CSB_nCoeff);
  printf("N coeff R1CSC : %d\n", r1cs_hdr->R1CSC_nCoeff);
  
  printf("end of constraints : %lld\n",ftell(ifp));
  fread(&offset, sizeof(uint32_t), 1, ifp); 
  printf("Lable section len : %lld\n",offset);
  */

  fclose(ifp);

  return;
}
  

void readR1CSFile_h(uint32_t *samples, const char *filename, r1csv1_t *r1cs, r1cs_idx_t r1cs_idx )
{
  FILE *ifp = fopen(filename,"rb");
  uint32_t tmp_word, n_coeff;
  uint32_t r1cs_offset=0, r1cs_coeff_offset=1+r1cs->nConstraints, r1cs_val_offset = 1+r1cs->nConstraints;
  uint32_t k=0, accum_coeffs=0, i,j;
  t_int64 offset = r1cs->constraintLen;

  samples[r1cs_offset++] = r1cs->nConstraints;
  
  //printf("constraint len : %lld\n",offset);
  fseek(ifp, r1cs->constraintOffset, SEEK_SET);

  while (!offset){
    fread(&n_coeff, sizeof(uint32_t), 1, ifp); 
    offset-=sizeof(uint32_t);
    if (k%3 == r1cs_idx) {
      accum_coeffs+= ((uint32_t) n_coeff);
      samples[r1cs_offset++] = accum_coeffs;
      r1cs_val_offset += n_coeff;
      for (i=0; i< n_coeff; i++){
        fread(&samples[r1cs_coeff_offset++], sizeof(uint32_t), 1, ifp); 
        fread(&tmp_word, 1,1, ifp);
	tmp_word &= tmp_word & 0xFF;
        for(j=0; j <tmp_word; j++){
           fread(&samples[r1cs_val_offset+j], 1, 1, ifp); 
        }
        offset-=(tmp_word + sizeof(char) + sizeof(uint32_t));
        r1cs_val_offset += NWORDS_256BIT;
      }
      r1cs_coeff_offset = r1cs_val_offset;

    }  else {
      for (i=0; i< n_coeff; i++){
        fseek(ifp, 4, SEEK_CUR);
        fread(&tmp_word, 1, 1, ifp); 
	tmp_word &=0xFF;
        fseek(ifp, tmp_word, SEEK_CUR);
        offset-=(tmp_word + sizeof(char) + sizeof(uint32_t));
      }
    }
    
    k++;
  }

  fclose(ifp);
}

#else
void readR1CSFileHeader_h(r1csv1_t *r1cs_hdr, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");
  uint32_t k=0,i;
  uint32_t tmp_word, n_coeff;
  t_int64 offset;
  uint32_t section_type;

  r1cs_hdr->R1CSA_nCoeff=0;
  r1cs_hdr->R1CSB_nCoeff=0;
  r1cs_hdr->R1CSC_nCoeff=0;

  fread(&r1cs_hdr->magic_number, sizeof(uint32_t), 1, ifp); 
  if (r1cs_hdr->magic_number != R1CS_HDR_MAGIC_NUMBER){
    printf("Unexpected R1CS header format\n");
    fclose(ifp);
    exit(1);
  }

  fread(&r1cs_hdr->version, sizeof(uint32_t), 1, ifp); 
  if (r1cs_hdr->version != R1CS_HDR_V01){
    printf("Unexpected R1CS version %d\n", r1cs_hdr->version);
    fclose(ifp);
    exit(1);
  }

  fread(&r1cs_hdr->nsections, sizeof(uint32_t), 1, ifp); 
  //printf("N sections : %d\n",r1cs_hdr->nsections);
  fread(&section_type, sizeof(uint32_t), 1, ifp); 
  //printf("Section type : %d\n",section_type);
  if (section_type != R1CS_HDR_SECTION_TYPE){
    printf("Unexpected section : %d\n",section_type);
    fclose(ifp);
    exit(1);
  }

  fread(&offset, sizeof(t_uint64), 1, ifp); 
  //printf("HEADER Section Length : %lld\n", offset);
  //fseek(ifp, R1CS_HDR_FIELDDEFSIZE_OFFSET_NBYTES * sizeof(char), SEEK_SET);
  fread(&offset, sizeof(uint32_t), 1, ifp); 
  //printf("Field Size : %lld\n", offset);
  fseek(ifp, offset , SEEK_CUR);

  //fread(&r1cs_hdr->word_width_bytes, sizeof(uint32_t), 1, ifp); 
  //if (r1cs_hdr->word_width_bytes != 4){
     //printf("Unexpected R1CS word width %d\n",r1cs_hdr->word_width_bytes);
     //fclose(ifp);
     //exit(1);
  //}

  fread(&r1cs_hdr->nVars, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPubOutputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPubInputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nPrivInputs, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->nLabels, sizeof(t_uint64), 1, ifp); 
  fread(&r1cs_hdr->nConstraints, sizeof(uint32_t), 1, ifp); 

  fread(&offset, sizeof(uint32_t), 1, ifp); 
  fread(&r1cs_hdr->constraintLen, sizeof(uint32_t), 2, ifp); 

  /*
  printf("word_width_bytes : %d\n", r1cs_hdr->word_width_bytes);
  printf("nVars : %d\n", r1cs_hdr->nVars);
  printf("nPubOutputs : %d\n", r1cs_hdr->nPubOutputs);
  printf("nPubInputs : %d\n", r1cs_hdr->nPubInputs);
  printf("nPrivInputs : %d\n", r1cs_hdr->nPrivInputs);
  printf("nLabels : %d\n", r1cs_hdr->nLabels);
  printf("nConstraints : %d\n", r1cs_hdr->nConstraints);
  printf("Const section len : %lld\n",r1cs_hdr->constraintLen);
  */
  

  r1cs_hdr->constraintOffset = ftell(ifp);

  r1cs_hdr->R1CSA_nCoeff = 0;
  r1cs_hdr->R1CSB_nCoeff = 0;
  r1cs_hdr->R1CSC_nCoeff = 0;

  offset = r1cs_hdr->constraintLen;

  while (offset > 0){
    fread(&n_coeff, sizeof(uint32_t), 1, ifp); 
    //printf("N coeff : %d\n",n_coeff);
    offset-=sizeof(uint32_t);
    if (k%3 == R1CSA_IDX){
      r1cs_hdr->R1CSA_nCoeff+= (n_coeff);
    } else if (k%3 == R1CSB_IDX){
      r1cs_hdr->R1CSB_nCoeff+= (n_coeff);
    } else {
      r1cs_hdr->R1CSC_nCoeff+= (n_coeff);
    }
    fseek(ifp, n_coeff*(32+4), SEEK_CUR);
    offset -= (n_coeff*36);
    /*for (i=0; i< n_coeff; i++){
      fseek(ifp, 4, SEEK_CUR);
      fread(&tmp_word, sizeof(char), 1, ifp); 
      tmp_word &= 0xFF;
      fseek(ifp, tmp_word, SEEK_CUR);
      offset-=(tmp_word + sizeof(char) + sizeof(uint32_t));
    }
    */
    k++;
  }

  /*
  printf("N coeff R1CSA : %d\n", r1cs_hdr->R1CSA_nCoeff);
  printf("N coeff R1CSB : %d\n", r1cs_hdr->R1CSB_nCoeff);
  printf("N coeff R1CSC : %d\n", r1cs_hdr->R1CSC_nCoeff);
  
  printf("end of constraints : %lld\n",ftell(ifp));
  */
  fread(&offset, sizeof(uint32_t), 1, ifp); 
  offset &=0xFFFF;
  //printf("Lable section len : %lld\n",offset);
  

  fclose(ifp);

  return;
}
  

void readR1CSFile_h(uint32_t *samples, const char *filename, r1csv1_t *r1cs, r1cs_idx_t r1cs_idx )
{
  FILE *ifp = fopen(filename,"rb");
  uint32_t tmp_word, n_coeff;
  uint32_t r1cs_offset=0, r1cs_coeff_offset=1+r1cs->nConstraints, r1cs_val_offset = 1+r1cs->nConstraints;
  uint32_t k=0, accum_coeffs=0, i,j;
  t_int64 offset = r1cs->constraintLen;

  samples[r1cs_offset++] = r1cs->nConstraints;
  
  /* printf("constraint LEN : %lld\n",offset);
  printf("constraint Offset : %lld\n",r1cs->constraintOffset);
  */
  fseek(ifp, r1cs->constraintOffset, SEEK_SET);
  //printf("Start\n");

  while (offset){
    fread(&n_coeff, sizeof(uint32_t), 1, ifp); 
    //printf("N COEFF : %d\n",n_coeff);
    offset-=sizeof(uint32_t);
    if (k%3 == r1cs_idx) {
      accum_coeffs+= ((uint32_t) n_coeff);
      samples[r1cs_offset++] = accum_coeffs;
      r1cs_val_offset += n_coeff;
      for (i=0; i< n_coeff; i++){
        fread(&samples[r1cs_coeff_offset++], sizeof(uint32_t), 1, ifp); 
        //fread(&tmp_word, 1,1, ifp);
	//tmp_word &= tmp_word & 0xFF;
	tmp_word = 8;
        //for(j=0; j <tmp_word; j++){
        fread(&samples[r1cs_val_offset], sizeof(uint32_t), tmp_word, ifp); 
        //}
        offset-=36;
        r1cs_val_offset += NWORDS_256BIT;
      }
      r1cs_coeff_offset = r1cs_val_offset;

    }  else {
      /*for (i=0; i< n_coeff; i++){
        fseek(ifp, 4, SEEK_CUR);
        fread(&tmp_word, 1, 1, ifp); 
	tmp_word &=0xFF;
        fseek(ifp, tmp_word, SEEK_CUR);
        offset-=(tmp_word + sizeof(char) + sizeof(uint32_t));
      } */
      fseek(ifp, n_coeff*(32+4), SEEK_CUR);
      offset -= (n_coeff*36);
    }
    
    k++;
  }

  fclose(ifp);

}
#endif

void readECTablesNElementsFile_h(ec_table_offset_t *table_offset, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");

  fread(&table_offset->table_order,     sizeof(uint32_t), 1, ifp); 
  fread(&table_offset->woffset_A,      sizeof(t_uint64), 1, ifp); 
  fread(&table_offset->woffset_B2,     sizeof(t_uint64), 1, ifp); 
  fread(&table_offset->woffset_B1,     sizeof(t_uint64), 1, ifp); 
  fread(&table_offset->woffset_C,      sizeof(t_uint64), 1, ifp); 
  fread(&table_offset->woffset_hExps,  sizeof(t_uint64), 1, ifp); 
  fread(&table_offset->nwords_tdata,   sizeof(t_uint64), 1, ifp); 

  /*
  printf("Order : %d\n",table_offset->table_order);
  printf("Woffset1 A : %ld\n", table_offset->woffset1_A);
  printf("Woffset1 B2 : %ld\n", table_offset->woffset1_B2);
  printf("Woffset1 B1 : %ld\n", table_offset->woffset1_B1);
  printf("Woffset C : %ld\n", table_offset->woffset_C);
  printf("Woffset hExps : %ld\n", table_offset->woffset_hExps);
  printf("N Words : %d\n", table_offset->nwords_tdata);
  */
  

  fclose(ifp);
}
/*
  Read header PK binary file

  char * filename      : location of file to be read

  circuit bin header file format:
*/
void readU256PKFileHeader_h(pkbin_hfile_t *hfile, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");
  fread(&hfile->nWords, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->ftype, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->protocol, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->Rbitlen, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->k_binformat, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->k_ecformat, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->nVars, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->nPublic, sizeof(uint32_t), 1, ifp); 
  fread(&hfile->domainSize, sizeof(uint32_t), 1, ifp); 
  fclose(ifp);

}
/*
  Read PK binary file
       
*/
void readU256PKFile_h(uint32_t *samples, const char *filename, unsigned long long nwords=0)
{
  readU256CircuitFile_h(samples, filename, nwords);
}



/*
  Write binary file

  t_uint32_t * samples : input vector containing samples. Vector is of length nwords 
  char * filename      : location of file to be written
  uint32_t nwords      : Number of samples to write.
*/
void writeU256DataFile_h(uint32_t *samples, const char *filename, unsigned long long nwords)
{
  FILE *ifp = fopen(filename,"wb");
  fwrite(samples, sizeof(uint32_t), nwords, ifp); 
  fclose(ifp);

}

void appendU256DataFile_h(uint32_t *samples, const char *filename, unsigned long long nwords)
{
  FILE *ifp = fopen(filename,"ab");
  fseek(ifp, 0, SEEK_END);
  fwrite(samples, sizeof(uint32_t), nwords, ifp); 
  fclose(ifp);

}

/*
  Read u256 data binary file and optionally decimate samples

  t_uint32_t * samples : output vector containing samples. Vector is of length outsize
  char * filename      : location of file containing samples
  uint32_t insize      : Number of samples from file to read. 
  uint32_t outsize     : Number of output samples. Samples are stored in vector with a 
                         insize/outsize ratio 
*/
void readU256DataFile_h(uint32_t *samples, const char *filename, uint32_t insize, uint32_t outsize)
{
  uint32_t i, j=0,k=0;
  uint32_t r[NWORDS_256BIT];
  FILE *ifp = fopen(filename,"rb");

  uint32_t count = insize/outsize;
  if (count != 1)
  {
    for (i=0;i<insize; i++){
      fread(r,sizeof(uint32_t),NWORDS_256BIT,ifp);
      if (j % count == 0){
        memcpy(&samples[k*NWORDS_256BIT], r, sizeof(uint32_t)*NWORDS_256BIT);
        k++;
      }
      j++;
    }
  } 
  else  {
    fread(samples, sizeof(uint32_t)*outsize, NWORDS_256BIT, ifp);
  }
 
  
  fclose(ifp);
}

void readU256DataFileFromOffset_h(uint32_t *samples, const char *filename, t_uint64 woffset, t_uint64 nwords)
{
  FILE *ifp = fopen(filename,"rb");

  fseek(ifp, woffset * sizeof(uint32_t), SEEK_SET);
  fread(samples,sizeof(uint32_t),nwords,ifp);

  /*
  printf("Offset : %ld, Nwords : %d\n",woffset, nwords);
  printU256Number(&samples[0]);
  printU256Number(&samples[NWORDS_256BIT]);
  printU256Number(&samples[2*NWORDS_256BIT]);
  printU256Number(&samples[3*NWORDS_256BIT]);
  */

  fclose(ifp);

}
void readWitnessFile_h(uint32_t *samples, const char *filename, uint32_t fmt,  const unsigned long long inlen)
{
  unsigned long long i;
  unsigned long long nwords;
  uint32_t wsize;
  uint32_t wmore;
  uint32_t nwords32;
  uint32_t r[NWORDS_256BIT];
  FILE *ifp = fopen(filename,"rb");
  

  fread(&nwords,sizeof(uint32_t),WITNESS_HEADER_N_LEN_NWORDS,ifp);
  fread(&wsize,sizeof(uint32_t),WITNESS_HEADER_SIZE_LEN_NWORDS,ifp);
  fread(&wmore,sizeof(uint32_t),WITNESS_HEADER_OTHER_LEN_NWORDS,ifp); 
  if (!fmt){
    fseek(ifp, 32, SEEK_SET);
  }

#if 0
  for (i=0;i<inlen; i++){
    fread(&samples[i*NWORDS_256BIT],sizeof(uint32_t),NWORDS_256BIT,ifp);
  }
#else
    fread(samples,sizeof(uint32_t)*NWORDS_256BIT,inlen,ifp);
#endif
  
  fclose(ifp);
}
void writeWitnessFile_h(uint32_t *samples, const char *filename, const unsigned long long nwords)
{
  uint32_t wsize = NWORDS_256BIT;
  uint32_t wmore = 0;
  FILE *ifp = fopen(filename,"wb");

  fwrite(&nwords, sizeof(uint64_t), 1, ifp); 
  fwrite(&wsize, sizeof(uint32_t), 1, ifp); 
  fwrite(&wmore, sizeof(uint32_t), 1, ifp); 

  fseek(ifp, WITNESS_HEADER_LEN_NWORDS * sizeof(uint32_t), SEEK_SET);

  fwrite(samples, sizeof(uint32_t), nwords*NWORDS_256BIT, ifp); 

}

