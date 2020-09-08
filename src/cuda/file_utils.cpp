
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
#include <sys/stat.h>
#include <sys/sysinfo.h>

#include "types.h"
#include "constants.h"
#include "rng.h"
#include "log.h"
#include "bigint.h"
#include "ff.h"
#include "utils_host.h"
#include "file_utils.h"

static void  zKeyInit_h(zkey_t *zkey, const char *filename);
static uint32_t *readZKeySection_h(zkey_t *zkey, uint32_t section_id, const char *filename);
static void  zKeyToPkFileAddHdr_h(uint32_t *buffer, zkey_t *zkey, const char *pkbin_filename);

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
        r1cs_val_offset += NWORDS_FR;
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
        r1cs_val_offset += NWORDS_FR;
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

void getDataFileSize(t_uint64 *nwords , const char *filename)
{
  struct stat st;
  stat(filename, &st);

  *nwords = st.st_size/sizeof(uint32_t);
}

void readDataFile(uint32_t *samples, const char *filename)
{
  t_uint64 nwords;
  getDataFileSize(&nwords, filename);

  FILE *ifp = fopen(filename,"rb");
 
  fread(samples,sizeof(uint32_t),nwords,ifp);

  fclose(ifp);

}

void readU256DataFileFromOffset_h(uint32_t *samples, const char *filename, t_uint64 woffset, t_uint64 nwords)
{
  FILE *ifp = fopen(filename,"rb");

  fseek(ifp, woffset * sizeof(uint32_t), SEEK_SET);
  fread(samples,sizeof(uint32_t),nwords,ifp);

  fclose(ifp);

}
void readWitnessFile_h(uint32_t *samples, const char *filename, uint32_t fmt,  const unsigned long long inlen)
{
  unsigned long long i;
  unsigned long long nwords;
  uint32_t wsize;
  uint32_t wmore;
  uint32_t nwords32;
  uint32_t r[NWORDS_FR];
  FILE *ifp = fopen(filename,"rb");
  

  fread(&nwords,sizeof(uint32_t),WITNESS_HEADER_N_LEN_NWORDS,ifp);
  fread(&wsize,sizeof(uint32_t),WITNESS_HEADER_SIZE_LEN_NWORDS,ifp);
  fread(&wmore,sizeof(uint32_t),WITNESS_HEADER_OTHER_LEN_NWORDS,ifp); 
  if (!fmt){
    fseek(ifp, 32, SEEK_SET);
  }

#if 0
  for (i=0;i<inlen; i++){
    fread(&samples[i*NWORDS_256BIT],sizeof(uint32_t),NWORDS_FR,ifp);
  }
#else
    fread(samples,sizeof(uint32_t)*NWORDS_FR,inlen,ifp);
#endif
  
  fclose(ifp);
}

void writeWitnessFile_h(uint32_t *samples, const char *filename, const unsigned long long nwords)
{
  uint32_t wsize = NWORDS_FR;
  uint32_t wmore = 0;
  FILE *ifp = fopen(filename,"wb");

  fwrite(&nwords, sizeof(uint64_t), 1, ifp); 
  fwrite(&wsize, sizeof(uint32_t), 1, ifp); 
  fwrite(&wmore, sizeof(uint32_t), 1, ifp); 

  fseek(ifp, WITNESS_HEADER_LEN_NWORDS * sizeof(uint32_t), SEEK_SET);

  fwrite(samples, sizeof(uint32_t), nwords*NWORDS_FR, ifp); 

}

void readWtnsFile_h(uint32_t *samples, unsigned long long nElems,  unsigned long long start, const char *filename)
{
  FILE *ifp = fopen(filename,"rb");

  fseek(ifp, start, SEEK_SET);
  fread(samples, sizeof(uint32_t), nElems*NWORDS_FR, ifp); 
  
  fclose(ifp);
}

uint32_t *readSharedMWtnsFile_h(uint32_t *samples_out, unsigned long long nElems, const char *filename)
{
  uint32_t *samples;
  int32_t status;
  uint32_t shKey, shID;

  FILE *ifp = fopen(filename,"rb");

  fseek(ifp, WTNS_SHARED_HDR_START_OFFSET_NBYTES, SEEK_SET);
  fread(&shKey, sizeof(uint32_t), 1, ifp); 
  fread(&status, sizeof(int32_t), 1, ifp); 
  fread(&shID, sizeof(uint32_t), 1, ifp); 

  if (status) {
	  return NULL;
  }

  int32_t shmid = shared_get_h((void **) &samples, nElems*NWORDS_FR*sizeof(uint32_t));
  if (shmid == -1){
	  return NULL;
  }

  fclose(ifp);
  memcpy(samples_out, samples, nElems*NWORDS_FR*sizeof(uint32_t));
  shared_detach_h(samples);
  
  return samples;
}

unsigned long long readNWtnsNEls_h(unsigned long long *start, const char *filename)
{
  uint32_t section_id;
  unsigned long long section_len, nElems;
  FILE *ifp = fopen(filename,"rb");

  fseek(ifp, WTNS_HDR_START_OFFSET_NBYTES, SEEK_SET);
  fread(&section_id, sizeof(uint32_t), 1, ifp); 
  fread(&section_len, sizeof(unsigned long long), 1, ifp); 

  // set pointer at the beginning of section 2
  fseek(ifp, section_len, SEEK_CUR);
  fread(&section_id, sizeof(uint32_t), 1, ifp); 
  fread(&section_len, sizeof(unsigned long long), 1, ifp); 
  
  nElems = section_len/(NWORDS_FR * sizeof(uint32_t));
  *start = ftell(ifp);

  fclose(ifp);

  return nElems;
}

void zKeyToPkFile_h(const char *pkbin_filename, const char *zkey_filename)
{
  uint32_t *buffer;
  uint32_t *buffer2;
  zkey_t zkey;
  FP_INIT_ARRZERO(zero);
  const uint32_t *ECInf = CusnarksG1InfGet();
  const uint32_t *EC2Inf = CusnarksG2InfGet();

  // initialize section offsets
  zKeyInit_h(&zkey, zkey_filename);

  // Section 2 -> 
  //      n8q (1W),  q(),  n8r,  r,   NVars,  NPub,   DomainSize  (multiple of 2
  //      alpha1 , beta1,  delta1,  beta2,   gamma2,     delta2
  buffer2 = readZKeySection_h(&zkey, ZKEY_HDR_SECTION_2, zkey_filename);
  zKeyToPkFileAddHdr_h(buffer2, &zkey, pkbin_filename);

  FILE *ofp = fopen(pkbin_filename,"a+");
  fseek(ofp, 0, SEEK_END);

  // Section 4 Coeffs
  buffer = readZKeySection_h(&zkey, ZKEY_HDR_SECTION_4, zkey_filename);
  fwrite(buffer, sizeof(uint32_t), zkey.section_len[ZKEY_HDR_SECTION_4]/sizeof(uint32_t), ofp); 
  unsigned long long polsA_nWords = zkey.section_len[ZKEY_HDR_SECTION_4]/sizeof(uint32_t);
  unsigned long long extra_words = 0;
  if (polsA_nWords < buffer2[ZKEY_HDR_SECTION2_DOMAINSIZE_OFFSET] * NWORDS_FR){
     extra_words = buffer2[ZKEY_HDR_SECTION2_DOMAINSIZE_OFFSET] * NWORDS_FR - polsA_nWords;
  }
  free(buffer);

  // alpha1
  fwrite(&buffer2[ZKEY_HDR_SECTION2_ALPHA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  // beta1 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_BETA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  // delta1 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_DELTA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  //  beta2
  fwrite(&buffer2[ZKEY_HDR_SECTION2_BETA2_OFFSET], sizeof(uint32_t), 4*NWORDS_FP, ofp); 
  // delta2 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_DELTA2_OFFSET], sizeof(uint32_t), 4*NWORDS_FP, ofp); 


  // Section 5 A
  buffer = readZKeySection_h(&zkey, ZKEY_HDR_SECTION_5, zkey_filename);
  for (t_uint64 i=0; i < zkey.section_len[ZKEY_HDR_SECTION_5]/(NWORDS_FP * sizeof(uint32_t) * 2) ; i++){
     if (equBI_h(&buffer[i*2*NWORDS_FP], zero, NWORDS_FP) && 
        (equBI_h(&buffer[i*2*NWORDS_FP+NWORDS_FP], zero, NWORDS_FP))) {
        memcpy(&buffer[i*2*NWORDS_FP],ECInf, 2*NWORDS_FP*sizeof(uint32_t));
     }
  }
  fwrite(buffer, sizeof(uint32_t), zkey.section_len[ZKEY_HDR_SECTION_5]/sizeof(uint32_t), ofp); 
  // alpha1
  fwrite(&buffer2[ZKEY_HDR_SECTION2_ALPHA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  // delta1 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_DELTA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  free(buffer);

  // Section 6 B1
  buffer = readZKeySection_h(&zkey, ZKEY_HDR_SECTION_6, zkey_filename);
  for (t_uint64 i=0; i < zkey.section_len[ZKEY_HDR_SECTION_6]/(NWORDS_FP * sizeof(uint32_t) * 2) ; i++){
     if (equBI_h(&buffer[i*2*NWORDS_FP], zero, NWORDS_FP) && 
        (equBI_h(&buffer[i*2*NWORDS_FP+NWORDS_FP], zero, NWORDS_FP))) {
        memcpy(&buffer[i*2*NWORDS_FP],ECInf, 2*NWORDS_FP*sizeof(uint32_t));
     }
  }
  fwrite(buffer, sizeof(uint32_t), zkey.section_len[ZKEY_HDR_SECTION_6]/sizeof(uint32_t), ofp); 
  // beta1 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_BETA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  // delta1 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_DELTA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  free(buffer);

  // Section 7 B2
  buffer = readZKeySection_h(&zkey, ZKEY_HDR_SECTION_7, zkey_filename);
  for (t_uint64 i=0; i < zkey.section_len[ZKEY_HDR_SECTION_7]/(NWORDS_FP * sizeof(uint32_t) * 4) ; i++){
     if (equBI_h(&buffer[i*2*NWORDS_FP], zero, NWORDS_FP) && 
        (equBI_h(&buffer[i*2*NWORDS_FP+NWORDS_FP], zero, NWORDS_FP)) &&
        (equBI_h(&buffer[i*2*NWORDS_FP+2*NWORDS_FP], zero, NWORDS_FP) && 
        (equBI_h(&buffer[i*2*NWORDS_FP+3*NWORDS_FP], zero, NWORDS_FP)))) {
        memcpy(&buffer[i*2*NWORDS_FP],EC2Inf, 4*NWORDS_FP*sizeof(uint32_t));
     }
  }
  fwrite(buffer, sizeof(uint32_t), zkey.section_len[ZKEY_HDR_SECTION_7]/sizeof(uint32_t), ofp); 
  //  beta2
  fwrite(&buffer2[ZKEY_HDR_SECTION2_BETA2_OFFSET], sizeof(uint32_t), 4*NWORDS_FP, ofp); 
  // delta2 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_DELTA2_OFFSET], sizeof(uint32_t), 4*NWORDS_FP, ofp); 
  free(buffer);

  // Section 8 C
  buffer = readZKeySection_h(&zkey, ZKEY_HDR_SECTION_8, zkey_filename);
  for (t_uint64 i=0; i < zkey.section_len[ZKEY_HDR_SECTION_8]/(NWORDS_FP * sizeof(uint32_t) * 2) ; i++){
     if (equBI_h(&buffer[i*2*NWORDS_FP], zero, NWORDS_FP) && 
        (equBI_h(&buffer[i*2*NWORDS_FP+NWORDS_FP], zero, NWORDS_FP))) {
        memcpy(&buffer[i*2*NWORDS_FP],ECInf, 2*NWORDS_FP*sizeof(uint32_t));
     }

  }
  fwrite(buffer, sizeof(uint32_t), zkey.section_len[ZKEY_HDR_SECTION_8]/sizeof(uint32_t), ofp); 
  free(buffer);

  // Section 9 H
  buffer = readZKeySection_h(&zkey, ZKEY_HDR_SECTION_9, zkey_filename);
  for (t_uint64 i=0; i < zkey.section_len[ZKEY_HDR_SECTION_9]/(NWORDS_FP * sizeof(uint32_t) * 2) ; i++){
     if (equBI_h(&buffer[i*2*NWORDS_FP], zero, NWORDS_FP) && 
        (equBI_h(&buffer[i*2*NWORDS_FP+NWORDS_FP], zero, NWORDS_FP))) {
        memcpy(&buffer[i*2*NWORDS_FP],ECInf, 2*NWORDS_FP*sizeof(uint32_t));
     }

  }
  fwrite(buffer, sizeof(uint32_t), zkey.section_len[ZKEY_HDR_SECTION_9]/sizeof(uint32_t), ofp); 
  // delta1 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_DELTA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_DELTA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_DELTA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_DELTA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_DELTA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 
  fwrite(&buffer2[ZKEY_HDR_SECTION2_DELTA1_OFFSET], sizeof(uint32_t), 2*NWORDS_FP, ofp); 

  free(buffer);
  free(buffer2);
  fclose(ofp);

}

static void  zKeyInit_h(zkey_t *zkey, const char *filename)
{
  unsigned long long section_len;
  uint32_t section_id;
  uint32_t i;
  FILE *ifp = fopen(filename,"rb");

  // init zkey
  memset(zkey, 0, sizeof(zkey_t));
  fseek(ifp, ZKEY_HDR_START_OFFSET_NBYTES, SEEK_SET);
  
  while (fread(&section_id, sizeof(uint32_t), 1, ifp)) {
      fread(&zkey->section_len[section_id], sizeof(unsigned long long), 1, ifp); 
      zkey->section_offset[section_id] = ftell(ifp);
      fseek(ifp, zkey->section_len[section_id], SEEK_CUR);
  }

  fclose(ifp);  
}

uint32_t *readZKeySection_h(uint32_t section_id, const char *filename)
{
  zkey_t zkey;
  uint32_t *buffer;

  zKeyInit_h(&zkey, filename);
  buffer = readZKeySection_h(&zkey, section_id, filename);

  return buffer;
}

static uint32_t *readZKeySection_h(zkey_t *zkey, uint32_t section_id, const char *filename)
{
  unsigned long long section_len;
  uint32_t *buffer;
  unsigned long long buffer_len;
  FILE *ifp = fopen(filename,"rb");

  fseek(ifp, zkey->section_offset[section_id] - 8, SEEK_SET);
  // allocate buffer size for section
  fread(&section_len, sizeof(unsigned long long), 1, ifp); 
  buffer_len =(section_len + sizeof(uint32_t) -1)/sizeof(uint32_t);
  buffer = (uint32_t *) malloc(buffer_len * sizeof(uint32_t));
  fread(buffer, sizeof(uint32_t), buffer_len , ifp); 

  fclose(ifp);  

  return buffer;
}

static void  zKeyToPkFileAddHdr_h(uint32_t *buffer, zkey_t *zkey, const char *pkbin_filename)
{
  uint32_t nVars = buffer[ZKEY_HDR_SECTION2_NVARS_OFFSET];
  uint32_t nPublic = buffer[ZKEY_HDR_SECTION2_NPUBLIC_OFFSET];
  uint32_t domainSize = buffer[ZKEY_HDR_SECTION2_DOMAINSIZE_OFFSET];
  uint32_t domainBits =  31-msbuBI_h(&domainSize,1);
  uint32_t tmp;
  buffer[ZKEY_HDR_SECTION2_DOMAINSIZE_OFFSET] = domainSize;

  FILE *ofp = fopen(pkbin_filename,"wb");
  //      n8q (1W),  q(),  n8r,  r,   NVars,  NPub,   DomainSize  (multiple of 2
  //      alpha1 , beta1,  delta1,  beta2,   gamma2,     delta2

  //nWords
  tmp=0;
  fwrite(&tmp, sizeof(uint32_t), 1, ofp); 
  // File TYPE
  tmp = SNARKSFILE_T_PK;
  fwrite(&tmp, sizeof(uint32_t), 1, ofp); 
  // Groth
  tmp = PROTOCOL_T_GROTH;
  fwrite(&tmp, sizeof(uint32_t), 1, ofp); 
  // RBitLen
  //TODO
  //tmp = NWORDS_FR*sizeof(uint32_t) | ((NWORDS_FR * sizeof(uint32_t))<< 16); 
  tmp = NWORDS_FR | (NWORDS_FR<< 16); 
  fwrite(&tmp, sizeof(uint32_t), 1, ofp); 
  // MOnt
  tmp = FMT_MONT;
  fwrite(&tmp, sizeof(uint32_t), 1, ofp); 
  // Affine 
  tmp = EC_T_AFFINE;
  fwrite(&tmp, sizeof(uint32_t), 1, ofp); 
  // nVars 
  fwrite(&nVars, sizeof(uint32_t), 1, ofp); 
  // nPublic
  fwrite(&nPublic, sizeof(uint32_t), 1, ofp); 
  // Domain Bits
  fwrite(&domainBits, sizeof(uint32_t), 1, ofp); 
  // Domain Size
  tmp = domainSize;
  fwrite(&domainSize, sizeof(uint32_t), 1, ofp); 
  // MOD_FP
  fwrite(&buffer[ZKEY_HDR_SECTION2_FP_OFFSET], sizeof(uint32_t), buffer[ZKEY_HDR_SECTION2_FPLEN_OFFSET]/sizeof(uint32_t), ofp); 
  // MOD_FR
  fwrite(&buffer[ZKEY_HDR_SECTION2_FR_OFFSET], sizeof(uint32_t), buffer[ZKEY_HDR_SECTION2_FRLEN_OFFSET]/sizeof(uint32_t), ofp); 
  unsigned long long polsA_nWords = zkey->section_len[ZKEY_HDR_SECTION_4]/sizeof(uint32_t);
  unsigned long long polsB_nWords = 0;
  unsigned long long extra_words = 0;
  if (polsA_nWords < domainSize * NWORDS_FR){
     extra_words = domainSize * NWORDS_FR - polsA_nWords;
  }
  polsA_nWords = polsA_nWords  + extra_words;

  // polsA_nWords
  fwrite(&polsA_nWords, sizeof(unsigned long long), 1, ofp); 
  fwrite(&polsB_nWords, sizeof(unsigned long long), 1, ofp); 

  unsigned long long AnWords = zkey->section_len[ZKEY_HDR_SECTION_5]/sizeof(uint32_t) + 4 * NWORDS_FP;
  unsigned long long B1nWords = zkey->section_len[ZKEY_HDR_SECTION_6]/sizeof(uint32_t) + 4 * NWORDS_FP;
  unsigned long long B2nWords = zkey->section_len[ZKEY_HDR_SECTION_7]/sizeof(uint32_t) + 8 * NWORDS_FP;
  unsigned long long CnWords = zkey->section_len[ZKEY_HDR_SECTION_8]/sizeof(uint32_t);
  unsigned long long HnWords = zkey->section_len[ZKEY_HDR_SECTION_9]/sizeof(uint32_t) + 12 * NWORDS_FP;

  // AnWords
  fwrite(&AnWords, sizeof(unsigned long long), 1, ofp); 
  // B1nWords
  fwrite(&B1nWords, sizeof(unsigned long long), 1, ofp); 
  // B2nWords
  fwrite(&B2nWords, sizeof(unsigned long long), 1, ofp); 
  // CnWords
  fwrite(&CnWords, sizeof(unsigned long long), 1, ofp); 
  // HnWords
  fwrite(&HnWords, sizeof(unsigned long long), 1, ofp); 

  fclose(ofp);
}
