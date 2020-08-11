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
// File name  : ff.h
//
// Date       : 06/03/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implementation 
// ------------------------------------------------------------------

*/
#ifndef _FILE_UTILS_H_
#define _FILE_UTILS_H_

#define ZKEY_HDR_START_OFFSET_NBYTES (12)

#define ZKEY_HDR_SECTION2_FPLEN_OFFSET (0)
#define ZKEY_HDR_SECTION2_FP_OFFSET (1)
#define ZKEY_HDR_SECTION2_FRLEN_OFFSET (9)
#define ZKEY_HDR_SECTION2_FR_OFFSET (10)
#define ZKEY_HDR_SECTION2_NVARS_OFFSET  (18)
#define ZKEY_HDR_SECTION2_NPUBLIC_OFFSET (19)
#define ZKEY_HDR_SECTION2_DOMAINSIZE_OFFSET (20)
#define ZKEY_HDR_SECTION2_ALPHA1_OFFSET (21)
#define ZKEY_HDR_SECTION2_BETA1_OFFSET (ZKEY_HDR_SECTION2_ALPHA1_OFFSET + 2*NWORDS_FP)
#define ZKEY_HDR_SECTION2_DELTA1_OFFSET (ZKEY_HDR_SECTION2_ALPHA1_OFFSET + 12*NWORDS_FP)
#define ZKEY_HDR_SECTION2_BETA2_OFFSET (ZKEY_HDR_SECTION2_ALPHA1_OFFSET + 4*NWORDS_FP)
#define ZKEY_HDR_SECTION2_GAMMA2_OFFSET (ZKEY_HDR_SECTION2_ALPHA1_OFFSET + 8*NWORDS_FP)
#define ZKEY_HDR_SECTION2_DELTA2_OFFSET (ZKEY_HDR_SECTION2_ALPHA1_OFFSET + 14*NWORDS_FP)


#define WTNS_HDR_START_OFFSET_NBYTES (12)
// 12B HDR2 + 12BHDR1 + 40B SECTION1 + 12B HDR2
#define WTNS_SHARED_HDR_START_OFFSET_NBYTES (76) 


typedef enum {
   ZKEY_HDR_SECTION_0 = 0, // Zkey
   ZKEY_HDR_SECTION_1,  // GROTH
   ZKEY_HDR_SECTION_2,  // Constants
   ZKEY_HDR_SECTION_3,  // IC
   ZKEY_HDR_SECTION_4,  // Coeffs
   ZKEY_HDR_SECTION_5,  // A
   ZKEY_HDR_SECTION_6,  // B1
   ZKEY_HDR_SECTION_7,  // B2
   ZKEY_HDR_SECTION_8,  // C
   ZKEY_HDR_SECTION_9,  // H
   ZKEY_HDR_SECTION_10, // Contributions

   ZKEY_HDR_NSECTIONS // Contributions
}zkey_sections_t;

typedef struct {
   unsigned long long section_offset[ZKEY_HDR_NSECTIONS];
   unsigned long long section_len[ZKEY_HDR_NSECTIONS];
}zkey_t;

void readU256DataFile_h(uint32_t *samples, const char *filename, uint32_t insize, uint32_t outsize);
void readU256DataFileFromOffset_h(uint32_t *samples, const char *filename, t_uint64 woffset, t_uint64 nwrods);
void readWitnessFile_h(uint32_t *samples, const char *filename, uint32_t fmt, const unsigned long long insize);
void writeU256DataFile_h(uint32_t *samples, const char *filename, unsigned long long nwords);
void appendU256DataFile_h(uint32_t *samples, const char *filename, unsigned long long nwords);
void writeWitnessFile_h(uint32_t *samples, const char *filename, const unsigned long long nwords);
void readU256CircuitFileHeader_h(cirbin_hfile_t *hfile, const char *filename);
void readU256CircuitFile_h(uint32_t *samples, const char *filename, unsigned long long nwords);
void readU256PKFileHeader_h(pkbin_hfile_t *hfile, const char *filename);
void readU256PKFile_h(uint32_t *samples, const char *filename, unsigned long long nwords);
void readR1CSFileHeader_h(r1csv1_t *r1cs_hdr, const char *filename);
void readR1CSFile_h(uint32_t *samples, const char *filename, r1csv1_t *r1cs, r1cs_idx_t r1cs_idx );
void readECTablesNElementsFile_h(ec_table_offset_t *table_offset, const char *filename);
void readDataFile(uint32_t *samples, const char *filename);
void getDataFileSize(t_uint64 *nwords , const char *filename);
void zKeyToPkFile_h(const char *pkbin_filename, const char *zkey_filename);
uint32_t *readZKeySection_h(uint32_t section_id, const char *zkey_filename);
unsigned long long readNWtnsNEls_h(unsigned long long *start, const char *filename);
void readWtnsFile_h(uint32_t *samples, unsigned long long nElems,  unsigned long long start, const char *filename);
uint32_t *readSharedMWtnsFile_h(unsigned long long nElems, const char *filename);


#endif
