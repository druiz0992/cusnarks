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


#endif
