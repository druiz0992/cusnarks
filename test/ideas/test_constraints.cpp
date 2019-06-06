#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "types.h"
#include "utils_host.h"


const char input_file[] = "../../data/prove-kyc.bin";
//const char input_file[] = "../../data/circuit.bin";
int main()
{
  uint32_t *constraints;
  uint32_t *poly = NULL;
  uint32_t *poly_len = NULL;
  uint32_t i;
  cirbin_hfile_t header;
  readU256CircuitFileHeader_h(&header, input_file);
  int ret_val, len=0;

  constraints = (uint32_t *)calloc(header.nWords, sizeof(uint32_t));
  readU256CircuitFile_h(constraints, input_file,0);

  poly_len = (uint32_t *)calloc(header.nVars, sizeof(uint32_t));

  r1cs_to_mpoly_len_h(poly_len, &constraints[CIRBIN_H_N_OFFSET], &header, 1);

  for(i=0; i < header.nVars; i++){
    len+=poly_len[i];
  }

  poly = (uint32_t *)calloc(1+len*(NWORDS_256BIT+2), sizeof(uint32_t));
  poly[0] = header.nVars;

  memcpy(&poly[1], poly_len,header.nVars*sizeof(uint32_t));

  r1cs_to_mpoly_h(poly,&constraints[CIRBIN_H_N_OFFSET], &header,1); 
  free(constraints);
  free(poly);
  free(poly_len);
  
  return 0;
}
