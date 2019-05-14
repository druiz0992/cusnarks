#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "utils_host.h"


const char input_file[] = "../../data/prove-kyc.bin";
int main()
{
  uint32_t *constraints;
  uint32_t *poly;
  cirbin_hfile_t header;
  readU256CircuitFileHeader_h(&header, input_file);

  constraints = (uint32_t *)calloc(header.nWords, sizeof(uint32_t));
  readU256CircuitFile_h(constraints, input_file,0);

  poly = (uint32_t *)calloc(header.nVars*header.nConstraints*(NWORDS_256BIT+1)+header.nVars, sizeof(uint32_t));
  constraints_to_zpoly(poly, &constraints[8], header.nVars, header.nConstraints); 

  free(constraints);
  free(poly);
  
  return 0;
}
