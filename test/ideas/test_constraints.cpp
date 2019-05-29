#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "utils_host.h"


const char input_file[] = "../../data/prove-kyc.bin";
int main()
{
  uint32_t *constraints;
  uint32_t *poly = NULL;
  cirbin_hfile_t header;
  readU256CircuitFileHeader_h(&header, input_file);
  uint32_t pwords= MAX_R1CSPOLY_NWORDS;
  int ret_val;

  constraints = (uint32_t *)calloc(header.nWords, sizeof(uint32_t));
  readU256CircuitFile_h(constraints, input_file,0);

  poly = (uint32_t *)calloc(pwords*NWORDS_256BIT, sizeof(uint32_t));
  ret_val = r1cs_to_mpoly_h(poly,&constraints[CIRBIN_H_N_OFFSET], &header,1); 
  if (ret_val < 0) {
    printf("error. Increase MAX_R1CSPOLY_NWORDS value to %d\n",-ret_val);
  } else if (ret_val > 0) {
     printf("error. Increase MAX_R1CSPOLYTMP_NWORDS value\n");
  } else {
    printf("OK\n");
    free(constraints);
    free(poly);
  }
  
  return 0;
}
