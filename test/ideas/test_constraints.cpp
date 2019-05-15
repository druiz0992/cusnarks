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
  uint32_t pwords= 9000;
  uint32_t ret_val;

  constraints = (uint32_t *)calloc(header.nWords, sizeof(uint32_t));
  readU256CircuitFile_h(constraints, input_file,0);

  do {
  printf("Iteration : %d\n",pwords);
  poly = (uint32_t *)calloc(pwords*(NWORDS_256BIT+1)+1, sizeof(uint32_t));
  ret_val = r1cs_to_zpoly_h(poly,&constraints[CIRBIN_H_N_OFFSET], &header,pwords,1); 
  if (ret_val == 0) {
    free (poly);
  }
  pwords+=pwords;
  } while (ret_val == 0);

  free(constraints);
  free(poly);
  
  return 0;
}
