
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "types.h"
#include "utils_host.h"

const char input_file[] = "../../circuits/prove-kyc_pk.bin";
int main()
{
   uint32_t *mpoly = (uint32_t *) calloc(25999272/4 , sizeof(uint32_t));
   readU256PKFile_h(mpoly, input_file,0);

   uint32_t polsA_nWords = mpoly[10+8+8] ;
   uint32_t *polsA = &mpoly[10+8+8+8];
   uint32_t reduce_coeff = 0;
   uint32_t m = 1<<16;
   uint32_t nVars = 41050;
   uint32_t pidx=1;
   mpoly_eval_t *args = (mpoly_eval_t *) malloc(sizeof(mpoly_eval_t));

   uint32_t *scl = (uint32_t *)calloc(nVars , NWORDS_256BIT * sizeof(uint32_t));
   uint32_t *mpoly_out = (uint32_t *) calloc(m , NWORDS_256BIT * sizeof(uint32_t));
   uint32_t i;
   for (i=0; i< nVars*NWORDS_256BIT; i++){
     scl[i] = (uint32_t)rand();
     if (i%NWORDS_256BIT == 7){
        scl[i] = 0;
     }
   }     
   args->pout = mpoly_out; 
   args->scalar = scl;
   args->pin = polsA;
   args->reduce_coeff = reduce_coeff;
   args->start_idx = 0;
   args->last_idx = nVars; 
   args->max_threads= 0;
   args->pidx = pidx;        

   mpoly_eval_h(args);

   free(mpoly);
   free(mpoly_out);
   free(scl);
   free(args);

}
