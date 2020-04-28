#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "types.h"
#include "constants.h"
#include "utils_host.h"

int main()
{
   shmem_t type = SHMEM_T_WITNESS_32M;
   uint32_t *shmp;
   unsigned long long nels;
   unsigned long long size;

   if (type == SHMEM_T_WITNESS_32M) {
      nels =  (1ull << 5);
   } else if (type == SHMEM_T_WITNESS_64M) { 
      nels =  (1ull << 6);
   } else{
      nels =  (1ull << 7);
   }
   size = nels * sizeof(uint32_t) * NWORDS_256BIT;


   uint32_t *buffer = (uint32_t *) malloc(size);
   const uint32_t *P = CusnarksPGet((mod_t)0);

   memset(buffer, 0, size);

   for (uint32_t i=0; i< 10; i++){
     printU256Number(&buffer[i*NWORDS_256BIT]);
   }

   // write access
   shared_new_h((void **) &shmp, size);

   printf("\nShared mem created\n");

   setRandom256(buffer, nels , P);
   memcpy(shmp, buffer, size);
   printf("\nCreate buffer\n");
   for (uint32_t i=0; i< 10; i++){
     printU256Number(&buffer[i*NWORDS_256BIT]);
   }
   printf("\nCreate shmem\n");
   for (uint32_t i=0; i< 10; i++){
     printU256Number(&shmp[i*NWORDS_256BIT]);
   }

   free(buffer);

   printf("\nCreate shmem after free buffer\n");
   for (uint32_t i=0; i< 10; i++){
     printU256Number(&shmp[i*NWORDS_256BIT]);
   }

   return 1;
}


