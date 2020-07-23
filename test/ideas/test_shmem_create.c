//g++ test_shmem_create.c -I ../../src/cuda/ -L ../../lib/ -lcusnarks -o create


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>

#include "types.h"
#include "constants.h"
#include "bigint.h"
#include "utils_host.h"
#include "ff.h"

int main(int argc, char *argv[])
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
     printUBINumber(&buffer[i*NWORDS_256BIT],8);
   }

   // write access
   int shmid = shared_new_h((void **) &shmp, 3*size);
   if (shmid == -1){
      return -1;
   }
   // check size

   printf("\nShared mem created\n");

   setRandomBI(buffer, nels , P, 8);
   memcpy(shmp, buffer, size);
   printf("\nCreate buffer\n");
   for (uint32_t i=0; i< 10; i++){
     printUBINumber(&buffer[i*NWORDS_256BIT],8);
   }
   printf("\nCreate shmem\n");
   for (uint32_t i=0; i< 10; i++){
     printUBINumber(&shmp[i*NWORDS_256BIT],8);
   }

   free(buffer);

   printf("\nCreate shmem after free buffer\n");
   for (uint32_t i=0; i< 10; i++){
     printUBINumber(&shmp[i*NWORDS_256BIT],8);
   }

   if (argc > 1){
     shared_free_h((void *) shmp, shmid);
   }

   return 1;
}


