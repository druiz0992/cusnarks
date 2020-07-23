//g++ test_shmem_read.c -I ../../src/cuda/ -L ../../lib/ -lcusnarks -o read 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "types.h"
#include "constants.h"
#include "utils_host.h"
#include "bigint.h"

int main(int argc, char *argv[])
{
   uint32_t *buffer;
   shmem_t type = SHMEM_T_WITNESS_32M;
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

   printf("\nClient Create shmem before\n");
   for (uint32_t i=0; i< 10; i++){
     printUBINumber(&buffer[i*NWORDS_256BIT],8);
   }

   int32_t shmid = shared_get_h((void **) &buffer, size);
   if (shmid == -1)
   {
     printf("Error");
   }

   printf("\nClient Create shmem\n");
   for (uint32_t i=0; i< 10; i++){
     printUBINumber(&buffer[i*NWORDS_256BIT],8);
   }

   if (argc > 1){
     shared_free_h((void *) buffer, shmid);
   }

   return 1;
}


