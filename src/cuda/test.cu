#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "bigint.h"
#include "bigint_device.h"

#define N_ELEMS (1000)

main()
{
  int i, j;

  uint32_t *a =(uint32_t *) malloc(N_ELEMS * sizeof(uint32_t *) * NWORDS_256BIT);
  uint32_t *r =(uint32_t *) malloc(N_ELEMS/2 * sizeof(uint32_t *) * NWORDS_256BIT);
  uint32_t p[] = {7,0,0,0,0,0,0,0};

  for (i=0; i < N_ELEMS; i++){
    for (j=0; j< NWORDS_256BIT; j++){
       a[i*NWORDS_256BIT + j] = (uint32_t)rand();
    }
  }

  BigInt *bn = new BigInt(a, p, N_ELEMS);
  bn->addm();
  bn->retrieve(r);
  for (i=0; i < N_ELEMS; i++){
     printf("0x");
    for (j=0; j< NWORDS_256BIT; j++){
       printf("%x",r[i]);
    }
    printf("\n");
  }
}
