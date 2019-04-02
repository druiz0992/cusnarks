
#include <stdio.h>
#include "types.h"
#include "zpoly_host.h"

main()
{
  CUZPolySps *p = new CUZPolySps(20);
  p->rand(23);
  //p->show();

  uint32_t n_coeff = p->getNCoeff();
  //printf("N Coeff : %d\n",n_coeff);

  uint32_t *p_sps = p->getZPolySpsRep();
  uint32_t *coeff = p->getDCoeff();
  uint32_t i,j;

  /*
  printf ("N Coeff : %d\n",p_sps[1]);
  for (i=2; i< n_coeff+2;i++){
     printf("C[%u] : ",coeff[i-2]);
    for(j=0;j<NWORDS_256BIT;j++){
      printf("%u ",p_sps[2+(i-2)*NWORDS_256BIT+j]);
    }
    printf("\n");
  }
  */

  delete p;

  #if 0
  uint32_t n_zpoly = 20;
  CUZPolySps **p_arr = new CUZPolySps *[n_zpoly];
  CUZPolySpsArray *P;
 
  for (i = 0; i < n_zpoly; i++){
    p_arr[i] = new CUZPolySps(i+5);
    p_arr[i]->rand(23);  
  }

  P = new CUZPolySpsArray(p_arr, n_zpoly);

  /*
  for (i=0; i < n_zpoly; i++){
    P->show(i);
  }
  */
  #endif
}
