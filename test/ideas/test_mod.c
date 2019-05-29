
#include <stdio.h>

main(){
  int Nx = 5;
  int Ny = 5;
  int Nxyn = (1 << (Nx+Ny)) - 1;

  int i;
  int r;

  for (i=0; i < 1<<(Nx+Ny); i++){
     r = i;
     r *= -1;
     r &= Nxyn;
     
     printf("%d - %d\n",i, r);
  }
}
 

  

