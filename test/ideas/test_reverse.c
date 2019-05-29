#include <stdio.h>

main()
{
  int d_in[32];
  int d_out[32];
  int i, tmp;

  for (i=0; i < 32; i++){
     d_in[i] = i;
     d_out[i] = ((((i * 0x802 & 0x22110) | ( i * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff);
     printf("in: %02x/%d -> Out : %02x/%d\n",d_in[i],d_in[i], d_out[i], d_out[i]);
  }

  
}
