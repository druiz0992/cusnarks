#include <stdio.h>
#include <stdlib.h>

//#define NX (10)
//#define NY (10)
//#define NXX  (5)
//#define NYX  (5)

#define NX (9)
#define NY (7)
#define NXX  (5)
#define NXY ((NX) - (NXX))
#define NYX  (4)
#define NYY  ((NY) - (NYX))
#define MSIZE ( 1 << (NX+NY))

#define BASE (19)
#define BASE32 (5)
#define MOD255 (0xff)

/*
In:
0    1     2 ....    1024
1024 1025.  ...      2048 
.........................
1024*1023 .....      1024*1024-1 


xx
0 1
*/

void fft3d2xx_kernel(int *out_vector_h, int *in_vector_h, int Nx, int Ny, int Nxx)
{
   int reverse_idx, new_ridx, new_cidx, new_kidx;
   const int Nxn = (1 << Nx) - 1;
   const int Nxxxn = (1 << (Nx-Nxx)) - 1;
   const int Nxxn = (1 << Nxx) - 1;
   const int base = (BASE + BASE32 - Nxx );
   const int base2 = (MOD255 >> (BASE32-Nxx));

   for (int i = 0; i < (1 << (Nx + Ny)); i++){
      reverse_idx = ((((((i & Nxxn) * 0x802 & 0x22110) | ( (i & Nxxn) * 0x8020 & 0x88440)) * 0x10101 >> base) & base2)) << (Ny + Nx - Nxx);
      new_cidx = ((i & Nxxn) << (Ny + Nx - Nxx));
      new_ridx = (((i >> Nxx) & Nxxxn) << Ny);
      new_kidx = (i >> Nx);
      //if (i < 2048){
        printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d, root_ridx: %d, root_cidx: %d, root_idx: %d\n",
                               i, new_ridx, new_cidx, new_kidx, reverse_idx, 
                               new_ridx >> Ny, reverse_idx >> (Ny + Nx - Nxx), 
                               ((new_ridx >> Ny) * (reverse_idx >> (Ny + Nx - Nxx)) ) << Ny);
        //printf("tid : %d, in_idx : %d, out_idx: %d, root_idx: %d\n",
                               //i, new_ridx + new_cidx + new_kidx, 
                               //reverse_idx + new_kidx + new_ridx, 
                               //(new_ridx >> Ny) * (reverse_idx >> (Ny + Nx - Nxx)) << Nx);
      //}

      out_vector_h[reverse_idx + new_kidx +new_ridx] = in_vector_h[new_cidx + new_kidx + new_ridx];
    }

}

void fft3d2xy_kernel(int *out_vector_h, int *in_vector_h, int Nx, int Ny, int Nxy)
{
  int reverse_idx, new_ridx, new_cidx, new_kidx;

   const int Nyn = (1 << Ny) - 1;
   const int Nxxyn = (1 << (Nx-Nxy)) - 1;
   const int Nxyn = (1 << Nxy) - 1;
   const int base = (BASE + BASE32 - Nxy );
   const int base2 = (MOD255 >> (BASE32-Nxy));

   #if 0
   int in[1024*64];
   int out[1024*64];
   for (int i=0; i< 1024*64; i++){
     in[i] = 0;
     out[i] = 0;
   }
   #endif

   for (int i = 0; i < (1 << (Nx + Ny)); i++){
      //reverse_idx = ((((((i & Nxxxn) * 0x802 & 0x22110) | ( (i & Nxxxn) * 0x8020 & 0x88440)) * 0x10101 >> base) & 0x7F)) << Ny;
      //new_ridx = ((i & Nxn) >> (Ny-Nxx) << Ny);
      //new_cidx = (i & Nxxn) << Ny;
      //new_kidx = (i >> Nx);
  //int new_ridx = ((tid / 32 )%1024);
  //int new_cidx = (tid % 32) * 1024;  
  //int new_kidx = (tid/(1024*32) * 32 * 1024);
      new_cidx = (i & Nxyn) << Ny;  
      new_ridx = ((i >> Nxy )& Nyn);
      new_kidx = (i >> (Nxy + Ny) << (Nxy + Ny));
      reverse_idx = (((((((i&Nxyn) ) * 0x802 & 0x22110) | ( ((i&Nxyn)) * 0x8020 & 0x88440)) * 0x10101 >> base) & base2))<<(Nx-Nxy+Ny);
        printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d, root_ridx: %d, root_cidx: %d, root_idx: %d\n",
                               i, new_ridx, new_cidx, new_kidx, reverse_idx,
                               //new_ridx >> Ny, reverse_idx >>Ny);
                               (reverse_idx + (new_kidx>>(Nxy)) + new_ridx)>>Ny ,
                              ((reverse_idx + (new_kidx>>(Nxy)) + new_ridx)&Nyn),
                               ((reverse_idx + (new_kidx>>(Nxy)) + new_ridx)>>Ny ) *
                              ((reverse_idx + (new_kidx>>(Nxy)) + new_ridx)&Nyn));
        //printf("tid : %d, in_idx : %d, out_idx: %d\n",
                               //i, new_cidx + new_ridx + new_kidx,
                               //reverse_idx + new_ridx + (new_kidx >> (Nxy)));
      //}

      out_vector_h[reverse_idx + new_ridx + (new_kidx>>(Nxy))] = in_vector_h[new_cidx + new_ridx + new_kidx];
#if 0
      if (out[reverse_idx + new_ridx + (new_kidx>>(Nxy))] > 0){
        printf("error out\n");
        
      } 
      if (in[new_cidx + new_ridx + new_kidx] > 0){
        printf("error in\n");
      }
      out[reverse_idx + new_ridx + (new_kidx>>(Nxy))]++;
      in[new_cidx + new_ridx + new_kidx]++;
#endif
    }

#if 0
    printf("\nPending In\n");
    int p=1;
    for (int i=0; i< 1024*64; i++){
       if (in[i] == 0) { printf("%d ",i);p++;}
       if (p%100 == 0) printf("\n");
    }
    printf("\nPending Out\n");
    p=1;
    for (int i=0; i< 1024*64; i++){
       if (out[i] == 0) { printf("%d ",i);p++;}
       if (p%100 == 0) printf("\n");
    }
    printf("\nRepeated In\n");
    p=1;
    for (int i=0; i< 1024*64; i++){
       if (in[i] > 1) { printf("%d(%d) ",i,in[i]);p++;}
       if (p%100 == 0) printf("\n");
    }
    printf("\nRepeated Out\n");
    p=1;
    for (int i=0; i< 1024*64; i++){
       if (out[i] > 1) { printf("%d(%d) ",i,out[i]);p++;}
       if (p%100 == 0) printf("\n");
    }

    printf("\n");
#endif

}

void fft3d2yx_kernel(int *out_vector_h, int *in_vector_h, int Nx, int Ny, int Nyx)
{
   int new_ridx,  new_cidx,  new_kidx, reverse_idx;
   const int Nyn = (1 << Ny) - 1;
   const int Nyyxn = (1 << (Ny-Nyx)) - 1;
   const int Nyxn = (1 << Nyx) - 1;
   const int base = (BASE + BASE32 - Nyx );
   const int base2 = (MOD255 >> (BASE32-Nyx));

   #if 0
   int in[1024*64];
   int out[1024*64];
   for (int i=0; i< 1024*64; i++){
     in[i] = 0;
     out[i] = 0;
   }
   #endif

   for (int i = 0; i < (1 << (Nx + Ny)); i++){

     new_cidx = (i & Nyxn ) << (Ny - Nyx);  
     new_ridx = ((i >> Nyx) & Nyyxn);
     new_kidx = ((i >> Ny) << Ny);
     reverse_idx = ((((((i&Nyxn) * 0x802 & 0x22110) | ( (i&Nyxn) * 0x8020 & 0x88440)) * 0x10101 >> base) & base2)) << (Ny - Nyx);
     printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d, root_ridx: %d, root_cidx: %d, Root_idx: %d\n",
               i, new_ridx, new_cidx, new_kidx, reverse_idx, new_ridx, reverse_idx >>(Ny - Nyx),
               (new_ridx * (reverse_idx >>(Ny - Nyx))) << Nx);
     printf("tid : %d, in_idx : %d, out_idx: %d, root_idx: %d\n",
        i, new_ridx + new_cidx + new_kidx, reverse_idx + new_kidx + new_ridx, (new_ridx * (reverse_idx >> (Ny - Nyx))) << Nx);

     out_vector_h[reverse_idx + new_kidx +new_ridx] = in_vector_h[new_cidx + new_kidx + new_ridx];

      #if 0
      if (out[reverse_idx + new_ridx + new_kidx] > 0){
        printf("error out\n");
        
      } 
      if (in[new_cidx + new_ridx + new_kidx] > 0){
        printf("error in\n");
      }
      out[reverse_idx + new_ridx + new_kidx]++;
      in[new_cidx + new_ridx + new_kidx]++;
      #endif
    }

    #if 0
    printf("\nPending In\n");
    int p=1;
    for (int i=0; i< 1024*64; i++){
       if (in[i] == 0) { printf("%d ",i);p++;}
       if (p%100 == 0) printf("\n");
    }
    printf("\nPending Out\n");
    p=1;
    for (int i=0; i< 1024*64; i++){
       if (out[i] == 0) { printf("%d ",i);p++;}
       if (p%100 == 0) printf("\n");
    }
    printf("\nRepeated In\n");
    p=1;
    for (int i=0; i< 1024*64; i++){
       if (in[i] > 1) { printf("%d(%d) ",i,in[i]);p++;}
       if (p%100 == 0) printf("\n");
    }
    printf("\nRepeated Out\n");
    p=1;
    for (int i=0; i< 1024*64; i++){
       if (out[i] > 1) { printf("%d(%d) ",i,out[i]);p++;}
       if (p%100 == 0) printf("\n");
    }

    printf("\n");
     #endif

}

void fft3d2yy_kernel(int *out_vector_h, int *in_vector_h, int Nx, int Ny, int Nyy)
{
    int new_ridx,  new_cidx, new_kidx, reverse_idx;
    const int Nyn = (1 << Ny) - 1;
    const int Nxyyn = (1 << (Nx-Nyy)) - 1;
    const int Nyyn = (1 << Nyy) - 1;
    const int Nyyyn = (1 << (Ny-Nyy)) - 1;
    const int base = (BASE + BASE32 - Nyy );
    const int base2 = (MOD255 >> (BASE32-Nyy));

   int out[1024*64];
   for (int i=0; i< 1024*64; i++){
     out[i] = 0;
   }

    for (int i = 0; i < (1 << (Nx + Ny)); i++){
      new_cidx = (i & Nyyn);  
      new_ridx = ((i >>  Nyy) & Nyyyn) << Nx;
      new_kidx = (i >> Ny);
      reverse_idx = ((((((i&Nyyn) * 0x802 & 0x22110) | ((i&Nyyn) * 0x8020 & 0x88440)) * 0x10101) >> base) &base2)<<(Nx + Ny - Nyy);

      printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d\n",
              i, new_ridx, new_cidx, new_kidx, reverse_idx);
      //printf("tid : %d, in_idx : %d, out_idx: %d\n",
             //i, i, reverse_idx + new_ridx + new_kidx);
      out_vector_h[reverse_idx + new_kidx +new_ridx] = in_vector_h[i];

      if (out[reverse_idx + new_ridx + new_kidx] > 0){
        printf("error out\n");
        
      } 
      out[reverse_idx + new_ridx + new_kidx]++;
   }
    printf("\nPending Out\n");
    int p=1;
    for (int i=0; i< 1024*64; i++){
       if (out[i] == 0) { printf("%d ",i);p++;}
       if (p%100 == 0) printf("\n");
    }
    printf("\nRepeated Out\n");
    p=1;
    for (int i=0; i< 1024*64; i++){
       if (out[i] > 1) { printf("%d(%d) ",i,out[i]);p++;}
       if (p%100 == 0) printf("\n");
    }

    printf("\n");
}


void fft3dxx_kernel(int *out_vector_h, int *in_vector_h, int Nx, int Ny, int Nxx)
{
   int reverse_idx, new_ridx, new_cidx, new_kidx;
   Nx = 10;
   Ny = 10;
   Nxx = 5;

   for (int i = 0; i < (1 << (Nx + Ny)); i++){
      reverse_idx = ((((((i % 32) * 0x802 & 0x22110) | ( (i%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff))*32*1024;
      new_ridx = ((i%1024) / 32) * 1024;
      new_cidx = (i % 32) * 32 * 1024;
      new_kidx = (i/1024);
        printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d, root_ridx: %d, root_cidx: %d\n",
                               i, new_ridx, new_cidx, new_kidx, reverse_idx,
                               new_ridx/1024, reverse_idx/(1024*32));
        //printf("tid : %d, in_idx : %d, out_idx: %d, root_idx: %d\n",
                               //i, new_ridx + new_cidx + new_kidx, 
                               //reverse_idx + new_kidx + new_ridx, 
                               //new_ridx/1024 * reverse_idx / (1024*32) * 1024);

      out_vector_h[reverse_idx + new_kidx +new_ridx] = 
                       in_vector_h[new_ridx + new_cidx + new_kidx];
    }

}
void fft3dxy_kernel(int *out_vector_h, int *in_vector_h, int Nx, int Ny, int Nxy)
{
   int new_ridx,new_cidx, new_kidx;
   int reverse_idx;

   Nx = 10;
   Ny = 10;
   Nxy = 5;

   for (int i = 0; i < (1 << (Nx + Ny)); i++){
    new_ridx = ((i/32) % 1024);
    new_cidx = (i % 32) * 1024;  
    new_kidx = (i/(1024*32) * 32*1024);
    reverse_idx = (((((((new_cidx/1024) % 32) * 0x802 & 0x22110) | ( ((new_cidx/1024)%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff))*1024*32;
    printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d, root_ridx: %d, root_cidx: %d\n",
         i, new_ridx, new_cidx, new_kidx, reverse_idx,
                               (reverse_idx + new_kidx/32 + new_ridx)/1024,
                               (reverse_idx + new_kidx/32 + new_ridx)%1024);
    //printf("tid : %d, in_idx : %d, out_idx: %d\n",
           //i, new_ridx + new_cidx + new_kidx, reverse_idx + new_kidx/(32)+new_ridx);

    out_vector_h[reverse_idx + new_ridx + new_kidx/32] = in_vector_h[new_ridx + new_cidx + new_kidx];
  }

}


void fft3dyx_kernel(int *out_vector_h, int *in_vector_h, int Nx, int Ny, int Nyx)
{
   Nx = 10;
   Ny = 10;
   Nyx = 5;
  
   int new_ridx,  new_cidx,  new_kidx, reverse_idx;

   for (int i = 0; i < (1 << (Nx + Ny)); i++){

     new_ridx = ((i/32) % 32);
     new_cidx = (i % 32) * 32;  
     new_kidx = (i/1024)*1024;
     reverse_idx = ((((((i % 32) * 0x802 & 0x22110) | ( (i%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff))*32;
     printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d, root_ridx: %d, root_cidx: %d\n",
               i, new_ridx, new_cidx, new_kidx, reverse_idx, new_ridx, reverse_idx/(32));
     //printf("tid : %d, in_idx : %d, out_idx: %d, root_idx: %d\n",
        //i, new_ridx + new_cidx + new_kidx, reverse_idx + new_kidx + new_ridx, new_ridx * reverse_idx / 32);

     out_vector_h[reverse_idx + new_kidx +new_ridx] = in_vector_h[new_ridx + new_cidx + new_kidx];
    }

}

void fft3dyy_kernel(int *out_vector_h, int *in_vector_h, int Nx, int Ny, int Nyy)
{
    Nx = 10;
    Ny = 10;
    Nyy = 5;
    int new_ridx,  new_cidx, new_kidx, reverse_idx;

    for (int i = 0; i < (1 << (Nx + Ny)); i++){
      new_cidx = (i % 32);  
      new_ridx = (((i / 32 ) % 32)* 1024);
      new_kidx = (i/1024);
      reverse_idx = ((((((i%32) * 0x802 & 0x22110) | ((i%32) * 0x8020 & 0x88440)) * 0x10101) >> 19) &0xff)*1024*32;
      //reverse_idx = i%32 * 1024 * 32;

      printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d\n",
              i, new_ridx, new_cidx, new_kidx, reverse_idx);
      //printf("tid : %d, in_idx : %d, out_idx: %d\n",
             //i, i, reverse_idx + new_ridx + new_kidx);
      out_vector_h[reverse_idx + new_kidx +new_ridx] = in_vector_h[i];
   }
}


main()
{
  int *in_vector_h, *out_vector_h;
  int i,j;
  int nrows = NY, ncols=NX;
  int gridS, blockS;

  in_vector_h = (int *)malloc(MSIZE * sizeof(int));
  out_vector_h = (int *)malloc(MSIZE * sizeof(int));

  for (i=0;i<(1 << nrows);i++){
     for (j=0;j< (1 << ncols); j++){
        in_vector_h[j + (i << ncols)] = j + (i<<ncols);
        out_vector_h[j + (i << ncols)] = 0;
     }
  }

  printf("XXXXXXXXXXXXXXXXXXXXX\n");
  printf("In\n");
  for (i=0;i<(1 << nrows);i++){
     for (j=0;j< (1 << ncols); j++){
        printf("%d ",in_vector_h[j+(i<<ncols)]);
     }
     printf("\n");
  }
  printf("XXXXXXXXXXXXXXXXXXXXX\n");

  fft3d2xx_kernel(out_vector_h, in_vector_h, NX, NY, NXX);

  printf("XXXXXXXXXXXXXXXXXXXXX\n");
  printf("Out XX\n");
  for ( i=0;i< (1 << nrows);i++){
     for (j=0;j< (1 << ncols); j++){
        printf("%d ",out_vector_h[j+(i<<ncols)]);
     }
     printf("\n");
  }
  printf("XXXXXXXXXXXXXXXXXXXXX\n");

  fft3d2xy_kernel(out_vector_h, in_vector_h, NX, NY, NXY);
  printf("XXXXXXXXXXXXXXXXXXXXX\n");
  printf("Out XY\n");
  for ( i=0;i< (1 << nrows);i++){
     for (j=0;j< (1 << ncols); j++){
        printf("%d ",out_vector_h[j+(i<<ncols)]);
     }
     printf("\n");
  }
  printf("XXXXXXXXXXXXXXXXXXXXX\n");

  fft3d2yx_kernel(out_vector_h, in_vector_h, NX, NY, NYX);

  printf("XXXXXXXXXXXXXXXXXXXXX\n");
  printf("Out YX\n");
  for ( i=0;i< (1 << nrows);i++){
     for (j=0;j< (1 << ncols); j++){
        printf("%d ",out_vector_h[j+(i<<ncols)]);
     }
     printf("\n");
  }
  printf("XXXXXXXXXXXXXXXXXXXXX\n");
  fft3d2yy_kernel(out_vector_h, in_vector_h, NX, NY, NYY);

  printf("XXXXXXXXXXXXXXXXXXXXX\n");
  printf("Out YY\n");
  for ( i=0;i< (1 << nrows);i++){
     for (j=0;j< (1 << ncols); j++){
        printf("%d ",out_vector_h[j+(i<<ncols)]);
     }
     printf("\n");
  }
  printf("XXXXXXXXXXXXXXXXXXXXX\n");

  free(in_vector_h);
  free(out_vector_h);
}

