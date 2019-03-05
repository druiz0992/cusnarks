#include <stdio.h>
#include <stdlib.h>

__global__ void fft3dxx_kernel(int *out_vector_d, int *in_vector_d, int Nx1, int Nx2, int Ny1, int Ny2)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int new_ridx = (tid << Ny2) & ((1 << (Nx2 + Ny2))-1);
    int new_cidx = tid >> Nx2;

    int reverse_idx = ((((((tid % 32) * 0x802 & 0x22110) | ( (tid%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff)+32*((tid%1024)/32)) ;

    out_vector_d[reverse_idx +1024*(tid/1024)] = in_vector_d[new_ridx + new_cidx];
}
__global__ void fft3dxy_kernel(int *out_vector_d, int *in_vector_d, int Nx1, int Nx2, int Ny1, int Ny2)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int new_ridx = (tid << Ny2) & ((1 << (Nx2 + Ny2))-1);
    int new_cidx = tid >> Nx2;

    int reverse_idx = ((((((tid % 32) * 0x802 & 0x22110) | ( (tid%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff)+32*((tid%1024)/32)) ;

    out_vector_d[reverse_idx +1024*(tid/1024)] = in_vector_d[tid];
}

#define NX  (5)
#define NY  (5)
#define NX2 (5)
#define NY2 (5)
#define MSIZE ( 1 << (NX2+NY2))

main()
{
  int *in_vector_d, *out_vector_d;
  int *in_vector_h, *out_vector_h;
  int i,j;
  int nrows = NY2, ncols=NX2;
  int gridS, blockS;

  in_vector_h = (int *)malloc(MSIZE * sizeof(int));
  out_vector_h = (int *)malloc(MSIZE * sizeof(int));

  for (j=0;j<(1 << ncols);j++){
     for (i=0;i< (1 << nrows); i++){
        in_vector_h[j + (i << ncols)] = j + (i<<ncols);
     }
  }

  cudaMalloc((void**) &in_vector_d, MSIZE * sizeof(int));
  cudaMalloc((void**) &out_vector_d, MSIZE * sizeof(int));

  cudaMemcpy(in_vector_d, in_vector_h, MSIZE * sizeof(int), cudaMemcpyHostToDevice);

  printf("In:\n");
  for (j=0;j<(1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",in_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }

  blockS =256;
  gridS = (MSIZE + blockS - 1)/blockS;
  fft3dxx_kernel<<<gridS,blockS>>>(out_vector_d, in_vector_d, NX, NX2, NY,NY2);
  cudaDeviceSynchronize();
  cudaMemcpy(out_vector_h, out_vector_d, MSIZE * sizeof(int), cudaMemcpyDeviceToHost);

  printf("xx\n");
  for ( j=0;j< (1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",out_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }

  cudaMemcpy(in_vector_d, out_vector_h, MSIZE * sizeof(int), cudaMemcpyHostToDevice);
  fft3dxy_kernel<<<gridS,blockS>>>(out_vector_d, in_vector_d, NX, NX2, NY,NY2);
  cudaDeviceSynchronize();
  cudaMemcpy(out_vector_h, out_vector_d, MSIZE * sizeof(int), cudaMemcpyDeviceToHost);
  printf("xy\n");
  for ( j=0;j< (1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",out_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }

  cudaMemcpy(in_vector_d, out_vector_h, MSIZE * sizeof(int), cudaMemcpyHostToDevice);
  fft3dxx_kernel<<<gridS,blockS>>>(out_vector_d, in_vector_d, NX, NX2, NY,NY2);
  cudaDeviceSynchronize();
  cudaMemcpy(out_vector_h, out_vector_d, MSIZE * sizeof(int), cudaMemcpyDeviceToHost);
  printf("yx\n");
  for ( j=0;j< (1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",out_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }

  cudaMemcpy(in_vector_d, out_vector_h, MSIZE * sizeof(int), cudaMemcpyHostToDevice);
  fft3dxy_kernel<<<gridS,blockS>>>(out_vector_d, in_vector_d, NX, NX2, NY,NY2);
  cudaDeviceSynchronize();
  cudaMemcpy(out_vector_h, out_vector_d, MSIZE * sizeof(int), cudaMemcpyDeviceToHost);
  printf("yy\n");
  for ( j=0;j< (1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",out_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }


  cudaFree(in_vector_d);
  cudaFree(out_vector_d);
  free(in_vector_h);
  free(out_vector_h);
}

