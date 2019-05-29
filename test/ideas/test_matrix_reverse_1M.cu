#include <stdio.h>
#include <stdlib.h>


/*
In:
0    1     2 ....    1024
1024 1025.  ...      2048 
.........................
1024*1023 .....      1024*1024-1 


xx
0 1
*/
__global__ void fft3dxx_kernel(int *out_vector_d, int *in_vector_d, int Nx1, int Nx2, int Ny1, int Ny2)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int new_ridx = ((tid%1024) / 32) * 1024;
    int new_cidx = (tid % 32) * 32 * 1024;  
    int new_kidx = (tid/1024);
    int new_ridx2, new_cidx2, new_kidx2;
    int reverse_idx;

    if (tid == 0){
       for (int i = 0; i < 2048; i++){
         reverse_idx = ((((((i % 32) * 0x802 & 0x22110) | ( (i%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff))*32*1024;
         new_ridx2 = ((i%1024) / 32) * 1024;
         new_cidx2 = (i % 32) * 32 * 1024;
         new_kidx2 = (i/1024);
         printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d, root_ridx: %d, root_cidx: %d\n",i, new_ridx2, new_cidx2, new_kidx2, reverse_idx, new_ridx2/1024, reverse_idx/(1024*32));
         printf("tid : %d, in_idx : %d, out_idx: %d, root_idx: %d\n", i, new_ridx2 + new_cidx2 + new_kidx2, reverse_idx + new_kidx2 + new_ridx2, new_ridx2/1024 * reverse_idx / (1024*32) * 1024);
       }
    }
    reverse_idx = ((((((tid % 32) * 0x802 & 0x22110) | ( (tid%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff))*32*1024;

    out_vector_d[reverse_idx + new_kidx +new_ridx] = in_vector_d[new_ridx + new_cidx + new_kidx];
}
__global__ void fft3dxy_kernel(int *out_vector_d, int *in_vector_d, int Nx1, int Nx2, int Ny1, int Ny2)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int new_ridx = ((tid/32) % 1024);
    int new_cidx = (tid % 32) * 1024;  
    int new_kidx = (tid/(1024*32) * 32*1024);
    int new_ridx2, new_cidx2, new_kidx2;
    //volatile int *in, *out;
    //in = in_vector_d;
    //out = out_vector_d;
    int reverse_idx;

    #if 0
    if (tid == 0){
       for (int i = 0; i <1024*1024; i++){
         new_ridx2 = ((i/32) % 1024); 
         new_cidx2 = (i % 32) * 1024;  
         new_kidx2 = (i/ (1024*32) * 32*1024);
         reverse_idx = (((((((new_cidx2/1024) % 32) * 0x802 & 0x22110) | ( ((new_cidx2/1024)%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff))*1024*32;
         printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d\n",i, new_ridx2, new_cidx2, new_kidx2, reverse_idx);
         printf("tid : %d, in_idx : %d, out_idx: %d\n", i, new_ridx2 + new_cidx2 + new_kidx2, reverse_idx + new_kidx2/(32)+new_ridx2);
       }
    }
    #endif
    reverse_idx = (((((((new_cidx/1024) % 32) * 0x802 & 0x22110) | ( ((tid/1024)%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff))*1024*32;

    #if 0
    if (tid == 1){
       printf("out[%d] : %d, in[%d]: %d\n", reverse_idx + new_ridx + new_kidx, 
                                            out_vector_d[reverse_idx + new_ridx + new_kidx], 
                                            new_ridx + new_cidx + new_kidx,
                                             in_vector_d[new_ridx + new_cidx + new_kidx]);
    }
    #endif
    out_vector_d[reverse_idx + new_ridx + new_kidx/32] = in_vector_d[new_ridx + new_cidx + new_kidx];
    #if 0
    //out[reverse_idx + new_ridx + new_kidx] = in[new_ridx + new_cidx + new_kidx];
    if (tid == 1){
       printf("out[%d] : %d, in[%d]: %d\n", reverse_idx + new_ridx + new_kidx, 
                                            out_vector_d[reverse_idx + new_ridx + new_kidx], 
                                            new_ridx + new_cidx + new_kidx,
                                             in_vector_d[new_ridx + new_cidx + new_kidx]);
    }
    #endif

}

__global__ void fft3dyx_kernel(int *out_vector_d, int *in_vector_d, int Nx1, int Nx2, int Ny1, int Ny2)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int new_ridx = ((tid/32) % 32);
    int new_cidx = (tid % 32) * 32;  
    int new_kidx = (tid/1024)*1024;
    int new_ridx2, new_cidx2, new_kidx2;
    int reverse_idx;

    #if 0
    if (tid == 0){
       for (int i = 0; i < 1024*1024; i++){
         reverse_idx = ((((((i % 32) * 0x802 & 0x22110) | ( (i%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff))*32;
         new_ridx2 = ((i/32) % 32);
         new_cidx2 = (i % 32) * 32;
         new_kidx2 = (i/1024)*1024;
         printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d, root_ridx: %d, root_cidx: %d\n",i, new_ridx2, new_cidx2, new_kidx2, reverse_idx,
                                                                                                         new_ridx2, reverse_idx/(32));
         printf("tid : %d, in_idx : %d, out_idx: %d, root_idx: %d\n", i, new_ridx2 + new_cidx2 + new_kidx2, reverse_idx + new_kidx2 + new_ridx2,
                                                                      new_ridx2 * reverse_idx / 32);
       }
    }
    #endif
    reverse_idx = ((((((tid % 32) * 0x802 & 0x22110) | ( (tid%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff))*32;

    out_vector_d[reverse_idx + new_kidx +new_ridx] = in_vector_d[new_ridx + new_cidx + new_kidx];
}

__global__ void fft3dyy_kernel(int *out_vector_d, int *in_vector_d, int Nx1, int Nx2, int Ny1, int Ny2)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int new_ridx = ((tid/32) * 32);
    int new_cidx = (tid) % 32;  
    int new_kidx = (tid/1024) *1024;
    int new_ridx2, new_cidx2, new_kidx2;
    //volatile int *in, *out;
    //in = in_vector_d;
    //out = out_vector_d;
    int reverse_idx;

    if (tid == 0){
       for (int i = 0; i <1024*1024; i++){
         new_cidx2 = (i  % 32);  
         new_ridx2 = (((i/32)%32) * 1024); 
         new_kidx2 = (i/ 1024);
         reverse_idx = ((((((i%32) * 0x802 & 0x22110) | ((i%32) * 0x8020 & 0x88440)) * 0x10101) >> 19) &0xff)*1024*32;
         printf("tid : %d, nr_idx: %d, nc_idx: %d, nk_idx: %d, ridx: %d\n",i, new_ridx2, new_cidx2, new_kidx2, reverse_idx);
         printf("tid : %d, in_idx : %d, out_idx: %d\n", i, i, reverse_idx + new_ridx2 + new_kidx2);
       }
    }
    reverse_idx = (((((((tid) % 32) * 0x802 & 0x22110) | ( ((tid)%32) * 0x8020 & 0x88440)) * 0x10101 >> 19) &0xff));

    #if 0
    if (tid == 1){
       printf("out[%d] : %d, in[%d]: %d\n", reverse_idx + new_ridx + new_kidx, 
                                            out_vector_d[reverse_idx + new_ridx + new_kidx], 
                                            new_ridx + new_cidx + new_kidx,
                                             in_vector_d[new_ridx + new_cidx + new_kidx]);
    }
    #endif
    out_vector_d[reverse_idx + new_ridx] = in_vector_d[tid];
    #if 0
    //out[reverse_idx + new_ridx + new_kidx] = in[new_ridx + new_cidx + new_kidx];
    if (tid == 1){
       printf("out[%d] : %d, in[%d]: %d\n", reverse_idx + new_ridx + new_kidx, 
                                            out_vector_d[reverse_idx + new_ridx + new_kidx], 
                                            new_ridx + new_cidx + new_kidx,
                                             in_vector_d[new_ridx + new_cidx + new_kidx]);
    }
    #endif

}

#define NX  (5)
#define NY  (5)
#define NX2 (10)
#define NY2 (10)
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

  printf("XXXXXXXXXXXXXXXXXXXXX\n");
  printf("In\n");
  for (j=0;j<(1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",in_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }
  printf("XXXXXXXXXXXXXXXXXXXXX\n");

  blockS =256;
  gridS = (MSIZE + blockS - 1)/blockS;
  fft3dxx_kernel<<<gridS,blockS>>>(out_vector_d, in_vector_d, NX, NX2, NY,NY2);
  cudaDeviceSynchronize();
  cudaMemcpy(out_vector_h, out_vector_d, MSIZE * sizeof(int), cudaMemcpyDeviceToHost);

  printf("XXXXXXXXXXXXXXXXXXXXX\n");
  printf("Out XX\n");
  for ( j=0;j< (1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",out_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }
  printf("XXXXXXXXXXXXXXXXXXXXX\n");

  #if 0
  cudaMemcpy(in_vector_d, out_vector_h, MSIZE * sizeof(int), cudaMemcpyHostToDevice);
  //cudaMemset(out_vector_d, 0, MSIZE*sizeof(int));
  fft3dxy_kernel<<<gridS,blockS>>>(out_vector_d, in_vector_d, NX, NX2, NY,NY2);
  cudaDeviceSynchronize();
  cudaMemcpy(out_vector_h, out_vector_d, MSIZE * sizeof(int), cudaMemcpyDeviceToHost);
  #if 0
  printf("xy\n");
  for ( j=0;j< (1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",out_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }
  #endif

  cudaMemcpy(in_vector_d, out_vector_h, MSIZE * sizeof(int), cudaMemcpyHostToDevice);
  fft3dyx_kernel<<<gridS,blockS>>>(out_vector_d, in_vector_d, NX, NX2, NY,NY2);
  cudaDeviceSynchronize();
  cudaMemcpy(out_vector_h, out_vector_d, MSIZE * sizeof(int), cudaMemcpyDeviceToHost);
  #if 0
  printf("yx\n");
  for ( j=0;j< (1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",out_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }

  #endif
  cudaMemcpy(in_vector_d, out_vector_h, MSIZE * sizeof(int), cudaMemcpyHostToDevice);
  fft3dyy_kernel<<<gridS,blockS>>>(out_vector_d, in_vector_d, NX, NX2, NY,NY2);
  cudaDeviceSynchronize();
  cudaMemcpy(out_vector_h, out_vector_d, MSIZE * sizeof(int), cudaMemcpyDeviceToHost);
  #if 0
  printf("yy\n");
  for ( j=0;j< (1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",out_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }

  #endif 

  #endif
  cudaFree(in_vector_d);
  cudaFree(out_vector_d);
  free(in_vector_h);
  free(out_vector_h);
}

