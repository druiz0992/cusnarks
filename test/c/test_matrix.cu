#include <stdio.h>
#include <stdlib.h>

__global__ void matrix_kernel(int *out_vector_d, int *in_vector_d, int nrows, int ncols)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int new_ridx = tid >> nrows; 
    int new_cidx = (tid << ncols) & ((1 << (nrows + ncols))-1);

    // 0 4 8  12 16 20 24 28 
    // 1 5 9  13 17 21 25 29
    // 2 6 10 14 18 22 26 30
    // 3 7 11 15 19 23 27 31

    //out_vector_d[new_ridx + new_cidx] = in_vector_d[tid];
    // 2^NROWS FFT of 2^NCOLS points indexed by tid
    out_vector_d[tid] = in_vector_d[(tid << nrows & ((1 << (nrows + ncols))-1 )) + (tid >> ncols)];

    // 2^NCOLS FFT of 2^NROWS points indexed by tid
    //out_vector_d[tid] = in_vectod_d[tid]

    //output as 
    //out_vector_d[tid] = in_vector_d[(tid << nrows & ((1 << (nrows + ncols))-1 )) + (tid >> ncols)];

    

}

#define NROWS 5
#define NCOLS 5
#define MSIZE ( 1 << (NROWS+NCOLS))

main()
{
  int *in_vector_d, *out_vector_d;
  int in_vector_h[MSIZE], out_vector_h[MSIZE];
  int i,j;
  int nrows = NROWS, ncols=NCOLS;

  for (i=0; i< (1 << (ncols+nrows)); i++){
    in_vector_h[i] = i;
  }
 

  cudaMalloc((void**) &in_vector_d, MSIZE * sizeof(int));
  cudaMalloc((void**) &out_vector_d, MSIZE * sizeof(int));

  cudaMemcpy(in_vector_d, in_vector_h, MSIZE * sizeof(int), cudaMemcpyHostToDevice);

  matrix_kernel<<<1,MSIZE>>>(out_vector_d, in_vector_d, nrows, ncols);
  cudaDeviceSynchronize();
  cudaMemcpy(out_vector_h, out_vector_d, MSIZE * sizeof(int), cudaMemcpyDeviceToHost);

  printf("In:\n");
  for (j=0;j<(1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",in_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }
  printf("Out\n");
  for ( j=0;j< (1 << nrows);j++){
     for (i=0;i< (1 << ncols); i++){
        printf("%d ",out_vector_h[i+(j<<ncols)]);
     }
     printf("\n");
  }

  cudaFree(in_vector_d);
  cudaFree(out_vector_d);
}

