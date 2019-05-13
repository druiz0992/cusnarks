#include <stdio.h>
#include <stdlib.h>

__global__ void memory_init_kernel(int *out_vector_d, int *in_vector_d, int len)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid > len){
      return;
    }
    out_vector_d[tid] = in_vector_d[tid];
}
__global__ void memory_check_kernel(int *out_vector_d, int *in_vector_d, int len)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (tid > len){
    return;
  }

  if (out_vector_d[tid] != in_vector_d[tid]) {
    printf("Error\n");
  }
}

#define NROWS 8
#define NCOLS 4
#define MSIZE ( NROWS * NCOLS)

main()
{
  int *in_vector_d, *out_vector_d;
  int *in_vector_d2, *out_vector_d2;
  int in_vector_h[MSIZE], out_vector_h[MSIZE];
  int i,j;
  int nrows = NROWS, ncols=NCOLS;

  for (i=0; i< MSIZE; i++){
    in_vector_h[i] = i;
    out_vector_h[i] = 0;
  }
 
  cudaMalloc((void**) &in_vector_d, MSIZE * sizeof(int));
  cudaMalloc((void**) &out_vector_d, MSIZE * sizeof(int));

  cudaMemcpy(in_vector_d, in_vector_h, MSIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(out_vector_d, out_vector_h, MSIZE * sizeof(int), cudaMemcpyHostToDevice);

  printf("In:\n");
  for (j=0;j< MSIZE;j++){
     printf("%d ",in_vector_h[j]);
  }
  printf("\n");
  printf("Out\n");
  for ( j=0;j< MSIZE;j++){
        printf("%d ",out_vector_h[j]);
  }
  printf("\n");

  memory_init_kernel<<<1, MSIZE>>>(out_vector_d, in_vector_d, MSIZE);
  cudaDeviceSynchronize();

  printf("In:\n");
  for (j=0;j< MSIZE;j++){
     printf("%d ",in_vector_h[j]);
  }
  printf("\n");
  printf("Out\n");
  for ( j=0;j< MSIZE;j++){
        printf("%d ",out_vector_h[j]);
  }
  printf("\n");

  memory_check_kernel<<<1, MSIZE>>>(out_vector_d, in_vector_d, MSIZE);
  cudaDeviceSynchronize();
  cudaMemcpy(out_vector_h, out_vector_d, MSIZE * sizeof(int), cudaMemcpyDeviceToHost);

  printf("In:\n");
  for (j=0;j< MSIZE;j++){
     printf("%d ",in_vector_h[j]);
  }
  printf("\n");
  printf("Out\n");
  for ( j=0;j< MSIZE;j++){
        printf("%d ",out_vector_h[j]);
  }
  printf("\n");

  printf("--------------------------------------\n");
  memory_check_kernel<<<1, MSIZE>>>(out_vector_d2, in_vector_d2, MSIZE);
  cudaDeviceSynchronize();
  cudaMemcpy(out_vector_h, out_vector_d, MSIZE * sizeof(int), cudaMemcpyDeviceToHost);

  printf("In:\n");
  for (j=0;j< MSIZE;j++){
     printf("%d ",in_vector_h[j]);
  }
  printf("\n");
  printf("Out\n");
  for ( j=0;j< MSIZE;j++){
        printf("%d ",out_vector_h[j]);
  }
  printf("\n");


  cudaFree(in_vector_d);
  cudaFree(out_vector_d);
}

