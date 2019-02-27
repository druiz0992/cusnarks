#include <stdio.h>
#include <stdlib.h>

__global__ void shl_xor_kernel(int *out_vector_d, int *in_vector_d, int lane)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    //if (tid % 2 == 0){
       out_vector_d[tid] = __shfl_xor_sync(0xffffffff, in_vector_d[tid], lane);
    //} else {
       //out_vector_d[tid] = __shfl_xor_sync(0xffffffff, in_vector_d[tid], lane);
    //}

}

main()
{
  int *in_vector_d, *out_vector_d;
  int in_vector_h[32], out_vector_h[32];
  int i;

  for (i=0; i< 32; i++){
    in_vector_h[i] = i;
  }
 

  cudaMalloc((void**) &in_vector_d, 32 * sizeof(int));
  cudaMalloc((void**) &out_vector_d, 32 * sizeof(int));

  cudaMemcpy(in_vector_d, in_vector_h, 32 * sizeof(int), cudaMemcpyHostToDevice);

  for (int j=0;j<5;j++){
     shl_xor_kernel<<<1,32>>>(out_vector_d, in_vector_d, 1<<j);

     cudaDeviceSynchronize();
     cudaMemcpy(out_vector_h, out_vector_d, 32 * sizeof(int), cudaMemcpyDeviceToHost);
     printf("In:\t");
     for (i=0;i< 32; i++){
        printf("%d ",in_vector_h[i]);
     }
     printf("\n");

     printf("Out[%d]:\t", 1<<j);
     for (i=0;i< 32; i++){
        printf("%d ",out_vector_h[i]);
     }
     printf("\n");
  }

  cudaFree(in_vector_d);
  cudaFree(out_vector_d);
}

