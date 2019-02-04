/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU
This class will get translated into python via swig
*/

#include <bigint_kernel.cu>
#include <bigint.hh>
#include <assert.h>
#include <iostream>
using namespace std;

BigInt::BigInt (uint32_t* array_host_, uint32_t length_) {
  array_host = array_host_;
  len = length_;
  uint32_t size = len * sizeof(uint32_t);
  cudaError_t err = cudaMalloc((void**) &array_device, size);
  assert(err == 0);
  err = cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
  assert(err == 0);
}

void BigInt::mod_add() {
  BigInt_ModAdd256<<<64, 64>>>(array_device, len);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

void BigInt::retreive() {
  int size = length * sizeof(int);
  cudaMemcpy(array_host, array_device, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if(err != 0) { cout << err << endl; assert(0); }
}

BigInt::~BigInt() {
  cudaFree(array_device);
}
