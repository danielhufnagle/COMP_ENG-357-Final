#include <iostream>
// header for cuda
#include <cuda_runtime.h>

// CUDA kernel function that runs on the GPU
__global__ void hello() {
  printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
  // print from CPU
  std::cout << "Hello from CPU" << std::endl;
  
  // launch GPU kernel
  // <<<num blocks, threads per block>>>
  hello<<<2, 2>>>();

  // wait for GPU to finish
  cudaDeviceSynchronize();

  return 0;
}
