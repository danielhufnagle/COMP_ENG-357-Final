// main.cu
#include <stdio.h>

// A very simple GPU kernel
__global__ void hello_from_gpu() {
    printf("Hello from thread %d!\n", threadIdx.x);
}

int main() {
    // Launch the kernel with 1 block of 5 threads
    hello_from_gpu<<<1, 5>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}
