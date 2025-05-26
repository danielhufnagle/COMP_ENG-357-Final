#include <cuda_runtime.h>
#include "gpu_gate_sizing.h"

__global__ void compute_gate_delay_kernel(float* delays, int* gate_types, float* input_slews, int num_gates) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_gates) {
    int type = gate_types[i];
    float slew = input_slews[i];
    float a = 0.02f * type;
    float b = 0.01f * type;
    delays[i] = a + b * sqrtf(slew);
  }
}

void GpuGateSizing::runGateSizingGPU(std::vector<GateData>& gates, std::vector<float>& delays) {
  int num_gates = gates.size();
  int* d_gate_types;
  float* d_input_slews;
  float* d_delays;

  std::vector<int> gate_types(num_gates);
  std::vector<float> input_slews(num_gates);
  delays.resize(num_gates);

  for (int i = 0; i < num_gates; ++i) {
    gate_types[i] = gates[i].gate_type;
    input_slews[i] = gates[i].input_slew;
  }

  cudaMalloc(&d_gate_types, num_gates * sizeof(int));
  cudaMalloc(&d_input_slews, num_gates * sizeof(float));
  cudaMalloc(&d_delays, num_gates * sizeof(float));

  cudaMemcpy(d_gate_types, gate_types.data(), num_gates * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_slews, input_slews.data(), num_gates * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (num_gates + blockSize - 1) / blockSize;
  compute_gate_delay_kernel<<<numBlocks, blockSize>>>(d_delays, d_gate_types, d_input_slews, num_gates);

  cudaMemcpy(delays.data(), d_delays, num_gates * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_gate_types);
  cudaFree(d_input_slews);
  cudaFree(d_delays);
}
