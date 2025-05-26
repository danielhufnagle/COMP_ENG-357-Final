// cuda_gate_sizing.cu
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "gpu_gate_sizing.h"

__global__ void assign_cells_kernel(
    int* chosen_sizes,
    float* input_slews,
    float* load_caps,
    float* slew_targets,
    int gate_count,
    int gate_type_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= gate_count) return;

  float best_error = 1e9f;
  int best_choice = 0;

  for (int i = 0; i < gate_type_count; ++i) {
    float estimated_slew = 0.02f * (i + 1) + 0.01f * sqrtf(input_slews[idx]) + 0.005f * load_caps[idx];
    float error = fabsf(estimated_slew - slew_targets[idx]);
    if (error < best_error) {
      best_error = error;
      best_choice = i;
    }
  }
  chosen_sizes[idx] = best_choice;
}

void cuda_assign_cells(const std::vector<GateInfo>& gates, std::vector<int>& chosen_sizes) {
  int n = gates.size();
  int gate_type_count = gates[0].gate_type_count;

  std::vector<float> input_slews(n), load_caps(n), slew_targets(n);
  for (int i = 0; i < n; ++i) {
    input_slews[i] = gates[i].input_slew;
    load_caps[i] = gates[i].load_cap;
    slew_targets[i] = gates[i].slew_target;
  }

  float *d_input_slews, *d_load_caps, *d_slew_targets;
  int *d_chosen_sizes;

  cudaMalloc(&d_input_slews, n * sizeof(float));
  cudaMalloc(&d_load_caps, n * sizeof(float));
  cudaMalloc(&d_slew_targets, n * sizeof(float));
  cudaMalloc(&d_chosen_sizes, n * sizeof(int));

  cudaMemcpy(d_input_slews, input_slews.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_load_caps, load_caps.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_slew_targets, slew_targets.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  assign_cells_kernel<<<numBlocks, blockSize>>>(
    d_chosen_sizes, d_input_slews, d_load_caps, d_slew_targets, n, gate_type_count);

  chosen_sizes.resize(n);
  cudaMemcpy(chosen_sizes.data(), d_chosen_sizes, n * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_input_slews);
  cudaFree(d_load_caps);
  cudaFree(d_slew_targets);
  cudaFree(d_chosen_sizes);
}


