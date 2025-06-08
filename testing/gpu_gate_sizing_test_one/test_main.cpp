#include <iostream>
#include "gpu_gate_sizing.h"

int main() {
  std::vector<GpuGateSizing::GateData> gates = {
    {1, 0.2f}, {2, 0.5f}, {3, 0.3f}
  };
  std::vector<float> delays;
  
  GpuGateSizing::runGateSizingGPU(gates, delays);

  for (size_t i = 0; i < delays.size(); i++) {
    std::cout << "Gate " << i << ": delay = " << delays[i] << "\n";
  }

  return 0;
}
