#pragma once

#include <vector>

namespace GpuGateSizing {
  struct GateData {
    int gate_type;
    float input_slew;
  };


  void runGateSizingGPU(std::vector<GateData>& gates, std::vector<float>& delays);
}
