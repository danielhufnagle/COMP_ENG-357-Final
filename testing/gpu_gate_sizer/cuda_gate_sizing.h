#pragma once
#include <vector>

struct GateInfo {
  int cell_id;
  int gate_type_count;
  float input_slew;
  float load_cap;
  float slew_target;
};

void cuda_assign_cells(const std::vector<GateInfo>& gates, std::vector<int>& chosen_sizes);
void cuda_refine_slew_targets(std::vector<GateInfo>& gates, const std::vector<float>& slacks, int iter);
