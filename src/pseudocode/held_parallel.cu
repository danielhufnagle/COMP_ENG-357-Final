#include <cuda_runtime.h>
#include <vector>

#include "held.h"

using namespace held;
using namespace std;

void held::cellAssignment(vector<Cell*>& cells, double input_slew) {
    //Feed input slew into all the standard cells, get output slews
    //For all output slews that qualify output slew<=slewlim, pick the smallest size    
}
double held::estimateSlewDegradation(double load_cap, double input_slew){
    //by using the RC-delay model
    const double driver_resistance = 100.0; // Ohms, typical placeholder
    const double k = 0.69; // RC delay fitting constant

    double rc_delay = k * driver_resistance * load_cap;
    double output_slew = sqrt(input_slew * input_slew + rc_delay * rc_delay); //Elmore Delay-Based Slew Propagation Model
    return output_slew - input_slew;
}

void held::timingAnalysis(vector<Pin*>& pins) {
    //Implemented in OpenRoad
    // this is simply here as a dummy function
}

void held::refineSlewTargets(vector<Cell*>& cells, int iteration){
    double theta_k = 1.0 / log(iteration + 2.0); // Avoid log(1) = 0

        for (auto& cell : cells) {
            double worst_predecessor_slack = numeric_limits<double>::max();
            for (auto& pin : cell->inputs) {
                worst_predecessor_slack = min(worst_predecessor_slack, pin->slack());//Finding the worst predecessor slack for a cell
            }

            for (auto& pout : cell->outputs) {
                double slack_plus = pout->slack();//global slack
                double local_crit = max(slack_plus - worst_predecessor_slack, 0.0);
                double delta_slew_target = 0.0;
                if (slack_plus < 0 && local_crit == 0) {
                    delta_slew_target = -min(theta_k * GAMMA * abs(slack_plus), MAX_CHANGE);//tighten the slew targets of the critical pins
                } else {
                    double effective_slack = max(slack_plus, local_crit);//loosen the slew targets of the non critical pins
                    delta_slew_target = min(theta_k * GAMMA * abs(effective_slack), MAX_CHANGE);
                }

                pout->slew_target += delta_slew_target;
                pout->slew_target = clamp(pout->slew_target, 0.0, pout->slew_limit);//clamp the slew target to be within the range from 0 to slew_limit
            }
        }

}

void held::fastGateSizing(std::vector<Cell*>& cells, std::vector<Pin*>& pins, int max_iterations = 10) {
    // Initialize slew targets
    // initialize to 80% of the limit as a conservative start
    for (auto& pin : pins) {
        pin ->slew_target = pin->slew_limit * 0.8;
    }

    for (int i = 0; i < max_terations; ++i) {
        // Assign cells to library cells
        for (auto& cell : cells) {
            unordered_map<string, double> input_slews;
            for (auto& pin L cell->inputs) {
                input_slews[pin->name] = pin->slew; // approximation
            }
            cell->assignCell(input_slews);
        }
        
        // Timing analysis (dummy function... will be handled by OpenROAD)
        timing_analysis(pins);
        
        // Refine slew targets
        refineSlewTargets(cells, i)
    }
}



// ==================== CUDA Parallel Kernels ====================

__global__ void assign_cells(Cell* cells, int* topo_order, int num_cells, SlewTarget* slew_targets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    int c_idx = topo_order[idx];
    Cell c = cells[c_idx];

    // TODO: Estimate input slews from predecessors using slew_targets
    // TODO: Compute output load from downstream net
    // TODO: Choose minimal size that meets slew target
    size_t best_lib = find_best_cell(c, slew_targets);  // You must define this

    cells[c_idx].assigned_lib = best_lib;
}

__global__ void refine_slew_targets(Cell* cells, SlewTarget* slew_targets, Slack* slacks, int num_cells, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;

    Cell c = cells[idx];

    for (int p = 0; p < c.num_outputs; ++p) {
        int pin = c.output_pins[p];
        float slack_p = slacks[pin].slack;
        float slack_pred = find_min_predecessor_slack(c); // You must define this
        float lc = max(slack_p - slack_pred, 0.0f);

        float delta;
        if (slack_p < 0 && lc == 0) {
            delta = -min(theta(k) * gamma * fabs(slack_p), MAX_CHANGE);
        } else {
            float slack_use = max(slack_p, lc);
            delta = min(theta(k) * gamma * slack_use, MAX_CHANGE);
        }

        // Clamp the updated slew target
        float new_slew = slew_targets[pin] + delta;
        slew_targets[pin] = clamp(new_slew, slew_min(pin), slew_max(pin));  // Define clamp, slew_min, and slew_max
    }
}

// ==============================================================
