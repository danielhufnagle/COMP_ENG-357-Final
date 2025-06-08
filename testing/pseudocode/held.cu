#include <cuda_runtime.h>
#include <vector>

#include "held.h"

using namespace held;
using namespace std;

// void initializeSlewTargets(const vector<Cell*>& cells) {
//     for (auto& cell : cells) {
//         for (auto& pout : cell->outputs) {
//             double min_slew_target = numeric_limits<double>::max();

//             // For each sink pin q connected to this output pin
//             for (Pin* q : pout->connected_sinks) {
//                 // Estimate degradation from pout to q
//                 double degr = estimateSlewDegradation(q->capacitance_limit, pout->slew);
//                 double allowed = q->slew_limit - degr;
//                 min_slew_target = min(min_slew_target, allowed);
//             }

//             // Clamp to min possible slew for this pin
//             double realistic_min = pout->realistic_min_slew;
//             pout->slew_target = clamp(min_slew_target, realistic_min, pout->slew_limit);
//         }
//     }
// }

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
