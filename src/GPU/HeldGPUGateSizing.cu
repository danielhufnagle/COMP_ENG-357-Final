// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "rsz/Resizer.hh"
#include "sta/Corner.hh"
#include "sta/Graph.hh"
#include "sta/GraphDelayCalc.hh"
#include "sta/Liberty.hh"
#include "sta/Network.hh"
#include "sta/PortDirection.hh"
#include "sta/Search.hh"
#include "sta/Units.hh"
#include "sta/Sta.hh"

namespace rsz {

using std::max;
using std::min;
using std::numeric_limits;
using std::sqrt;

using utl::RSZ;

using sta::dbStaState;
using sta::Graph;
using sta::LibertyCellSeq;
using sta::NetIterator;
using sta::InstancePinIterator;
using sta::PinConnectedPinIterator;
using sta::TimingArc;
using sta::TimingArcSet;
using sta::GateTimingModel;
using sta::MinMax;
using sta::ArcDelay;

// Device-side data structures
struct DeviceCellInfo {
    int instance_id;
    bool is_fixed;
    int current_variant_idx;
    float distance_from_src;
    float slack_minus;
    
    // Output pin data
    int output_pin_id;
    float slew_target;
    float wire_cap;
    float downstream_cap;
    float slew_limit_from_sinks;
    float min_achievable_slew;
    float arrival_time;
    float required_time;
    float slack;
    float slew_actual;
    
    // Input pins
    int input_pins_count;
    int input_pins[8]; // Assuming max 8 inputs per cell
};

// Host-side data structures
struct HostCellInfo {
    Instance* instance;
    bool is_fixed;
    int current_variant_idx;
    double distance_from_src;
    double slack_minus;
    
    struct {
        Pin* pin;
        double slew_target;
        double wire_cap;
        double downstream_cap;
        double slew_limit_from_sinks;
        double min_achievable_slew;
        double arrival_time;
        double required_time;
        double slack;
        double slew_actual;
        std::vector<Pin*> sinks;
    } output;
    
    std::vector<Pin*> input_pins;
    std::vector<LibertyCell*> variants;
};

class HeldGPUGateSizing {
public:
    HeldGPUGateSizing(Resizer* resizer,
                      double gamma,
                      double max_change,
                      int max_iterations);
    
    bool optimizeGateSizing(double clock_period);

private:
    // Host-side members
    Resizer* resizer_;
    Logger* logger_;
    dbNetwork* network_;
    Sta* sta_;
    double gamma_;
    double max_change_;
    int max_iterations_;
    double log_constant_;
    int iteration_;
    int endpoints_count_;
    double best_ws_;
    double best_sns_;
    double best_avg_area_;
    double prev_ws_;
    double prev_sns_;
    double prev_avg_area_;
    double default_max_slew_;
    double default_min_slew_;
    double default_input_slew_;
    Corner* corner_;
    DcalcAnalysisPt* dcalc_ap_;
    
    // Data structures
    std::unordered_map<Instance*, HostCellInfo> cell_info_map_;
    std::unordered_map<Instance*, std::vector<Instance*>> cell_graph_;
    std::unordered_map<Instance*, std::vector<Instance*>> cell_preds_;
    std::vector<HostCellInfo*> topo_sorted_cells_;
    
    // Device-side data
    thrust::device_vector<DeviceCellInfo> d_cell_info_;
    thrust::device_vector<int> d_cell_variants_;
    thrust::device_vector<float> d_variant_areas_;
    
    // Helper methods
    void buildCellInfoMap();
    void buildCellGraph();
    void computeLongestDistanceFromRegisters();
    void initializeSlewTargets();
    void assignCellsToMeetSlewTargets();
    void performTimingAnalysis();
    void refineSlewTargets();
    double computeOutputSlew(LibertyCell* cell, double input_slew, double load_cap);
    double estimateInputSlew(const std::vector<Pin*>& input_pins);
    double computeWireSlewDegradation(Pin* driver_pin, Pin* sink_pin);
    double computeMinAchievableSlew(HostCellInfo* cell_info, double load_cap, double input_slew);
    double computeSlewLimitFromSinks(HostCellInfo* cell_info);
    bool checkCapacitanceLimits(LibertyCell* cell, double load_cap);
    double computeWorstSlack();
    double computeTotalNegativeSlack();
    double computeAverageCellArea();
    void saveCurrentAssignmentAsBest();
    void restoreBestAssignment();
    bool checkHeldStoppingCriterion(double prev_ws, double prev_sns, double prev_avg_area);
    
    // CUDA kernels
    __global__ void assignCellsKernel(DeviceCellInfo* cell_info,
                                     int* cell_variants,
                                     float* variant_areas,
                                     int num_cells,
                                     float* input_slews,
                                     float* load_caps);
                                     
    __global__ void refineSlewTargetsKernel(DeviceCellInfo* cell_info,
                                           int* cell_preds,
                                           int* pred_offsets,
                                           int num_cells,
                                           float theta_k,
                                           float gamma,
                                           float max_change);
                                           
    __global__ void updateTimingKernel(DeviceCellInfo* cell_info,
                                      int num_cells,
                                      float* arrival_times,
                                      float* required_times,
                                      float* slews);
};

HeldGPUGateSizing::HeldGPUGateSizing(Resizer* resizer,
                                     double gamma,
                                     double max_change,
                                     int max_iterations)
    : resizer_(resizer),
      logger_(resizer->logger_),
      network_(static_cast<dbNetwork*>(resizer->network_)),
      sta_(resizer->sta_),
      gamma_(gamma),
      max_change_(max_change),
      max_iterations_(max_iterations),
      log_constant_(2.0), // Held's default
      iteration_(0),
      endpoints_count_(0),
      best_ws_(numeric_limits<double>::infinity()),
      best_sns_(numeric_limits<double>::infinity()),
      best_avg_area_(numeric_limits<double>::infinity()),
      prev_ws_(numeric_limits<double>::infinity()),
      prev_sns_(numeric_limits<double>::infinity()),
      prev_avg_area_(numeric_limits<double>::infinity()),
      default_max_slew_(0.5e-9),  // 500ps
      default_min_slew_(10e-12),  // 10ps
      default_input_slew_(50e-12), // 50ps
      corner_(nullptr),
      dcalc_ap_(nullptr) {
}

bool HeldGPUGateSizing::optimizeGateSizing(double clock_period) {
    clock_period_ = clock_period;
    
    // Find the slowest corner for sizing
    corner_ = resizer_->tgt_slew_corner_;
    if (!corner_) {
        for (Corner* corner : *sta_->corners()) {
            corner_ = corner;
            break;
        }
    }
    
    if (!corner_) {
        corner_ = sta_->findCorner("default");
        if (!corner_) {
            logger_->warn(RSZ, 100, "No timing corner found for gate sizing");
            return false;
        }
    }
    
    dcalc_ap_ = corner_->findDcalcAnalysisPt(MinMax::max());
    if (!dcalc_ap_) {
        logger_->warn(RSZ, 121, "No delay calculation analysis point found");
        return false;
    }
    
    logger_->info(RSZ, 101, "Starting Held's GPU-accelerated gate sizing algorithm");
    
    // Initialize data structures
    buildCellInfoMap();
    buildCellGraph();
    computeLongestDistanceFromRegisters();
    initializeSlewTargets();
    
    // Prepare device data
    int num_cells = cell_info_map_.size();
    d_cell_info_.resize(num_cells);
    
    // Convert host data to device format
    std::vector<DeviceCellInfo> h_cell_info(num_cells);
    int cell_idx = 0;
    for (auto& [inst, cell_info] : cell_info_map_) {
        DeviceCellInfo& d_info = h_cell_info[cell_idx];
        d_info.instance_id = cell_idx;
        d_info.is_fixed = cell_info.is_fixed;
        d_info.current_variant_idx = cell_info.current_variant_idx;
        d_info.distance_from_src = cell_info.distance_from_src;
        d_info.slack_minus = cell_info.slack_minus;
        
        if (cell_info.output.pin) {
            d_info.output_pin_id = network_->id(cell_info.output.pin);
            d_info.slew_target = cell_info.output.slew_target;
            d_info.wire_cap = cell_info.output.wire_cap;
            d_info.downstream_cap = cell_info.output.downstream_cap;
            d_info.slew_limit_from_sinks = cell_info.output.slew_limit_from_sinks;
            d_info.min_achievable_slew = cell_info.output.min_achievable_slew;
        }
        
        d_info.input_pins_count = cell_info.input_pins.size();
        for (int i = 0; i < d_info.input_pins_count && i < 8; ++i) {
            d_info.input_pins[i] = network_->id(cell_info.input_pins[i]);
        }
        
        cell_idx++;
    }
    
    // Copy to device
    thrust::copy(h_cell_info.begin(), h_cell_info.end(), d_cell_info_.begin());
    
    // Save initial state as best
    saveCurrentAssignmentAsBest();
    
    // Main optimization loop
    bool improved = false;
    
    for (iteration_ = 1; iteration_ <= max_iterations_; ++iteration_) {
        logger_->info(RSZ, 114, "Held GPU sizing iteration {}", iteration_);
        
        // Assign cells to meet slew targets
        assignCellsToMeetSlewTargets();
        
        // Timing analysis
        performTimingAnalysis();
        
        // Evaluate current solution
        double curr_ws = abs(computeWorstSlack());
        double curr_sns = computeTotalNegativeSlack() / std::max(1.0, (double)endpoints_count_);
        double curr_avg_area = computeAverageCellArea();
        
        logger_->info(RSZ, 115, 
                     "Iteration {} - WS: {:.3f} ps, SNS: {:.3f} ps, Area: {:.3f}",
                     iteration_, 
                     curr_ws * 1e12,
                     curr_sns * 1e12,
                     curr_avg_area);
        
        // Check if this is the best solution so far
        double curr_held_obj = curr_ws + curr_sns + curr_avg_area;
        double best_held_obj = best_ws_ + best_sns_ + best_avg_area_;
        
        if (curr_held_obj < best_held_obj) {
            best_ws_ = curr_ws;
            best_sns_ = curr_sns;
            best_avg_area_ = curr_avg_area;
            saveCurrentAssignmentAsBest();
            improved = true;
        }
        
        // Check stopping criteria
        if (iteration_ > 1) {
            if (checkHeldStoppingCriterion(prev_ws_, prev_sns_, prev_avg_area_)) {
                logger_->info(RSZ, 122, "Held stopping criterion met - restoring best solution");
                restoreBestAssignment();
                break;
            }
            
            // Additional numerical convergence check
            if (abs(curr_ws - prev_ws_) < 1e-12 && 
                abs(curr_sns - prev_sns_) < 1e-12 && 
                abs(curr_avg_area - prev_avg_area_) < 1e-6) {
                logger_->info(RSZ, 124, "Numerical convergence achieved");
                break;
            }
        }
        
        // Store current metrics for next iteration
        prev_ws_ = curr_ws;
        prev_sns_ = curr_sns;
        prev_avg_area_ = curr_avg_area;
        
        // Refine slew targets
        refineSlewTargets();
    }
    
    // Report final results
    double final_wns = computeWorstSlack();
    double final_tns = computeTotalNegativeSlack();
    double final_area = computeAverageCellArea();
    
    logger_->info(RSZ, 117,
                 "Held GPU sizing completed - WNS: {:.3f} ps, TNS: {:.3f} ps, "
                 "Avg Area: {:.3f}, Iterations: {}",
                 final_wns * 1e12,
                 final_tns * 1e12, 
                 final_area,
                 iteration_);
    
    return improved;
}

void HeldGPUGateSizing::assignCellsToMeetSlewTargets() {
    int num_cells = cell_info_map_.size();
    
    // Prepare input data
    std::vector<float> h_input_slews(num_cells);
    std::vector<float> h_load_caps(num_cells);
    
    // Compute input slews and load caps for each cell
    int cell_idx = 0;
    for (auto& [inst, cell_info] : cell_info_map_) {
        h_input_slews[cell_idx] = estimateInputSlew(cell_info.input_pins);
        h_load_caps[cell_idx] = resizer_->graph_delay_calc_->loadCap(cell_info.output.pin, dcalc_ap_);
        cell_idx++;
    }
    
    // Allocate device memory
    thrust::device_vector<float> d_input_slews(h_input_slews);
    thrust::device_vector<float> d_load_caps(h_load_caps);
    
    // Prepare cell variants data
    std::vector<int> h_cell_variants(num_cells * 8, -1); // -1 indicates no variant
    std::vector<float> h_variant_areas(num_cells * 8, 0.0f);
    
    cell_idx = 0;
    for (auto& [inst, cell_info] : cell_info_map_) {
        for (size_t i = 0; i < cell_info.variants.size() && i < 8; ++i) {
            h_cell_variants[cell_idx * 8 + i] = i; // Store variant index
            h_variant_areas[cell_idx * 8 + i] = cell_info.variants[i]->area();
        }
        cell_idx++;
    }
    
    // Copy to device
    thrust::device_vector<int> d_cell_variants(h_cell_variants);
    thrust::device_vector<float> d_variant_areas(h_variant_areas);
    
    // Launch kernel
    int block_size = 256;
    int num_blocks = (num_cells + block_size - 1) / block_size;
    
    assignCellsKernel<<<num_blocks, block_size>>>(
        thrust::raw_pointer_cast(d_cell_info_.data()),
        thrust::raw_pointer_cast(d_cell_variants.data()),
        thrust::raw_pointer_cast(d_variant_areas.data()),
        num_cells,
        thrust::raw_pointer_cast(d_input_slews.data()),
        thrust::raw_pointer_cast(d_load_caps.data())
    );
    
    // Synchronize and check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        logger_->error(RSZ, 129, "CUDA error in assignCellsKernel: {}", cudaGetErrorString(error));
        return;
    }
    cudaDeviceSynchronize();
    
    // Copy results back to host
    std::vector<DeviceCellInfo> h_cell_info(num_cells);
    thrust::copy(d_cell_info_.begin(), d_cell_info_.end(), h_cell_info.begin());
    
    // Update host data structures
    cell_idx = 0;
    for (auto& [inst, cell_info] : cell_info_map_) {
        const DeviceCellInfo& d_info = h_cell_info[cell_idx];
        if (d_info.current_variant_idx != cell_info.current_variant_idx) {
            // Cell was changed - update the instance
            LibertyCell* new_cell = cell_info.variants[d_info.current_variant_idx];
            if (resizer_->replaceCell(inst, new_cell, true)) {
                cell_info.current_variant_idx = d_info.current_variant_idx;
            }
        }
        cell_idx++;
    }
}

void HeldGPUGateSizing::performTimingAnalysis() {
    // Perform STA on host
    sta_->findDelays();
    sta_->findRequireds();
    
    int num_cells = cell_info_map_.size();
    
    // Prepare timing data
    std::vector<float> h_arrival_times(num_cells);
    std::vector<float> h_required_times(num_cells);
    std::vector<float> h_slews(num_cells);
    
    // Collect timing data from STA
    int cell_idx = 0;
    for (auto& [inst, cell_info] : cell_info_map_) {
        if (cell_info.output.pin) {
            h_arrival_times[cell_idx] = sta_->pinArrival(cell_info.output.pin, RiseFall::rise(), MinMax::max());
            
            sta::Vertex* vertex = resizer_->graph_->pinDrvrVertex(cell_info.output.pin);
            if (vertex) {
                h_required_times[cell_idx] = sta_->vertexRequired(vertex, RiseFall::rise(), MinMax::max());
                h_slews[cell_idx] = sta_->vertexSlew(vertex, RiseFall::rise(), MinMax::max());
            }
        }
        cell_idx++;
    }
    
    // Copy to device
    thrust::device_vector<float> d_arrival_times(h_arrival_times);
    thrust::device_vector<float> d_required_times(h_required_times);
    thrust::device_vector<float> d_slews(h_slews);
    
    // Launch kernel
    int block_size = 256;
    int num_blocks = (num_cells + block_size - 1) / block_size;
    
    updateTimingKernel<<<num_blocks, block_size>>>(
        thrust::raw_pointer_cast(d_cell_info_.data()),
        num_cells,
        thrust::raw_pointer_cast(d_arrival_times.data()),
        thrust::raw_pointer_cast(d_required_times.data()),
        thrust::raw_pointer_cast(d_slews.data())
    );
    
    // Synchronize and check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        logger_->error(RSZ, 130, "CUDA error in updateTimingKernel: {}", cudaGetErrorString(error));
        return;
    }
    cudaDeviceSynchronize();
    
    // Copy results back to host
    std::vector<DeviceCellInfo> h_cell_info(num_cells);
    thrust::copy(d_cell_info_.begin(), d_cell_info_.end(), h_cell_info.begin());
    
    // Update host data structures
    cell_idx = 0;
    for (auto& [inst, cell_info] : cell_info_map_) {
        const DeviceCellInfo& d_info = h_cell_info[cell_idx];
        if (cell_info.output.pin) {
            cell_info.output.arrival_time = d_info.arrival_time;
            cell_info.output.required_time = d_info.required_time;
            cell_info.output.slack = d_info.slack;
            cell_info.output.slew_actual = d_info.slew_actual;
        }
        cell_idx++;
    }
}

void HeldGPUGateSizing::refineSlewTargets() {
    int num_cells = cell_info_map_.size();
    
    // Prepare predecessor data
    std::vector<int> h_cell_preds;
    std::vector<int> h_pred_offsets(num_cells + 1, 0);
    
    // Build predecessor lists
    int offset = 0;
    for (auto& [inst, cell_info] : cell_info_map_) {
        h_pred_offsets[offset] = h_cell_preds.size();
        for (Instance* pred : cell_preds_[inst]) {
            auto it = cell_info_map_.find(pred);
            if (it != cell_info_map_.end()) {
                h_cell_preds.push_back(std::distance(cell_info_map_.begin(), it));
            }
        }
        offset++;
    }
    h_pred_offsets[num_cells] = h_cell_preds.size();
    
    // Copy to device
    thrust::device_vector<int> d_cell_preds(h_cell_preds);
    thrust::device_vector<int> d_pred_offsets(h_pred_offsets);
    
    // Compute theta_k
    double theta_k = 1.0 / log(iteration_ + log_constant_);
    
    // Launch kernel
    int block_size = 256;
    int num_blocks = (num_cells + block_size - 1) / block_size;
    
    refineSlewTargetsKernel<<<num_blocks, block_size>>>(
        thrust::raw_pointer_cast(d_cell_info_.data()),
        thrust::raw_pointer_cast(d_cell_preds.data()),
        thrust::raw_pointer_cast(d_pred_offsets.data()),
        num_cells,
        theta_k,
        gamma_,
        max_change_
    );
    
    // Synchronize and check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        logger_->error(RSZ, 131, "CUDA error in refineSlewTargetsKernel: {}", cudaGetErrorString(error));
        return;
    }
    cudaDeviceSynchronize();
    
    // Copy results back to host
    std::vector<DeviceCellInfo> h_cell_info(num_cells);
    thrust::copy(d_cell_info_.begin(), d_cell_info_.end(), h_cell_info.begin());
    
    // Update host data structures
    int cell_idx = 0;
    for (auto& [inst, cell_info] : cell_info_map_) {
        const DeviceCellInfo& d_info = h_cell_info[cell_idx];
        cell_info.slack_minus = d_info.slack_minus;
        if (cell_info.output.pin) {
            cell_info.output.slew_target = d_info.slew_target;
        }
        cell_idx++;
    }
}

// CUDA kernel for assigning cells to meet slew targets
__global__ void assignCellsKernel(DeviceCellInfo* cell_info,
                                 int* cell_variants,
                                 float* variant_areas,
                                 int num_cells,
                                 float* input_slews,
                                 float* load_caps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    
    DeviceCellInfo& info = cell_info[idx];
    if (info.is_fixed || info.output_pin_id == -1) return;
    
    // Get current load and input slew
    float load_cap = load_caps[idx];
    float input_slew = input_slews[idx];
    
    // Find smallest variant that meets slew target
    int best_variant_idx = -1;
    float best_slew = info.slew_target;
    
    // Iterate through variants (sorted by area)
    for (int i = 0; i < 8; ++i) { // Assuming max 8 variants per cell
        int variant_idx = cell_variants[idx * 8 + i];
        if (variant_idx == -1) break;
        
        // Compute output slew for this variant
        float output_slew = computeOutputSlew(variant_idx, input_slew, load_cap);
        
        // Check if this variant meets the slew target
        if (output_slew <= info.slew_target) {
            best_variant_idx = i;
            best_slew = output_slew;
            break; // Take the first (smallest) variant that meets target
        }
    }
    
    // Update cell info if we found a better variant
    if (best_variant_idx >= 0 && best_variant_idx != info.current_variant_idx) {
        info.current_variant_idx = best_variant_idx;
        info.slew_actual = best_slew;
    }
}

// CUDA kernel for refining slew targets
__global__ void refineSlewTargetsKernel(DeviceCellInfo* cell_info,
                                       int* cell_preds,
                                       int* pred_offsets,
                                       int num_cells,
                                       float theta_k,
                                       float gamma,
                                       float max_change) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    
    DeviceCellInfo& info = cell_info[idx];
    if (info.output_pin_id == -1) return;
    
    // Find minimum slack among predecessors
    float slk_minus = std::numeric_limits<float>::infinity();
    int pred_start = pred_offsets[idx];
    int pred_end = pred_offsets[idx + 1];
    
    for (int i = pred_start; i < pred_end; ++i) {
        int pred_idx = cell_preds[i];
        if (pred_idx >= 0) {
            float pred_slack = cell_info[pred_idx].slack;
            slk_minus = min(slk_minus, pred_slack);
        }
    }
    
    if (slk_minus == std::numeric_limits<float>::infinity()) {
        slk_minus = 0.0f; // No predecessors
    }
    
    // Update slack_minus
    info.slack_minus = slk_minus;
    
    // Get current slack
    float slk_plus = info.slack;
    
    // Compute local constraint
    float lc = max(slk_plus - slk_minus, 0.0f);
    
    // Compute slew target adjustment
    float delta_slew_target = 0.0f;
    
    if (slk_plus < 0.0f && lc == 0.0f) {
        // Negative slack case
        delta_slew_target = -min(theta_k * gamma * abs(slk_plus), max_change);
    } else {
        // Positive slack case
        slk_plus = max(slk_plus, lc);
        delta_slew_target = +min(theta_k * gamma * abs(slk_plus), max_change);
    }
    
    // Update slew target with bounds checking
    float new_slew_target = info.slew_target + delta_slew_target;
    new_slew_target = max(info.min_achievable_slew, 
                         min(new_slew_target, info.slew_limit_from_sinks));
    
    info.slew_target = new_slew_target;
}

// CUDA kernel for updating timing information
__global__ void updateTimingKernel(DeviceCellInfo* cell_info,
                                  int num_cells,
                                  float* arrival_times,
                                  float* required_times,
                                  float* slews) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells) return;
    
    DeviceCellInfo& info = cell_info[idx];
    if (info.output_pin_id == -1) return;
    
    // Update timing information
    info.arrival_time = arrival_times[idx];
    info.required_time = required_times[idx];
    info.slack = info.required_time - info.arrival_time;
    info.slew_actual = slews[idx];
}

// Helper function to compute output slew on device
__device__ float computeOutputSlew(int variant_idx, float input_slew, float load_cap) {
    // This is a simplified model - in practice, you would use a lookup table
    // or a more sophisticated model based on the cell's characteristics
    float base_slew = 50e-12f; // 50ps base slew
    float slew_factor = 1.0f + (load_cap * 1e12f) * 0.1f; // 0.1ps per fF
    float input_factor = 1.0f + (input_slew * 1e12f) * 0.05f; // 5% slew degradation per ps
    
    return base_slew * slew_factor * input_factor;
}

double HeldGPUGateSizing::computeWorstSlack() {
    return sta_->worstSlack(MinMax::max());
}

double HeldGPUGateSizing::computeTotalNegativeSlack() {
    double total_negative_slack = 0.0;
    endpoints_count_ = 0;
    
    for (auto& [inst, cell_info] : cell_info_map_) {
        if (cell_info.output.pin) {
            bool is_endpoint = false;
            if (cell_info.output.sinks.empty()) {
                is_endpoint = true; // Primary output
            } else {
                // Check if any sink is a register input
                for (Pin* sink_pin : cell_info.output.sinks) {
                    Instance* sink_inst = network_->instance(sink_pin);
                    LibertyCell* sink_cell = network_->libertyCell(sink_inst);
                    if (sink_cell && sink_cell->hasSequentials()) {
                        is_endpoint = true;
                        break;
                    }
                }
            }
            
            if (is_endpoint) {
                endpoints_count_++;
                double slack = sta_->pinSlack(cell_info.output.pin, MinMax::max());
                if (slack < 0.0) {
                    total_negative_slack += slack;
                }
            }
        }
    }
    
    return total_negative_slack;
}

double HeldGPUGateSizing::computeAverageCellArea() {
    double total_area = 0.0;
    int cell_count = 0;
    
    for (auto& [inst, cell_info] : cell_info_map_) {
        if (!cell_info.is_fixed) {
            LibertyCell* cell = network_->libertyCell(inst);
            if (cell) {
                total_area += cell->area();
                cell_count++;
            }
        }
    }
    
    return cell_count > 0 ? total_area / cell_count : 0.0;
}

void HeldGPUGateSizing::saveCurrentAssignmentAsBest() {
    // Store current cell assignments
    for (auto& [inst, cell_info] : cell_info_map_) {
        LibertyCell* cell = network_->libertyCell(inst);
        if (cell) {
            cell_info.variants[cell_info.current_variant_idx] = cell;
        }
    }
}

void HeldGPUGateSizing::restoreBestAssignment() {
    // Restore best cell assignments
    for (auto& [inst, cell_info] : cell_info_map_) {
        if (!cell_info.is_fixed) {
            LibertyCell* best_cell = cell_info.variants[cell_info.current_variant_idx];
            if (best_cell) {
                resizer_->replaceCell(inst, best_cell, true);
            }
        }
    }
}

bool HeldGPUGateSizing::checkHeldStoppingCriterion(double prev_ws, double prev_sns, double prev_avg_area) {
    // Held's stopping criterion: Check if WS worsened AND overall objective worsened
    double curr_ws = abs(computeWorstSlack());
    double curr_sns = computeTotalNegativeSlack() / std::max(1.0, (double)endpoints_count_);
    double curr_avg_area = computeAverageCellArea();
    
    double curr_obj = curr_ws + curr_sns + curr_avg_area;
    double prev_obj = prev_ws + prev_sns + prev_avg_area;
    
    return curr_ws > prev_ws && curr_obj > prev_obj;
}

} // namespace rsz
