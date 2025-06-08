// SPDX-License-Identifier: BSD-3-Clause

#include "HeldGateSizing.hh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>

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

HeldGateSizing::HeldGateSizing(Resizer* resizer,
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

bool HeldGateSizing::optimizeGateSizing(double clock_period) {
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
  
  logger_->info(RSZ, 101, "Starting Held's gate sizing algorithm");
  
  // Held's Algorithm 1, Line 1: Initialize slew targets
  buildCellInfoMap();
  buildCellGraph(); 
  computeLongestDistanceFromRegisters();
  initializeSlewTargets();
  
  // Save initial state as best
  saveCurrentAssignmentAsBest();
  
  // Held's Algorithm 1, Lines 2-6: Main optimization loop
  bool improved = false;
  
  for (iteration_ = 1; iteration_ <= max_iterations_; ++iteration_) {
    logger_->info(RSZ, 114, "Held sizing iteration {}", iteration_);
    
    // Held's Algorithm 1, Line 3: Assign cells to library cells
    assignCellsToMeetSlewTargets();
    
    // Held's Algorithm 1, Line 4: Timing analysis
    performTimingAnalysis();
    
    // Evaluate current solution using Held's metrics
    double curr_ws = abs(computeWorstSlack());
    double curr_sns = computeTotalNegativeSlack() / std::max(1.0, (double)endpoints_count_);
    double curr_avg_area = computeAverageCellArea();
    
    logger_->info(RSZ, 115, 
                 "Iteration {} - WS: {:.3f} ps, SNS: {:.3f} ps, Area: {:.3f}",
                 iteration_, 
                 curr_ws * 1e12,
                 curr_sns * 1e12,
                 curr_avg_area);
    
    // Check if this is the best solution so far (Held's objective: WS + SNS + area)
    double curr_held_obj = curr_ws + curr_sns + curr_avg_area;
    double best_held_obj = best_ws_ + best_sns_ + best_avg_area_;
    
    if (curr_held_obj < best_held_obj) {
      best_ws_ = curr_ws;
      best_sns_ = curr_sns;
      best_avg_area_ = curr_avg_area;
      saveCurrentAssignmentAsBest();
      improved = true;
    }
    
    // Held's stopping criterion: Check if WS worsened AND overall objective worsened
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
    
    // Held's Algorithm 1, Line 5: Refine slew targets
    refineSlewTargets();
  }
  
  // Held's Algorithm 1, Line 7: Return best assignment (already restored if needed)
  double final_wns = computeWorstSlack();
  double final_tns = computeTotalNegativeSlack();
  double final_area = computeAverageCellArea();
  
  logger_->info(RSZ, 117,
               "Held sizing completed - WNS: {:.3f} ps, TNS: {:.3f} ps, "
               "Avg Area: {:.3f}, Iterations: {}",
               final_wns * 1e12,
               final_tns * 1e12, 
               final_area,
               iteration_);
  
  return improved;
}

void HeldGateSizing::buildCellInfoMap() {
  cell_info_map_.clear();
  
  int total_instances = 0;
  int sizeable_instances = 0;
  
  // Iterate through all instances in the design
  sta::LeafInstanceIterator* inst_iter = network_->leafInstanceIterator();
  while (inst_iter->hasNext()) {
    Instance* inst = inst_iter->next();
    total_instances++;
    
    // Skip if this is not a standard cell or is dont_touch
    if (resizer_->dontTouch(inst) || !resizer_->isLogicStdCell(inst)) {
      continue;
    }
    
    LibertyCell* cell = network_->libertyCell(inst);
    if (!cell) continue;
    
    sizeable_instances++;
    
    HeldCellInfo cell_info;
    cell_info.instance = inst;
    cell_info.is_fixed = cell->hasSequentials(); // Registers are fixed
    cell_info.current_variant_idx = 0;
    cell_info.distance_from_src = 0.0;
    cell_info.slack_minus = 0.0;
    
    // Get equivalent cells (swappable variants) - Held's [c]
    cell_info.variants = getEquivalentCells(inst);
    
    // Find current variant index
    for (size_t i = 0; i < cell_info.variants.size(); ++i) {
      if (cell_info.variants[i] == cell) {
        cell_info.current_variant_idx = i;
        break;
      }
    }
    
    // Initialize output pin data
    InstancePinIterator* pin_iter = network_->pinIterator(inst);
    while (pin_iter->hasNext()) {
      Pin* pin = pin_iter->next();
      if (network_->direction(pin)->isOutput()) {
        cell_info.output.pin = pin;
        cell_info.output.slew_target = default_max_slew_;
        cell_info.output.wire_cap = 0.0;
        cell_info.output.downstream_cap = 0.0;
        cell_info.output.slew_limit_from_sinks = default_max_slew_;
        cell_info.output.min_achievable_slew = default_min_slew_;
        
        // Collect sink pins
        PinConnectedPinIterator* conn_iter = network_->connectedPinIterator(pin);
        while (conn_iter->hasNext()) {
          const Pin* conn_pin = conn_iter->next();
          if (conn_pin != pin && network_->direction(conn_pin)->isInput()) {
            cell_info.output.sinks.push_back(const_cast<Pin*>(conn_pin));
          }
        }
        delete conn_iter;
        break; // Assume single output for simplicity
      } else if (network_->direction(pin)->isInput()) {
        cell_info.input_pins.push_back(pin);
      }
    }
    delete pin_iter;
    
    cell_info_map_[inst] = cell_info;
  }
  delete inst_iter;
  
  logger_->info(RSZ, 118, "Built cell info map: {} total instances, {} sizeable", 
               total_instances, sizeable_instances);
}

std::vector<LibertyCell*> HeldGateSizing::getEquivalentCells(Instance* inst) {
  LibertyCell* cell = network_->libertyCell(inst);
  if (!cell) return {};
  
  // Use resizer's existing logic to get swappable cells
  LibertyCellSeq equiv_cells = resizer_->getSwappableCells(cell);
  std::vector<LibertyCell*> result(equiv_cells.begin(), equiv_cells.end());
  
  // Held requires sorting by area (smallest first for greedy selection)
  std::sort(result.begin(), result.end(), 
           [](const LibertyCell* a, const LibertyCell* b) {
             return a->area() < b->area();
           });
  
  return result;
}

void HeldGateSizing::buildCellGraph() {
  // Build cell-to-cell adjacency for topological ordering
  cell_graph_.clear();
  cell_preds_.clear();
  
  for (auto& [inst, cell_info] : cell_info_map_) {
    cell_graph_[inst] = std::vector<Instance*>();
    cell_preds_[inst] = std::vector<Instance*>();
  }
  
  // Build edges: if cell A's output drives cell B's input, add A->B edge
  for (auto& [inst, cell_info] : cell_info_map_) {
    if (cell_info.output.pin) {
      for (Pin* sink_pin : cell_info.output.sinks) {
        Instance* sink_inst = network_->instance(sink_pin);
        if (cell_info_map_.find(sink_inst) != cell_info_map_.end()) {
          // Add edge inst -> sink_inst
          cell_graph_[inst].push_back(sink_inst);
          cell_preds_[sink_inst].push_back(inst);
        }
      }
    }
  }
}

void HeldGateSizing::computeLongestDistanceFromRegisters() {
  // Held's requirement: "longest distance (under unit edge lengths) from a register"
  // Initialize all distances to 0
  for (auto& [inst, cell_info] : cell_info_map_) {
    cell_info.distance_from_src = 0.0;
  }
  
  // Start BFS/DP from all register outputs (fixed cells)
  std::queue<Instance*> queue;
  std::unordered_map<Instance*, double> distances;
  std::unordered_map<Instance*, int> visit_count; // Prevent infinite loops
  
  // Initialize: all registers have distance 0
  for (auto& [inst, cell_info] : cell_info_map_) {
    if (cell_info.is_fixed) {
      distances[inst] = 0.0;
      visit_count[inst] = 0;
      queue.push(inst);
    } else {
      distances[inst] = -1.0; // unvisited
      visit_count[inst] = 0;
    }
  }
  
  // BFS to find longest distance (unit edge weights) with cycle protection
  const int MAX_VISITS = 1000; // Prevent infinite loops
  const double MAX_DISTANCE = 100.0; // Reasonable upper bound for circuit depth
  
  while (!queue.empty()) {
    Instance* current = queue.front();
    queue.pop();
    
    // Prevent infinite loops - if a node is visited too many times, skip it
    if (visit_count[current] > MAX_VISITS) {
      logger_->warn(RSZ, 132, "Potential cycle detected in cell graph, limiting distance computation");
      continue;
    }
    visit_count[current]++;
    
    double current_dist = distances[current];
    
    // Sanity check - prevent distance from growing too large
    if (current_dist > MAX_DISTANCE) {
      continue;
    }
    
    // Update all successors
    for (Instance* successor : cell_graph_[current]) {
      double new_dist = current_dist + 1.0;
      
      // Only update if we found a genuinely longer path and haven't exceeded limits
      if (distances[successor] < new_dist && new_dist <= MAX_DISTANCE && visit_count[successor] <= MAX_VISITS) {
        distances[successor] = new_dist;
        queue.push(successor);
      }
    }
  }
  
  // Copy back to cell_info and sort
  topo_sorted_cells_.clear();
  for (auto& [inst, cell_info] : cell_info_map_) {
    cell_info.distance_from_src = distances[inst];
    if (distances[inst] < 0) {
      // Unreachable from registers - assign large distance
      cell_info.distance_from_src = 1000.0;
    }
    topo_sorted_cells_.push_back(&cell_info);
  }
  
  // Sort by distance (ascending) - we'll process in reverse order (decreasing distance)
  std::sort(topo_sorted_cells_.begin(), topo_sorted_cells_.end(),
           [](const HeldCellInfo* a, const HeldCellInfo* b) {
             return a->distance_from_src < b->distance_from_src;
           });
           
  logger_->info(RSZ, 123, "Computed longest distances: max_distance={:.1f}", 
                topo_sorted_cells_.empty() ? 0.0 : topo_sorted_cells_.back()->distance_from_src);
}

void HeldGateSizing::initializeSlewTargets() {
  // Held's specification: "Initialize slew targets for all cell output pins such that 
  // the slew limits will just be met at subsequent sinks (accounting for the slew 
  // degradations on the wires)"
  
  for (auto& [inst, cell_info] : cell_info_map_) {
    if (cell_info.output.pin) {
      // Compute slew_limit_from_sinks = min over sinks of (slewlim(q) - degradation(p->q))
      double min_sink_limit = std::numeric_limits<double>::infinity();
      bool any_sink_has_limit = false;
      
      for (Pin* sink_pin : cell_info.output.sinks) {
        LibertyPort* sink_port = network_->libertyPort(sink_pin);
        if (sink_port) {
          float sink_slew_limit;
          bool exists;
          sta_->findSlewLimit(sink_port, corner_, MinMax::max(), sink_slew_limit, exists);
          if (exists && sink_slew_limit > 0) {
            any_sink_has_limit = true;
            // Account for wire degradation
            double degradation = computeWireSlewDegradation(cell_info.output.pin, sink_pin);
            double effective_limit = (double)sink_slew_limit - degradation;
            min_sink_limit = std::min(min_sink_limit, effective_limit);
          }
        }
      }
      
      if (!any_sink_has_limit) {
        // No explicit constraints found - use technology default
        min_sink_limit = default_max_slew_;
      }
      
      // Store the computed limit
      cell_info.output.slew_limit_from_sinks = min_sink_limit;
      
      // Initialize slew target to meet the tightest sink constraint
      cell_info.output.slew_target = std::max(min_sink_limit, default_min_slew_);
      cell_info.output.slew_target = std::min(cell_info.output.slew_target, default_max_slew_);
    }
  }
}

void HeldGateSizing::assignCellsToMeetSlewTargets() {
  // Held's specification: "Cells are assigned to the smallest equivalent library cell 
  // such that the slew targets at all output pins are met"
  
  int cells_changed = 0;
  int cells_processed = 0;
  
  // Process cells in decreasing longest distance from registers
  for (auto it = topo_sorted_cells_.rbegin(); it != topo_sorted_cells_.rend(); ++it) {
    HeldCellInfo* cell_info = *it;
    cells_processed++;
    
    if (cell_info->is_fixed || !cell_info->output.pin) {
      continue; // Skip fixed cells (registers)
    }
    
    if (cell_info->variants.size() <= 1) {
      // Even single variants need verification
      if (cell_info->variants.size() == 1) {
        // Check if current single variant meets slew target
        double load_cap = resizer_->graph_delay_calc_->loadCap(cell_info->output.pin, dcalc_ap_);
        double input_slew = estimateInputSlew(cell_info->input_pins);
        double output_slew = computeOutputSlew(cell_info->variants[0], input_slew, load_cap);
        
        if (output_slew > cell_info->output.slew_target) {
          // Single variant violates target - will be addressed in next iteration
          logger_->warn(RSZ, 130, "Single variant violates slew target, will refine");
        }
      }
      continue;
    }
    
    // Calculate current load capacitance and input slew
    double load_cap = resizer_->graph_delay_calc_->loadCap(cell_info->output.pin, dcalc_ap_);
    cell_info->output.downstream_cap = load_cap;
    double input_slew = estimateInputSlew(cell_info->input_pins);
    
    // Find smallest variant that meets slew target and capacitance limits
    int best_variant_idx = -1;
    
    for (size_t i = 0; i < cell_info->variants.size(); ++i) {
      LibertyCell* variant = cell_info->variants[i];
      
      // Check capacitance limits first
      if (!checkCapacitanceLimits(variant, load_cap)) {
        continue;
      }
      
      // Compute output slew for this variant
      double output_slew = computeOutputSlew(variant, input_slew, load_cap);
      
      // Check if this variant meets the slew target
      if (output_slew <= cell_info->output.slew_target) {
        best_variant_idx = i;
        break; // Take the first (smallest) variant that meets target
      }
    }
    
    // Apply the change if we found a valid variant different from current
    if (best_variant_idx >= 0 && best_variant_idx != cell_info->current_variant_idx) {
      LibertyCell* best_cell = cell_info->variants[best_variant_idx];
      
      if (resizer_->replaceCell(cell_info->instance, best_cell, true)) {
        cell_info->current_variant_idx = best_variant_idx;
        cells_changed++;
      }
    }
    // If no variant meets the target, leave unchanged (Held: "left to next global iteration")
  }
  
  logger_->info(RSZ, 127, "Assignment: processed {} cells, changed {} cells", 
                cells_processed, cells_changed);
}

void HeldGateSizing::performTimingAnalysis() {
  // Held's "Timing analysis" - complete STA with forward and backward passes
  sta_->findDelays();      // Forward pass: compute arrival times
  sta_->findRequireds();   // Backward pass: compute required times
  
  // Update our data structures with timing results
  endpoints_count_ = 0;
  for (auto& [inst, cell_info] : cell_info_map_) {
    if (cell_info.output.pin) {
      Pin* pin = cell_info.output.pin;
      
      // Get timing values from STA
      cell_info.output.arrival_time = sta_->pinArrival(pin, RiseFall::rise(), MinMax::max());
      
      sta::Vertex* vertex = resizer_->graph_->pinDrvrVertex(pin);
      if (vertex) {
        cell_info.output.required_time = sta_->vertexRequired(vertex, RiseFall::rise(), MinMax::max());
        cell_info.output.slack = cell_info.output.required_time - cell_info.output.arrival_time;
        cell_info.output.slew_actual = sta_->vertexSlew(vertex, RiseFall::rise(), MinMax::max());
      }
      
      // Count endpoints (register inputs or primary outputs)
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
      }
    }
  }
}

void HeldGateSizing::refineSlewTargets() {
  // Held's Algorithm 2: Refine slew targets
  
  // Algorithm 2, Line 1: θ_k ← 1 / log(k + const)
  double theta_k = 1.0 / log(iteration_ + log_constant_);
  
  int targets_changed = 0;
  
  for (auto& [inst, cell_info] : cell_info_map_) {
    if (!cell_info.output.pin) continue;
    
    // Algorithm 2, Line 2: slk^-(c) ← min{slack(p') | p' ∈ preds of c}
    double slk_minus = std::numeric_limits<double>::infinity();
    for (Instance* pred_inst : cell_preds_[inst]) {
      auto pred_it = cell_info_map_.find(pred_inst);
      if (pred_it != cell_info_map_.end() && pred_it->second.output.pin) {
        double pred_slack = sta_->pinSlack(pred_it->second.output.pin, MinMax::max());
        slk_minus = std::min(slk_minus, pred_slack);
      }
    }
    if (slk_minus == std::numeric_limits<double>::infinity()) {
      slk_minus = 0.0; // No predecessors
    }
    cell_info.slack_minus = slk_minus;
    
    // Algorithm 2, Lines 3-14: For all p ∈ Pout(c)
    Pin* output_pin = cell_info.output.pin;
    
    // Algorithm 2, Line 4: slk^+(p) ← slack(p)
    double slk_plus = sta_->pinSlack(output_pin, MinMax::max());
    
    // Algorithm 2, Line 5: lc(p) ← max{slk^+(p) - slk^-(c), 0}
    double lc = std::max(slk_plus - slk_minus, 0.0);
    
    double delta_slew_target = 0.0;
    
    // Algorithm 2, Lines 6-11: Conditional logic
    if (slk_plus < 0.0 && lc == 0.0) {
      // Algorithm 2, Line 7: Δslewt ← -min{θ_k·γ·|slk^+(p)|, max_change}
      delta_slew_target = -std::min(theta_k * gamma_ * std::abs(slk_plus), max_change_);
    } else {
      // Algorithm 2, Line 9: slk^+(p) ← max{slk^+(p), lc(p)}
      slk_plus = std::max(slk_plus, lc);
      // Algorithm 2, Line 10: Δslewt ← +min{θ_k·γ·|slk^+(p)|, max_change}
      delta_slew_target = +std::min(theta_k * gamma_ * std::abs(slk_plus), max_change_);
    }
    
    // Algorithm 2, Line 12: slewt(p) ← slewt(p) + Δslewt
    double new_slew_target = cell_info.output.slew_target + delta_slew_target;
    
    // Algorithm 2, Line 13: Project slewt(p) into [slewt_min([p]), slewlim(p)]
    // Compute bounds for this specific cell
    double load_cap = resizer_->graph_delay_calc_->loadCap(output_pin, dcalc_ap_);
    double input_slew = estimateInputSlew(cell_info.input_pins);
    double slew_min = computeMinAchievableSlew(&cell_info, load_cap, input_slew);
    double slew_max = cell_info.output.slew_limit_from_sinks;
    
    // Clamp to bounds
    new_slew_target = std::max(slew_min, std::min(new_slew_target, slew_max));
    
    if (std::abs(new_slew_target - cell_info.output.slew_target) > 1e-15) {
      targets_changed++;
    }
    
    cell_info.output.slew_target = new_slew_target;
  }
  
  logger_->info(RSZ, 128, "Refinement: θ_k={:.3f}, {} targets changed", 
                theta_k, targets_changed);
}

double HeldGateSizing::computeOutputSlew(LibertyCell* cell,
                                        double input_slew,
                                        double load_cap) {
  // Find output port
  LibertyPort* output_port = nullptr;
  sta::LibertyCellPortIterator port_iter(cell);
  while (port_iter.hasNext()) {
    LibertyPort* port = port_iter.next();
    if (port->direction()->isOutput()) {
      output_port = port;
      break;
    }
  }
  
  if (!output_port) return default_max_slew_;
  
  // Use resizer's gate delay calculation
  ArcDelay delays[RiseFall::index_count];
  Slew slews[RiseFall::index_count];
  resizer_->gateDelays(output_port, load_cap, dcalc_ap_, delays, slews);
  
  return std::max(slews[RiseFall::riseIndex()], slews[RiseFall::fallIndex()]);
}

double HeldGateSizing::estimateInputSlew(const std::vector<Pin*>& input_pins) {
  // Held's specification: "Input slews are estimated from predecessor targets by:
  // est_slew(p') := θ·slewt(p') + (1-θ)·slew(p')"
  
  if (input_pins.empty()) {
    return default_input_slew_;
  }
  
  double theta_k = 1.0 / log(iteration_ + log_constant_);
  double worst_input_slew = default_input_slew_;
  
  for (Pin* input_pin : input_pins) {
    // Find driving pin
    Pin* driving_pin = nullptr;
    PinConnectedPinIterator* conn_iter = network_->connectedPinIterator(input_pin);
    while (conn_iter->hasNext()) {
      const Pin* conn_pin = conn_iter->next();
      if (conn_pin != input_pin && network_->direction(conn_pin)->isOutput()) {
        driving_pin = const_cast<Pin*>(conn_pin);
        break;
      }
    }
    delete conn_iter;
    
    if (driving_pin) {
      // Get actual slew from STA
      sta::Vertex* drvr_vertex = resizer_->graph_->pinDrvrVertex(driving_pin);
      double actual_slew = default_input_slew_;
      double target_slew = default_input_slew_;
      
      if (drvr_vertex) {
        actual_slew = sta_->vertexSlew(drvr_vertex, RiseFall::rise(), MinMax::max());
      }
      
      // Find target slew from our cell info map
      Instance* driving_inst = network_->instance(driving_pin);
      auto it = cell_info_map_.find(driving_inst);
      if (it != cell_info_map_.end()) {
        target_slew = it->second.output.slew_target;
      }
      
      // Apply Held's mixing formula
      double est_slew = theta_k * target_slew + (1.0 - theta_k) * actual_slew;
      
      // Add wire degradation
      double wire_degradation = computeWireSlewDegradation(driving_pin, input_pin);
      est_slew += wire_degradation;
      
      worst_input_slew = std::max(worst_input_slew, est_slew);
    }
  }
  
  return worst_input_slew;
}

double HeldGateSizing::computeWireSlewDegradation(Pin* driver_pin, Pin* sink_pin) {
  // Use OpenROAD's existing infrastructure for wire slew degradation
  // This is a simplified model - could be enhanced with actual RC calculation
  Net* net = network_->net(driver_pin);
  if (!net) return 0.0;
  
  // Use wire capacitance as proxy for RC delay degradation
  double total_cap = resizer_->graph_delay_calc_->loadCap(driver_pin, dcalc_ap_);
  
  // Simplified RC model: degradation proportional to wire capacitance
  // Use a conservative estimate of wire capacitance contribution
  double estimated_wire_cap = total_cap * 0.1; // Assume 10% of load is wire
  return estimated_wire_cap * 1e11; // Convert to reasonable slew degradation (empirical factor)
}

double HeldGateSizing::computeMinAchievableSlew(HeldCellInfo* cell_info, 
                                               double load_cap, 
                                               double input_slew) {
  if (cell_info->variants.empty()) return default_min_slew_;
  
  // Find minimum slew achievable by the largest (strongest) variant
  LibertyCell* largest_variant = cell_info->variants.back(); // Sorted by area, so last is largest
  return computeOutputSlew(largest_variant, input_slew, load_cap);
}

double HeldGateSizing::computeSlewLimitFromSinks(HeldCellInfo* cell_info) {
  // Recompute the slew limit based on current sink constraints
  return cell_info->output.slew_limit_from_sinks; // Already computed in initialization
}

bool HeldGateSizing::checkCapacitanceLimits(LibertyCell* cell, double load_cap) {
  // Check if the cell can drive the given load capacitance
  // Find output port and check its capacitance limit
  LibertyPort* output_port = nullptr;
  sta::LibertyCellPortIterator port_iter(cell);
  while (port_iter.hasNext()) {
    LibertyPort* port = port_iter.next();
    if (port->direction()->isOutput()) {
      output_port = port;
      break;
    }
  }
  
  if (!output_port) return false;
  
  // Check capacitance limit
  float cap_limit;
  bool exists;
  output_port->capacitanceLimit(MinMax::max(), cap_limit, exists);
  if (exists && load_cap > cap_limit) {
    return false;
  }
  
  return true;
}

double HeldGateSizing::computeWorstSlack() {
  return sta_->worstSlack(MinMax::max());
}

double HeldGateSizing::computeTotalNegativeSlack() {
  return sta_->totalNegativeSlack(MinMax::max());
}

double HeldGateSizing::computeAverageCellArea() {
  if (cell_info_map_.empty()) return 0.0;
  
  double total_area = 0.0;
  for (const auto& [inst, cell_info] : cell_info_map_) {
    LibertyCell* cell = network_->libertyCell(inst);
    if (cell) {
      total_area += cell->area();
    }
  }
  
  return total_area / cell_info_map_.size();
}

double HeldGateSizing::computeHeldObjective() {
  // Held's objective: WS + SNS + avg_area
  double ws = std::abs(computeWorstSlack());
  double sns = computeTotalNegativeSlack() / std::max(1.0, (double)endpoints_count_);
  double avg_area = computeAverageCellArea();
  
  return ws + sns + avg_area;
}

bool HeldGateSizing::checkHeldStoppingCriterion(double prev_ws, double prev_sns, double prev_area) {
  // Held's stopping criterion: if WS worsened AND overall objective worsened
  double curr_ws = std::abs(computeWorstSlack());
  double curr_sns = computeTotalNegativeSlack() / std::max(1.0, (double)endpoints_count_);
  double curr_area = computeAverageCellArea();
  
  bool ws_worsened = curr_ws > prev_ws;
  bool obj_worsened = (curr_ws + curr_sns + curr_area) > (prev_ws + prev_sns + prev_area);
  
  return ws_worsened && obj_worsened;
}

void HeldGateSizing::saveCurrentAssignmentAsBest() {
  best_assignment_.clear();
  for (const auto& [inst, cell_info] : cell_info_map_) {
    best_assignment_[inst] = cell_info.current_variant_idx;
  }
}

void HeldGateSizing::restoreBestAssignment() {
  for (auto& [inst, cell_info] : cell_info_map_) {
    auto it = best_assignment_.find(inst);
    if (it != best_assignment_.end()) {
      int best_idx = it->second;
      if (best_idx != cell_info.current_variant_idx && 
          best_idx < (int)cell_info.variants.size()) {
        LibertyCell* best_cell = cell_info.variants[best_idx];
        LibertyCell* current_cell = network_->libertyCell(inst);
        if (best_cell != current_cell) {
          resizer_->replaceCell(inst, best_cell, true);
          cell_info.current_variant_idx = best_idx;
        }
      }
    }
  }
  
  // Rerun timing analysis after restoration
  performTimingAnalysis();
}

void HeldGateSizing::setDefaultSlewLimits(double max_slew, double min_slew) {
  default_max_slew_ = max_slew;
  default_min_slew_ = min_slew;
}

}  // namespace rsz