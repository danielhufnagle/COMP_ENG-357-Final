// GPU Implementation of HeldGateSizing.hh
// Written by Daniel Hufnagle, Xuyi Zhou, Michael Mao

#pragma once

// standard library dependencies
#include <array>
#include <limits>
#include <unordered_map>
#include <vector>

// openROAD internal dependencies
// static timing analysis dependencies
#include "db_sta/dbNetwork.hh"
#include "db_sta/dbSta.hh"
#include "sta/Liberty.hh"
#include "sta/Sta.hh"
#include "sta/PortDirection.hh"
// Logger
#include "utl/Logger.h"

// Held resizer namespace
namespace rsz {
    
using ult::Logger;
    
using sta::Corner;
using sta::DcalcAnalysusPt;
using sta::dbNetwork;
using sta::dbSta;
using sta::Instance;
using sta::LibertyCell;
using sta::Net;
using sta::Pin;
using sta::RiseFall;
using sta::Slew;

class Resizer;


// Structure to represent a cell instance in Held's algorithm
struct HeldCellInfo {
    Instance* instance; // OpenROAD instance
    std::vector<LibertyCell*> variants; // Available equivalent library cells
    int current_variant_idx; // Index of current library cell
    bool is_fixed; // is the cell fixed? True for registers and fixed cells
    double distance_from_src; // longest distance from register

    // Timing data for the cell's output
    struct OutputData {
        Pin* pin; // output pin
        double slew_target; // Current slew target
        double slew_actual; // Actual slew from static timing analysis
        double arrival_time; // arrival time
        double required_time; // required arrival time
        double slack; // slack = required_time - arrival_time
        double wire_cap; // wire capacitance
        double downstream_cap; // total load capacitance
        std::vector<Pin*> sinks; // sink pins driven by this output
        double slew_limit_from_sinks; // computed from sink limits minus wire degradation
        doble min_achievable_slew // minimum slew achievable by any variant (library cell)
    } output;

    // Input pins data
    std::vector<Pin*> input_pins;

    // Predecessor slack for Held's refinement algorithm
    double slack_minus;
};


// Held's fast global gate sizing algorithm implementation
// Integrated with OpenROAD's resizer infrastructure
class HeldGateSizing {
public:
    HeldGateSizing(Resizer* resizer, double gamma = 0.5, double max_change = 0.05, int max_iterations = 50);
    ~HeldGateSizing() = default;

    // Main algorithm entry point
    bool optimizeGateSizing(double clock_period);

    // Configuration
    void setGamma(double gamma) { gamma_ = gamma; }
    void setMaxChange(double max_change) { max_change_ = max_change; }
    void setMaxIterations(int max_iters) { max_iterations_ = max_iters; }
    void setDefaultSlewLimits(double max_slew, double min_slew = 0.0);
    void setLogConstant(double log_const) { log_constant_ = log_const; }

private:
    // Algorithm phases (Held's Algorithm 1)
    void initializeSlewTargets();
    void assignCellsToMeetSlewTargets();
    void performTimingAnalysis();
    void refineSlewTargets();

    // Helper functions
    void buildCellInfoMap();
    void computeTopologicalLevels();
    std::vector<LibertyCell*> getEquivalentCells(Instance* inst);

    // Held's specific calculations
    double computeOutputSlew(LibertyCell* cell, double input_slew, double load_cap);
    double estimateInputSlew(const std::vector<Pin*>& input_pins);
    double computeWireSlewDegradation(Pin* driver_pin, Pin* sink_pin);
    double computeMinAchievableSlew(HeldCellInfo* cell_info, double load_cap, double input_slew);
    double computeSlewLimitFromSinks(HeldCellInfo* cell_info);
    bool checkCapacitanceLimits(LibertyCell* cell, double load_cap);

    // Topological distance computation (longest path from register)
    void computeLongestDistanceFromRegisters();
    void buildCellGraph();

    // Objective evaluation
    double computeWorstSlack();
    double comoputeTotalNegativeSlack();
    double computeAverageCellArea();
    double computeHeldObjecttive(); // Worst Slack + Sum Negative Slacks + Average Area

    // best solutiont racking
    void saveCurrentAssignmentAsBest();
    void restoreBestAssignment();

    // Held's stopping criterion
    bool checkHeldStoppingCriterion(double prev_ws, double prev_sns, double prev_area);

    // OpenROAD integration
    Resizer* resizer_;
    Logger* logger_;
    dbNetwork* network_;
    dbSta* sta_;

    // Algorithm parameters
    double clock_period_;
    double gamma_; // tuning parameter for slew target adjustmnet
    double max_change_; // maximum allowed slew target change per iteration
    int max_iterations_;
    double log_constant_; // constant for theta_k = 1/log(k + const)
    
    // Current state
    int iteration_;
    int endpoints_count_;
    std::unordered_map<Instance*, HeldCellInfo> cell_info_map_;
    std::vector<HeldCellInfo*> topo_sorted_cells_;

    // Cell graph for topological ordering
    std::unordered_map<Instance*, std::vector<Instance*>> cell_graph_; // adjacency list
    std::unordered_map<Instance*, std::vector<Instance*>> cell_preds_; // predecessors

    // Best solution tracking
    std::unordered_map<Instance*, int> best_assignment_;
    double best_ws_;
    double best_sns_;
    double best_avg_area_;

    // previous iteration metrics for stopping criterion
    double prev_ws_;
    double prev_sns_;
    doulbe prev_avg_area_;

    // Default values
    double default_max_slew_;
    double default_min_slew_;
    double default_input_slew_;

    // Timing corners
    const Corner* corner_;
    const DcalcAnalysisPt* dcalc_ap_;
};
} // namespace rsz
