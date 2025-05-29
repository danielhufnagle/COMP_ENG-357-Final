/*held.h*/

//
// Header for all structures and the like needed for the Held Gate Sizing Algorithm
//

#pragma once

#include <vector>

using namespace std;

namespace held {

    // ----- CONSTANTS ----- //
    /* Used in refining slew targets */
    const double MAX_CHANGE = 5.0;
    const double GAMMA = 0.1;
    const double LAMBDA = 0.7;

    // ----- PIN ----- //
    /* Defines a pin of a cell */
    struct Pin {
        string name;
        double arrival_time;
        double required_arrival_time;
        double slew;
        double slew_target;
        double slew_limit;
        double capacitance_limit;

        double slack() const {
            return required_arrival_time - arrival_time;
        }
    };

    // ----- CELL ----- //
    /* Defines a cell */
    struct Cell {
        string name;
        vector<Pin*> inputs;
        vector<Pin*> outputs;
        vector<std::string> equivalent_cells;
        string assigned_cell;

        // Dummy method to choose a minimum size cell meeting slew targets
        void assignCell(const std::unordered_map<std::string, double>& input_slews) {
            // Real implementation would evaluate delay/slew models here
            assigned_cell = equivalent_cells.front();  // Simplified for now
        }
    };

    // ----- Held Algorithm Functions ----- //
    
    void initializeSlewTargets(std::vector<Cell*>& cells);
    void cellAssignment(std::vector<Cell*>& cells, double input_slew);
    double estimateSlewDegradation(double load_cap, double input_slew);
    void timingAnalysis(std::vector<Pin*>& pins);
    void refineSlewTargets(std::vector<Cell*>& cells, int iteration);
    void fastGateSizing(std::vector<Cell*>& cells, std::vector<Pin*>& pins, int max_iterations = 10);
}