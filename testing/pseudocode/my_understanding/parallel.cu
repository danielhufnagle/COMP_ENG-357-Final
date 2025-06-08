// section 1: initializing slew targets
void initialize_slew_targets() {
	for (output_pin : output_pins) {
		double corresponding_input_slewlim = max(output_pin.next_input_pin.slew_lim);
		double slew_degradation = slew_degrad(output_pin, output_pin.next_input_pin);
		output_pin.slew_target = corresponding_input_slewlim - slew_degradation;
	}
}
// cuda kernel
__global__ void initialize_slew_targets_kernel(OutputPin* output_pins, int num_pins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pins) {
        OutputPin& output_pin = output_pins[idx];
        double corresponding_input_slewlim = max(output_pin.next_input_pin.slew_lim);
        double slew_degradation = slew_degrad(output_pin, output_pin.next_input_pin);
        output_pin.slew_target = corresponding_input_slewlim - slew_degradation;
    }
}

// launching the cuda kernel
int threadsPerBlock = 256;
int blocksPerGrid = (num_output_pins + threadsPerBlock - 1) / threadsPerBlock;
initialize_slew_targets_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output_pins, num_output_pins);


// section 2: assigning cells to library cells
void input_slew_estimation() {
	// theta shifts from 1 to 0 with each progressive iteration
	input_pin.estimated_slew = theta * input_pin.predecessor.slew_target - (1-theta) * input_pin.predecessor.slew + slew_degrad(input_pin, input_pin.predecessor);
}

void assign_cells_to_library_cells() {
	// assume that library cells are not sorted by size
	library_idx;
	double cell_size = DOUBLE_MAX; // from <climits>
	for (size_t i = 0; i < library_cells.size(); i++) {
		bool slew_targets_met = local_timing_analysis();
		if (slew_targets_met && library_cells[i].size < cell_size) {
			library_idx = i;
			cell_size = library_cells[i].size;
		}
	}
}

// data structures for CUDA implementation
struct LibraryCell {
    double size;
    // Add other necessary fields
};

struct Cell {
    int best_library_idx;
    double best_size;
    // Any other relevant fields
};

// cuda kernel
__global__ void assign_cells_kernel(Cell* cells, const LibraryCell* library_cells, int num_cells, int num_library_cells) {
    int cell_idx = blockIdx.x;
    int lib_idx = threadIdx.x;

    extern __shared__ double shared_sizes[];
    __shared__ int shared_indices[1024]; // Assuming up to 1024 library cells

    if (cell_idx >= num_cells || lib_idx >= num_library_cells) return;

    const Cell cell = cells[cell_idx];
    const LibraryCell libcell = library_cells[lib_idx];

    bool valid = local_timing_analysis(cell, libcell);
    shared_sizes[lib_idx] = valid ? libcell.size : DBL_MAX;
    shared_indices[lib_idx] = valid ? lib_idx : -1;

    __syncthreads();

    // Parallel reduction to find min size
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lib_idx < stride && shared_sizes[lib_idx + stride] < shared_sizes[lib_idx]) {
            shared_sizes[lib_idx] = shared_sizes[lib_idx + stride];
            shared_indices[lib_idx] = shared_indices[lib_idx + stride];
        }
        __syncthreads();
    }

    if (lib_idx == 0) {
        cells[cell_idx].best_size = shared_sizes[0];
        cells[cell_idx].best_library_idx = shared_indices[0];
    }
}

// host-side code for assignment
void assign_cells_to_library_cells(Cell* h_cells, int num_cells, LibraryCell* h_libcells, int num_library_cells) {
    Cell* d_cells;
    LibraryCell* d_libcells;

    cudaMalloc(&d_cells, sizeof(Cell) * num_cells);
    cudaMalloc(&d_libcells, sizeof(LibraryCell) * num_library_cells);

    cudaMemcpy(d_cells, h_cells, sizeof(Cell) * num_cells, cudaMemcpyHostToDevice);
    cudaMemcpy(d_libcells, h_libcells, sizeof(LibraryCell) * num_library_cells, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024; // max threads per block
    assign_cells_kernel<<<num_cells, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_cells, d_libcells, num_cells, num_library_cells);

    cudaMemcpy(h_cells, d_cells, sizeof(Cell) * num_cells, cudaMemcpyDeviceToHost);

    cudaFree(d_cells);
    cudaFree(d_libcells);
}
// note that this assumes that local timing analysis can be done on the GPU or simplified to a fast evaluable condition

// section 3: refine slew targets
void refine_slew_targets() {
	double theta = 1 / (log(k + GAMMA)) // k is iteration number, GAMMA is ...
	double predecessor_criticality = cell.predecessor_criticality();
	for (output_pin : cell.output_pins) {
		double global_criticality = output_pin.slack;
		double local_criticality = max(global_criticality - predecessor_criticality, 0);
		double delta_slew_target;
		if (global_criticality < 0 && local_criticality = 0) {
			delta_slew_target = -1 * min(theta * lambda * abs(global_criticality), max_change);
		}
		else {
			global_criticality = max(global_criticality, local_criticality);
			delta_slew_target = min(theta * lambda * abs(global_criticality), max_change);
		}
		output_pin.slew_target += delta_slew_target;
		if (output_pin.slew_target > output_pin.slew_limit) {
			output_pin.slew_target = output_pin.slew_limit;
		}
		else if (output_pin.slew_target < min_slew) {
			output_pin.slew_target = min_slew;
		}
	}
}

__global__ void refine_slew_targets_kernel(Cell* cells, int num_cells, double theta, double lambda, double max_change, double min_slew) {
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx < num_cells) {
        Cell& cell = cells[cell_idx];
        double predecessor_criticality = cell.predecessor_criticality();
        for (int i = 0; i < cell.num_output_pins; ++i) {
            OutputPin& output_pin = cell.output_pins[i];
            double global_criticality = output_pin.slack;
            double local_criticality = max(global_criticality - predecessor_criticality, 0.0);
            double delta_slew_target;
            if (global_criticality < 0 && local_criticality == 0) {
                delta_slew_target = -1.0 * min(theta * lambda * abs(global_criticality), max_change);
            } else {
                global_criticality = max(global_criticality, local_criticality);
                delta_slew_target = min(theta * lambda * abs(global_criticality), max_change);
            }
            output_pin.slew_target += delta_slew_target;
            output_pin.slew_target = max(min(output_pin.slew_target, output_pin.slew_limit), min_slew);
        }
    }
}

