// HeldCudaGateSizing.cu
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cfloat>

//----------------------------------------------------------------------
// Device Kernels
//----------------------------------------------------------------------

// 1) Slew Target Initialization
//slew_lim for each pin type is defined in the cell library
//slew_degrad is estimated based on the RC delay model
__global__
void initSlewTargets(
    int    numPins,
    const int*    rowPtr,     // size numPins+1
    const int*    colIdx,     // size numEdges
    const float*  slewLim,    // [numPins]
    const float*  slewDeg,    // [numEdges]
    float*        slewT       // [numPins] out
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numPins) return;

    float m = FLT_MAX;
    for (int e = rowPtr[p]; e < rowPtr[p+1]; ++e) {
        float cand = slewLim[colIdx[e]] - slewDeg[e];
        m = (cand < m ? cand : m);
    }
    slewT[p] = m;
}

// 2) Cell Assignment to Library Cells
__global__
void assignCells(
    int    numGates,
    const float* slewT,       // [numGates]
    const float* currSlew,    // [numGates]
    const int  numLib,
    const float* libSizes,    // [numLib]
    const float* libSlewModels,// [numLib] simplistic
    int*        chosenLib     // [numGates] out
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numGates) return;

    float target = slewT[gid];
    float bestSize = FLT_MAX;
    int   bestIdx  = -1;

    // simple linear search; could be optimized
    for (int i = 0; i < numLib; ++i) {
        // estimate output slew = θ*target + (1-θ)*currSlew + model
        float est = libSlewModels[i] + target; // placeholder
        if (est <= target && libSizes[i] < bestSize) {
            bestSize = libSizes[i];
            bestIdx  = i;
        }
    }
    chosenLib[gid] = bestIdx;
}

// 3a) Forward Timing (levelized)
__global__
void forwardTiming(
    int    startPin,
    int    endPin,
    const int*    predPtr,    // size numPins+1
    const int*    predList,   // size numEdges
    const float*  delays,     // [numEdges]
    const float*  atIn,       // [numPins]
    float*        atOut       // [numPins] out
) {
    int p = startPin + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= endPin) return;

    float maxAT = 0.0f;
    for (int e = predPtr[p]; e < predPtr[p+1]; ++e) {
        int pr = predList[e];
        float a  = atIn[pr] + delays[e];
        maxAT = fmaxf(maxAT, a);
    }
    atOut[p] = maxAT;
}

// 3b) Backward Timing & Slack
__global__
void backwardTiming(
    int    startPin,
    int    endPin,
    const int*    succPtr,    // size numPins+1
    const int*    succList,   // size numEdges
    const float*  delays,     // [numEdges]
    const float*  ratIn,      // [numPins]
    float*        ratOut,     // [numPins] out
    const float*  at,         // [numPins]
    float*        slack       // [numPins] out
) {
    int p = startPin + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= endPin) return;

    float minR = FLT_MAX;
    for (int e = succPtr[p]; e < succPtr[p+1]; ++e) {
        int sq = succList[e];
        float r  = ratIn[sq] - delays[e];
        minR = fminf(minR, r);
    }
    ratOut[p] = minR;
    slack[p]  = minR - at[p];
}

// 4) Slew Target Refinement
__global__
void refineSlewTargets(
    int    numPins,
    const float*  slack,      // [numPins]
    const int*    predPtr,    // for local criticality
    const float*  at,         // [numPins]
    const float*  rat,        // [numPins]
    float        gamma,
    float        rfac,
    float*       slewT       // [numPins] in/out
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numPins) return;

    // compute local criticality: lc = max(slack[p] - worstPredSlack, 0)
    float worstPred = FLT_MAX;
    for (int e = predPtr[p]; e < predPtr[p+1]; ++e) {
        int pr = e; // if predList provided, use that
        worstPred = fminf(worstPred, rat[pr]-at[pr] /* slack[pr] */);
    }
    float lc = fmaxf(slack[p] - worstPred, 0.0f);

    // adjust slewT
    if (slack[p] <= 0) {
        slewT[p] -= gamma * fabsf(slack[p]);
    } else {
        slewT[p] += rfac * fmaxf(slack[p], lc);
    }
}

//----------------------------------------------------------------------
// Host Orchestration
//----------------------------------------------------------------------

void runHeldOnGPU(
    /* host data pointers: adjacency, initial slewLim, etc. */
) {
    // 1) Allocate & copy data to device
    //    cudaMalloc(...); cudaMemcpy(...);

    dim3 block(256), gridPins((numPins+block.x-1)/block.x),
               gridGates((numGates+block.x-1)/block.x);

    // Precompute level boundaries on host: levelStart[i], levelEnd[i]
    // ...

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        // 1. init slews
        initSlewTargets<<<gridPins, block>>>(numPins, d_rowPtr, d_colIdx, d_slewLim, d_slewDeg, d_slewT);

        // 2. assign cells
        assignCells<<<gridGates, block>>>(numGates, d_slewT, d_currSlew, numLib, d_libSizes, d_libModels, d_chosen);

        // 3a. forward timing
        for (int lvl = 0; lvl < numLevels; ++lvl) {
            int s = levelStart[lvl], e = levelEnd[lvl];
            int cnt = e - s;
            forwardTiming<<<(cnt+block.x-1)/block.x, block>>>(s, e, d_predPtr, d_predList, d_delays, d_atOld, d_atNew);
            std::swap(d_atOld, d_atNew);
        }

        // 3b. backward timing + slack
        for (int lvl = numLevels-1; lvl >= 0; --lvl) {
            int s = levelStart[lvl], e = levelEnd[lvl];
            int cnt = e - s;
            backwardTiming<<<(cnt+block.x-1)/block.x, block>>>(s, e, d_succPtr, d_succList, d_delays, d_ratOld, d_ratNew, d_atOld, d_slack);
            std::swap(d_ratOld, d_ratNew);
        }

        // 4. refine slew targets
        refineSlewTargets<<<gridPins, block>>>(numPins, d_slack, d_predPtr, d_atOld, d_ratOld, gamma, rfac, d_slewT);

        // 5. stopping: copy back worst slack or use reduction
        cudaDeviceSynchronize();
        float worstSlack = reduceMaxOnHost(d_slack, numPins);
        if (fabs(prevWorst - worstSlack) < EPS) {
            break;
        }
        prevWorst = worstSlack;
    }

    // copy final slewT, chosenLib, etc. back to host
}

