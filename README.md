# COMP_ENG-357-Final
Final Project for COMP_ENG 357: Design Automation in VLSI

Xuyi Zhou, Michael Mao, Daniel Hufnagle (Northwestern University)

The project aims to develop a GPU-accelerated parallel optimization framework for VLSI design, focusing on one of gate-sizing, Vth assignment, or placement optimization (More than one may be parallelized and accelerated if time permits, but for now, we will aim for accomplishing just one of the three). Parallel CPU-based optimization, while offering tremendous improvements to the speed at which circuit optimization can be done, still has room for performance improvements that may be doable with a GPU.

We will go through [OpenRoad](https://github.com/the-openroad-project), looking for algorithms in these stages of VLSI physical design that will be efficiently parallelizable on a GPU. As OpenRoad already implements these algorithms in C++, utilizing CUDA for parallelization on an Nvidia GPU should be doable given our time constraints. We will maintain our own fork of OpenRoad, where we will aim to implement our GPU-accelerated algorithms and then test against OpenRoad’s.

Overleaf Link: https://www.overleaf.com/6493982245rtmtzqskkryx#5f39fb

