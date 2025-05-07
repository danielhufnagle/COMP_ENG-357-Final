# Literature for CE 357 Final Project

### Sapatnekar et al., TCAD 1993
Uses convex optimization tehcniques to address gate sizing. Gate delays are modeled withn the Elmore delay model, which is convex in terms of the gate sizes. The sizing problem can then be solved under this model by expressing it as a convex optimization task subject to design constraints such as maximum area, power, and timing bounds. This method guarantees convergence to a globally optimal solution.

### Fishburn and Dunlop, ICCAD 1985
Uses a greedy strategy by identifying the critical path and resizing gates along this path based on delay sensitivities. This is called Timing Optimization by Iterative Local Sizing (TILOS). Will not produce global optimum, but can produce good results in practice and is the foundation for algorithms developed later

### Boyd et al., Operations Research 2005
Uses geometric programming, a class of convex optimization that can handle posymonial (think of this as an extension of polynomial) functions. Gate delays and constraints are modelled using posynomials and GP is performed to optimize. 

### Khatkhate et al. DAC 2004
Uses principles of lagrangian relaxation and network flow theory. Relax hard timing constraints by introducing lagrance multipliers and incorporate them into sizing objective function. Circuit becomes modeled by timing graph where gate sizing decisions affect propagation delay. Using network flow formulations, available slack is distributed across the circuit to minimize delay violations while reducing area and power.s
