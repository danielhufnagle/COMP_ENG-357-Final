# Held's Algorithm Summary
## Introduction
The algorithm is a two-step process:
1. Run a fast global gate sizing that falls into the category of delay budgeting heuristics, but instead of delay budgets, we distribute slew (transition) time targets. Traverse netlist reverse to signal direction. This means that input slews are not known as preceding cells still have to be sized. The slew targets at the predecessors provide the capability to compute good estimates for the input slews.
2. Run a local search on the most critical paths to guide the worst path delay further into a local optimum.
## The gate sizing problem
Let:
- $C$: the set of cells
- $P$: the set of pins on a chip
- $P(c)$: the set of pins on a cell
- $P_{in}(c)$: the set of input pins on a cell
- $P_{out}(c)$: the set of output pins on a cell
</a>

Also assume that $C$ contains one fixed cell to represent the primary inputs and outputs of the chip.

The timing constraints considered are based on static timing analysis with slew propagation where:
- $G^T$: directed timing graph that describes signal propagation on the set of pins
- $at(p)$: latest arrival time of a signal arriving at pin $p$
- $slew(p)$: slew of a signal arriving at pin $p$
- $rat(p)$: required arrival time of a signal at pin $p$
</a>

Note that all three of $at(p)$, $slew(p)$ and $rat(p)$ are propagated to each pin $p$ in the design.
- $slack(p)$: defined as $rat(p)-at(p)$, it indicates the criticality of the timing signal in $p$. $slack(p)\geq 0$ implies that the paths through $p$ could be delayed by that amount without violating the timing constraints. Otherwise, the most critical path needs to be accelerated by at least $-slack(p)$.
</a>

We will also assume the propagation of single slew values throughout this paper.

The gate sizing problem consists of taking each cell in set of cells ($c\in C$) and assigning them to a library cell in the discrete cell library ($B\in \mathfrak{B}$). \ Note also that
- $[c] \subset \mathfrak{B}$: the set of logically equivalent cells to which $c$ may be assigned
</a>

In typical formulations, the assignment should be chosen such that some objective function, e.g. the total power or area consumption is minimized while all timing constraints are met, i.e. $slack(p)\geq0$ for all $p\in P$.\
Besides slacks, slew limits $slew(q)\leq slewlim(q)$ for all input pins $q\in P_{in}(c),\;c\in C$ as well as capacitance limits $downcap(p)\leq caplim(p)$ for all output pins $p\in P_{out}(c)$ must be preserved.
Gate sizing is mostly applied when no feasible solution exists, a practical objective is to maximize the worst slack, but to also push less critical negative slacks towards 0. This solution reduces the need for other more intensive optimization routines.
## Fast global gate sizing
