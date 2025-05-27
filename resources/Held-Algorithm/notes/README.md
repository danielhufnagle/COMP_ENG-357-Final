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
Besides slacks, slew limits $slew(q)\leq slewlim(q)$ for all input pins $q\in P_{in}(c),c\in C$ as well as capacitance limits $downcap(p)\leq caplim(p)$ for all output pins $p\in P_{out}(c)$ must be preserved.
Gate sizing is mostly applied when no feasible solution exists, a practical objective is to maximize the worst slack, but to also push less critical negative slacks towards 0. This solution reduces the need for other more intensive optimization routines.
## Fast global gate sizing
The slew targeting algorithm is as follows
```python
initialize_slew_targets(output_pins)
while (!stopping_criterion):
  assign_library_cells(cells)
  timing_analysis()
  refine_slew_targets()
return best_assignment
```
Breaking this down a little more:

```python
initialize_slew_targets(output_pins)
```

Slew targets $slewt(p)$ is assigned to all output pins $p\in P_{out}(c), c\in C$. Slew targets are initialized such that the slew limits will just be met at subsequent sinks (accounting for slew degradation on the wires).

```python
assign_library_cells(cells)
```

Cell sizes are chosen such that the slew targets are met. The slew targets are updated based on an estimate of the slew gradient that guides the cell to a locally optimum solution when refined.

To bound running time, the algorithm avoids incremental timing updates. Instead the timing is updated for the complete design by a timing oracle in line 4 once per iteration. This is to allow the algorithm to take advantage of a parallel timing engine. Assigning library cells and refining slew targets can be parallelized as well.

```python
stopping_criterion
```

The stopping criterion is met when all of the following are true:
- Current cell assignment worsens the worst slack
- increases a weighted sum of the absolute worst negative slack (WS), the absolute sum of negative slacks (SNS) divided by the number of endpoints, and the average cell area
</a>

If the criterion i smet, the assignment of the previous iteration, which achieves the best present objective value, is recovered.

### Assigning cells to library cells - further breakdown
Cells are assigned to the smallest equivalent library cell such that the slew targets at all of their output pins (usually just output pin - singular) are met. This choice depends on the input slews and output loads, respectively the4 layout and sink pin capacitances of the output nets. These values depend on the sizes of other cells.

As the downstream network usually has a bigger impact on the cell timing than the input slew, it is preferable to know the exact downstream capacitances when sizing a cell. Let:
- $G_c$ be the directed graph with vertices for each $c\in C$ and an edge connecting the predecessor cells $c'$ with $c$ if there is a connection between an output pin of $c'$ and an input pin of $c$.
</a>

Cells are processed in order of decreasing distance (under unit edge lengths) from a register in the acyclic subgraph, which arises from $G_c$ by omitting edges entering register vertices.
#### Input slew estimation
When a cell $c$ is sized, the successor sizes are already known except for the registers as successors. While the slews of the input pins of cell $c$ are dependedt on the unknown predecessor sizes, they can be reasonably estimated by the slew targets.

For a pin $q\in P_{in}(c)$ with predecessor $p'\in\delta^-_{G_T}(q)$ (predecessor has a directed edge to c), the estimated final slew in $p'$ is the following:

$$
est\textunderscore slew(p')=\theta slewt(p')-(1-\theta)slew(p')
$$

Where $\theta$ is a weight that starts at 1 and progressively moves down to 0 with each global iteration. This means that in the beginning, predecessor slews will end up closer to $est\textunderscore slew(p')$ than $slew(p')$. When changes if the predecessor size become less likely with further iterations, the computed slews begin to dominate the estimate.

To get the estimate for $slew(q)$, we need to add the slew degradation on the wire $slew\textunderscore degrad(p',q)$ to $est\textunderscore slew(p')$. When the pin capacitance of $q$ changes, the slew degradation will too, which can be approximated quickly by an RC-delay model for the wire
#### Sizing
Now given the estimated predecessor slews and the layout of the output network, the minimum cell size for $c$ preserving the slew targets (and load capacitance limits) can be computed performing a local timing analysis through $c$ and its downstream wires for all available library cells in $[c]$. 

Note a level of cells with equal distance levels can be sized in parallel.

If delay and slew propagation through a cell are parameterized by load capacitance and input slews, sizing can be accelerated by look-up table, but this becomes far too inefficient for more complex delay models

To speed up the overall algorithm, we only perform one iteration of assignment, leaving it to the next global iteration to remove sub-optimal or illegal assignments.
### Refining slew targets - further breakdown
The slew target $slewt(p)$ is refined based on the *global* and *local criticality* of $p$. The global criticality $slk^+(p)$ is just the slack at $p$, with $p$ being globally critical if the slack is negative.

$$
slk^+(p)=slack(p)=rat(p)-at(p)
$$

Local criticality indicates whether the worst of the slacks in $p$ and its direct predecessors can be improved by either accelerating $c$ by decreasing $slewt(p)$, or by decreasing the input pin capacitances of $c$ by increasing $slewt(p)$.

The predecessor criticality of cell $c$ is defined as follows

$$
slk^-(c)=min(slack(p'))
$$

Given that $p'$ is a direct predecessor of $c$.

Then we can define the local criticality of $p$ by the following:

$$
lc(p)=max(slk^+(p)-slk^-(c), 0)
$$

$p$ is locally critical if $lc(p)=0$, meaning that $p$ is either located on a worst-slack path through a most critical predecessor of $c$, or $p$ is an output pin of a register whose output path is at least as critical as any path through its predecessors.

The algorithm to update the slew targets of a cell $c$ in a global iteration is as follows. 
```python
theta_k = 1/log(k + CONSTANT)
slk_minus = min(slack(p')) given p' in c.predecessors
for output_pin in output_pins(c):
  slk_plus = slack(p)
  lc(p)=max(slk_plus - slk_minus, 0)
  if slk_plus < 0 and lc(p) == 0:
    delta_slewt = -min(theta_k * gamma * slk_plus(p), max_change)
  else:
    slk_plus = max(slk_plus, lc(p))
    delta_slewt = min(theta_k * gamma * slk_plus(p), max_change)
  slewt(p) = slewt(p) + delta_slewt
  project slewt(p) into [slewt([p]), slewlim(p)]
```
Essentially, if $p$ is globally and locally critical, we decrease $slewt(p)$ by subtracting a number proportional to $slk^+(p)$ but not exceeding the constant $max\textunderscore change$. Otherwise, we increase $slewt(p)$ by adding a number proportional to the maximum of $slk^+(p)$ and $lc(p)$. The gamma ($\gamma$) constant is an estimate of $\frac{\partial slew(p)}{\partial slk^+}$. This means that $\gamma\cdot |slk^+|$ is the required slew change to reach a non-negative slack in $p$. Realistically, $\gamma$ is just set to a small constant. $\theta_k$ is a damping factor that reduces potential oscillation.

The `project slewt(p) into [slewt([p]), slewlim(p)]` means that we project the slew target into the feasible range. The maximum slew limit $slewlim(p)$ for the output pin $p$ is induced by the attached sinks:

$$
slewlim(p)=min(slewlim(q)-slewdegrad(p,q))
$$

Given that $q$ is a predecessor pin of $p$ and $slewdegrad(p,q)$ is the same as before. $slewt([p])$ is the lowest possible slew that is achievable with any equivalent cell given the current load capacitance and input slew, preventing unrealistically small slew targets.

#### Enhanced slew target refinement
In some cases this algorithm leads to overloaded cells that cannot be further enlarged, or to locally non-critical cells that cannot be downsized sufficiently because of two large successors. In an entire timing optimization flow, giate sizing is alternated with repreater insertion absorbing such situations, but this can lead to unnecessary repeaters.

These situations mean that slew targets of successors should be relaxed to enable smaller sizes, which the slew target refining algorithm already handles for locally uncritical cells. Now, when refining the slew target of a locally critical output pint $p\in P_{out}(c),c\in C$, we consider the largest estimated slew $est\textunderscore slew(p')$ of a most critical precessor pin

$$
p' \in argmax(est\textunderscore slew(r)| r\in\Gamma^-_{G_T}(Pin(c)), slack(r)=slk^-(c))
$$

If $est\textunderscore slew(p')>slewt(p)$, we increse the slew target in $p$ by

$$
slewt(p)=\lambda\cdot slewt(p)+(1-\lambda)\cdot est\textunderscore slew(p')
$$

where $0<\lambda<1$. The effect of an extraordinarily high value of $est\textunderscore slew(p')$ declines exponentially in the number of subsequent cell stages. $\lambda =0.7$ was found to be good. Note that less critical predecessors are not considered for relaxing the slew targets.

To enable the enhanced slew target computation, the cells must be traversed in signal direction, reverse to the sizing step. Again, the slew targets of all cells in a lever of equal longest path distance from a register can be updated in parallel.

## Local search gate sizing
The local search is applied to further improbe the result of the fast global gate sizing. It collects a small set of cells attached to the most critical nets and sizes them one after another to their local optimum based on more accurate slack evaluations. The next iteration starts with collecting cells from scratch.

First, we traverse all nets by increasing slack at their source pins and select all cells that are attached to the current net and to all nets that have the same slack at their sources. As soon as more than $K\in\mathbb{N}$ cells are collected the traversal of the net stops. Note this procedure collects at least the cells on the most critical paths and their direct successors for any choice of $K$. It is important to select not only the critical cells but all cells attachefd to a net, because the pin capacitances of the noncritical cells affect the timing. $K=0.2%$ of the total number of cells in the current design was found to be good.

Cells are then traversed in the order of decreasing longest path distance from a register. A cell $c\in C$ is assigned to a library cell $B\in [c]$ of minimum size such that

$$
min(0, slk^-(c),slk^+(p)|p\in P_{out}(c))
$$

is maximized. Slacks are computed by an exact analysis within a small neighborhood around $c$. The neighborhood contains its predecessor cells and all direct successors of $c$ and its predecessor pins. The algorithm stops when the worst slack could not be improved in the last iteration. For runtime reasons, this criterion could be modified to stop when the worst slack improvement falls below some threshold or when some maximum number of iterations is reached.
