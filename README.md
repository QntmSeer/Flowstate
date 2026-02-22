# StateFlow

**A Hybrid Dynamical Systems Engine for Cellular State Modeling**

StateFlow treats cells not as static clusters, but as stochastic dynamical systems. It moves beyond "What cell types exist?" and answers "How do cells transition between states?"

By modeling single-cell evolution as a Switching Linear Dynamical System (SLDS), StateFlow infers:
1. **Discrete Markov states** (metastable transcriptional programs)
2. **Continuous regulatory dynamics** (within-state pathway drift)

## Core Insight
Cells behave like stochastic dynamical systems occupying metastable states, transitioning probabilistically, and drifting continuously within regulatory regimes.

StateFlow captures this using a fully probabilistic, generative **Hybrid State-Space Model**:
$$z_{t+1} = A_{s_t} z_t + \epsilon$$
$$x_t = C z_t + \delta$$

Outputs include:
- Transition probabilities between cell programs
- Biological stability eigenvalues (identifying attractor basins)
- Stationary distributions (equilibrium population fates)

## Installation

```bash
git clone https://github.com/yourusername/stateflow.git
cd stateflow
pip install -e .
```

*Requires JAX for high-performance tensor operations and AutoDiff.*

## Built With

- **JAX**: For `vmap` parallelization and `lax.scan` exact inference.

## Current Status
**Phase 1** (Single-Cell SLDS Inference Engine) is conceptually proven and mathematically validated on simulated data using a Variational Expectation-Maximization (vEM) algorithm.

*Next up: Application to real biological continuous trajectories (e.g. myeloid differentiation).*
