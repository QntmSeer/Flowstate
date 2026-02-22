# Flowstate
**A Hybrid Dynamical Systems Engine for Cellular State Modeling**

Flowstate treats cells not as static clusters, but as stochastic dynamical systems. It moves beyond "What cell types exist?" and answers "How do cells transition between states?"

By modeling single-cell evolution as a Switching Linear Dynamical System (SLDS), Flowstate infers:
1. **Discrete Markov states** (metastable transcriptional programs)
2. **Continuous regulatory dynamics** (within-state pathway drift)

![Ground Truth Bifurcation](assets/benchmark_bifurcation_gt.png)

## Core Insight
Cells behave like stochastic dynamical systems occupying metastable states, transitioning probabilistically, and drifting continuously within regulatory regimes.

Flowstate captures this using a fully probabilistic, generative **Hybrid State-Space Model**:
$$z_{t+1} = A_{s_t} z_t + \epsilon$$
$$x_t = C z_t + \delta$$

Outputs include:
- Transition probabilities between cell programs mathematically extracted directly from continuous expression
- Biological stability eigenvalues (identifying attractor basins vs transit amplifying points)
- Stationary distributions (equilibrium population fates)

## Features & Engines

### 1. Discrete Variational EM Inference
Infers the continuous trajectory $z$ and discrete state $s$ using a fully scalable sequence of Kalman Smoothing (Continuous E-Step) and Forward-Backward HMM (Discrete E-Step) operations over the observed gene expression trajectories, natively discovering topologies (bifurcations, trees) without arbitrary graph kNN limits.

![Inferred Topology Graph](assets/benchmark_bifurcation_inferred.png)

### 2. Continuous-Time Neural SDEs
Because biological transcriptomic data is rarely evenly spaced, Flowstate implements continuous-time dynamics. Using explicit Runge-Kutta Diffrax ODE solvers, Flowstate continuously integrates latent neural drift and diffusion fields $dz_t = f_{s_t}(z)dt + g_{s_t}(z)dW_t$ directly from unaligned scRNA-seq.

![Neural SDE Filtering](assets/test_sde_inference.png)

### 3. Integrated Lineage Barcoding
Modern clonal lineage tags (LARRY, CellTagging) provide actual ground truth familial relationships. Flowstate utilizes a Clonal Regularization Penalty during the E-Step to mathematically force cells sharing an ancestral barcode to respect descent constraints, preventing scattered artifactual projections.

![Lineage Barcode Penalty](assets/test_lineage_inference.png)

## Installation & Tests

Flowstate relies heavily on `jax`, `diffrax`, and `equinox`.

```bash
git clone https://github.com/QntmSeer/Flowstate.git
cd Flowstate
pip install -e .
```

Flowstate comes with a rigorous end-to-end stress testing suite that ensures mathematical validity across the discrete EM, the barcode regularizer, and the Optax SDE gradient integrations.

```bash
python stress_test_engine.py
```

### Real Biological Data Benchmarks
Flowstate successfully processes unaligned single-cell trajectories like the **Moignard 2015 Hematopoiesis** differentiating dataset, extracting probabilistic differentiation stages without KNN graphs:
![Moignard Application](assets/real_data_moignard.png)

## Current Status
**Phase 1** (Discrete SLDS) and **Phase 2** (Continuous SDEs + Barcodes) are formally complete, mathematically guaranteed via the VEM benchmarking suite.

---

## About & Disclaimer
**Flowstate is a highly experimental, exploratory project.** 
While the underlying mathematical engines (Variational EM, Continuous-Discrete Kalman Filters, Neural SDEs) are implemented based on rigorous theoretical frameworks, their application to biological single-cell transcriptomic modeling in this repository involves significant assumptions. 

The models presented here may be mathematically simplified or biologically unverified in certain complex real-world edge cases. This repository serves as a proof-of-concept for integrating complex continuous-time dynamics and lineage regularization into single-cell trajectory inference, and should not be used for critical clinical or diagnostic decisions without extensive further validation.
