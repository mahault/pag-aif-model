# PAG-AIF Model: Multi-World Active Inference Simulation

A computational framework for **Predictive Action Generation** using the **Active Inference Framework (AIF)**, featuring multi-dimensional action spaces across parallel "worlds" -- spatial navigation, pose configuration, and social interaction.

## Overview

This project implements a multi-world active inference simulation where agents operate across multiple action dimensions simultaneously:

- **Spatial World** -- World actions are translated into movements of a dot on a grid/continuous space
- **Pose World** -- Pose actions are translated into body/agent configuration representations
- **Social World** -- Social actions are represented as calls, signals, or other interaction modalities

The simulation runs these worlds in parallel, with cross-modal dependencies allowing each dimension to influence the others (e.g., social signals depend on spatial proximity; pose may constrain navigation).

## Architecture

The model uses a **Factorized Hierarchical POMDP** generative model:

```
                    [Meta-Level Controller]
                   (policy arbitration across worlds)
                              |
               +--------------+--------------+
               |              |              |
       [Spatial World]  [Pose World]   [Social World]
       (navigation)     (body config)  (communication)
```

### Generative Model Components

| Component | Description |
|-----------|-------------|
| **A matrix** (likelihood) | Maps hidden states to observations, encoding cross-modal dependencies |
| **B matrix** (transitions) | Per-world dynamics with controlled coupling between worlds |
| **C matrix** (preferences) | Per-modality preferred observations encoding agent goals |
| **D matrix** (priors) | Initial beliefs about each state factor |

### State Space Factorization

```python
num_states   = [N_spatial, N_pose, N_social]       # Hidden state factors
num_obs      = [N_visual, N_proprioceptive, N_social_obs]  # Observation modalities
num_controls = [N_move, N_pose_act, N_call]         # Control factors (action dims)
```

## Approach

### Multi-World Design Rationale

The factorized state space formalism in active inference natively supports multiple hidden state factors and control factors. Each "world" corresponds to a control factor dimension, allowing:

- **Independent inference** within each world (mean-field factorization)
- **Cross-modal coupling** through shared states or likelihood dependencies
- **Hierarchical control** via meta-level policy arbitration

This design is grounded in recent work on scale-free active inference (AXIOM, 2025) and factorized multi-agent active inference (Ruiz-Serra et al., 2024).

### Implementation Strategy

1. **Phase 1** -- Spatial navigation with pymdp (dot on grid). Validate state inference and policy selection.
2. **Phase 2** -- Add pose as a second control factor. Verify factorized inference across two worlds.
3. **Phase 3** -- Add social world and introduce a second agent. Implement cross-agent observations.
4. **Phase 4** -- Scale to multiple agents and simultaneous environments. Introduce hierarchical control.
5. **Phase 5** -- Evaluate against baselines, tune precision parameters, explore deep AIF extensions if needed.

### Key Technical Considerations

- **Combinatorial policy space**: Factorized mean-field policy selection mitigates the explosion of joint policies across action dimensions.
- **Temporal scale mismatches**: Spatial navigation may operate at faster time scales than social interactions -- hierarchical temporal models address this.
- **Cross-modal dependencies**: Social signals depend on spatial proximity; pose may constrain navigation. Encoded as sparse conditional dependencies in the A matrix.
- **Ablation-first validation**: Each world is tested independently before coupling.

## Tech Stack

- **[pymdp](https://github.com/infer-actively/pymdp)** -- Active inference in discrete state spaces (JAX backend for GPU acceleration)
- **Python 3.10+**
- **JAX/NumPy** -- Numerical computation and parallelization
- **Matplotlib/Pygame** -- Visualization of multi-world simulation

## Project Structure

```
pag-aif-model/
├── src/                  # Source code
│   ├── worlds/           # World implementations (spatial, pose, social)
│   ├── agents/           # Active inference agent definitions
│   └── utils/            # Shared utilities
├── notebooks/            # Jupyter notebooks for analysis and Colab exports
├── experiments/          # Experiment configurations and scripts
├── tests/                # Unit and integration tests
├── docs/                 # Documentation and reports
└── README.md
```

## Literature Review

### Foundational Active Inference

- Friston, K. (2010). *The free-energy principle: a rough guide to the brain?* [PDF](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20rough%20guide%20to%20the%20brain.pdf)
- Parr, T., Pezzulo, G., & Friston, K. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.* MIT Press. [Open Access](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)
- Friston, K. et al. (2016). *Active inference and learning.* [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5167251/)
- Sajid, N. et al. *Active inference: demystified and compared.* [PDF](https://activeinference.github.io/papers/sajid.pdf)

### Computational Implementations

- Heins, C. et al. (2022). *pymdp: A Python library for active inference in discrete state spaces.* [arXiv](https://arxiv.org/abs/2201.03904) | [GitHub](https://github.com/infer-actively/pymdp)
- ActiveInference.jl (2025). *A Julia Library for Simulation and Parameter Estimation with Active Inference Models.* [MDPI Entropy](https://www.mdpi.com/1099-4300/27/1/62)
- Smith, R. et al. (2022). *A step-by-step tutorial on active inference and its application to empirical data.* [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0022249621000973)

### Deep Active Inference

- Fountas, Z. et al. (2020). *Deep active inference agents using Monte-Carlo methods.* NeurIPS. [PDF](https://proceedings.neurips.cc/paper/2020/file/865dfbde8a344b44095495f3591f7407-Paper.pdf)
- Yeganeh, Y. T. et al. (2025). *Deep Active Inference Agents for Delayed and Long-Horizon Environments.* [arXiv](https://arxiv.org/abs/2505.19867)

### Multi-Agent and Social Active Inference

- Ruiz-Serra, J., Sweeney, P., & Harre, M. S. (2024). *Factorised Active Inference for Strategic Multi-Agent Interactions.* AAMAS 2025. [arXiv](https://arxiv.org/abs/2411.07362)
- Legaspi, R. & Toyoizumi, T. (2022). *Interactive inference: a multi-agent model of cooperative joint actions.* [arXiv](https://arxiv.org/abs/2210.13113)
- Tison, R. & Poirier, P. (2025). *As One and Many: Relating Individual and Emergent Group-Level Generative Models in Active Inference.* [MDPI Entropy](https://www.mdpi.com/1099-4300/27/2/143)
- Kaufmann, R. et al. (2021). *An Active Inference Model of Collective Intelligence.* [MDPI Entropy](https://www.mdpi.com/1099-4300/23/7/830)
- Loewe, J. L. et al. (2025). *Orchestrator: Active Inference for Multi-Agent Systems in Long-Horizon Tasks.* [arXiv](https://arxiv.org/abs/2509.05651)

### Embodied Active Inference and Navigation

- Ciria, A. et al. (2021). *Robot navigation as hierarchical active inference.* [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608021002021)
- Koudahl, M. T. et al. (2024). *Spatial and Temporal Hierarchy for Autonomous Navigation Using Active Inference.* [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11154534/)
- Seth, A. K. & Friston, K. (2016). *Active interoceptive inference and the emotional brain.* [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5062097/)
- Friston, K. et al. (2012). *Action understanding and active inference.* [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3491875/)
- Bruineberg, J. et al. (2018). *The Active Inference Approach to Ecological Perception.* [Frontiers](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00021/full)

### Robotics (AXIOM)

- VERSES AI / Friston, K. et al. (2025). *AXIOM: Active Inference Robotics Architecture.* [Blog](https://www.verses.ai/research-blog/why-learn-if-you-can-infer-active-inference-for-robot-planning-control)

### Reference Collections

- Millidge, B. *FEP Active Inference Papers.* [GitHub](https://github.com/BerenMillidge/FEP_Active_Inference_Papers)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/mahault/pag-aif-model.git
cd pag-aif-model

# Install dependencies
pip install pymdp jax jaxlib numpy matplotlib

# Run a basic spatial navigation example (Phase 1)
python src/main.py
```

## Contributors

- **Mao** (mahault)
- **Alejandro Jimenez** (aljiro)
- **Tony** (TBD)

## License

MIT
