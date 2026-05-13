# PAG-AIF Model: Periaqueductal Gray + Active Inference Simulation

A computational model of the **Periaqueductal Gray (PAG)** as a competitive attractor network driven by **Active Inference (AIF)**, generating coordinated multi-dimensional behavior across spatial, postural, and social action spaces.

## Overview

The PAG orchestrates defensive and affiliative behaviors by selecting coordinated action patterns across multiple output dimensions. This project models that process as:

1. **POMDP / AIF layer** -- Infers the agent's context from multi-dimensional sensory input and computes precision (domain salience) and arousal (active/passive gating)
2. **PAG competitive attractor network** -- A 6x2 matrix of neurons (6 domains x active/passive) that compete to select a behavioral pattern
3. **Output dynamics** -- Winning attractors bias the agent's behavior in parallel action spaces (pose, world movement, social signals)

### PAG Domain / Policy Structure

| Domain | Active Policy | Passive/Modulatory Policy |
|--------|--------------|--------------------------|
| **Pain/body** | withdraw, escape, seek care | analgesia, guarding, immobility |
| **Peripersonal risk** | flee, evade, attack | freeze, hide, risk-assess |
| **Social threat** | vocalize, appease, attack, leave | submission, social freezing, collapse |
| **Sex/reproduction** | approach, solicitation | lordosis, receptive posture |
| **Care/attachment** | contact-seeking, distress calling | huddling, stillness, dependence |
| **Panic/entrapment** | frantic escape, struggle | freeze, shutdown, inhibition |

## Architecture

```
   Input Spaces (K=4)              PAG Network (6x2)            Output Spaces
 ┌───────────────────┐          ┌─────────────────────┐       ┌──────────────┐
 │ Spatial (2D)      │─┐       │ Body  Perip  World  │       │ Pose         │
 │  safety/exposure  │ │       │ Other Social Panic  │──────▶│ World Action │
 │ Body (2D)         │─┤─AIF──▶│                     │       │ Social/Call  │
 │  pain/arousal     │ │ bias  │ Row 1: Active       │       └──────────────┘
 │ Relational (1D)   │─┤       │ Row 2: Passive      │
 │  agent proximity  │ │       └─────────────────────┘
 │ Emotional (1D)    │─┘              ▲
 │  friend/predator  │               │
 └───────────────────┘        AIF_bias_ij =
                              precision_i * arousal_j
```

### PAG Neuron Dynamics

Each neuron x_ij (domain i, mode j) evolves as:

```
dx_ij/dt = -x_ij + f(input_ij + AIF_bias_ij - inhibition_ij + compatibility_ij)
```

Where:
- **input_ij**: Gaussian sensory gradient fields (e.g., e^(-(x-p)^2))
- **AIF_bias_ij**: precision_column_i * arousal_j -- domain salience x active/passive gating
- **inhibition_ij**: lateral competition between incompatible attractors
- **compatibility_ij**: cooperation between compatible patterns (flee + vocalize can co-activate)

### Output Mapping

Winning attractors bias dynamics in each output space:

```
x_dot = p_winner * (x - g_winner) + noise
```

Each attractor defines goal points g in the pose, spatial, and social planes. The winning attractor's parameter p dominates, steering the agent's trajectory.

### Input Spaces

| Space | Dimensions | Gradient Encodes |
|-------|-----------|-----------------|
| **Spatial** | 2D Cartesian | Shelter vs. open, proximity to boundaries |
| **Body** | 2D (pain, arousal) | Interoceptive state, homeostatic distance |
| **Relational** | 1D | Physical distance to other agents |
| **Emotional** | 1D | Appraisal of other: friend (-1) to predator (+1) |

## Implementation

### Division of Labor

- **POMDP / AIF** (`src/pomdp_specification.py`): State inference, observation model, preference model, precision/arousal computation
- **PAG Network** (Alejandro): Competitive attractor dynamics, compatibility structure, winner selection
- **Integration**: POMDP precision/arousal → PAG bias; PAG winners → action dynamics → state transitions

### Key Equations

**Precision** (posterior over domains given observations):
```
precision_i = exp(-2 * ||obs - profile_i||^2) / Z
```

**Arousal gating**:
```
arousal_active  = body_arousal
arousal_passive = 1 - body_arousal
```

**AIF bias** (outer product):
```
bias_ij = precision_i * arousal_j    shape: (6, 2)
```

## Tech Stack

- **[pymdp](https://github.com/infer-actively/pymdp)** -- Active inference in discrete state spaces (JAX backend for GPU acceleration)
- **Python 3.10+**
- **JAX/NumPy** -- Numerical computation and parallelization
- **Matplotlib/Pygame** -- Visualization of multi-world simulation

## Project Structure

```
pag-aif-model/
├── src/
│   ├── pomdp_specification.py  # POMDP: input spaces, observations, precision/arousal
│   ├── pag_network.py          # PAG: competitive attractor dynamics (Alejandro)
│   ├── output_dynamics.py      # Output: attractor-weighted action generation
│   ├── simulation.py           # Full loop integration
│   ├── worlds/                 # Environment definitions
│   ├── agents/                 # Agent wrappers
│   └── utils/                  # Shared utilities
├── notebooks/                  # Jupyter notebooks for analysis and Colab exports
├── experiments/                # Experiment configurations and scripts
├── tests/                      # Unit and integration tests
├── docs/                       # Documentation and reports
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

### Allostatic / Multi-Gradient Navigation

- Csar Hernandez-Castellanos, O. et al. (2022). *Allostatic control of behavioral flexibility in goal-driven survival circuits.* Frontiers in Robotics and AI. [Full text](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.1052998/full)

### Reference Collections

- Millidge, B. *FEP Active Inference Papers.* [GitHub](https://github.com/BerenMillidge/FEP_Active_Inference_Papers)

## Visualization Roadmap

Inspired by the multi-gradient approach in Hernandez-Castellanos et al. (2022, Paul's lab), the simulation results will be reported as coordinated multi-space visualizations:

### Per-run panel layout

1. **Spatial trajectory** (2D) -- Agent path in the world with shelter/threat zones shaded as gradient fields. Path color-coded by dominant PAG domain at each timestep.

2. **Pose trajectory** (2D) -- Path through (upright/crouch x tense/relaxed) space. Shows behavioral mode transitions: clusters during freeze, arcs during flee.

3. **Interoceptive trajectory** (2D) -- Path through (heart rate x respiration) space. Shows autonomic state regulation and arousal dynamics.

4. **Social signal trajectory** (2D) -- Path through (call intensity x call valence) space. Shows vocalization patterns over time.

5. **Occupancy heatmaps** -- One per output space. Directly compare where the agent "lives" in each space and how threat/context shifts the distribution. This extends beyond the single physical-space trajectory shown in Hernandez-Castellanos et al.

6. **PAG activation time series** -- The (6x2) neuron activation matrix over time, rendered as a heatmap strip or stacked area plot. Shows which domains win competition and when transitions occur.

7. **Precision / arousal time series** -- AIF layer context inference over time. Shows how domain salience and active/passive gating evolve.

### Comparative analysis

- **Condition contrasts**: Same agent in different environments (safe vs. threatening), showing occupancy shifts across all 4 output spaces simultaneously.
- **Ablation panels**: Remove individual PAG columns or disable active/passive gating to isolate each component's contribution.
- **Multi-agent**: Show two agents' trajectories in the same spatial field with social signal exchange highlighted.

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
