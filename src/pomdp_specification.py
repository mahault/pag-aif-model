"""
PAG-AIF POMDP Specification
============================

Defines the generative model (POMDP) that provides inputs to the PAG
competitive attractor network. The POMDP handles:
  - State inference (where am I in each input space?)
  - Observation likelihoods (Gaussian gradient fields)
  - Preference encoding (what does the agent want?)
  - Precision/arousal computation (biases for PAG columns/rows)

The PAG network (Alejandro's side) handles:
  - Competitive attractor dynamics
  - Active vs. passive policy selection
  - Output action generation in each space

This module closes the loop by mapping POMDP posteriors → PAG biases
and PAG outputs → state transitions.

Architecture
------------
    Input Spaces (K=4 dims)        PAG Network (6 cols x 2 rows)
    ┌─────────────────┐            ┌──────────────────────────┐
    │ Spatial (2D)     │──┐        │  Body │Perip│World│Other│Social│
    │ Body (2D)        │──┤─→ AIF ─┤  act  │ act │ act │ act │ act  │
    │ Relational (1D)  │──┤  bias  │  pas  │ pas │ pas │ pas │ pas  │
    │ Emotional (1D)   │──┘        └──────────────────────────┘
                                          │
                                          ▼
                                   Output Actions
                                   (Pose, World, Social)

References
----------
- Friston et al. (2016). Active inference and learning.
- Heins et al. (2022). pymdp: Active inference in discrete state spaces.
- Seth & Friston (2016). Active interoceptive inference.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# 1. INPUT SPACE DEFINITIONS
# =============================================================================

@dataclass
class SpatialSpace:
    """
    2D Cartesian space representing the world/environment.

    Gradients encode:
      - Shelter vs. open space (safety gradient)
      - Proximity to landmarks, boundaries, or resources

    State: (x, y) position of the agent on the grid.
    """
    grid_size: int = 20
    shelter_positions: list = field(default_factory=lambda: [(2, 2), (17, 17)])
    open_positions: list = field(default_factory=lambda: [(10, 10)])

    def safety_gradient(self, position: np.ndarray) -> float:
        """How safe is this position? High near shelter, low in open space."""
        min_dist_to_shelter = min(
            np.linalg.norm(position - np.array(s))
            for s in self.shelter_positions
        )
        return np.exp(-0.1 * min_dist_to_shelter**2)

    def exposure_gradient(self, position: np.ndarray) -> float:
        """How exposed is this position? Inverse of safety."""
        return 1.0 - self.safety_gradient(position)


@dataclass
class BodySpace:
    """
    2D space representing interoceptive/body states.

    Dimensions:
      - Pain ←→ Pleasure (valence axis)
      - Low arousal ←→ High arousal (arousal axis)

    This maps to the body/pain PAG column and provides the arousal
    parameter that gates active vs. passive rows.
    """
    pain_level: float = 0.0      # [0, 1] — 0 = no pain, 1 = max pain
    arousal_level: float = 0.5   # [0, 1] — 0 = dorsal vagal shutdown, 1 = max sympathetic

    def body_gradient(self, state: np.ndarray) -> np.ndarray:
        """
        Gaussian gradient field around current body state.
        state: [pain, arousal]
        Returns gradient vector pointing toward homeostatic setpoint.
        """
        setpoint = np.array([0.0, 0.5])  # no pain, moderate arousal
        diff = state - setpoint
        return np.exp(-np.sum(diff**2))


@dataclass
class RelationalSpace:
    """
    1D space representing physical/relational distance to other agents.

    Low values = close proximity to other agent.
    High values = far from other agents.

    This drives the other-agent and peripersonal PAG columns.
    """
    max_distance: float = 20.0

    def proximity_gradient(self, self_pos: np.ndarray, other_pos: np.ndarray) -> float:
        """
        Gaussian proximity field: how close is the other agent?
        Returns value in [0, 1], high when close.
        """
        dist = np.linalg.norm(self_pos - other_pos)
        return np.exp(-0.05 * dist**2)


@dataclass
class EmotionalSpace:
    """
    1D space representing the agent's appraisal of another agent.

    Dimension: Friend ←→ Predator (valence toward other)
      -1 = predator/threat
       0 = neutral/unknown
      +1 = friend/ally

    This modulates which PAG columns are activated when another
    agent is detected (social vs. predator response).
    """
    valence: float = 0.0  # [-1, 1]

    def threat_gradient(self, valence: float) -> float:
        """How threatening is the other? High when valence is negative."""
        return np.exp(-2.0 * (valence + 1)**2)  # peaks at valence = -1

    def affiliation_gradient(self, valence: float) -> float:
        """How affiliative is the other? High when valence is positive."""
        return np.exp(-2.0 * (valence - 1)**2)  # peaks at valence = +1


# =============================================================================
# 2. OBSERVATION MODEL (A MATRIX - LIKELIHOODS)
# =============================================================================

@dataclass
class ObservationModel:
    """
    Maps hidden states to observations using Gaussian gradient fields.

    Each observation modality corresponds to one or more input spaces.
    Observations are continuous gradient values, not discrete categories.

    For the discrete pymdp implementation, these are discretized into bins.
    For the continuous/hybrid implementation, they feed directly as
    real-valued inputs.

    Modalities:
      0: Spatial observation   — safety/exposure gradient at agent position
      1: Body observation      — pain/arousal gradient
      2: Proximity observation — relational distance gradient to nearest other
      3: Valence observation   — emotional appraisal of nearest other
    """
    num_modalities: int = 4
    num_bins: int = 10  # discretization bins per modality

    def compute_observations(
        self,
        spatial: SpatialSpace,
        body: BodySpace,
        relational: RelationalSpace,
        emotional: EmotionalSpace,
        agent_pos: np.ndarray,
        body_state: np.ndarray,
        other_pos: Optional[np.ndarray] = None,
        other_valence: float = 0.0,
    ) -> np.ndarray:
        """
        Compute raw continuous observations from all input spaces.

        Returns: [safety, body_homeostasis, proximity, threat] all in [0, 1]
        """
        obs = np.zeros(self.num_modalities)

        # Spatial: how safe is current position?
        obs[0] = spatial.safety_gradient(agent_pos)

        # Body: how close to homeostatic setpoint?
        obs[1] = body.body_gradient(body_state)

        # Relational: how close is the nearest other?
        if other_pos is not None:
            obs[2] = relational.proximity_gradient(agent_pos, other_pos)
        else:
            obs[2] = 0.0  # no other detected

        # Emotional: threat level of nearest other
        obs[3] = emotional.threat_gradient(other_valence)

        return obs

    def discretize(self, continuous_obs: np.ndarray) -> np.ndarray:
        """Convert continuous [0,1] observations to discrete bin indices."""
        bins = np.clip(
            (continuous_obs * self.num_bins).astype(int),
            0, self.num_bins - 1
        )
        return bins


# =============================================================================
# 3. TRANSITION MODEL (B MATRIX - DYNAMICS)
# =============================================================================

@dataclass
class TransitionModel:
    """
    Defines how hidden states evolve given actions.

    In the full loop:
      PAG output → action in each space → state transition → new observations

    Each input space has its own transition dynamics:
      - Spatial: agent moves on grid (up/down/left/right/stay)
      - Body: pain/arousal drift toward homeostasis + perturbations
      - Relational: changes as agents move (derived from spatial)
      - Emotional: valence updates based on interaction history

    The PAG winning attractor determines which action is taken,
    biasing the random walk: ẋ = p*(x - g_winner) + noise
    """
    drift_rate: float = 0.1      # speed of body homeostatic drift
    noise_scale: float = 0.05    # stochastic perturbation

    def spatial_transition(
        self, position: np.ndarray, action: np.ndarray, dt: float = 1.0
    ) -> np.ndarray:
        """
        Move agent in spatial world.
        action: 2D velocity vector from PAG output dynamics.
        """
        noise = np.random.randn(2) * self.noise_scale
        new_pos = position + action * dt + noise
        return new_pos

    def body_transition(
        self,
        body_state: np.ndarray,
        external_perturbation: np.ndarray,
        dt: float = 1.0,
    ) -> np.ndarray:
        """
        Body state drifts toward homeostasis + external perturbations.
        body_state: [pain, arousal]
        external_perturbation: e.g., injury increases pain, threat increases arousal
        """
        setpoint = np.array([0.0, 0.5])
        drift = -self.drift_rate * (body_state - setpoint)
        noise = np.random.randn(2) * self.noise_scale
        new_state = body_state + (drift + external_perturbation) * dt + noise
        return np.clip(new_state, 0.0, 1.0)

    def emotional_transition(
        self, valence: float, interaction_signal: float, dt: float = 1.0
    ) -> float:
        """
        Update emotional valence toward other agent based on interactions.
        interaction_signal: positive = friendly interaction, negative = hostile
        """
        decay = -0.05 * valence  # slow decay toward neutral
        update = 0.1 * interaction_signal
        noise = np.random.randn() * self.noise_scale * 0.5
        new_valence = valence + (decay + update) * dt + noise
        return float(np.clip(new_valence, -1.0, 1.0))


# =============================================================================
# 4. PREFERENCE MODEL (C MATRIX)
# =============================================================================

@dataclass
class PreferenceModel:
    """
    Encodes what the agent prefers to observe in each modality.

    In AIF, the C matrix defines preferred observations. The agent
    acts to make its observations match these preferences (pragmatic
    value component of expected free energy).

    Preferences are defined per PAG domain — when a domain is active,
    its associated preferences dominate.

    Domain mapping to preferred observations:
      Body/Pain:        prefer high body_homeostasis, low pain
      Peripersonal:     prefer high safety, moderate proximity
      World/Escape:     prefer high safety (seek shelter)
      Other-agent:      prefer low proximity (distance from unknown)
      Social:           prefer high proximity + high affiliation
      Panic:            prefer high safety + low arousal
    """

    def get_domain_preferences(self, domain_idx: int) -> np.ndarray:
        """
        Returns preferred observation vector [safety, body, proximity, threat]
        for a given PAG domain.

        Domain indices:
          0 = Body/Pain
          1 = Peripersonal risk
          2 = World/Escape
          3 = Other-agent
          4 = Social
          5 = Panic/Entrapment
        """
        # Columns: [safety_pref, body_pref, proximity_pref, low_threat_pref]
        # Values in [0, 1]: 1 = strongly prefer high, 0 = prefer low
        preference_table = np.array([
            # safety  body  proximity  low_threat
            [0.3,     1.0,  0.2,       0.5],   # 0: Body/Pain — prioritize body homeostasis
            [1.0,     0.5,  0.3,       0.8],   # 1: Peripersonal — seek safety, avoid threat
            [1.0,     0.3,  0.1,       0.5],   # 2: World/Escape — maximize safety (flee to shelter)
            [0.5,     0.5,  0.3,       0.9],   # 3: Other-agent — distance + assess threat
            [0.3,     0.5,  0.9,       0.1],   # 4: Social — seek proximity, low threat (affiliative)
            [1.0,     0.8,  0.1,       0.9],   # 5: Panic — safety + body regulation + isolation
        ])
        return preference_table[domain_idx]


# =============================================================================
# 5. PRECISION AND AROUSAL → PAG BIAS
# =============================================================================

@dataclass
class PAGBiasComputer:
    """
    Computes AIF_bias_ij = precision_column_i * arousal_j

    This is the interface between the POMDP and the PAG network.

    precision_column_i:
      Posterior confidence that domain i is the active context.
      Derived from observation likelihoods — how well do current
      observations match each domain's expected sensory profile?

    arousal_j:
      Global arousal state that gates active (j=0) vs passive (j=1).
      Derived from the body space arousal dimension.
        High arousal → active row favored (fight, flee, vocalize)
        Low arousal  → passive row favored (freeze, submit, collapse)

    The bias matrix has shape (num_domains, 2) = (6, 2).
    """
    num_domains: int = 6
    num_modes: int = 2  # active, passive

    # Sensory profile per domain: expected [safety, body, proximity, threat]
    # when that domain context is truly active
    domain_sensory_profiles: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.domain_sensory_profiles is None:
            self.domain_sensory_profiles = np.array([
                # safety  body  proximity  threat
                [0.5,     0.1,  0.5,       0.3],   # Body/Pain: low body homeostasis
                [0.2,     0.5,  0.7,       0.6],   # Peripersonal: low safety, something close & threatening
                [0.1,     0.3,  0.3,       0.7],   # World/Escape: very low safety, high threat
                [0.5,     0.5,  0.8,       0.5],   # Other-agent: something close, ambiguous
                [0.5,     0.5,  0.8,       0.1],   # Social: something close, non-threatening
                [0.1,     0.2,  0.5,       0.8],   # Panic: low safety, low body, high threat
            ])

    def compute_precision(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute precision_column_i for each domain.

        Uses Gaussian similarity between current observations and each
        domain's expected sensory profile. High similarity → high precision
        for that domain.

        Returns: array of shape (num_domains,) with values in [0, 1]
        """
        precisions = np.zeros(self.num_domains)
        for i in range(self.num_domains):
            diff = observations - self.domain_sensory_profiles[i]
            precisions[i] = np.exp(-2.0 * np.sum(diff**2))

        # Normalize to sum to 1 (posterior over domains)
        total = precisions.sum()
        if total > 0:
            precisions /= total
        else:
            precisions[:] = 1.0 / self.num_domains

        return precisions

    def compute_arousal_weights(self, arousal: float) -> np.ndarray:
        """
        Compute arousal_j gating for active (j=0) vs passive (j=1).

        arousal: scalar in [0, 1] from body space
          High arousal → [high, low]  → active row favored
          Low arousal  → [low, high]  → passive row favored

        Returns: array of shape (2,)
        """
        active_weight = arousal
        passive_weight = 1.0 - arousal
        return np.array([active_weight, passive_weight])

    def compute_bias_matrix(
        self, observations: np.ndarray, arousal: float
    ) -> np.ndarray:
        """
        Compute the full AIF bias matrix for the PAG network.

        AIF_bias_ij = precision_column_i * arousal_j

        Returns: array of shape (num_domains, 2)
        """
        precision = self.compute_precision(observations)
        arousal_weights = self.compute_arousal_weights(arousal)

        # Outer product: (num_domains,) x (2,) → (num_domains, 2)
        bias_matrix = np.outer(precision, arousal_weights)

        return bias_matrix


# =============================================================================
# 6. FULL POMDP AGENT
# =============================================================================

class POMDPAgent:
    """
    The POMDP side of the PAG-AIF loop.

    Each timestep:
      1. Receive observations from all input spaces
      2. Compute posterior over domains (precision per column)
      3. Extract arousal from body state
      4. Compute AIF bias matrix → send to PAG network
      5. Receive PAG output (winning attractors)
      6. Execute actions, transition states
      7. Loop

    Parameters
    ----------
    agent_id : str
        Identifier for this agent.
    spatial : SpatialSpace
        The spatial environment configuration.
    """

    def __init__(self, agent_id: str, spatial: SpatialSpace):
        self.agent_id = agent_id
        self.spatial = spatial
        self.body = BodySpace()
        self.relational = RelationalSpace()
        self.emotional = EmotionalSpace()
        self.obs_model = ObservationModel()
        self.transition_model = TransitionModel()
        self.preference_model = PreferenceModel()
        self.bias_computer = PAGBiasComputer()

        # Agent state
        self.position = np.array([10.0, 10.0])  # start at center
        self.body_state = np.array([0.0, 0.5])   # [pain, arousal]
        self.other_valence = 0.0                  # neutral toward other

        # History for analysis
        self.history = {
            "positions": [],
            "body_states": [],
            "observations": [],
            "precisions": [],
            "bias_matrices": [],
            "actions": [],
        }

    def observe(self, other_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute observations from all input spaces."""
        obs = self.obs_model.compute_observations(
            spatial=self.spatial,
            body=self.body,
            relational=self.relational,
            emotional=self.emotional,
            agent_pos=self.position,
            body_state=self.body_state,
            other_pos=other_pos,
            other_valence=self.other_valence,
        )
        return obs

    def compute_pag_bias(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute AIF bias matrix to send to PAG network.

        Returns: (6, 2) matrix — precision_i * arousal_j
        """
        arousal = float(self.body_state[1])
        bias = self.bias_computer.compute_bias_matrix(observations, arousal)
        return bias

    def step(
        self,
        pag_action: dict,
        other_pos: Optional[np.ndarray] = None,
        external_perturbation: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Execute one timestep of the POMDP loop.

        Parameters
        ----------
        pag_action : dict
            Output from PAG network:
              'spatial_action': np.ndarray (2D velocity)
              'interaction_signal': float (social action effect)
        other_pos : np.ndarray or None
            Position of the other agent (if any).
        external_perturbation : np.ndarray or None
            External effects on body state (injury, threat arousal, etc.)

        Returns
        -------
        dict with:
            'observations': current obs
            'bias_matrix': AIF bias for PAG
            'precision': domain precisions
            'arousal': current arousal level
        """
        if external_perturbation is None:
            external_perturbation = np.array([0.0, 0.0])

        # 1. Transition states based on PAG actions
        self.position = self.transition_model.spatial_transition(
            self.position, pag_action.get("spatial_action", np.zeros(2))
        )

        self.body_state = self.transition_model.body_transition(
            self.body_state, external_perturbation
        )

        self.other_valence = self.transition_model.emotional_transition(
            self.other_valence,
            pag_action.get("interaction_signal", 0.0),
        )

        # 2. Observe
        observations = self.observe(other_pos)

        # 3. Compute PAG bias
        bias_matrix = self.compute_pag_bias(observations)
        precision = self.bias_computer.compute_precision(observations)
        arousal = float(self.body_state[1])

        # 4. Record history
        self.history["positions"].append(self.position.copy())
        self.history["body_states"].append(self.body_state.copy())
        self.history["observations"].append(observations.copy())
        self.history["precisions"].append(precision.copy())
        self.history["bias_matrices"].append(bias_matrix.copy())

        return {
            "observations": observations,
            "bias_matrix": bias_matrix,
            "precision": precision,
            "arousal": arousal,
        }


# =============================================================================
# 7. QUICK DEMO / SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    print("PAG-AIF POMDP Specification — Sanity Check")
    print("=" * 50)

    # Create environment and agent
    spatial = SpatialSpace()
    agent = POMDPAgent(agent_id="agent_0", spatial=spatial)

    # Place another agent nearby
    other_pos = np.array([12.0, 11.0])

    # Simulate a few steps with no PAG output yet (null actions)
    null_action = {"spatial_action": np.zeros(2), "interaction_signal": 0.0}

    print("\nBaseline (no threat, center of grid, neutral other):")
    result = agent.step(null_action, other_pos=other_pos)
    print(f"  Observations:  {np.round(result['observations'], 3)}")
    print(f"  Precision:     {np.round(result['precision'], 3)}")
    print(f"  Arousal:       {result['arousal']:.3f}")
    print(f"  Bias matrix:\n{np.round(result['bias_matrix'], 3)}")

    # Simulate threat: increase pain and arousal
    print("\nAfter injury (pain spike + arousal increase):")
    perturbation = np.array([0.6, 0.3])  # pain + arousal surge
    result = agent.step(null_action, other_pos=other_pos,
                        external_perturbation=perturbation)
    print(f"  Body state:    {np.round(agent.body_state, 3)}")
    print(f"  Observations:  {np.round(result['observations'], 3)}")
    print(f"  Precision:     {np.round(result['precision'], 3)}")
    print(f"  Arousal:       {result['arousal']:.3f}")
    print(f"  Bias matrix:\n{np.round(result['bias_matrix'], 3)}")

    # Change emotional valence: other becomes threatening
    print("\nOther agent becomes threatening (predator):")
    agent.other_valence = -0.8
    result = agent.step(null_action, other_pos=other_pos)
    print(f"  Valence:       {agent.other_valence:.3f}")
    print(f"  Observations:  {np.round(result['observations'], 3)}")
    print(f"  Precision:     {np.round(result['precision'], 3)}")
    print(f"  Arousal:       {result['arousal']:.3f}")
    print(f"  Bias matrix:\n{np.round(result['bias_matrix'], 3)}")

    print("\n" + "=" * 50)
    print("Sanity check complete. Bias matrix shape:", result["bias_matrix"].shape)
    print("Ready to connect to PAG attractor network.")
