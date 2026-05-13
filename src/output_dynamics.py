"""
PAG-AIF Output Dynamics
========================

Translates PAG network activations (winning attractors) into continuous
actions across 4 output spaces:

  1. Spatial action space  — 2D movement (flee, approach, wander)
  2. Pose space            — 2D body configuration (upright/crouch, tense/relaxed)
  3. Interoceptive space   — 2D autonomic regulation (heart rate, respiration)
  4. Social signalling     — 2D vocalization (intensity, valence)

Each output space follows attractor-weighted dynamics:

    x_dot = sum_k( w_k * (g_k - x) ) / sum_k(w_k) + damping + noise

Where:
  - k indexes the active PAG neurons
  - w_k is the activation level of neuron k (from competitive network)
  - g_k is the goal point of attractor k in this output space
  - noise provides stochastic exploration

The PAG network (6 domains x 2 modes = 12 neurons) maps to these 4
output spaces via an attractor goal table: each neuron defines a goal
point in every output space, encoding the coordinated behavioral pattern.

Architecture (from Alejandro's updated diagram)
------------------------------------------------
    PAG activations (12 neurons)
           │
           ▼
    ┌──────────────────────────────────────────┐
    │         Attractor Goal Table             │
    │  (12 neurons x 4 spaces x 2 dims)       │
    │                                          │
    │  neuron_k → [g_spatial, g_pose,          │
    │              g_intero, g_social]          │
    └──────────────────────────────────────────┘
           │
           ▼
    ┌──────┬──────┬──────┬──────┐
    │Spatial│ Pose │Intero│Social│
    │action │space │ space│signal│
    └──────┴──────┴──────┴──────┘
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# PAG NEURON INDEXING
# =============================================================================
# 6 domains x 2 modes = 12 neurons
# PAG activation matrix shape: (6, 2)
#
# Domain indices:
#   0 = Body/Pain
#   1 = Peripersonal Risk
#   2 = World/Escape
#   3 = Other-agent/Confrontation
#   4 = Social field/Signalling
#   5 = Panic/Entrapment
#
# Mode indices:
#   0 = Active (dorsal)
#   1 = Passive (ventrolateral)

NUM_DOMAINS = 6
NUM_MODES = 2
NUM_NEURONS = NUM_DOMAINS * NUM_MODES

# Output space names and dimensions
OUTPUT_SPACES = ["spatial", "pose", "interoceptive", "social"]
OUTPUT_DIMS = 2  # each output space is 2D


# =============================================================================
# ATTRACTOR GOAL TABLE
# =============================================================================

@dataclass
class AttractorGoalTable:
    """
    Maps each PAG neuron to goal points in each output space.

    Shape: (num_domains, num_modes, num_output_spaces, output_dims)
           = (6, 2, 4, 2)

    Each entry g[i, j, s, :] is the 2D goal point for domain i, mode j,
    in output space s.

    Output space conventions:
      Spatial:       (flee_drive, shelter_drive) — mapped to world coords later
      Pose:          (upright↔crouch, tense↔relaxed) — [-1, 1] each
      Interoceptive: (heart_rate_target, respiration_target) — [0, 1] each
      Social:        (call_intensity, call_valence) — [0,1] x [-1,1]
    """
    goals: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.goals is None:
            self.goals = self._default_goals()

    def _default_goals(self) -> np.ndarray:
        """
        Default attractor goals for each neuron in each output space.

        Each row: [spatial(2D), pose(2D), interoceptive(2D), social(2D)]
        """
        g = np.zeros((NUM_DOMAINS, NUM_MODES, len(OUTPUT_SPACES), OUTPUT_DIMS))

        # ── Domain 0: Body/Pain ──────────────────────────────────
        # Active: withdraw, escape despite injury, seek care
        g[0, 0] = [
            [0.3, 0.0],     # spatial: slow withdrawal
            [0.5, 0.8],     # pose: slightly crouched, tense (guarding)
            [0.7, 0.6],     # intero: elevated HR, faster breathing
            [0.6, 0.5],     # social: moderate distress call
        ]
        # Passive: analgesia, guarding, immobility, recuperation
        g[0, 1] = [
            [0.0, 0.0],     # spatial: stay still
            [-0.8, -0.5],   # pose: crouched, somewhat relaxed (conserving)
            [0.3, 0.3],     # intero: reduced HR, slow breathing
            [0.1, 0.0],     # social: quiet, neutral
        ]

        # ── Domain 1: Peripersonal Risk ──────────────────────────
        # Active: flee, evade, attack
        g[1, 0] = [
            [1.0, 0.0],     # spatial: fast movement away
            [1.0, 1.0],     # pose: upright, maximally tense
            [0.9, 0.9],     # intero: high HR, rapid breathing
            [0.4, -0.5],    # social: moderate alarm call
        ]
        # Passive: freeze, hide, risk-assess
        g[1, 1] = [
            [0.0, 0.0],     # spatial: frozen in place
            [-0.5, 0.9],    # pose: crouched, very tense (freeze)
            [0.4, 0.2],     # intero: bradycardia, held breath
            [0.0, 0.0],     # social: silent
        ]

        # ── Domain 2: World/Escape ───────────────────────────────
        # Active: flee toward shelter
        g[2, 0] = [
            [1.0, 1.0],     # spatial: fast directed flight
            [1.0, 0.8],     # pose: upright, tense (running)
            [1.0, 1.0],     # intero: maximum HR, maximum breathing
            [0.3, -0.8],    # social: alarm vocalization
        ]
        # Passive: freeze, hide
        g[2, 1] = [
            [0.0, 0.0],     # spatial: immobile
            [-1.0, 0.5],    # pose: fully crouched, moderate tension
            [0.2, 0.2],     # intero: suppressed HR, shallow breathing
            [0.0, 0.0],     # social: silent
        ]

        # ── Domain 3: Other-agent/Confrontation ──────────────────
        # Active: vocalize, appease, attack, leave
        g[3, 0] = [
            [0.5, 0.3],     # spatial: moderate repositioning
            [0.8, 0.7],     # pose: upright, tense (assertive)
            [0.8, 0.7],     # intero: elevated HR, faster breathing
            [0.9, -0.3],    # social: loud, slightly aggressive call
        ]
        # Passive: submission, social freezing, collapse, inhibition
        g[3, 1] = [
            [0.0, 0.0],     # spatial: stay / cower
            [-0.7, -0.3],   # pose: crouched, somewhat relaxed (submissive)
            [0.3, 0.3],     # intero: moderate suppression
            [0.2, 0.3],     # social: quiet appeasement signal
        ]

        # ── Domain 4: Social field/Signalling ────────────────────
        # Active: contact-seeking, approach, solicitation
        g[4, 0] = [
            [-0.5, -0.3],   # spatial: approach other (negative = toward)
            [0.6, -0.3],    # pose: upright, relaxed (open posture)
            [0.5, 0.5],     # intero: moderate HR, normal breathing
            [0.8, 1.0],     # social: loud affiliative call
        ]
        # Passive: huddling, stillness, dependence posture
        g[4, 1] = [
            [-0.2, 0.0],    # spatial: slight approach, mostly still
            [-0.4, -0.7],   # pose: crouched, relaxed (huddling)
            [0.3, 0.3],     # intero: calm
            [0.3, 0.8],     # social: soft affiliative signal
        ]

        # ── Domain 5: Panic/Entrapment ───────────────────────────
        # Active: frantic escape, struggle
        g[5, 0] = [
            [0.8, 0.8],     # spatial: frantic undirected movement
            [0.3, 1.0],     # pose: variable, maximum tension
            [1.0, 1.0],     # intero: maximum HR, hyperventilation
            [1.0, -1.0],    # social: maximum alarm / distress scream
        ]
        # Passive: freeze, shutdown, breath-holding inhibition
        g[5, 1] = [
            [0.0, 0.0],     # spatial: collapsed, immobile
            [-1.0, -0.8],   # pose: collapsed, limp
            [0.1, 0.1],     # intero: near-shutdown, minimal vitals
            [0.0, 0.0],     # social: silent (dissociative)
        ]

        return g

    def get_goal(self, domain: int, mode: int, space_idx: int) -> np.ndarray:
        """Get the 2D goal for a specific neuron in a specific output space."""
        return self.goals[domain, mode, space_idx]


# =============================================================================
# OUTPUT SPACE DYNAMICS
# =============================================================================

@dataclass
class OutputSpace:
    """
    A single 2D output space with attractor-weighted dynamics.

    State evolves as:
        x_dot = coupling * weighted_force + damping + noise

    Where weighted_force = sum(w_k * (g_k - x)) / sum(w_k), so the
    force points TOWARD the weighted-average goal.

    Parameters
    ----------
    name : str
        Name of this output space.
    space_idx : int
        Index in OUTPUT_SPACES list.
    coupling : float
        How fast state moves toward attractors.
    noise_scale : float
        Stochastic perturbation magnitude.
    damping : float
        Decay rate pulling state toward origin when no attractors active.
    bounds : tuple
        (min, max) clipping bounds for state values.
    """
    name: str = ""
    space_idx: int = 0
    state: np.ndarray = field(default_factory=lambda: np.zeros(2))
    coupling: float = 1.0
    noise_scale: float = 0.05
    damping: float = 0.1
    bounds: tuple = (-1.0, 1.0)

    def step(
        self,
        pag_activations: np.ndarray,
        goal_table: AttractorGoalTable,
        dt: float = 0.1,
    ) -> np.ndarray:
        """
        Evolve this output space by one timestep.

        Parameters
        ----------
        pag_activations : np.ndarray, shape (num_domains, num_modes)
            Activation levels of all PAG neurons.
        goal_table : AttractorGoalTable
            Neuron-to-goal mapping.
        dt : float
            Integration timestep.

        Returns
        -------
        np.ndarray : the new 2D state
        """
        force = np.zeros(2)
        total_weight = 0.0

        for i in range(NUM_DOMAINS):
            for j in range(NUM_MODES):
                w = pag_activations[i, j]
                if w > 0.01:
                    goal = goal_table.get_goal(i, j, self.space_idx)
                    force += w * (goal - self.state)
                    total_weight += w

        # Normalize to prevent runaway when many neurons co-activate
        if total_weight > 0:
            force = force / total_weight

        damping_force = -self.damping * self.state
        noise = np.random.randn(2) * self.noise_scale

        x_dot = self.coupling * force + damping_force + noise
        self.state = self.state + x_dot * dt
        self.state = np.clip(self.state, self.bounds[0], self.bounds[1])

        return self.state.copy()


# =============================================================================
# OUTPUT DYNAMICS MANAGER
# =============================================================================

class OutputDynamics:
    """
    Manages all 4 output spaces and converts PAG activations into actions.

    Each timestep:
      1. Receive PAG activations (6, 2)
      2. Evolve each output space toward winning attractor goals
      3. Convert output states into action dict for POMDPAgent

    The spatial output is special: raw (flee_drive, shelter_drive) needs
    to be projected onto actual threat/shelter directions in the world.
    """

    def __init__(
        self,
        goal_table: Optional[AttractorGoalTable] = None,
        dt: float = 0.1,
        spatial_speed: float = 1.0,
    ):
        self.goal_table = goal_table or AttractorGoalTable()
        self.dt = dt
        self.spatial_speed = spatial_speed

        self.spaces = {
            "spatial": OutputSpace(
                name="spatial", space_idx=0,
                coupling=1.5, noise_scale=0.08, damping=0.05,
                bounds=(-1.0, 1.0),
            ),
            "pose": OutputSpace(
                name="pose", space_idx=1,
                coupling=0.8, noise_scale=0.03, damping=0.15,
                bounds=(-1.0, 1.0),
            ),
            "interoceptive": OutputSpace(
                name="interoceptive", space_idx=2,
                coupling=0.5, noise_scale=0.02, damping=0.2,
                bounds=(0.0, 1.0),
            ),
            "social": OutputSpace(
                name="social", space_idx=3,
                coupling=1.0, noise_scale=0.05, damping=0.3,
                bounds=(-1.0, 1.0),
            ),
        }

        self.history = {name: [] for name in OUTPUT_SPACES}

    def step(
        self,
        pag_activations: np.ndarray,
        threat_direction: Optional[np.ndarray] = None,
        shelter_direction: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Evolve all output spaces and produce actions.

        Parameters
        ----------
        pag_activations : np.ndarray, shape (6, 2)
        threat_direction : np.ndarray or None, shape (2,)
            Unit vector from agent toward threat.
        shelter_direction : np.ndarray or None, shape (2,)
            Unit vector from agent toward nearest shelter.

        Returns
        -------
        dict with all output states and derived action signals.
        """
        results = {}
        for name, space in self.spaces.items():
            new_state = space.step(pag_activations, self.goal_table, self.dt)
            results[name] = new_state.copy()
            self.history[name].append(new_state.copy())

        spatial_action = self._spatial_to_world_velocity(
            results["spatial"], threat_direction, shelter_direction
        )

        # Interaction signal: intensity * valence
        # Positive = affiliative, negative = aggressive/alarm
        social_intensity = np.clip(results["social"][0], 0, 1)
        social_valence = results["social"][1]
        interaction_signal = social_intensity * social_valence

        return {
            "spatial_action": spatial_action,
            "pose_state": results["pose"],
            "interoceptive_state": results["interoceptive"],
            "social_signal": results["social"],
            "interaction_signal": float(interaction_signal),
        }

    def _spatial_to_world_velocity(
        self,
        raw_spatial: np.ndarray,
        threat_direction: Optional[np.ndarray],
        shelter_direction: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Convert raw spatial output (flee_drive, shelter_drive) to world
        frame velocity.

        flee_drive (dim 0):    movement AWAY from threat
        shelter_drive (dim 1): movement TOWARD shelter

        Falls back to raw output as velocity if no directions given.
        """
        if threat_direction is None and shelter_direction is None:
            return raw_spatial * self.spatial_speed

        velocity = np.zeros(2)

        if threat_direction is not None:
            flee_dir = -threat_direction
            norm = np.linalg.norm(flee_dir)
            if norm > 0:
                flee_dir = flee_dir / norm
            velocity += raw_spatial[0] * flee_dir

        if shelter_direction is not None:
            norm = np.linalg.norm(shelter_direction)
            if norm > 0:
                shelter_dir = shelter_direction / norm
            else:
                shelter_dir = np.zeros(2)
            velocity += raw_spatial[1] * shelter_dir

        return velocity * self.spatial_speed

    def get_pag_action_dict(
        self,
        pag_activations: np.ndarray,
        agent_pos: np.ndarray,
        threat_pos: Optional[np.ndarray] = None,
        shelter_positions: Optional[list] = None,
    ) -> dict:
        """
        Convenience: compute output and format for POMDPAgent.step().

        Computes threat/shelter directions from positions, then calls step().
        """
        threat_direction = None
        shelter_direction = None

        if threat_pos is not None:
            diff = threat_pos - agent_pos
            norm = np.linalg.norm(diff)
            if norm > 0.01:
                threat_direction = diff / norm

        if shelter_positions:
            dists = [np.linalg.norm(agent_pos - np.array(s))
                     for s in shelter_positions]
            nearest_idx = int(np.argmin(dists))
            shelter_diff = np.array(shelter_positions[nearest_idx]) - agent_pos
            norm = np.linalg.norm(shelter_diff)
            if norm > 0.01:
                shelter_direction = shelter_diff / norm

        return self.step(pag_activations, threat_direction, shelter_direction)

    def reset(self):
        """Reset all output spaces to neutral."""
        for space in self.spaces.values():
            space.state = np.zeros(2)
        self.history = {name: [] for name in OUTPUT_SPACES}

    def get_history_arrays(self) -> dict:
        """Return history as numpy arrays for plotting."""
        return {
            name: np.array(states) for name, states in self.history.items()
            if len(states) > 0
        }


# =============================================================================
# SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    print("PAG-AIF Output Dynamics — Sanity Check")
    print("=" * 55)

    output = OutputDynamics(dt=0.1)

    def run_scenario(name, pag, steps=50, threat_dir=None, shelter_dir=None):
        output.reset()
        for _ in range(steps):
            result = output.step(pag, threat_dir, shelter_dir)
        print(f"\n--- {name} ---")
        print(f"  Spatial:       {np.round(result['spatial_action'], 3)}")
        print(f"  Pose:          {np.round(result['pose_state'], 3)}")
        print(f"  Interoceptive: {np.round(result['interoceptive_state'], 3)}")
        print(f"  Social:        {np.round(result['social_signal'], 3)}")
        print(f"  Interaction:   {result['interaction_signal']:.3f}")
        return result

    # 1. Resting
    pag = np.zeros((NUM_DOMAINS, NUM_MODES))
    run_scenario("Resting (no PAG activity)", pag, steps=20)

    # 2. Active flee
    pag = np.zeros((NUM_DOMAINS, NUM_MODES))
    pag[2, 0] = 1.0
    run_scenario(
        "Active flee (World/Escape, active)",
        pag,
        threat_dir=np.array([1.0, 0.0]),
        shelter_dir=np.array([-1.0, -0.5]),
    )

    # 3. Freeze
    pag = np.zeros((NUM_DOMAINS, NUM_MODES))
    pag[1, 1] = 1.0
    run_scenario("Freeze (Peripersonal Risk, passive)", pag)

    # 4. Affiliative approach
    pag = np.zeros((NUM_DOMAINS, NUM_MODES))
    pag[4, 0] = 1.0
    run_scenario("Affiliative approach (Social, active)", pag)

    # 5. Panic active
    pag = np.zeros((NUM_DOMAINS, NUM_MODES))
    pag[5, 0] = 1.0
    run_scenario("Panic escape (Panic, active)", pag)

    # 6. Panic passive (shutdown)
    pag = np.zeros((NUM_DOMAINS, NUM_MODES))
    pag[5, 1] = 1.0
    run_scenario("Shutdown (Panic, passive)", pag)

    # 7. Competition: flee vs freeze
    pag = np.zeros((NUM_DOMAINS, NUM_MODES))
    pag[2, 0] = 0.6
    pag[1, 1] = 0.4
    run_scenario("Competing: flee(0.6) vs freeze(0.4)", pag)

    # 8. Compatible co-activation: flee + alarm call
    pag = np.zeros((NUM_DOMAINS, NUM_MODES))
    pag[2, 0] = 0.7
    pag[3, 0] = 0.5
    run_scenario("Compatible: flee(0.7) + confront(0.5)", pag)

    print("\n" + "=" * 55)
    print("Output dynamics sanity check complete.")
