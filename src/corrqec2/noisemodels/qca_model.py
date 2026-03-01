import numpy as np
from numba import njit
import stim

from ..experiments import Experiment, SurfaceCodeMemory, SurfaceCodeStability
from ..experiments import combine_split_circuits
from .base_noise_model import NoiseModel


@njit
def _sample_batches(
    n_samples: int,
    n_sites: int,
    n_rounds: int,
    initial_probs: np.ndarray,  # shape (2,)
    a: float,
    b: float,
    red_indices: np.ndarray,
    black_indices: np.ndarray,
    adjacencies: np.ndarray,
    update_probs: np.ndarray,
    emission_probs: np.ndarray,
) -> np.ndarray:
    """Numba JIT implementation of batch sampling from QCA + storm HMM model.

    Args:
        n_samples (int): _description_
        n_sites (int): _description_
        n_rounds (int): _description_
        initial_probs (np.ndarray): _description_
        a (float): _description_
        b (float): _description_
        red_indices (np.ndarray): _description_
        black_indices (np.ndarray): _description_
        adjacencies (np.ndarray): _description_
        update_probs (np.ndarray): _description_
        emission_probs (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """

    samples = np.empty((n_samples, n_sites, n_rounds), dtype=np.int8)

    for i in range(n_samples):
        # Initialize bath state
        bath_state = np.empty(n_sites, dtype=np.int8)
        for k in range(n_sites):
            u = np.random.rand()
            if u < initial_probs[0]:
                bath_state[k] = 0
            else:
                bath_state[k] = 1

        # Initial emission
        samples[i, :, 0] = _emission_step(bath_state, emission_probs)

        # Generate samples for this chain
        for j in range(1, n_rounds):
            # Storm update
            bath_state = _storm_step(bath_state, a, b)
            # Quantum cellular automaton update
            bath_state = _qca_step(
                bath_state,
                red_indices,
                black_indices,
                adjacencies,
                update_probs,
            )
            # Emission
            samples[i, :, j] = _emission_step(bath_state, emission_probs)

    return samples


@njit
def _storm_step(
    bath_state: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Perform single storm transition on a classical bath configuration.

    Args:
        bath_state (np.ndarray): np.int8, shape (n_sites,)
        a (float): _description_
        b (float): _description_

    Returns:
        np.ndarray: _description_
    """
    new_state = bath_state.copy()
    for i in range(bath_state.size):
        if bath_state[i] == 0:
            if np.random.rand() < a:
                new_state[i] = 1
        else:
            if np.random.rand() < b:
                new_state[i] = 0
    return new_state


@njit
def _qca_step(
    bath_state: np.ndarray,
    red_indices: np.ndarray,
    black_indices: np.ndarray,
    adjacencies: np.ndarray,
    update_probs: np.ndarray,
) -> np.ndarray:
    """Perform single checkerboard QCA update on a classical bath configuration.

    Args:
        bath_state (np.ndarray): np.int8, shape (n_sites,)
        red_indices (np.ndarray): np.int32, shape (n_red,)
        black_indices (np.ndarray): np.int32, shape (n_black,)
        adjacencies (np.ndarray): np.int32, shape (n_sites, max_deg)
        update_probs (np.ndarray): np.float64, shape (max_deg+1,)

    Returns:
        np.ndarray: np.int8, shape (n_sites,)
    """
    new_state = bath_state.copy()
    max_deg = adjacencies.shape[1]

    # --- Layer 1: red controls -> update black targets ---
    for bi in range(black_indices.shape[0]):
        j = black_indices[bi]
        k = 0
        for ni in range(max_deg):
            nb = adjacencies[j, ni]
            if nb == -1:
                break
            k += bath_state[nb]
        if np.random.rand() < update_probs[k]:
            new_state[j] ^= 1

    # --- Layer 2: black controls -> update red targets ---
    for ri in range(red_indices.shape[0]):
        i = red_indices[ri]
        k = 0
        for ni in range(max_deg):
            nb = adjacencies[i, ni]
            if nb == -1:
                break
            k += new_state[nb]
        if np.random.rand() < update_probs[k]:
            new_state[i] ^= 1

    return new_state


@njit
def _emission_step(
    bath_state: np.ndarray,
    emission_probs: np.ndarray,
) -> np.ndarray:
    """Perform emission step for classical bath configuration.

    Args:
        bath_state (np.ndarray): np.int8, shape (n_sites,)
        emission_probs (np.ndarray): np.float64, shape (2, 4)
    """
    emissions = np.empty(bath_state.size, dtype=np.int8)
    for i in range(bath_state.size):
        u = np.random.rand()
        cumulative = 0.0
        k_chosen = 3  # Default fallback value
        for k in range(4):
            cumulative += emission_probs[bath_state[i], k]
            if u < cumulative:
                k_chosen = k
                break
        emissions[i] = k_chosen
    return emissions


def _get_red_black_indices(experiment: Experiment) -> tuple[np.ndarray, np.ndarray]:
    """Get the indices of red (data) and black (syndrome) qubits."""

    qubit_list = experiment.all_qubits
    red_qubits = experiment.qubits["data"]
    black_qubits = experiment.qubits["syndrome"]

    red_indices = np.array([qubit_list.index(q) for q in red_qubits], dtype=np.int32)
    black_indices = np.array(
        [qubit_list.index(q) for q in black_qubits], dtype=np.int32
    )
    return red_indices, black_indices


def _get_adjacencies(experiment: Experiment, tol: float = 1e-5) -> np.ndarray:
    """Build the compact adjacency matrix for the environment qubits in the experiment."""
    qubit_list = experiment.all_qubits
    coords = experiment.circuit.get_final_qubit_coordinates()
    coords_array = np.array([coords[q] for q in qubit_list])

    # -1 means no neighbor in that direction
    adjacencies = np.zeros((len(qubit_list), 255), dtype=np.int32) - 1
    max_deg = 0
    for i, q in enumerate(qubit_list):
        coord = coords_array[i]
        distances = np.linalg.norm(coords_array - coord, axis=1)
        distances[i] = np.inf  # Ignore self-distance
        min_dist = np.min(distances)
        neighbours = np.where(np.abs(distances - min_dist) < tol)[0]
        adjacencies[i, 0 : len(neighbours)] = neighbours
        if len(neighbours) > max_deg:
            max_deg = len(neighbours)

    adjacencies = adjacencies[:, 0:max_deg]

    return adjacencies


def _get_update_probs(theta: float, max_deg: int) -> np.ndarray:
    """Compute update probabilities vector for QCA model."""
    ks = np.arange(0, max_deg + 1, dtype=np.float64)
    update_probs = np.sin(ks * theta * 0.5) ** 2
    return update_probs


def _estimate_qca_marginal_error_rate(
    a: float,
    b: float,
    theta: float,
    emissions: np.ndarray | list[list[float]] | None = None,
    degree: int = 4,
    n_iterations: int = 3,
) -> float:
    """Estimate marginal per-site error probability for the Storm-QCA model.

    This is a first-order mean-field heuristic:
    1) storm update on mean excited density,
    2) checkerboard QCA update approximated from an average flip probability,
    3) mapping excited-state occupancy to emitted non-identity probability.

    Args:
        a (float): Storm excitation probability 0->1.
        b (float): Storm relaxation probability 1->0.
        theta (float): QCA angle in radians.
        emissions (np.ndarray | list[list[float]] | None): Emission table of shape
            (2, 4), where column 0 is probability of I. If None, defaults to
            model convention [[1,0,0,0],[0,1/3,1/3,1/3]].
        degree (int): Effective nearest-neighbour degree in the lattice.
        n_iterations (int): Number of fixed-point iterations for density estimate.

    Returns:
        float: Estimated marginal non-identity error probability in [0, 1].
    """
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("a and b must lie in [0, 1].")
    if a + b <= 0:
        raise ValueError("a + b must be > 0.")
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1.")

    if emissions is None:
        emissions_arr = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1 / 3, 1 / 3, 1 / 3]])
    else:
        emissions_arr = np.asarray(emissions, dtype=np.float64)
    if emissions_arr.shape != (2, 4):
        raise ValueError("emissions must have shape (2, 4).")

    # Initial storm-only fixed point.
    rho = a / (a + b)

    for _ in range(n_iterations):
        # Storm mean-field update.
        rho_storm = a + (1.0 - a - b) * rho

        # Mean QCA flip probability:
        # E[sin^2(K*theta/2)], K ~ Binomial(degree, rho_storm)
        z = degree
        c = (1.0 - rho_storm) + rho_storm * np.exp(1j * theta)
        mean_cos = np.real(c**z)
        flip_prob = 0.5 * (1.0 - mean_cos)
        flip_prob = float(np.clip(flip_prob, 0.0, 1.0))

        # Two half-steps approximation on occupancy.
        rho = (1.0 - 2.0 * flip_prob) ** 2 * rho_storm + 2.0 * flip_prob * (
            1.0 - flip_prob
        )
        rho = float(np.clip(rho, 0.0, 1.0))

    p_emit_if_calm = 1.0 - float(emissions_arr[0, 0])
    p_emit_if_excited = 1.0 - float(emissions_arr[1, 0])
    p_marg = (1.0 - rho) * p_emit_if_calm + rho * p_emit_if_excited

    return float(np.clip(p_marg, 0.0, 1.0))


class StormQCAModel(NoiseModel):
    """
    Spatiotemporal noise model based on a quantum cellular automaton (QCA) combined with a storm HMM.
    """

    def __init__(
        self,
        model_params: dict,
        gate_noise: dict | None = None,
    ):
        super().__init__(gate_noise=gate_noise, noisy_qubit_types="all")
        self._no_error_matrix = False
        self.model_params = model_params

        # Extract model parameters
        a, b = model_params["a"], model_params["b"]
        self._a = a
        self._b = b
        self._theta = model_params["theta"]
        self._emissions = np.array(model_params["emissions"])
        # TODO: Stationary probabilities (approximate), may not be accurate for QCA
        self._pi_a = a / (a + b)
        self._pi_b = b / (a + b)
        self._initial_probs = np.array([self._pi_b, self._pi_a])

        # Setup experiment-specific geometry placeholders
        self._red_indices = None
        self._black_indices = None
        self._adjacencies = None
        self._update_probs = None

    def _setup_geometry(self, experiment: Experiment):

        self._red_indices, self._black_indices = _get_red_black_indices(experiment)
        self._adjacencies = _get_adjacencies(experiment)
        max_deg = self._adjacencies.shape[1]
        self._update_probs = _get_update_probs(self._theta, max_deg)

    def gen_error_matrix(
        self, experiment: Experiment, n_samples: int = 1
    ) -> np.ndarray:
        n_qubits, n_rounds = experiment.get_error_matrix_shape(self.noisy_qubit_types)

        if self._red_indices is None:
            self._setup_geometry(experiment)

        samples = _sample_batches(
            n_samples=n_samples,
            n_sites=n_qubits,
            n_rounds=n_rounds,
            initial_probs=self._initial_probs,
            a=self._a,
            b=self._b,
            red_indices=self._red_indices,
            black_indices=self._black_indices,
            adjacencies=self._adjacencies,
            update_probs=self._update_probs,
            emission_probs=self._emissions,
        )

        return samples

    def gen_marginalized_circuit(self, experiment) -> stim.Circuit:
        # Get base circuit
        if self.gate_noise is not None:
            split_circuits = self.gen_noisy_circuit(experiment, split_circuit=True)
        else:
            split_circuits = experiment.split_circuits

        p_D = _estimate_qca_marginal_error_rate(
            a=self._a,
            b=self._b,
            theta=self._theta,
            emissions=self._emissions,
            degree=self._adjacencies.shape[1] if self._adjacencies is not None else 4,
            n_iterations=10,
        )

        # Get qubit targets for marginal channel injection
        targets = experiment.get_qubits_by_type(self.noisy_qubit_types)

        subcircuits_new = []
        for subcircuit in split_circuits[1:-1]:
            if isinstance(subcircuit, stim.Circuit):
                subcircuit_new = stim.Circuit()
                subcircuit_new.append("DEPOLARIZE1", targets, p_D)
                subcircuit_new += subcircuit
            elif isinstance(subcircuit, tuple):
                repeat_count, repeat_block = subcircuit
                repeat_block_new = stim.Circuit()
                repeat_block_new.append("DEPOLARIZE1", targets, p_D)
                repeat_block_new += repeat_block
                subcircuit_new = (repeat_count, repeat_block_new)
            subcircuits_new.append(subcircuit_new)

        # Combine all parts back into a single circuit
        new_circuit = combine_split_circuits(
            [split_circuits[0]] + subcircuits_new + [split_circuits[-1]]
        )

        return new_circuit

    def gen_detector_error_model(
        self, experiment: Experiment
    ) -> stim.DetectorErrorModel:
        pass
