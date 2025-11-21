import numpy as np
from numba import njit
import stim

from ..experiments import Experiment, SurfaceCodeMemory, SurfaceCodeStability
from ..experiments import combine_split_circuits
from .base_noise_model import NoiseModel


@njit
def _qca_update(
    bath_state: np.ndarray,
    red_indices: np.ndarray,
    black_indices: np.ndarray,
    adjacencies: np.ndarray,
    update_probs: np.ndarray,
) -> np.ndarray:
    """Perform one bipartite (checkerboard) QCA update on a classical bath configuration.

    Args:
        bath_state (np.ndarray): _description_
        red_indices (np.ndarray): _description_
        black_indices (np.ndarray): _description_
        adjacencies (np.ndarray): _description_
        update_probs (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    new_state = bath_state.copy()

    # --- Layer 1: red controls -> update black targets ---
    for j in black_indices:
        # Count number of red neighbours in state 1
        k = 0


def _get_adjacencies(
    experiment: Experiment, max_deg: int, tol: float = 1e-5
) -> np.ndarray:
    """Build the compact adjacency matrix for the environment qubits in the experiment.

    Args:
        experiment (Experiment): _description_
        tol (float, optional): _description_. Defaults to 1e-5.

    Returns:
        np.ndarray: _description_
    """

    qubit_list = experiment.all_qubits
    coords = experiment.circuit.get_final_qubit_coordinates()
    coords_array = np.array([coords[q] for q in qubit_list])

    # -1 means no neighbor in that direction
    adjacencies = np.zeros((len(qubit_list), max_deg), dtype=np.int32) - 1

    for i, q in enumerate(qubit_list):
        coord = coords[q]
        distances = np.linalg.norm(coords_array - coord, axis=1)
        distances[i] = np.inf  # Ignore self-distance
        min_dist = np.min(distances)
        neighbours = np.where(np.abs(distances - min_dist) < tol)[0]
        adjacencies[i, 0 : len(neighbours)] = neighbours

    return adjacencies


class QCAModel(NoiseModel):
    def __init__(
        self,
        model_params: dict,
        gate_noise: dict | None = None,
    ):
        super().__init__(gate_noise=gate_noise, noisy_qubit_types="all")
        self._no_error_matrix = False
        self.model_params = model_params
