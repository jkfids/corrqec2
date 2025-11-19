import os
import numpy as np
from numba import njit
import stim

from ..experiments.base_experiment import Experiment
from ..experiments.experiment_utils import combine_split_circuits
from .base_noise_model import NoiseModel


@njit
def _sample_storm_hmm_batch(
    n_chains: int,
    n_rounds: int,
    initial_probs: np.ndarray,  # shape (2,)
    transition_matrix: np.ndarray,  # shape (2, 2)
    emission_probs: np.ndarray,  # shape (2, 4)
) -> np.ndarray:
    """_summary_

    Args:
        n_chains (int): _description_
        n_rounds (int): _description_
        initial_probs (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    samples = np.empty((n_chains, n_rounds), dtype=np.int8)

    for i in range(n_chains):
        u = np.random.rand()
        if u < initial_probs[0]:
            state = 0
        else:
            state = 1

        for j in range(n_rounds):
            v = np.random.rand()
            cumulative = 0.0
            for k in range(4):
                cumulative += emission_probs[state, k]
                if v < cumulative:
                    break
            samples[i, j] = k

            w = np.random.rand()
            if w < transition_matrix[state, 0]:
                state = 0
            else:
                state = 1

    return samples


class StormModel(NoiseModel):
    def __init__(
        self,
        model_params: dict,
        gate_noise: dict | None = None,
        noisy_qubit_types: str | list[str] = "all",
    ):
        super().__init__(gate_noise, noisy_qubit_types)
        self._no_error_matrix = False
        self.model_params = model_params

        # Extract model parameters
        a, b = model_params["a"], model_params["b"]
        self._a = a
        self._b = b
        # Transfer/transition matrix
        self._T = np.array([[1.0 - a, a], [b, 1.0 - b]])
        # Emission probabilities
        self._emissions = np.array(model_params["emissions"])
        # stationary prob. of being in stormy state
        self._pi_a = a / (a + b)
        # stationary prob. of being in calm state
        self._pi_b = b / (a + b)
        # Initial probabilities
        self._initial_probs = np.array([self._pi_b, self._pi_a])

    def gen_error_matrix(
        self, experiment: Experiment, n_samples: int = 1
    ) -> np.ndarray:
        """Generate error matrix for custom Pauli noise model for experiment batches.

        Args:
            experiment (Experiment): _description_
            n_samples (int, optional): _description_. Defaults to 1.

        Raises:
            NotImplementedError: _description_

        Returns:
            np.ndarray: _description_
        """
        n_qubits, n_rounds = experiment.get_error_matrix_shape(
            qubit_types=self.noisy_qubit_types
        )

        samples = _sample_storm_hmm_batch(
            n_chains=n_qubits * n_samples,
            n_rounds=n_rounds,
            initial_probs=self._initial_probs,
            transition_matrix=self._T,
            emission_probs=self._emissions,
        )

        error_matrix = samples.reshape(n_samples, n_qubits, n_rounds)

        return error_matrix

    def gen_marginalized_circuit(self, experiment: Experiment) -> stim.Circuit:
        """_summary_

        Args:
            experiment (Experiment): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            stim.Circuit: _description_
        """

        # Get base circuit
        if self.gate_noise is not None:
            split_circuits = self.gen_noisy_circuit(experiment, split_circuit=True)
        else:
            split_circuits = experiment.split_circuits

        # Calculate marginal error probabilities
        emissions = self._emissions
        calm_fraction = self._pi_b
        storm_fraction = self._pi_a
        p_I = calm_fraction * emissions[0][0] + storm_fraction * emissions[1][0]
        p_D = 1.0 - p_I  # Marginal independent depolarizing probability

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
