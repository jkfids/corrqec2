import numpy as np
from numba import njit
import stim

from ..experiments import Experiment
from .base_noise_model import NoiseModel


@njit
def _sample_storm_hmm_batch(
    n_chains: int,
    n_rounds: int,
    initial_probs: np.ndarray,  # shape (2,)
    transfer_matrix: np.ndarray,  # shape (2, 2)
    emission_probs: np.ndarray,  # shape (2, 4)
) -> np.ndarray:
    """Numba JIT implementation of batch sampling from storm HMM model.

    Args:
        n_chains: Number of independent chains to sample.
        n_rounds: Number of rounds per chain.
        initial_probs: Initial probabilities of the calm and stormy states,
            shape (2,).
        transfer_matrix: State transition probabilities, shape (2, 2).
        emission_probs: Pauli emission probabilities for the calm and stormy
            states, shape (2, 4).

    Returns:
        Pauli indices (0=I, 1=X, 2=Y, 3=Z), shape (n_chains, n_rounds).
    """
    samples = np.empty((n_chains, n_rounds), dtype=np.int8)

    for i in range(n_chains):
        # Initial bath state
        u = np.random.rand()
        if u < initial_probs[0]:
            state = 0
        else:
            state = 1

        # Generate samples for this chain
        for j in range(n_rounds):
            # Emission
            v = np.random.rand()
            cumulative = 0.0
            k_chosen = 3  # Default fallback value
            for k in range(4):
                cumulative += emission_probs[state, k]
                if v < cumulative:
                    k_chosen = k
                    break
            samples[i, j] = k_chosen

            # Transition
            w = np.random.rand()
            if w < transfer_matrix[state, 0]:
                state = 0
            else:
                state = 1

    return samples


class StormModel(NoiseModel):
    """
    Noise model based on a two-state hidden Markov model (HMM) representing 'stormy' and 'calm' states.
    """

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
        # Stationary prob. of being in stormy state
        self._pi_a = a / (a + b)
        # Stationary prob. of being in calm state
        self._pi_b = b / (a + b)
        # Initial probabilities
        self._initial_probs = np.array([self._pi_b, self._pi_a])

    def gen_error_matrix(
        self, experiment: Experiment, n_samples: int = 1
    ) -> np.ndarray:
        """Generate error matrix for custom Pauli noise model for experiment batches.

        Args:
            experiment: Experiment fixing the qubit layout and number of rounds.
            n_samples: Number of samples in the batch. Defaults to 1.

        Raises:
            NotImplementedError: If the experiment type is not supported.

        Returns:
            Pauli indices (0=I, 1=X, 2=Y, 3=Z), shape
            (n_samples, n_noisy_qubits, n_rounds).
        """
        n_qubits, n_rounds = experiment.get_error_matrix_shape(self.noisy_qubit_types)

        samples = _sample_storm_hmm_batch(
            n_chains=n_qubits * n_samples,
            n_rounds=n_rounds,
            initial_probs=self._initial_probs,
            transfer_matrix=self._T,
            emission_probs=self._emissions,
        )

        error_matrix = samples.reshape(n_samples, n_qubits, n_rounds)

        return error_matrix

    def gen_marginalized_circuit(self, experiment: Experiment) -> stim.Circuit:
        """Generate noisy circuit with marginalized, independent noise.

        Args:
            experiment: Experiment fixing the qubit layout and number of rounds.

        Raises:
            NotImplementedError: If the experiment type is not supported.

        Returns:
            Circuit carrying independent Stim noise channels at the model's
            marginal error rates.
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
        new_circuit = Experiment.combine_split_circuits(
            [split_circuits[0]] + subcircuits_new + [split_circuits[-1]]
        )

        return new_circuit

    def gen_detector_error_model(
        self, experiment: Experiment
    ) -> stim.DetectorErrorModel:
        pass
