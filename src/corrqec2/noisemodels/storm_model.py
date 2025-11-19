import os
import numpy as np
import stim

# Set JAX platform before import
_JAX_DEVICE = os.environ.get("JAX_PLATFORMS", "cpu")
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = _JAX_DEVICE

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from dynamax.hidden_markov_model import CategoricalHMM

from ..experiments.base_experiment import Experiment
from ..experiments.experiment_utils import combine_split_circuits
from .base_noise_model import NoiseModel


class StormModel(NoiseModel):
    def __init__(
        self,
        model_params: dict,
        gate_noise: dict | None = None,
        noisy_qubit_types: str | list[str] = "all",
    ):
        super().__init__(gate_noise, noisy_qubit_types)
        self.model_params = model_params
        self._no_error_matrix = False

        self.hmm, self.hmm_params = self._init_hmm()

    def _init_hmm(self):
        a = self.model_params["a"]  # transition prob. from calm state to stormy state
        b = self.model_params["b"]  # transition prob. from stormy state to calm state
        emissions = self.model_params["emissions"]
        pi_a = a / (a + b)  # stationary prob. of being in stormy state
        pi_b = b / (a + b)  # stationary prob. of being in calm state

        # num_states: Number of hidden states
        # emission_dim: Dimension of the emission space
        # num_classes: Size of the discrete emission alphabet
        hmm = CategoricalHMM(num_states=2, emission_dim=1, num_classes=4)
        params, _ = hmm.initialize(
            initial_probs=jnp.array([pi_b, pi_a]),
            transition_matrix=jnp.array([[1.0 - a, a], [b, 1.0 - b]]),
            emission_probs=jnp.array(emissions).reshape(2, 1, 4),
        )

        return hmm, params

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

        hmm = self.hmm
        params = self.hmm_params

        key = jr.PRNGKey(np.random.randint(0, 2**32))
        keys = jr.split(key, n_qubits * n_samples)
        _, samples = vmap(lambda k: hmm.sample(params, k, n_rounds))(keys)
        error_matrix = samples

        return error_matrix.reshape(n_samples, n_qubits, n_rounds)

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
        a = self.model_params["a"]
        b = self.model_params["b"]
        emissions = self.model_params["emissions"]
        calm_fraction = b / (a + b)
        storm_fraction = a / (a + b)
        p_I = calm_fraction * emissions[0][0] + storm_fraction * emissions[1][0]
        p_D = 1 - p_I  # Marginal independent depolarizing probability

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

    # def gen_marginalized_circuit(self, experiment: Experiment) -> stim.Circuit:
    #     """_summary_

    #     Args:
    #         experiment (Experiment): _description_

    #     Raises:
    #         NotImplementedError: _description_

    #     Returns:
    #         stim.Circuit: _description_
    #     """

    #     # Get base circuit
    #     if self.gate_noise is not None:
    #         init, init_round, (repeat_count, repeat_block), final = (
    #             self.gen_noisy_circuit(experiment, split_circuit=True)
    #         )
    #     else:
    #         init, init_round, (repeat_count, repeat_block), final = (
    #             experiment.split_circuits
    #         )

    #     # Calculate marginal error probabilities
    #     a = self.model_params["a"]
    #     b = self.model_params["b"]
    #     emissions = self.model_params["emissions"]
    #     calm_fraction = b / (a + b)
    #     storm_fraction = a / (a + b)
    #     p_I = calm_fraction * emissions[0][0] + storm_fraction * emissions[1][0]
    #     p_D = 1 - p_I  # Marginal independent depolarizing probability

    #     # Get qubit targets for marginal channel injection
    #     targets = experiment.get_qubits_by_type(self.noisy_qubit_types)

    #     # Append marginalized depolarization to the start of init_round
    #     init_round_new = stim.Circuit()
    #     init_round_new.append("DEPOLARIZE1", targets, p_D)
    #     init_round_new += init_round

    #     # Append marginalized depolarization to the start of repeat_block
    #     repeat_block_new = stim.Circuit()
    #     repeat_block_new.append("DEPOLARIZE1", targets, p_D)
    #     repeat_block_new += repeat_block

    #     # Combine all parts back into a single circuit
    #     new_circuit = combine_split_circuits(
    #         [init, init_round_new, (repeat_count, repeat_block_new), final]
    #     )

    #     return new_circuit

    def gen_detector_error_model(
        self, experiment: Experiment
    ) -> stim.DetectorErrorModel:
        pass
