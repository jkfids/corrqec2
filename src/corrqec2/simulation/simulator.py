import numpy as np
import stim

from ..experiments.base_experiment import Experiment
from ..noisemodels.base_noise_model import NoiseModel
from .error_masks import ErrorMasks


class CircuitSimulator:
    """Simulates stabilizer circuits with custom Pauli errors using Stim's FlipSimulator class."""

    def __init__(
        self,
        experiment: Experiment,
        noise_model: NoiseModel,
    ):
        self._experiment = experiment

        if noise_model._no_gate_noise:
            self._split_circuits = experiment.split_circuits
        else:
            self._split_circuits = noise_model.gen_noisy_circuit(
                experiment, split_circuit=True
            )
        self._noise_model = noise_model

    def simulate_batch(
        self,
        error_masks: ErrorMasks,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a batch of circuits with the given error masks.

        Args:
            error_masks: Masks specifying where to apply Pauli errors.

        Returns:
            Detection events and observable flips for the batch.
        """

        batch_size = error_masks.n_samples
        sim = stim.FlipSimulator(batch_size=batch_size)

        # Initial circuit segment
        sim.do(self._split_circuits[0])

        # Apply errors at the start of each round
        round_idx = 0
        for subcircuit in self._split_circuits[1:-1]:
            if isinstance(subcircuit, stim.Circuit):
                self._apply_pauli_flips(sim, error_masks, round_idx)
                sim.do(subcircuit)
                round_idx += 1
            elif isinstance(subcircuit, tuple):
                repeat_count, repeat_circuit = subcircuit
                for _ in range(repeat_count):
                    self._apply_pauli_flips(sim, error_masks, round_idx)
                    sim.do(repeat_circuit)
                    round_idx += 1
        sim.do(self._split_circuits[-1])
        detection_events = sim.get_detector_flips().T
        observable_flips = sim.get_observable_flips().T

        return detection_events, observable_flips

    @staticmethod
    def _apply_pauli_flips(
        sim: stim.FlipSimulator, error_masks: ErrorMasks, round_idx: int
    ):
        """Apply Pauli errors for the specific round."""
        sim.broadcast_pauli_errors(pauli="X", mask=error_masks.X_mask[round_idx])
        sim.broadcast_pauli_errors(pauli="Y", mask=error_masks.Y_mask[round_idx])
        sim.broadcast_pauli_errors(pauli="Z", mask=error_masks.Z_mask[round_idx])
