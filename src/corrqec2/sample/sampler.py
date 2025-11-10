from typing import List

import numpy as np
import stim

from ..experiments import Experiment, get_noisy_qubits, error_matrix_shape
from ..noisemodels import NoiseModel

class Sampler:
    def __init__(self, experiment: Experiment, noise_model: NoiseModel, batch_size: int = 1):
        self._sim = stim.FlipSimulator(batch_size=batch_size)
        
        if not experiment._circuit_generated:
            experiment.gen_stim_circuit()
        self._experiment = experiment
        
        if noise_model._no_gate_noise:
            self._split_circuits = experiment.split_circuits
        else:
            self._split_circuits = noise_model.gen_noisy_circuit(experiment, split_circuit=True)
        self._noise_model = noise_model
        
        # self._split_circuits = None  # Ensure attribute exists
        
    def sample(self, n_samples: int):
        pass
    
    def _sample_batch(self):
        error_matrix = self._noise_model.gen_error_matrix(self._experiment, self._sim.batch_size)
        error_masks = self._masks_from_error_matrix(error_matrix)
        return self._sample_error_matrix(error_masks)

    @classmethod
    def sample_with_custom_errors(cls, experiment: Experiment, error_matrix: np.ndarray, gate_noise: dict | None = None, noisy_qubit_types: str | List[str] = 'all'):
        noise_model = NoiseModel(gate_noise=gate_noise, noisy_qubit_types=noisy_qubit_types)
        if len(error_matrix.shape) == 2:
            error_matrix = error_matrix[np.newaxis, ...]
        sampler = cls(experiment, noise_model, batch_size=error_matrix.shape[0])

        expected_shape = error_matrix_shape(experiment, noise_model.noisy_qubit_types)
        if error_matrix.shape[1:] != expected_shape:
            raise ValueError(f"Error matrix dimensions {error_matrix.shape[1:]} do not match experiment error matrix dimensions {expected_shape}")
        
        error_masks = sampler._masks_from_error_matrix(error_matrix)
        return sampler._sample_error_matrix(error_masks)

    def _sample_error_matrix(self, error_masks: tuple[np.ndarray, np.ndarray, np.ndarray]):
        # if self._split_circuits is None:
        #     raise RuntimeError("split_circuits not initialized before sampling")
        self._sim.do(self._split_circuits[0])
        i = 0
        for subcircuit in self._split_circuits[1:-1]:
            if isinstance(subcircuit, stim.Circuit):
                self._apply_pauli_flips(error_masks, i)
                self._sim.do(subcircuit)
                i += 1
            elif isinstance(subcircuit, tuple):
                repeat_count, repeat_circuit = subcircuit
                for _ in range(repeat_count):
                    self._apply_pauli_flips(error_masks, i)
                    self._sim.do(repeat_circuit)
                    i += 1
        self._sim.do(self._split_circuits[-1])
        detection_events = self._sim.get_detector_flips().transpose()  # .transpose() so that it's right shape to input into PyMatching
        observable_flips = self._sim.get_observable_flips().flatten()
        self._sim.clear()
        return detection_events, observable_flips

    def _apply_pauli_flips(self, error_masks: tuple[np.ndarray, np.ndarray, np.ndarray], i: int):
        """Helper function to broadcast Pauli errors in simulation instance."""
        self._sim.broadcast_pauli_errors(pauli='X', mask=error_masks[0][i])
        self._sim.broadcast_pauli_errors(pauli='Y', mask=error_masks[1][i])
        self._sim.broadcast_pauli_errors(pauli='Z', mask=error_masks[2][i])

    def _masks_from_error_matrix(self, error_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert error matrix to boolean masks for X, Y, and Z errors to input in self._apply_pauli_flips().

        Args:
            error_matrix (np.ndarray): _description_

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
        """
        full_error_matrix = self._full_error_matrix_from_error_matrix(error_matrix)
        return self._masks_from_full_error_matrix(full_error_matrix)

    def _full_error_matrix_from_error_matrix(self, error_matrix: np.ndarray) -> np.ndarray:
        """Generate full error matrix from error matrix by padding error_matrix with zeros.

        Args:
            error_matrix (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        n_stim_qubits = self._experiment.circuit.num_qubits
        noisy_qubits = get_noisy_qubits(self._experiment, self._noise_model.noisy_qubit_types)
        full_error_matrix = np.zeros((error_matrix.shape[0], n_stim_qubits, error_matrix.shape[2]))
        full_error_matrix[:, noisy_qubits, :] = error_matrix
        return full_error_matrix

    @staticmethod
    def _masks_from_full_error_matrix(full_error_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert full error matrix to boolean masks for X, Y, and Z errors."""
        transposed = full_error_matrix.transpose(2, 1, 0)
        return transposed == 1, transposed == 2, transposed == 3
