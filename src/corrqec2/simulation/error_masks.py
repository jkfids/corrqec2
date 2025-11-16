from dataclasses import dataclass
import numpy as np

from ..experiments import Experiment


@dataclass(frozen=True)
class ErrorMasks:
    """Boolean masks for X, Y, Z Pauli errors. Each mask as shape (n_rounds, n_qubits, n_samples)"""

    X_mask: np.ndarray
    Y_mask: np.ndarray
    Z_mask: np.ndarray

    def __post_init__(self):
        shapes = {self.X_mask.shape, self.Y_mask.shape, self.Z_mask.shape}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent mask shapes: {shapes}")

    @property
    def shape(self):
        return self.X_mask.shape

    @property
    def n_samples(self):
        return self.X_mask.shape[2]


class ErrorMaskConverter:
    """Convert between error matrix and boolean mask representations for input into stim FlipSimulator."""

    @staticmethod
    def error_matrix_to_error_masks(
        error_matrix: np.ndarray,
        experiment: Experiment,
        noisy_qubit_types: list[str],
    ) -> ErrorMasks:
        """_summary_

        Args:
            error_matrix (np.ndarray): _description_
            experiment (Experiment): _description_
            noisy_qubit_types (list[str]): _description_

        Returns:
            ErrorMasks: _description_
        """
        padded_error_matrix = ErrorMaskConverter._pad_to_full_qubits(
            error_matrix, experiment, noisy_qubit_types
        )
        return ErrorMaskConverter._masks_from_full_error_matrix(padded_error_matrix)

    @staticmethod
    def _pad_to_full_qubits(
        error_matrix: np.ndarray,
        experiment: Experiment,
        noisy_qubit_types: list[str],
    ) -> np.ndarray:
        """Pad error matrix with zeros for non-noisy qubits."""

        n_stim_qubits = experiment.circuit.num_qubits
        noisy_qubits = experiment.get_qubits_by_type(noisy_qubit_types)
        n_samples, _, n_rounds = error_matrix.shape
        padded_error_matrix = np.zeros((n_samples, n_stim_qubits, n_rounds))
        padded_error_matrix[:, noisy_qubits, :] = error_matrix
        return padded_error_matrix

    @staticmethod
    def _masks_from_full_error_matrix(padded_error_matrix: np.ndarray) -> ErrorMasks:
        """Convert full error matrix to boolean masks."""
        # Transpose to (n_rounds, n_qubits, n_samples)
        transposed = padded_error_matrix.transpose(2, 1, 0)
        return ErrorMasks(
            X_mask=(transposed == 1), Y_mask=(transposed == 2), Z_mask=(transposed == 3)
        )
