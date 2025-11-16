from abc import ABC, abstractmethod
from typing import List

import numpy as np
import stim

from ..experiments.base_experiment import Experiment
from ..experiments.experiment_utils import (
    combine_split_circuits,
    get_noisy_qubits,
    error_matrix_shape,
    format_noisy_qubits,
    parse_noisy_qubit_types,
)

# CURRENTLY MISSING MPP
from ..stim_gates import (
    single_qubit_clifford,
    two_qubit_clifford,
    measure_X,
    measure_Y,
    measure_Z,
    reset_X,
    reset_Y,
    reset_Z,
)

gate_noise_special_strings = [
    "after_identity_depolarization",
    "after_clifford_depolarization",
    "before_measure_flip_probability",
    "after_reset_flip_probability",
]


class NoiseModel(ABC):
    def __init__(
        self, gate_noise: dict | None = None, noisy_qubit_types: str | List[str] = "all"
    ):
        if isinstance(noisy_qubit_types, str):
            noisy_qubit_types = [noisy_qubit_types]
        self.noisy_qubit_types = noisy_qubit_types
        self.gate_noise = gate_noise
        if gate_noise is None or gate_noise == {}:
            self._no_gate_noise = True
        else:
            self._no_gate_noise = False

    def gen_noisy_circuit(
        self, experiment: Experiment, split_circuit: bool = False
    ) -> stim.Circuit | List[stim.Circuit | tuple[int, stim.Circuit]]:
        """Inject Stim circuit with built-in Stim noise channels

        Args:
            circuit (stim.Circuit | Experiment): _description_
            split_circuit (bool, optional): _description_. Defaults to False.

        Returns:
            stim.Circuit | List[stim.Circuit | tuple[int, stim.Circuit]]: _description_
        """

        if not isinstance(experiment, Experiment):
            raise ValueError(f"Invalid experiment type: {type(experiment)}")

        noisy_split_circuits = []
        for subcircuit in experiment.split_circuits:
            repeat_count, base_circuit = (
                subcircuit if isinstance(subcircuit, tuple) else (None, subcircuit)
            )
            noisy_circuit = self._inject_gate_noise(base_circuit, self.gate_noise)
            noisy_split_circuits.append(
                (repeat_count, noisy_circuit) if repeat_count else noisy_circuit
            )

        if not split_circuit:
            noisy_circuit = combine_split_circuits(noisy_split_circuits)
            return noisy_circuit
        else:
            return noisy_split_circuits

    @abstractmethod
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
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def gen_marginalized_circuit(self, experiment: Experiment) -> stim.Circuit:
        """Generate noisy circuit with marginalized, independent noise.

        Args:
            experiment (Experiment): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            stim.Circuit: _description_
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def gen_detector_error_model(
        self, experiment: Experiment
    ) -> stim.DetectorErrorModel:
        """Generate the detector error model for the correlated noise model.

        Args:
            experiment (Experiment): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            stim.DetectorErrorModel: _description_
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def gen_marginalized_detector_error_model(
        self, experiment: Experiment
    ) -> stim.DetectorErrorModel:
        """Generate the detector error model corresponding to marginalized independent noise."""
        circuit = self.gen_marginalized_circuit(experiment)
        return circuit.detector_error_model()

    @staticmethod
    def _inject_gate_noise(
        circuit: stim.Circuit, gate_noise_dict: dict
    ) -> stim.Circuit:
        noisy_circuit = stim.Circuit()
        for instr in circuit:
            # Handle before_measure_flip_probability
            # CURRENTLY APPLIES Z FLIP IF MEASURING IN THE X OR Y BASIS
            if (
                instr.name in measure_X + measure_Y + measure_Z
                and "before_measure_flip_probability" in gate_noise_dict
            ):
                channel_name = "X_ERROR" if instr.name in measure_Z else "Z_ERROR"
                noisy_circuit.append(
                    channel_name,
                    instr.targets_copy(),
                    gate_noise_dict["before_measure_flip_probability"],
                )

            # Append the original instruction
            noisy_circuit.append(instr)

            # Handle after_identity_depolarization
            if (
                instr.name in ["I", "II"]
                and "after_identity_depolarization" in gate_noise_dict
            ):
                channel_name = "DEPOLARIZE1" if instr.name == "I" else "DEPOLARIZE2"
                noisy_circuit.append(
                    channel_name,
                    instr.targets_copy(),
                    gate_noise_dict["after_identity_depolarization"],
                )

            # Handle after_clifford_depolarization (doesn't include identity gates)
            elif (
                instr.name in single_qubit_clifford + two_qubit_clifford
                and "after_clifford_depolarization" in gate_noise_dict
            ):
                channel_name = (
                    "DEPOLARIZE1"
                    if instr.name in single_qubit_clifford
                    else "DEPOLARIZE2"
                )
                noisy_circuit.append(
                    channel_name,
                    instr.targets_copy(),
                    gate_noise_dict["after_clifford_depolarization"],
                )

            # Handle after_reset_flip_probability
            # CURRENTLY APPLIES Z FLIP IF RESETING IN THE X OR Y BASIS
            elif (
                instr.name in reset_X + reset_Y + reset_Z
                and "after_reset_flip_probability" in gate_noise_dict
            ):
                channel_name = "X_ERROR" if instr.name in reset_Z else "Z_ERROR"
                noisy_circuit.append(
                    channel_name,
                    instr.targets_copy(),
                    gate_noise_dict["after_reset_flip_probability"],
                )

        return noisy_circuit


if __name__ == "__main__":
    pass
