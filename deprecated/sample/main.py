import numpy as np
import stim

from ..experiments.base_experiment import Experiment
from ..noisemodels.base_noise_model import NoiseModel


def sample_with_custom_errors(
    experiment: Experiment,
    error_matrix: np.ndarray,
    gate_noise: dict | None = {},
    seed: int | None = None,
):

    if not experiment._circuit_generated:
        experiment.gen_stim_circuit()
    if experiment.error_matrix_dims != error_matrix.shape[1:]:
        raise ValueError(
            f"Error matrix dimensions {error_matrix.shape} do not match experiment error matrix dimensions {experiment.error_matrix_dims}"
        )

    batch_size = error_matrix.shape[0]
    sim = stim.FlipSimulator(batch_size=batch_size)

    if gate_noise is None or gate_noise == {}:
        split_circuits = experiment.split_circuits
    else:
        noise_model = NoiseModel(gate_noise)
        split_circuits = noise_model.gen_noisy_circuit(experiment, split_circuit=True)

    n_stim_qubits = split_circuits[0].num_qubits
    full_error_matrix = np.zeros(
        (error_matrix.shape[0], n_stim_qubits, error_matrix.shape[2])
    )
    full_error_matrix[:, experiment.noisy_qubits, :] = error_matrix

    X_mask, Y_mask, Z_mask = error_matrix_to_masks(full_error_matrix)

    sim.do(split_circuits[0])
    for subcircuit in split_circuits[1:-1]:
        if isinstance(subcircuit, stim.Circuit):
            sim.broadcast_pauli_errors(pauli="X", mask=X_mask[0])
            sim.broadcast_pauli_errors(pauli="Y", mask=Y_mask[0])
            sim.broadcast_pauli_errors(pauli="Z", mask=Z_mask[0])
            sim.do(subcircuit)
        elif isinstance(subcircuit, tuple):
            repeat_count, repeat_circuit = subcircuit
            for i in range(repeat_count):
                sim.broadcast_pauli_errors(pauli="X", mask=X_mask[i + 1])
                sim.broadcast_pauli_errors(pauli="Y", mask=Y_mask[i + 1])
                sim.broadcast_pauli_errors(pauli="Z", mask=Z_mask[i + 1])
                sim.do(repeat_circuit)
    sim.do(split_circuits[-1])

    detection_events = sim.get_detector_flips()
    observable_flips = sim.get_observable_flips()

    return detection_events, observable_flips


def error_matrix_to_masks(
    error_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transposed = error_matrix.transpose(2, 1, 0)
    X_mask = transposed == 1
    Y_mask = transposed == 2
    Z_mask = transposed == 3
    return X_mask, Y_mask, Z_mask


if __name__ == "__main__":
    pass
