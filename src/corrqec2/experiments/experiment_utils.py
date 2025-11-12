from typing import List, Tuple
import numpy as np
import stim
from .base_experiment import Experiment


def combine_split_circuits(
    split_circuits: list[stim.Circuit | tuple[int, stim.Circuit]],
) -> stim.Circuit:
    """_summary_

    Args:
        split_circuits (_type_): _description_

    Returns:
        _type_: _description_
    """
    combined_circuit = stim.Circuit()
    for subcircuit in split_circuits:
        if isinstance(subcircuit, stim.Circuit):
            combined_circuit += subcircuit
        elif isinstance(subcircuit, tuple):
            repeat_count, repeat_circuit = subcircuit
            combined_circuit.append(
                stim.CircuitRepeatBlock(repeat_count, repeat_circuit)
            )
    return combined_circuit


def get_noisy_qubits(
    experiment: Experiment, noisy_qubit_types: str | List[str]
) -> List[int]:
    """Returns a list of noisy qubit indices based on the specified qubit types."""
    noisy_qubit_types = parse_noisy_qubit_types(
        experiment, noisy_qubit_types
    )  # Ensure it's a list
    unsorted = []
    for qubit_type in noisy_qubit_types:
        unsorted += experiment.qubits[qubit_type]
    return [q for q in experiment.all_qubits if q in unsorted]


def error_matrix_shape(
    experiment: Experiment, noisy_qubit_types: str | List[str]
) -> Tuple[int, int]:
    """Returns the dimensions of the error matrix: (# noisy qubits, # rounds)."""
    noisy_qubit_types = parse_noisy_qubit_types(experiment, noisy_qubit_types)
    return len(get_noisy_qubits(experiment, noisy_qubit_types)), experiment.rounds


def format_noisy_qubits(
    experiment: Experiment, noisy_qubit_types: str | List[str]
) -> List[str]:
    """Returns a human-readable list of noisy qubits with their types and coordinates."""
    noisy_qubit_types = parse_noisy_qubit_types(experiment, noisy_qubit_types)
    noisy_qubits = get_noisy_qubits(experiment, noisy_qubit_types)
    qubit_list = np.empty(len(noisy_qubits), dtype=object)
    for qubit_type in noisy_qubit_types:
        for q, coords in experiment.qubit_coords[qubit_type].items():
            index = noisy_qubits.index(q)
            qubit_list[index] = f"{qubit_type}{tuple(coords)}"
    return qubit_list.tolist()


def parse_noisy_qubit_types(
    experiment: Experiment, noisy_qubit_types: str | List[str]
) -> List[str]:
    """_summary_

    Args:
        noisy_qubit_types (Union[str, List[str]]): _description_

    Raises:
        ValueError: _description_

    Returns:
        List[str]: _description_
    """
    if isinstance(noisy_qubit_types, str):
        if noisy_qubit_types == "all":
            return experiment.all_qubit_types
        elif noisy_qubit_types in experiment.all_qubit_types:
            return [noisy_qubit_types]
    elif isinstance(noisy_qubit_types, list):
        if all(type in experiment.all_qubit_types for type in noisy_qubit_types):
            return noisy_qubit_types

    raise ValueError(f"Invalid noisy_qubit_types value: {noisy_qubit_types}")
