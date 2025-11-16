from abc import ABC, abstractmethod
import numpy as np
import stim

# from ..noisemodels.base_noise_model import NoiseModel
# from ..sample.sampler import Sampler


class Experiment(ABC):
    """
    Base class for QEC experiments.

    An Experiment defines the structure of QEC protocol, including:
    - Its Stim circuit
    - Qubit organization and types
    - How the circuit is split for error injection
    - The dimensions of the error matrix
    """

    def __init__(self, distance: int, rounds: int | str):

        self._distance = distance
        self._rounds = self._parse_rounds(rounds)

        # Main initializations
        self._circuit = self.gen_stim_circuit()
        self._split_circuits = self.gen_split_circuits()
        self._qubit_coords = self.group_qubit_coords_by_type()

    def _parse_rounds(self, rounds: int | str) -> int:
        """Parse the rounds parameter, which can be an integer or a string like 'd' or '2d'.

        Args:
            rounds (Union[int, str]): Number of rounds or a string representing rounds relative to distance.

        Raises:
            ValueError: _description_

        Returns:
            int: _description_
        """
        if isinstance(rounds, int):
            return rounds
        elif isinstance(rounds, str):
            if rounds == "d":
                return self.distance
            elif (
                rounds.endswith("d") and rounds[:-1].isdigit() and int(rounds[:-1]) > 0
            ):
                return self.distance * int(rounds[:-1])

        raise ValueError(f"Invalid rounds value: {rounds}")

    # PROPERTIES
    @property
    def distance(self) -> int:
        return self._distance

    @property
    def rounds(self) -> int:
        return self._rounds

    @property
    def circuit(self) -> stim.Circuit:
        return self._circuit

    @property
    def split_circuits(self) -> list[stim.Circuit | tuple[int, stim.Circuit]]:
        return self._split_circuits

    @property
    def qubit_coords(self) -> dict[str, dict[int, list[float]]]:
        return self._qubit_coords

    @property
    def qubits(self) -> dict[str, list[int]]:
        return {
            qtype: list(coords.keys()) for qtype, coords in self._qubit_coords.items()
        }

    @property
    def all_qubits(self) -> list[int]:
        """Sorted list of all qubit indices."""
        all_qubits = []
        for qubit_list in self.qubits.values():
            all_qubits.extend(qubit_list)
        return sorted(set(all_qubits))  # Remove duplicates

    # ABSTRACT PROPERTIES (NEED TO BE IMPLEMENTED IN SUBCLASS)

    @property
    @abstractmethod
    def all_qubit_types(self) -> list[str]:
        """List of all qubit types in this experiment (e.g., ['data', 'syndrome'])."""
        pass

    # INITIALIZATION METHODS (NEED TO BE IMPLEMENTED IN SUBCLASS)

    @abstractmethod
    def gen_stim_circuit(self) -> stim.Circuit:
        """Generate the main Stim circuit for the experiment."""
        pass

    @abstractmethod
    def gen_split_circuits(self) -> list[stim.Circuit | tuple[int, stim.Circuit]]:
        """Generate the list of split circuits for error injection."""
        pass

    @abstractmethod
    def group_qubit_coords_by_type(self) -> dict[str, dict[int, list[float]]]:
        """Group qubit coordinates by their types."""
        pass

    # QUERY METHODS

    def get_qubits_by_type(self, qubit_types: list[str]) -> list[int]:
        """Returns a list of qubit indices for the specified qubit type(s)."""
        if "all" in qubit_types:
            qubits = self.all_qubits
        else:
            qubits = []
            for qtype in qubit_types:
                qubits.extend(self.qubits[qtype])
        return sorted(set(qubits))  # Remove duplicates and sort

    def get_error_matrix_shape(self, qubit_types: list[str]) -> tuple[int, ...]:
        """Returns the dimensions of the error matrix: (# noisy qubits, # rounds)."""
        n_target_qubits = len(self.get_qubits_by_type(qubit_types))
        return n_target_qubits, self.rounds
