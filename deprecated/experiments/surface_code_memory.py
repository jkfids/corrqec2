# Stand imports
from typing import List, Union

# Third-party imports
import numpy as np
import stim

# Local imports
from .base_experiment import Experiment


class SurfaceCodeMemory(Experiment):
    def __init__(
        self, distance: int, rounds: Union[int, str] = "d", memory_type: str = "Z"
    ):
        self.distance = distance
        self.rounds = self._parse_rounds(rounds)

        self._circuit_generated = False
        self.all_qubit_types = ["data", "syndrome"]
        self.memory_type = memory_type

        self.circuit = None
        self.split_circuits = None

        self.all_qubit_coords = None
        self.all_qubits = None
        self.qubit_coords = None
        self.qubits = None

        self.gen_stim_circuit()

    def _parse_rounds(self, rounds: Union[int, str]) -> int:
        """_summary_

        Args:
            rounds (Union[int, str]): _description_

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

    def gen_stim_circuit(self) -> stim.Circuit:
        self.circuit = self._gen_stim_circuit()
        self.split_circuits = self._get_split_circuits()

        self.all_qubit_coords = self.circuit.get_final_qubit_coordinates()
        self.all_qubits = list(self.all_qubit_coords.keys())
        self.qubit_coords = self._group_qubit_coords_by_type()
        self.qubits = {
            type: list(qubit_coords.keys())
            for type, qubit_coords in self.qubit_coords.items()
        }

        self._circuit_generated = True

        return self.circuit

    def _gen_stim_circuit(self) -> stim.Circuit:
        """_summary_

        Returns:
            _type_: _description_
        """
        circuit = stim.Circuit()
        code_task = "surface_code:rotated_memory_" + self.memory_type.lower()

        # Borrow the base circuit from Stim
        base_circuit = stim.Circuit.generated(
            code_task=code_task, distance=self.distance, rounds=self.rounds
        )

        # We insert identity gates on data qubits at the start of each round for convenience when injecting gate noise later on
        # Find data qubits
        for instr in base_circuit[::-1]:
            if instr.name == "MX" or instr.name == "M":
                targets = instr.targets_copy()
                break

        # Insert identity gates
        first_tick = False
        for instr in base_circuit:
            if isinstance(instr, stim.CircuitInstruction):
                circuit.append(instr)
                if instr.name == "TICK" and not first_tick:
                    first_tick = True
                    circuit.append("I", targets)
            elif isinstance(instr, stim.CircuitRepeatBlock):
                repeat_count = instr.repeat_count
                repeat_circuit = instr.body_copy()
                repeat_circuit.insert(1, stim.CircuitInstruction("I", targets))
                circuit.append(stim.CircuitRepeatBlock(repeat_count, repeat_circuit))

        return circuit

    def _get_split_circuits(
        self,
    ) -> List[Union[stim.Circuit, tuple[int, stim.Circuit]]]:
        """_summary_

        Returns:
            _type_: _description_
        """
        i_tick = None
        i_block = None

        for i, instr in enumerate(self.circuit):
            if instr.name == "TICK" and i_tick is None:
                i_tick = i
            elif isinstance(instr, stim.CircuitRepeatBlock) and i_block is None:
                i_block = i
                break

        circuit_init = self.circuit[: i_tick + 1]
        circuit_init_round = self.circuit[i_tick + 1 : i_block]
        circuit_repeat_block = self.circuit[i_block].body_copy()
        repeat_count = self.circuit[i_block].repeat_count
        circuit_final = self.circuit[i_block + 1 :]

        return [
            circuit_init,
            circuit_init_round,
            (repeat_count, circuit_repeat_block),
            circuit_final,
        ]

    def _group_qubit_coords_by_type(self) -> dict:
        """_summary_

        Returns:
            _type_: _description_
        """
        qubit_coords = {type: {} for type in self.all_qubit_types}
        for instr in self.circuit[::-1]:
            if instr.name == "MX" or instr.name == "M":
                qubit_coords["data"] = {
                    q.value: self.all_qubit_coords[q.value]
                    for q in instr.targets_copy()
                }
                break

        qubit_coords["syndrome"] = {
            q: self.all_qubit_coords[q]
            for q in self.all_qubits
            if q not in qubit_coords["data"]
        }
        return qubit_coords

    def get_qubits_by_type(self, qubit_types: str | List) -> List[int]:
        """Returns a list of qubit indices for the specified qubit type(s)."""
        if isinstance(qubit_types, str):
            qubit_types = [qubit_types]
        if "all" in qubit_types:
            return self.all_qubits
        qubits = []
        for qt in qubit_types:
            qubits += self.qubits[qt]
        return qubits

    def error_matrix_shape(self, qubit_types: str | List) -> tuple[int, int]:
        """Returns the dimensions of the error matrix: (# noisy qubits, # rounds)."""
        if isinstance(qubit_types, str):
            qubit_types = [qubit_types]
        n_qubits = len(self.get_qubits_by_type(qubit_types))
        return n_qubits, self.rounds


if __name__ == "__main__":
    pass
