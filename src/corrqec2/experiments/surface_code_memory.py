# Stand imports
from typing import List, Union

# Third-party imports
import numpy as np
import stim

# Local imports
from .base_experiment import Experiment


class SurfaceCodeMemory(Experiment):
    def __init__(self, distance: int, rounds: int | str, memory_type: str = "Z"):

        self.memory_type = memory_type
        super().__init__(distance=distance, rounds=rounds)

    @property
    def all_qubit_types(self) -> list[str]:
        return ["data", "syndrome"]

    def gen_stim_circuit(self) -> stim.Circuit:
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

    def gen_split_circuits(self) -> list[stim.Circuit | tuple[int, stim.Circuit]]:
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

    def group_qubit_coords_by_type(self) -> dict[str, dict[int, list[float]]]:
        qubit_coords = {type: {} for type in self.all_qubit_types}
        all_qubit_coords = self.circuit.get_final_qubit_coordinates()
        all_qubits = list(all_qubit_coords.keys())
        for instr in self.circuit[::-1]:
            if instr.name == "MX" or instr.name == "M":
                qubit_coords["data"] = {
                    q.value: all_qubit_coords[q.value] for q in instr.targets_copy()
                }
                break

        qubit_coords["syndrome"] = {
            q: all_qubit_coords[q] for q in all_qubits if q not in qubit_coords["data"]
        }
        return qubit_coords
