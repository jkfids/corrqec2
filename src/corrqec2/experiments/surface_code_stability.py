# Stand imports

# Third-party imports
import numpy as np
import stim

# Local imports
from .base_experiment import Experiment
from .stability_paper.circuits import surface_code_stability_experiment_circuit


class SurfaceCodeStability(Experiment):
    def __init__(self, distance: int, rounds: int | str, basis: str = "Z"):

        self.basis = basis
        super().__init__(distance=distance, rounds=rounds)

    @property
    def all_qubit_types(self) -> list[str]:
        return ["data", "syndrome"]

    def gen_stim_circuit(self) -> stim.Circuit:

        circuit = surface_code_stability_experiment_circuit(
            diam=self.distance,
            rounds=self.rounds,
            basis=self.basis,
        )

        return circuit

    def gen_split_circuits(self) -> list[stim.Circuit | tuple[int, stim.Circuit]]:

        i_tick = None
        i_block = None
        i_tick_reverse = None

        for i, instr in enumerate(self.circuit):
            if instr.name == "TICK" and i_tick is None:
                i_tick = i
            elif isinstance(instr, stim.CircuitRepeatBlock) and i_block is None:
                i_block = i
                break
        for i, instr in enumerate(self.circuit[::-1]):
            if instr.name == "TICK" and i_tick_reverse is None:
                i_tick_reverse = len(self.circuit) - i - 1
                break

        circuit_init = self.circuit[: i_tick + 1]
        circuit_init_round = self.circuit[i_tick + 1 : i_block]
        circuit_repeat_block = self.circuit[i_block].body_copy()
        repeat_count = self.circuit[i_block].repeat_count
        circuit_final_round = self.circuit[i_block + 1 : i_tick_reverse + 1]
        circuit_final = self.circuit[i_tick_reverse + 1 :]

        return [
            circuit_init,
            circuit_init_round,
            (repeat_count, circuit_repeat_block),
            circuit_final_round,
            circuit_final,
        ]

    def group_qubit_coords_by_type(self) -> dict[str, dict[int, list[float]]]:
        qubit_coords = {type: {} for type in self.all_qubit_types}
        all_qubit_coords = self.circuit.get_final_qubit_coordinates()
        all_qubits = list(all_qubit_coords.keys())
        for instr in self.circuit:
            if instr.name == "I":
                qubit_coords["data"] = {
                    q.value: all_qubit_coords[q.value] for q in instr.targets_copy()
                }
                break
        qubit_coords["syndrome"] = {
            q: all_qubit_coords[q] for q in all_qubits if q not in qubit_coords["data"]
        }
        return qubit_coords
