# Adapted from the source code accompanying:
#   C. Gidney, "Stability Experiments: The Overlooked Dual of Memory
#   Experiments", Quantum 6, 786 (2022).
#   Archived at https://doi.org/10.5281/zenodo.6859486
# Copyright (c) 2022 Craig Gidney.
# Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
# Modified for integration into corrqec2. Provided "as is", without warranties
# or conditions of any kind. See the "Third-party code" section of the README.

from typing import (
    Iterable,
    Dict,
    Callable,
    Any,
    Optional,
    List,
    Tuple,
    Generic,
    TypeVar,
)

import dataclasses

import stim

from ._util import complex_key, sorted_complex


T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class AtLayer(Generic[T]):
    """A special class that indicates the layer to read a measurement key from."""

    key: Any
    layer: int


class MeasurementTracker:
    """Tracks measurements and groups of measurements, for producing stim record targets."""

    def __init__(self):
        self.recorded: Dict[Any, Optional[List[int]]] = {}
        self.next_measurement_index = 0

    def copy(self) -> "MeasurementTracker":
        result = MeasurementTracker()
        result.recorded = {k: list(v) for k, v in self.recorded.items()}
        result.next_measurement_index = self.next_measurement_index
        return result

    def _rec(self, key: Any, value: Optional[List[int]]) -> None:
        if key in self.recorded:
            raise ValueError(f"Measurement key collision: {key=}")
        self.recorded[key] = value

    def record_measurement(self, key: Any) -> None:
        self._rec(key, [self.next_measurement_index])
        self.next_measurement_index += 1

    def make_measurement_group(self, sub_keys: Iterable[Any], *, key: Any) -> None:
        self._rec(key, self.measurement_indices(sub_keys))

    def record_obstacle(self, key: Any) -> None:
        self._rec(key, None)

    def measurement_indices(self, keys: Iterable[Any]) -> List[int]:
        result = set()
        for key in keys:
            if key not in self.recorded:
                raise ValueError(f"No such measurement: {key=}")
            for v in self.recorded[key]:
                if v is None:
                    raise ValueError(f"Obstacle at {key=}")
                if v in result:
                    result.remove(v)
                else:
                    result.add(v)
        return sorted(result)

    def current_measurement_record_targets_for(
        self, keys: Iterable[Any]
    ) -> List[stim.GateTarget]:
        t0 = self.next_measurement_index
        times = self.measurement_indices(keys)
        return [stim.target_rec(t - t0) for t in sorted(times)]


class Builder:
    """Helper class for building stim circuits.

    Handles qubit indexing (complex -> int conversion).
    Handles measurement tracking (naming results and referring to them by name).
    """

    def __init__(
        self,
        *,
        q2i: Dict[complex, int],
        circuit: stim.Circuit,
        tracker: MeasurementTracker,
    ):
        self.q2i = q2i
        self.circuit = circuit
        self.tracker = tracker

    def copy(self) -> "Builder":
        return Builder(
            q2i=dict(self.q2i), circuit=self.circuit.copy(), tracker=self.tracker.copy()
        )

    @staticmethod
    def for_qubits(qubits: Iterable[complex]) -> "Builder":
        q2i = {q: i for i, q in enumerate(sorted_complex(set(qubits)))}
        circuit = stim.Circuit()
        for q, i in q2i.items():
            circuit.append("QUBIT_COORDS", [i], [q.real, q.imag])
        return Builder(
            q2i=q2i,
            circuit=circuit,
            tracker=MeasurementTracker(),
        )

    def gate(self, name: str, qubits: Iterable[complex], sorted=True) -> None:
        if sorted:
            qubits = sorted_complex(qubits)
        self.circuit.append(name, [self.q2i[q] for q in qubits])

    def shift_coords(self, *, dp: complex = 0, dt: int):
        self.circuit.append("SHIFT_COORDS", [], [dp.real, dp.imag, dt])

    def measure(
        self,
        qubits: Iterable[complex],
        *,
        basis: str = "Z",
        tracker_key: Callable[[complex], Any] = lambda e: e,
        layer: int,
    ) -> None:
        qubits = sorted_complex(qubits)
        self.circuit.append(f"M{basis}", [self.q2i[q] for q in qubits])
        for q in qubits:
            self.tracker.record_measurement(AtLayer(tracker_key(q), layer))

    def measure_pauli_product(
        self,
        *,
        xs: Iterable[complex] = (),
        ys: Iterable[complex] = (),
        zs: Iterable[complex] = (),
        qs: Dict[str, Iterable[complex]] = None,
        key: Any,
        layer: int = -1,
    ):
        x = set(xs)
        y = set(ys)
        z = set(zs)
        if qs is not None:
            for b, bqs in qs.items():
                if b == "X":
                    x |= set(bqs)
                elif b == "Y":
                    y |= set(bqs)
                elif b == "Z":
                    z |= set(bqs)
                else:
                    raise NotImplementedError(f"{b=}")
        xz = x & z
        xy = x & y
        yz = y & z
        x -= xz
        x -= xy
        z -= xz
        z -= yz
        y -= xy
        y -= yz
        x |= yz
        y |= xz
        z |= xy
        vals = {}
        for q in x:
            vals[q] = stim.target_x(self.q2i[q])
        for q in y:
            vals[q] = stim.target_y(self.q2i[q])
        for q in z:
            vals[q] = stim.target_z(self.q2i[q])

        targets = []
        comb = stim.target_combiner()
        for q in sorted_complex(vals.keys()):
            targets.append(vals[q])
            targets.append(comb)
        if targets:
            targets.pop()
            self.circuit.append("MPP", targets)
            self.tracker.record_measurement(AtLayer(key, layer))
        else:
            self.tracker.make_measurement_group([], key=AtLayer(key, layer))

    def detector(
        self,
        keys: Iterable[Any],
        *,
        pos: Optional[complex] = None,
        t: int = 0,
        mark_as_post_selected: bool = False,
        ignore_non_existent: bool = False,
    ) -> None:
        if pos is not None:
            coords = [pos.real, pos.imag, t]
            if mark_as_post_selected:
                coords.append(1)
        else:
            if mark_as_post_selected:
                raise ValueError("pos is None and mark_as_post_selected")
            coords = None

        if ignore_non_existent:
            keys = [k for k in keys if k in self.tracker.recorded]
        targets = self.tracker.current_measurement_record_targets_for(keys)
        self.circuit.append("DETECTOR", targets, coords)

    def obs_include(self, keys: Iterable[Any], *, obs_index: int) -> None:
        self.circuit.append(
            "OBSERVABLE_INCLUDE",
            self.tracker.current_measurement_record_targets_for(keys),
            obs_index,
        )

    def tick(self) -> None:
        self.circuit.append("TICK")

    def cz(self, pairs: List[Tuple[complex, complex]]) -> None:
        sorted_pairs = []
        for a, b in pairs:
            if complex_key(a) > complex_key(b):
                a, b = b, a
            sorted_pairs.append((a, b))
        sorted_pairs = sorted(
            sorted_pairs, key=lambda e: (complex_key(e[0]), complex_key(e[1]))
        )
        # for a, b in sorted_pairs:
        #     self.circuit.append("CZ", [self.q2i[a], self.q2i[b]])
        for a, b in sorted_pairs:
            self.circuit.append("H", [self.q2i[b]])
            self.circuit.append("CX", [self.q2i[a], self.q2i[b]])
            self.circuit.append("H", [self.q2i[b]])

    def classical_paulis(
        self, *, control_keys: Iterable[Any], targets: Iterable[complex], basis: str
    ) -> None:
        gate = f"C{basis}"
        indices = [self.q2i[q] for q in sorted_complex(targets)]
        for rec in self.tracker.current_measurement_record_targets_for(control_keys):
            for i in indices:
                self.circuit.append(gate, [rec, i])
