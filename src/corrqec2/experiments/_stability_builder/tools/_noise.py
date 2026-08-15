# Adapted from the source code accompanying:
#   C. Gidney, "Stability Experiments: The Overlooked Dual of Memory
#   Experiments", Quantum 6, 786 (2022).
#   Archived at https://doi.org/10.5281/zenodo.6859486
# Copyright (c) 2022 Craig Gidney.
# Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
# Modified for integration into corrqec2. Provided "as is", without warranties
# or conditions of any kind. See the "Third-party code" section of the README.

from typing import (
    Optional,
    Dict,
    Set,
    List,
    Iterator,
    Union,
    AbstractSet,
    DefaultDict,
    Any,
)

import collections

import stim

CLIFFORD_1Q = "C1"
CLIFFORD_2Q = "C2"
ANNOTATION = "info"
MPP = "MPP"
MEASURE_RESET_1Q = "MR1"
JUST_MEASURE_1Q = "M1"
JUST_RESET_1Q = "R1"
NOISE = "!?"

OP_TYPES = {
    "I": CLIFFORD_1Q,
    "X": CLIFFORD_1Q,
    "Y": CLIFFORD_1Q,
    "Z": CLIFFORD_1Q,
    "C_XYZ": CLIFFORD_1Q,
    "C_ZYX": CLIFFORD_1Q,
    "H": CLIFFORD_1Q,
    "H_XY": CLIFFORD_1Q,
    "H_XZ": CLIFFORD_1Q,
    "H_YZ": CLIFFORD_1Q,
    "S": CLIFFORD_1Q,
    "SQRT_X": CLIFFORD_1Q,
    "SQRT_X_DAG": CLIFFORD_1Q,
    "SQRT_Y": CLIFFORD_1Q,
    "SQRT_Y_DAG": CLIFFORD_1Q,
    "SQRT_Z": CLIFFORD_1Q,
    "SQRT_Z_DAG": CLIFFORD_1Q,
    "S_DAG": CLIFFORD_1Q,
    "CNOT": CLIFFORD_2Q,
    "CX": CLIFFORD_2Q,
    "CY": CLIFFORD_2Q,
    "CZ": CLIFFORD_2Q,
    "ISWAP": CLIFFORD_2Q,
    "ISWAP_DAG": CLIFFORD_2Q,
    "SQRT_XX": CLIFFORD_2Q,
    "SQRT_XX_DAG": CLIFFORD_2Q,
    "SQRT_YY": CLIFFORD_2Q,
    "SQRT_YY_DAG": CLIFFORD_2Q,
    "SQRT_ZZ": CLIFFORD_2Q,
    "SQRT_ZZ_DAG": CLIFFORD_2Q,
    "SWAP": CLIFFORD_2Q,
    "XCX": CLIFFORD_2Q,
    "XCY": CLIFFORD_2Q,
    "XCZ": CLIFFORD_2Q,
    "YCX": CLIFFORD_2Q,
    "YCY": CLIFFORD_2Q,
    "YCZ": CLIFFORD_2Q,
    "ZCX": CLIFFORD_2Q,
    "ZCY": CLIFFORD_2Q,
    "ZCZ": CLIFFORD_2Q,
    "MPP": MPP,
    "MR": MEASURE_RESET_1Q,
    "MRX": MEASURE_RESET_1Q,
    "MRY": MEASURE_RESET_1Q,
    "MRZ": MEASURE_RESET_1Q,
    "M": JUST_MEASURE_1Q,
    "MX": JUST_MEASURE_1Q,
    "MY": JUST_MEASURE_1Q,
    "MZ": JUST_MEASURE_1Q,
    "R": JUST_RESET_1Q,
    "RX": JUST_RESET_1Q,
    "RY": JUST_RESET_1Q,
    "RZ": JUST_RESET_1Q,
    "DETECTOR": ANNOTATION,
    "OBSERVABLE_INCLUDE": ANNOTATION,
    "QUBIT_COORDS": ANNOTATION,
    "SHIFT_COORDS": ANNOTATION,
    "TICK": ANNOTATION,
    "DEPOLARIZE1": NOISE,
    "DEPOLARIZE2": NOISE,
    "PAULI_CHANNEL_1": NOISE,
    "PAULI_CHANNEL_2": NOISE,
    "X_ERROR": NOISE,
    "Y_ERROR": NOISE,
    "Z_ERROR": NOISE,
    # Not supported.
    # 'CORRELATED_ERROR': NOISE,
    # 'E': NOISE,
    # 'ELSE_CORRELATED_ERROR',
}
OP_MEASURE_BASES = {
    "M": "Z",
    "MX": "X",
    "MY": "Y",
    "MZ": "Z",
    "MPP": "",
}
COLLAPSING_OPS = {
    op
    for op, t in OP_TYPES.items()
    if t == JUST_RESET_1Q or t == JUST_MEASURE_1Q or t == MPP or t == MEASURE_RESET_1Q
}


class NoiseRule:
    """Describes how to add noise to an operation."""

    def __init__(self, *, after: Dict[str, float], flip_result: float = 0):
        """
        Args:
            after: A dictionary mapping noise rule names to their probability argument.
                For example, {"DEPOLARIZE2": 0.01, "X_ERROR": 0.02} will add two qubit
                depolarization with parameter 0.01 and also add 2% bit flip noise. These
                noise channels occur after all other operations in the moment and are applied
                to the same targets as the relevant operation.
            flip_result: The probability that a measurement result should be reported incorrectly.
                Only valid when applied to operations that produce measurement results.
        """
        if not (0 <= flip_result <= 1):
            raise ValueError(f"not (0 <= {flip_result=} <= 1)")
        for k, p in after.items():
            if OP_TYPES[k] != NOISE:
                raise ValueError(f"not a noise channel: {k} from {after=}")
            if not (0 <= p <= 1):
                raise ValueError(f"not (0 <= {p} <= 1) from {after=}")
        self.after = after
        self.flip_result = flip_result

    def append_noisy_version_of(
        self,
        *,
        split_op: stim.CircuitInstruction,
        out_during_moment: stim.Circuit,
        after_moments: DefaultDict[Any, stim.Circuit],
    ) -> None:
        targets = split_op.targets_copy()
        args = split_op.gate_args_copy()
        if self.flip_result:
            t = OP_TYPES[split_op.name]
            assert t == MPP or t == JUST_MEASURE_1Q or t == MEASURE_RESET_1Q
            assert len(args) == 0
            args = [self.flip_result]

        out_during_moment.append(split_op.name, targets, args)
        raw_targets = [t.value for t in targets if not t.is_combiner]
        for op_name, arg in self.after.items():
            after_moments[(op_name, arg)].append(op_name, raw_targets, arg)


class NoiseModel:
    def __init__(
        self,
        idle_depolarization: float,
        additional_depolarization_waiting_for_mr: float = 0,
        gate_rules: Optional[Dict[str, NoiseRule]] = None,
        measure_rules: Optional[Dict[str, NoiseRule]] = None,
        any_clifford_1q_rule: Optional[NoiseRule] = None,
        any_clifford_2q_rule: Optional[NoiseRule] = None,
    ):
        self.idle_depolarization = idle_depolarization
        self.additional_depolarization_waiting_for_mr = (
            additional_depolarization_waiting_for_mr
        )
        self.gate_rules = gate_rules
        self.measure_rules = measure_rules
        self.any_clifford_1q_rule = any_clifford_1q_rule
        self.any_clifford_2q_rule = any_clifford_2q_rule

    @staticmethod
    def depolarizing_cz_noise(p: float) -> "NoiseModel":
        return NoiseModel(
            idle_depolarization=p,
            any_clifford_1q_rule=NoiseRule(after={"DEPOLARIZE1": p}),
            measure_rules={
                "Z": NoiseRule(after={"DEPOLARIZE1": p}, flip_result=p),
            },
            gate_rules={
                "R": NoiseRule(after={"X_ERROR": p}),
                "CZ": NoiseRule(after={"DEPOLARIZE2": p}),
            },
        )

    @staticmethod
    def depolarizing_two_body_measurement_noise(p: float) -> "NoiseModel":
        return NoiseModel(
            idle_depolarization=p,
            any_clifford_1q_rule=NoiseRule(after={"DEPOLARIZE1": p}),
            measure_rules={
                "XX": NoiseRule(after={"DEPOLARIZE2": p}, flip_result=p),
                "YY": NoiseRule(after={"DEPOLARIZE2": p}, flip_result=p),
                "ZZ": NoiseRule(after={"DEPOLARIZE2": p}, flip_result=p),
                "X": NoiseRule(after={"DEPOLARIZE1": p}, flip_result=p),
                "Y": NoiseRule(after={"DEPOLARIZE1": p}, flip_result=p),
                "Z": NoiseRule(after={"DEPOLARIZE1": p}, flip_result=p),
            },
            gate_rules={
                "RX": NoiseRule(after={"Z_ERROR": p}),
                "RY": NoiseRule(after={"X_ERROR": p}),
                "R": NoiseRule(after={"X_ERROR": p}),
            },
        )

    def _noise_rule_for_split_operation(
        self, *, split_op: stim.CircuitInstruction
    ) -> Optional[NoiseRule]:
        if _occurs_in_classical_control_system(split_op=split_op):
            return None

        rule = self.gate_rules.get(split_op.name)
        if rule is not None:
            return rule

        t = OP_TYPES[split_op.name]

        if self.any_clifford_1q_rule is not None and t == CLIFFORD_1Q:
            return self.any_clifford_1q_rule
        if self.any_clifford_2q_rule is not None and t == CLIFFORD_2Q:
            return self.any_clifford_2q_rule
        if self.measure_rules is not None:
            rule = self.measure_rules.get(_measure_basis(split_op=split_op))
            if rule is not None:
                return rule

        raise ValueError(f"No noise (or lack of noise) specified for {split_op=}.")

    def _append_idle_error(
        self,
        *,
        moment_split_ops: List[stim.CircuitInstruction],
        out: stim.Circuit,
        system_qubits: AbstractSet[int],
    ) -> None:
        collapse_qubits = []
        clifford_qubits = []
        for split_op in moment_split_ops:
            if _occurs_in_classical_control_system(split_op=split_op):
                continue
            if split_op.name in COLLAPSING_OPS:
                qubits_out = collapse_qubits
            else:
                qubits_out = clifford_qubits
            for target in split_op.targets_copy():
                if not target.is_combiner:
                    qubits_out.append(target.value)

        # Safety check for operation collisions.
        usage_counts = collections.Counter(collapse_qubits + clifford_qubits)
        qubits_used_multiple_times = {q for q, c in usage_counts.items() if c != 1}
        if qubits_used_multiple_times:
            moment = stim.Circuit()
            for op in moment_split_ops:
                moment.append(op)
            raise ValueError(
                f"Qubits were operated on multiple times without a TICK in between:\n"
                f"multiple uses: {sorted(qubits_used_multiple_times)}\n"
                f"moment:\n"
                f"{moment}"
            )

        collapse_qubits_set = set(collapse_qubits)
        clifford_qubits_set = set(clifford_qubits)
        idle = sorted(system_qubits - collapse_qubits_set - clifford_qubits_set)
        if idle and self.idle_depolarization:
            out.append("DEPOLARIZE1", idle, self.idle_depolarization)

        wait = sorted(system_qubits - collapse_qubits_set)
        if wait and self.additional_depolarization_waiting_for_mr:
            out.append(
                "DEPOLARIZE1", idle, self.additional_depolarization_waiting_for_mr
            )

    def _append_noisy_moment(
        self,
        *,
        moment_split_ops: List[stim.CircuitInstruction],
        out: stim.Circuit,
        system_qubits: AbstractSet[int],
    ) -> None:
        after = collections.defaultdict(stim.Circuit)
        for split_op in moment_split_ops:
            rule = self._noise_rule_for_split_operation(split_op=split_op)
            if rule is None:
                out.append(split_op)
            else:
                rule.append_noisy_version_of(
                    split_op=split_op, out_during_moment=out, after_moments=after
                )
        for k in sorted(after.keys()):
            out += after[k]

        self._append_idle_error(
            moment_split_ops=moment_split_ops, out=out, system_qubits=system_qubits
        )

    def noisy_circuit(
        self,
        circuit: stim.Circuit,
        *,
        system_qubits: Optional[Set[int]] = None,
    ) -> stim.Circuit:
        """Returns a noisy version of the given circuit, by applying the receiving noise model.

        Args:
            circuit: The circuit to layer noise over.
            system_qubits: All qubits used by the circuit. These are the qubits eligible for idling noise.

        Returns:
            The noisy version of the circuit.
        """
        if system_qubits is None:
            system_qubits = set(range(circuit.num_qubits))

        result = stim.Circuit()

        first = True
        for moment_split_ops in _iter_split_op_moments(circuit):
            if first:
                first = False
            elif result and isinstance(result[-1], stim.CircuitRepeatBlock):
                pass
            else:
                result.append("TICK")
            if isinstance(moment_split_ops, stim.CircuitRepeatBlock):
                noisy_body = self.noisy_circuit(
                    moment_split_ops.body_copy(), system_qubits=system_qubits
                )
                noisy_body.append("TICK")
                result.append(
                    stim.CircuitRepeatBlock(
                        repeat_count=moment_split_ops.repeat_count, body=noisy_body
                    )
                )
            else:
                self._append_noisy_moment(
                    moment_split_ops=moment_split_ops,
                    out=result,
                    system_qubits=system_qubits,
                )

        return result


def _occurs_in_classical_control_system(*, split_op: stim.CircuitInstruction) -> bool:
    """Determines if an operation is an annotation or a classical control system update."""
    t = OP_TYPES[split_op.name]
    if t == ANNOTATION:
        return True
    if t == CLIFFORD_2Q:
        targets = split_op.targets_copy()
        for k in range(0, len(targets), 2):
            a = targets[k]
            b = targets[k + 1]
            if not (a.is_measurement_record_target or b.is_measurement_record_target):
                return False
        return True
    return False


def _split_targets_if_needed(
    op: stim.CircuitInstruction,
) -> List[stim.CircuitInstruction]:
    """Splits operations into pieces as needed (e.g. MPP into each product, classical control away from quantum ops)."""
    t = OP_TYPES[op.name]
    if t == CLIFFORD_2Q:
        yield from _split_targets_if_needed_clifford_2q(op)
    elif t == MPP:
        yield from _split_targets_if_needed_m_basis(op)
    else:
        yield op


def _split_targets_if_needed_clifford_2q(
    op: stim.CircuitInstruction,
) -> List[stim.CircuitInstruction]:
    """Splits classical control system operations away from things actually happening on the quantum computer."""
    assert OP_TYPES[op.name] == CLIFFORD_2Q
    targets = op.targets_copy()
    if any(t.is_measurement_record_target for t in targets):
        args = op.gate_args_copy()
        for k in range(0, len(targets), 2):
            yield stim.CircuitInstruction(op.name, targets[k : k + 2], args)
    else:
        yield op


def _split_targets_if_needed_m_basis(
    op: stim.CircuitInstruction,
) -> List[stim.CircuitInstruction]:
    """Splits an MPP operation into one operation for each Pauli product it measures."""
    targets = op.targets_copy()
    args = op.gate_args_copy()
    k = 0
    start = k
    while k < len(targets):
        if k + 1 == len(targets) or not targets[k + 1].is_combiner:
            yield stim.CircuitInstruction(op.name, targets[start : k + 1], args)
            k += 1
            start = k
        else:
            k += 2
    assert k == len(targets)


def _iter_split_op_moments(
    circuit: stim.Circuit,
) -> Iterator[Union[stim.CircuitRepeatBlock, List[stim.CircuitInstruction]]]:
    """Splits a circuit into moments and some operations into pieces.

    Classical control system operations like CX rec[-1] 0 are split from quantum operations like CX 1 0.

    MPP operations are split into one operation per Pauli product.

    Yields:
        Lists of operations corresponding to one moment in the circuit, with any problematic operations
        like MPPs split into pieces.

        (A moment is the time between two TICKs.)
    """
    cur_moment = []

    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            yield op
        elif isinstance(op, stim.CircuitInstruction):
            if op.name == "TICK":
                yield cur_moment
                cur_moment = []
            else:
                cur_moment.extend(_split_targets_if_needed(op))
    if cur_moment:
        yield cur_moment


def _measure_basis(*, split_op: stim.CircuitInstruction) -> Optional[str]:
    """Converts an operation into a string describing the Pauli product basis it measures.

    Returns:
        None: This is not a measurement (or not *just* a measurement).
        str: Pauli product string that the operation measures (e.g. "XX" or "Y").
    """
    result = OP_MEASURE_BASES.get(split_op.name)
    targets = split_op.targets_copy()
    if result == "":
        for k in range(0, len(targets), 2):
            t = targets[k]
            if t.is_x_target:
                result += "X"
            elif t.is_y_target:
                result += "Y"
            elif t.is_z_target:
                result += "Z"
            else:
                raise NotImplementedError(f"{targets=}")
    return result
