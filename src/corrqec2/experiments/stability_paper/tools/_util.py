from typing import List, Callable, Iterable, TypeVar, Any, Tuple, Dict

import numpy as np
import stim

TItem = TypeVar('TItem')


def complex_key(c: complex) -> Any:
    return c.real != int(c.real), c.real, c.imag


def sorted_complex(
        values: Iterable[TItem],
        *,
        key: Callable[[TItem], Any] = lambda e: e) -> List[TItem]:
    return sorted(values, key=lambda e: complex_key(key(e)))


def not_nones(vs) -> List[Any]:
    return [v for v in vs if v is not None]


def circuit_has_unsigned_stabilizers(
        circuit: stim.Circuit,
        stabilizers: Iterable[Tuple[Dict[str, Iterable[Any]], Dict[str, Iterable[Any]], Iterable[stim.GateTarget]]],
        *,
        q2i: Dict[Any, int] = None) -> bool:
    """Verifies that a circuit operates on stabilizers in the desired way.

    Args:
        circuit: The circuit to test.
        stabilizers: (inputs, outputs, measurements) triplets encoding a rule to check.
            inputs: A dictionary from Pauli to list of qubits with that Pauli for the stabilizer before the circuit.
            outputs: A dictionary from Pauli to list of qubits with that Pauli for the stabilizer after the circuit.
            measurements: A list of stim.target_rec targets identifying measurements multiplied into the travelling stabilizer to get it through the circuit.
        q2i: Optional dictionary for mapping qubit objects to qubit indices in the circuit. If not specified, directly use qubit indices.

    Returns:
        True: The circuit has all the requested stabilizers.
        False: The circuit is bad and should feel bad.
    """
    if q2i is None:
        q2i = {}
    for before, after, measurements in stabilizers:
        assert all(m.is_measurement_record_target for m in measurements)
        nq = circuit.num_qubits
        nm = circuit.num_measurements
        for init_basis in 'YXZ':
            offsets = [m.value for m in measurements]
            case = stim.Circuit()
            case.append(f"R{init_basis}", range(nq))

            if before:
                assert set("XYZ").issuperset(before.keys())
                targets = []
                for t in before.get("X", []):
                    targets.append(stim.target_x(q2i.get(t, t)))
                    targets.append(stim.target_combiner())
                for t in before.get("Y", []):
                    targets.append(stim.target_y(q2i.get(t, t)))
                    targets.append(stim.target_combiner())
                for t in before.get("Z", []):
                    targets.append(stim.target_z(q2i.get(t, t)))
                    targets.append(stim.target_combiner())
                if targets:
                    targets.pop()
                    case.append("MPP", targets)
                    offsets.append(-nm - 1)

            case += circuit

            if after:
                assert set("XYZ").issuperset(after.keys())
                targets = []
                for t in after.get("X", []):
                    targets.append(stim.target_x(q2i.get(t, t)))
                    targets.append(stim.target_combiner())
                for t in after.get("Y", []):
                    targets.append(stim.target_y(q2i.get(t, t)))
                    targets.append(stim.target_combiner())
                for t in after.get("Z", []):
                    targets.append(stim.target_z(q2i.get(t, t)))
                    targets.append(stim.target_combiner())
                if targets:
                    targets.pop()
                    case.append("MPP", targets)
                    offsets.append(0)
                    offsets = [e - 1 for e in offsets]

            case.append("DETECTOR", [stim.target_rec(k) for k in offsets])

            try:
                # Verify detector is deterministic.
                case.detector_error_model()
            except ValueError:
                return False
    return True


def score_binomial_line(*,
                        min_x_lim: float,
                        max_x_lim: float,
                        min_p_lim: float,
                        max_p_lim: float,
                        xs: List[float],
                        shots: List[int],
                        errors: List[int],
                        max_likelihood_factor: float,
                        y_distortion: Callable[[float], float] = lambda e: e) -> Any:
    from sinter.probability_util import log_binomial

    top_left_points = []
    bottom_right_points = []
    min_y_lim = np.log(min_p_lim)
    max_y_lim = np.log(max_p_lim)
    n = 2000
    for x in np.linspace(min_x_lim, max_x_lim, n):
        top_left_points.append(x + 1j*max_y_lim)
        bottom_right_points.append(x + 1j*min_y_lim)
    for y in np.linspace(min_y_lim, max_y_lim, n):
        top_left_points.append(min_x_lim + 1j*y)
        bottom_right_points.append(max_x_lim + 1j*y)

    v1 = np.zeros(shape=(2*n, 2*n), dtype=np.complex64)
    v2 = np.zeros(shape=(2*n, 2*n), dtype=np.complex64)
    v1 += top_left_points
    v2 += bottom_right_points
    v2 = v2.transpose()
    v1 = v1.flatten()
    v2 = v2.flatten()
    positive_slope = (np.real(v2) > np.real(v1)) & (np.imag(v2) < np.imag(v1))
    v1 = np.copy(v1[positive_slope])
    v2 = np.copy(v2[positive_slope])
    dv = v2 - v1
    di = np.imag(dv)
    dr = np.real(dv)
    possible_slopes = di / dr
    possible_offsets = v1.imag - possible_slopes * v1.real
    reasonable_offsets = possible_offsets < 100
    possible_slopes = possible_slopes[reasonable_offsets].astype(np.float64)
    possible_offsets = possible_offsets[reasonable_offsets].astype(np.float64)

    scores = np.zeros(possible_offsets.shape, dtype=np.float64)
    for k in range(len(xs)):
        p = np.exp(possible_offsets + possible_slopes * xs[k])
        scores += log_binomial(p=p, n=shots[k], hits=errors[k])
    best_k = np.argmax(scores)
    best_offset = possible_offsets[best_k]
    best_slope = possible_slopes[best_k]
    max_likelihood = scores[best_k]
    kept = scores >= max_likelihood - np.log(max_likelihood_factor)
    possible_offsets = np.copy(possible_offsets[kept])
    possible_slopes = np.copy(possible_slopes[kept])
    xs2, ys = outline(
        min_x=min_x_lim,
        max_x=max_x_lim,
        min_y=np.log(min_p_lim),
        max_y=np.log(max_p_lim),
        best_offset=best_offset,
        best_slope=best_slope,
        offsets=possible_offsets,
        slopes=possible_slopes,
    )
    ys = [y_distortion(e) for e in np.exp(ys)]

    x1 = np.min(xs)
    x2 = max_x_lim
    while not (-25 < best_offset + best_slope * x2 < 25) and x2 > np.max(xs):
        x2 -= 1
    y1 = np.log(y_distortion(np.exp(best_offset + best_slope * x1)))
    y2 = np.log(y_distortion(np.exp(best_offset + best_slope * x2)))
    dy = (y2 - y1) / (x2 - x1)
    y1 += (min_x_lim - x1) * dy
    x1 = min_x_lim
    while x1 > min_x_lim and min_p_lim < np.exp(y1) < max_p_lim:
        y1 -= dy
        x1 -= 1
    while x2 < max_x_lim and min_p_lim < np.exp(y2) < max_p_lim:
        y2 += dy
        x2 += 1
    return [x1, x2], [np.exp(y1), np.exp(y2)], xs2, ys


def outline(*,
            min_y: float,
            max_y: float,
            min_x: float,
            max_x: float,
            best_offset: float,
            best_slope: float,
            offsets: np.ndarray,
            slopes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    in1s = offsets * 1j
    in2s = 1 + (offsets + slopes) * 1j
    ref = 1 + best_slope * 1j
    ref /= abs(ref)
    in1s -= best_offset * 1j
    in2s -= best_offset * 1j
    corners = np.array([
        min_x + min_y * 1j,
        max_x + min_y * 1j,
        min_x + max_y * 1j,
        max_x + max_y * 1j,
    ])
    corners -= best_offset * 1j
    in1s *= ref.conjugate()
    in2s *= ref.conjugate()
    corners *= ref.conjugate()
    rot_slopes = (in2s.imag - in1s.imag) / (in2s.real - in1s.real)
    rot_offsets = in1s.imag - rot_slopes * in1s.real

    outs1 = []
    outs2 = []
    for x in np.linspace(np.min(np.real(corners)), np.max(np.real(corners)), 128):
        vs = rot_offsets + rot_slopes * x
        outs1.append(x + 1j*np.min(vs))
        outs2.append(x + 1j*np.max(vs))
    outs = np.array(outs1 + outs2[::-1], dtype=np.complex128)
    outs *= ref
    outs += best_offset * 1j
    return np.real(outs), np.imag(outs)
