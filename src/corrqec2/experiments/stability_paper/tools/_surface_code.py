import collections
import dataclasses
import functools
from typing import FrozenSet, Optional, List, Tuple


def checkerboard_basis(c: complex) -> str:
    if (c.real + c.imag) % 2 == 0:
        return 'X'
    return 'Z'


@dataclasses.dataclass(frozen=True)
class Tile:
    """A stabilizer from a surface code.
    """

    ordered_data: Tuple[Optional[complex], ...]
    measure_qubit: complex
    basis: str

    @functools.cached_property
    def used_set(self) -> FrozenSet[complex]:
        """All qubits used by the tile."""
        return self.data_set | frozenset([self.measure_qubit])

    @functools.cached_property
    def data_set(self) -> FrozenSet[complex]:
        """All data qubits used by the tile."""
        return frozenset(t for t in self.ordered_data if t is not None)


def surface_code_tiles(*,
                       diam: int,
                       side_basis: str,
                       top_bot_basis: str,
                       x_order: List[complex],
                       z_order: List[complex]) -> List[Tile]:
    data_qubits = {
        x + 1j * y
        for x in range(diam)
        for y in range(diam)
    }

    tiles = []
    for x in range(-1, diam):
        for y in range(-1, diam):
            tl = x + 1j*y
            basis = checkerboard_basis(tl)

            # Omit tiles on the boundary that don't match the boundary type.
            if x in [-1, diam - 1] and basis != side_basis:
                continue
            if y in [-1, diam - 1] and basis != top_bot_basis:
                continue

            # Pick the orientation that avoids bad hook errors.
            d_order = z_order if basis == 'Z' else x_order
            order = [tl + d for d in d_order]
            kept = tuple((d if d in data_qubits else None) for d in order)
            if sum(d is not None for d in kept) < 2:
                continue
            tiles.append(Tile(
                ordered_data=kept,
                measure_qubit=tl + 0.5 + 0.5j,
                basis=basis,
            ))

    # Clip exposed corner squares into triangles.
    usage_count = collections.Counter(d for tile in tiles for d in tile.data_set)
    clipped_data_quits = {d for d, c in usage_count.items() if c < 2}
    tiles = [
        Tile(
            ordered_data=tuple(None if d in clipped_data_quits else d for d in tile.ordered_data),
            measure_qubit=tile.measure_qubit,
            basis=tile.basis,
        )
        for tile in tiles
    ]
    tiles = [
        tile for tile in tiles if tile.data_set
    ]
    if not tiles:
        raise NotImplementedError("Spec resulted in no tiles.")

    return tiles
