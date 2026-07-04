"""Candidate talc mask generation.

This is not a trained talc model. It creates conservative dark-region candidate
masks inside non-ore matrix so a human can correct/save masks for active
learning.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence, Tuple, Union

GrayMask = Sequence[Sequence[Union[int, float]]]
BinaryMask = Sequence[Sequence[int]]
Coord = Tuple[int, int]

_NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _validate_rectangular(mask: Sequence[Sequence[object]], name: str) -> tuple[int, int]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    for row in mask:
        if len(row) != width:
            raise ValueError(f"{name} rows must all have the same width")
    return height, width


def _components(binary: list[list[int]]) -> list[set[Coord]]:
    height, width = _validate_rectangular(binary, "binary")
    seen: set[Coord] = set()
    components: list[set[Coord]] = []
    for row in range(height):
        for col in range(width):
            coord = (row, col)
            if binary[row][col] == 0 or coord in seen:
                continue
            queue: deque[Coord] = deque([coord])
            seen.add(coord)
            component: set[Coord] = set()
            while queue:
                cr, cc = queue.popleft()
                component.add((cr, cc))
                for dr, dc in _NEIGHBORS_4:
                    nr, nc = cr + dr, cc + dc
                    neighbor = (nr, nc)
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and binary[nr][nc]
                        and neighbor not in seen
                    ):
                        seen.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    return components


def detect_dark_matrix_candidates(
    grayscale: GrayMask,
    *,
    ore_mask: BinaryMask | None = None,
    dark_offset: float = 25.0,
    min_component_area: int = 12,
) -> list[list[int]]:
    """Detect dark scattered matrix regions as talc candidates.

    Pixels are candidates when they are at least ``dark_offset`` darker than the
    mean non-ore matrix intensity. Connected components smaller than
    ``min_component_area`` are removed.
    """
    height, width = _validate_rectangular(grayscale, "grayscale")
    if ore_mask is not None:
        oh, ow = _validate_rectangular(ore_mask, "ore_mask")
        if (oh, ow) != (height, width):
            raise ValueError("ore_mask must have the same shape as grayscale")

    matrix_values: list[float] = []
    for row in range(height):
        for col in range(width):
            if ore_mask is not None and ore_mask[row][col]:
                continue
            matrix_values.append(float(grayscale[row][col]))
    if not matrix_values:
        return [[0 for _ in range(width)] for _ in range(height)]

    threshold = (sum(matrix_values) / len(matrix_values)) - dark_offset
    raw = [[0 for _ in range(width)] for _ in range(height)]
    for row in range(height):
        for col in range(width):
            if ore_mask is not None and ore_mask[row][col]:
                continue
            if float(grayscale[row][col]) <= threshold:
                raw[row][col] = 1

    cleaned = [[0 for _ in range(width)] for _ in range(height)]
    for component in _components(raw):
        if len(component) >= min_component_area:
            for row, col in component:
                cleaned[row][col] = 1
    return cleaned
