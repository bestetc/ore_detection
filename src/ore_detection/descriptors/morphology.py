"""Component-level morphology descriptors for ore masks.

These functions intentionally avoid heavy numeric dependencies so they can run in
lightweight audit/active-learning scripts. They operate on small in-memory masks
represented as nested row sequences.
"""

from __future__ import annotations

from collections import deque
from math import pi
from typing import Any, Sequence, Set, Tuple

Coord = Tuple[int, int]
Mask = Sequence[Sequence[Any]]

_NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _height_width(mask: Mask) -> Tuple[int, int]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    return height, width


def _is_foreground(value: Any, foreground_values: Set[Any]) -> bool:
    return value in foreground_values


def _component_perimeter(component: set[Coord], height: int, width: int) -> int:
    perimeter = 0
    for row, col in component:
        for dr, dc in _NEIGHBORS_4:
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width or (nr, nc) not in component:
                perimeter += 1
    return perimeter


def connected_components(mask: Mask, *, foreground_values: Set[Any]) -> list[set[Coord]]:
    """Return 4-connected components for selected foreground values."""
    height, width = _height_width(mask)
    seen: set[Coord] = set()
    components: list[set[Coord]] = []

    for row in range(height):
        if len(mask[row]) != width:
            raise ValueError("mask rows must all have the same width")
        for col in range(width):
            coord = (row, col)
            if coord in seen or not _is_foreground(mask[row][col], foreground_values):
                continue
            queue: deque[Coord] = deque([coord])
            seen.add(coord)
            component: set[Coord] = set()
            while queue:
                current = queue.popleft()
                component.add(current)
                cr, cc = current
                for dr, dc in _NEIGHBORS_4:
                    nr, nc = cr + dr, cc + dc
                    neighbor = (nr, nc)
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and neighbor not in seen
                        and _is_foreground(mask[nr][nc], foreground_values)
                    ):
                        seen.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    return components


def component_stats(mask: Mask, *, foreground_values: Set[Any]) -> list[dict[str, float]]:
    """Compute morphology descriptors for each foreground component.

    ``bbox_fill`` is used as a dependency-light solidity proxy. True convex-hull
    solidity can be added later once scipy/scikit-image are available.
    """
    height, width = _height_width(mask)
    stats: list[dict[str, float]] = []
    for component in connected_components(mask, foreground_values=foreground_values):
        rows = [coord[0] for coord in component]
        cols = [coord[1] for coord in component]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        area = len(component)
        perimeter = _component_perimeter(component, height, width)
        bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
        perimeter2_over_area = (perimeter * perimeter) / area if area else 0.0
        circularity = (4 * pi * area / (perimeter * perimeter)) if perimeter else 0.0
        stats.append(
            {
                "area": float(area),
                "perimeter": float(perimeter),
                "perimeter2_over_area": float(perimeter2_over_area),
                "bbox_fill": float(area / bbox_area) if bbox_area else 0.0,
                "circularity": float(circularity),
                "min_row": float(min_row),
                "max_row": float(max_row),
                "min_col": float(min_col),
                "max_col": float(max_col),
            }
        )
    return stats


def summarize_components(
    mask: Mask,
    *,
    foreground_values: Set[Any],
    small_area_threshold: int = 25,
) -> dict[str, float]:
    """Aggregate component descriptors into image/tile-level features."""
    stats = component_stats(mask, foreground_values=foreground_values)
    if not stats:
        return {
            "component_count": 0.0,
            "foreground_area": 0.0,
            "small_component_area_fraction": 0.0,
            "perimeter2_over_area_weighted_mean": 0.0,
            "bbox_fill_area_weighted_mean": 0.0,
            "circularity_area_weighted_mean": 0.0,
        }

    foreground_area = sum(item["area"] for item in stats)
    small_area = sum(item["area"] for item in stats if item["area"] <= small_area_threshold)

    def weighted_mean(key: str) -> float:
        return sum(item[key] * item["area"] for item in stats) / foreground_area

    return {
        "component_count": float(len(stats)),
        "foreground_area": float(foreground_area),
        "small_component_area_fraction": float(small_area / foreground_area),
        "perimeter2_over_area_weighted_mean": float(weighted_mean("perimeter2_over_area")),
        "bbox_fill_area_weighted_mean": float(weighted_mean("bbox_fill")),
        "circularity_area_weighted_mean": float(weighted_mean("circularity")),
    }
