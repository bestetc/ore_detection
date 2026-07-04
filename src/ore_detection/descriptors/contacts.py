"""Contact and attachment descriptors for species/coarse masks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence, Set, Tuple

Mask = Sequence[Sequence[Any]]


def _canonical_pair(a: Any, b: Any) -> Tuple[Any, Any]:
    return (a, b) if a <= b else (b, a)


def contact_lengths(mask: Mask, *, classes: Set[Any]) -> dict[Tuple[Any, Any], int]:
    """Count 4-neighbor contact edges between different selected classes.

    Each undirected edge is counted once by checking only right and down
    neighbors. Background is ignored unless included in ``classes``.
    """
    height = len(mask)
    width = len(mask[0]) if height else 0
    counts: Counter[tuple[Any, Any]] = Counter()

    for row in range(height):
        if len(mask[row]) != width:
            raise ValueError("mask rows must all have the same width")
        for col in range(width):
            current = mask[row][col]
            if current not in classes:
                continue
            for nr, nc in ((row, col + 1), (row + 1, col)):
                if nr >= height or nc >= width:
                    continue
                neighbor = mask[nr][nc]
                if neighbor in classes and neighbor != current:
                    counts[_canonical_pair(current, neighbor)] += 1
    return dict(counts)


def hetero_sulfide_contact_length(mask: Mask, *, sulfide_classes: Set[Any]) -> int:
    """Total contact length between different sulfide/mineral classes."""
    return sum(contact_lengths(mask, classes=sulfide_classes).values())
