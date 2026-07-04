"""Dataset-specific mask label mappings for ore segmentation.

The source datasets store masks as small integer IDs, often in RGB images where
all channels contain the same ID value. This module keeps the mapping explicit so
we can train coarse ore/background models while preserving a species-aware view
for attachment/contact descriptors.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal

TargetTaxonomy = Literal["coarse", "species"]

COARSE_LABELS: dict[str, int] = {
    "background_matrix": 0,
    "sulfide_ore": 1,
    "oxide_magnetite_hematite": 2,
    "ignore": 255,
}

SPECIES_LABELS: dict[str, int] = {
    "background_matrix": 0,
    "pyrite_like": 1,
    "chalcopyrite_like": 2,
    "bornite_like": 3,
    "sphalerite_like": 4,
    "galena_like": 5,
    "tennantite_tetrahedrite_like": 6,
    "pyrrhotite_like": 7,
    "pentlandite_like": 8,
    "arsenopyrite_like": 9,
    "covelline_like": 10,
    "oxide_magnetite_hematite": 11,
    "ignore": 255,
}

# NOTE: These ID mappings are derived from the project dataset masks and the
# user-provided class lists. They should be verified against original dataset
# legends before final model training. Unknown values intentionally map to
# ignore rather than being treated as background.
SOURCE_CLASS_NAMES: dict[str, dict[int, str]] = {
    "set_1": {
        0: "background_matrix",
        1: "sphalerite_like",
        2: "pyrite_like",
        6: "galena_like",
        8: "bornite_like",
        11: "tennantite_tetrahedrite_like",
        # The set_1 class list includes chalcopyrite minerals, but that class
        # was not present in the sampled masks. Add the concrete ID here once
        # confirmed from the dataset legend.
    },
    "set_2": {
        0: "background_matrix",
        1: "pyrrhotite_like",
        3: "chalcopyrite_like",
        5: "oxide_magnetite_hematite",
        7: "pentlandite_like",
    },
    "set_3": {
        0: "background_matrix",
        1: "pyrite_like",
        2: "chalcopyrite_like",
        3: "arsenopyrite_like",
        5: "covelline_like",
        6: "bornite_like",
        8: "chalcopyrite_like",
        9: "oxide_magnetite_hematite",
        10: "oxide_magnetite_hematite",
        13: "oxide_magnetite_hematite",
        14: "oxide_magnetite_hematite",
    },
}

SULFIDE_SPECIES = {
    "pyrite_like",
    "chalcopyrite_like",
    "bornite_like",
    "sphalerite_like",
    "galena_like",
    "tennantite_tetrahedrite_like",
    "pyrrhotite_like",
    "pentlandite_like",
    "arsenopyrite_like",
    "covelline_like",
}


def _scalar_label(value: Any) -> int:
    """Extract an integer label from grayscale or RGB-style mask values."""
    if isinstance(value, tuple):
        if not value:
            raise ValueError("empty pixel tuple cannot be converted to a label")
        if len(set(value[:3])) == 1:
            return int(value[0])
        raise ValueError(f"RGB color {value!r} is not an RGB-encoded scalar label")
    return int(value)


def map_value(
    value: Any,
    *,
    source_dataset: str,
    target_taxonomy: TargetTaxonomy,
    unknown_value: int | None = None,
) -> int:
    """Map one source mask value into the requested target taxonomy.

    Unknown source values are mapped to the taxonomy's ignore ID by default.
    Pass ``unknown_value`` to override that behavior.
    """
    if source_dataset not in SOURCE_CLASS_NAMES:
        raise KeyError(f"unknown source dataset: {source_dataset}")
    if target_taxonomy not in {"coarse", "species"}:
        raise KeyError(f"unknown target taxonomy: {target_taxonomy}")

    source_value = _scalar_label(value)
    source_name = SOURCE_CLASS_NAMES[source_dataset].get(source_value)
    if source_name is None:
        if unknown_value is not None:
            return unknown_value
        labels = COARSE_LABELS if target_taxonomy == "coarse" else SPECIES_LABELS
        return labels["ignore"]

    if target_taxonomy == "species":
        return SPECIES_LABELS.get(source_name, SPECIES_LABELS["ignore"])

    if source_name == "background_matrix":
        return COARSE_LABELS["background_matrix"]
    if source_name == "oxide_magnetite_hematite":
        return COARSE_LABELS["oxide_magnetite_hematite"]
    if source_name in SULFIDE_SPECIES:
        return COARSE_LABELS["sulfide_ore"]
    return COARSE_LABELS["ignore"]


def map_mask_values(
    mask: Sequence[Sequence[Any]],
    *,
    source_dataset: str,
    target_taxonomy: TargetTaxonomy,
    unknown_value: int | None = None,
) -> list[list[int]]:
    """Map a 2D in-memory mask represented as nested rows."""
    return [
        [
            map_value(
                pixel,
                source_dataset=source_dataset,
                target_taxonomy=target_taxonomy,
                unknown_value=unknown_value,
            )
            for pixel in row
        ]
        for row in mask
    ]


def class_names_for_taxonomy(target_taxonomy: TargetTaxonomy) -> dict[int, str]:
    """Return reverse class names for display/reporting."""
    labels = COARSE_LABELS if target_taxonomy == "coarse" else SPECIES_LABELS
    return {value: name for name, value in labels.items()}
