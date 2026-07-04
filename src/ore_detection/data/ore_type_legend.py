"""Legend-driven class mappings for source ore-type segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Tuple

RgbColor = Tuple[int, int, int]

DEFAULT_LEGEND_PATH = Path(__file__).resolve().parents[3] / "configs" / "source_ore_type_legend.json"


def _as_rgb(value: Any) -> RgbColor:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"RGB color must have exactly three integer channels, got {value!r}")
    rgb = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"RGB channels must be in [0, 255], got {rgb!r}")
    return rgb  # type: ignore[return-value]


def format_rgb(rgb: RgbColor) -> str:
    """Return a stable string form for diagnostics and JSON metadata."""
    return f"{rgb[0]},{rgb[1]},{rgb[2]}"


@dataclass(frozen=True)
class LegendEntry:
    """One color entry from a human-readable dataset legend."""

    dataset: str
    rgb: RgbColor
    short: str
    legend_name: str
    target: str
    class_index: int
    note: str | None = None

    @classmethod
    def from_config(cls, *, dataset: str, raw: dict[str, Any], class_to_index: dict[str, int]) -> "LegendEntry":
        target = str(raw["target"])
        if target not in class_to_index:
            raise ValueError(f"{dataset} color {raw.get('rgb')!r} targets unknown class {target!r}")
        return cls(
            dataset=dataset,
            rgb=_as_rgb(raw["rgb"]),
            short=str(raw["short"]),
            legend_name=str(raw["legend_name"]),
            target=target,
            class_index=class_to_index[target],
            note=str(raw["note"]) if raw.get("note") else None,
        )

    def as_metadata(self) -> dict[str, Any]:
        """Serialize this entry for saved mask metadata."""
        data: dict[str, Any] = {
            "dataset": self.dataset,
            "rgb": list(self.rgb),
            "short": self.short,
            "legend_name": self.legend_name,
            "target": self.target,
            "class_index": self.class_index,
        }
        if self.note:
            data["note"] = self.note
        return data


@dataclass(frozen=True)
class OreTypeLegend:
    """Validated multiclass mapping shared by all supervised source datasets."""

    class_order: tuple[str, ...]
    background_class: str
    dataset_entries: dict[str, tuple[LegendEntry, ...]]
    dummy_outputs: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OreTypeLegend":
        class_order = tuple(str(name) for name in data["class_order"])
        background_class = str(data["background_class"])
        if background_class not in class_order:
            raise ValueError(f"background class {background_class!r} is missing from class_order")
        if len(set(class_order)) != len(class_order):
            raise ValueError("class_order contains duplicate class names")

        class_to_index = {name: index for index, name in enumerate(class_order)}
        dataset_entries: dict[str, tuple[LegendEntry, ...]] = {}
        for dataset, raw_dataset in data["datasets"].items():
            entries = tuple(
                LegendEntry.from_config(dataset=dataset, raw=raw_entry, class_to_index=class_to_index)
                for raw_entry in raw_dataset["colors"]
            )
            _validate_dataset_entries(dataset, entries)
            dataset_entries[dataset] = entries

        legend = cls(
            class_order=class_order,
            background_class=background_class,
            dataset_entries=dataset_entries,
            dummy_outputs=tuple(dict(item) for item in data.get("dummy_outputs", ())),
        )
        legend.validate()
        return legend

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.class_order

    @property
    def class_count(self) -> int:
        return len(self.class_order)

    @property
    def background_index(self) -> int:
        return self.class_order.index(self.background_class)

    @property
    def non_background_indices(self) -> tuple[int, ...]:
        return tuple(index for index, name in enumerate(self.class_order) if name != self.background_class)

    def class_index(self, target: str) -> int:
        return self.class_order.index(target)

    def entries_for_dataset(self, dataset: str) -> tuple[LegendEntry, ...]:
        try:
            return self.dataset_entries[dataset]
        except KeyError as exc:
            raise KeyError(f"unknown source dataset {dataset!r}") from exc

    def color_to_entry(self, dataset: str) -> dict[RgbColor, LegendEntry]:
        return {entry.rgb: entry for entry in self.entries_for_dataset(dataset)}

    def color_to_index(self, dataset: str) -> dict[RgbColor, int]:
        return {entry.rgb: entry.class_index for entry in self.entries_for_dataset(dataset)}

    def metadata(self) -> dict[str, Any]:
        return {
            "background_class": self.background_class,
            "background_index": self.background_index,
            "class_order": list(self.class_order),
            "non_background_indices": list(self.non_background_indices),
            "dummy_outputs": list(self.dummy_outputs),
            "datasets": {
                dataset: [entry.as_metadata() for entry in entries]
                for dataset, entries in sorted(self.dataset_entries.items())
            },
        }

    def validate(self) -> None:
        for dataset, entries in self.dataset_entries.items():
            _validate_dataset_entries(dataset, entries)
            if not any(entry.target == self.background_class for entry in entries):
                raise ValueError(f"{dataset} has no entry for background class {self.background_class!r}")


def _validate_dataset_entries(dataset: str, entries: tuple[LegendEntry, ...]) -> None:
    seen_colors: set[RgbColor] = set()
    for entry in entries:
        if entry.rgb in seen_colors:
            raise ValueError(f"{dataset} maps RGB {entry.rgb!r} more than once")
        seen_colors.add(entry.rgb)


def load_legend_config(path: str | Path = DEFAULT_LEGEND_PATH) -> OreTypeLegend:
    """Load the audited source ore-type legend mapping."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return OreTypeLegend.from_dict(data)


def write_legend_metadata(legend: OreTypeLegend, target_path: str | Path) -> None:
    """Write the resolved class map next to generated tensors."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(legend.metadata(), indent=2, sort_keys=True), encoding="utf-8")
