# Label Mapping

The project uses two target taxonomies.

## Coarse taxonomy

| ID | Class |
|---:|---|
| 0 | `background_matrix` |
| 1 | `sulfide_ore` |
| 2 | `oxide_magnetite_hematite` |
| 255 | `ignore` |

This taxonomy is the first target for robust ore/background segmentation and morphology descriptors.

## Species-aware taxonomy

Used for R&D contact/attachment metrics, especially cases like small chalcopyrite attachments inside or against larger pyrite blocks.

| Class |
|---|
| `background_matrix` |
| `pyrite_like` |
| `chalcopyrite_like` |
| `bornite_like` |
| `sphalerite_like` |
| `galena_like` |
| `tennantite_tetrahedrite_like` |
| `pyrrhotite_like` |
| `pentlandite_like` |
| `arsenopyrite_like` |
| `covelline_like` |
| `oxide_magnetite_hematite` |
| `ignore` |

## Training rule

Unknown source mask values map to `ignore`, not background. This avoids silently treating unlabeled or unexpected classes as non-ore.

## UI annotation taxonomy

The backend/UI annotation layer uses a **single-class-per-pixel** mask. A pixel cannot be both ore and talc; brush operations must overwrite the previous class at that pixel.

Recommended UI class IDs for the HSV/binary editor path:

| ID | Class | Display color | Meaning |
|---:|---|---|---|
| 0 | `background` | black `#000000` | non-ore matrix/background |
| 1 | `sulfide_ore` | green `#00dc00` | sulfide/ore regions; binary ore mask writes this class |
| 2 | `oxide_magnetite_hematite` | red `#ff4040` | optional oxide/magnetite/hematite class for later review |
| 3 | `talc` | white `#ffffff` | user-reviewed talc/dark-region class |
| 4 | `normal_ore` | blue `#0078ff` | reviewed normal/coarse intergrowth region |
| 5 | `hard_ore` | yellow `#ffdc00` | reviewed hard/thin/ragged intergrowth region |
| 255 | `ignore` | magenta `#ff00ff` | unknown/unreviewed/invalid pixels |

For trained multiclass ore-model predictions, the UI preserves model class IDs from the checkpoint and appends UI-only classes (`talc`, `normal_ore`, `hard_ore`, `ignore`) after the model channels.

Initial UI masks may be created from dummy segmentation:

- ore proposal: HSV Value threshold or source/binary mask proposal → class `sulfide_ore`
- talc proposal: manual/reviewed UI annotation → class `talc`

Manual brush editing must support:

- add selected class
- erase selected class back to `background_matrix`
- overwrite any existing class with the selected class
