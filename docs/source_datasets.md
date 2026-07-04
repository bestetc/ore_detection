# Source Dataset Notes

This project currently uses three supervised source datasets for mineral/ore segmentation and one weak-label baseline dataset.

## Baseline

- Path: `datasets/baseline`
- Magnification: `10x`
- Labels: folder-level weak labels (`Normal ore`, `Hard ore`, `Talc ore`, etc.)
- Pixel masks: not available for hard/normal/talc.
- Use: weak supervision for descriptor-based classification and target-domain visual validation.

## set_1

- Path: `datasets/set_1`
- Magnification: `50x`
- Masks: pixel-level mineral masks.
- Classes from project brief: sphalerite, pyrite, galena, bornite, tennantite-tetrahedrite group, chalcopyrite minerals, background.
- Use: supervised source segmentation.

## set_2

- Path: `datasets/set_2`
- Magnification: `50x`
- Masks: pixel-level mineral masks.
- Classes from project brief: pyrrhotite, chalcopyrite, pentlandite, magnetite, background.
- Use: supervised source segmentation, with magnetite mapped to the oxide group.

## set_3

- Path: `datasets/set_3`
- Magnification: `50x`
- Masks: pixel-level mineral masks.
- Classes from project brief: pyrite, arsenopyrite, covelline, bornite, chalcopyrite, ordinary magnetite, copper-bearing magnetite, hematite, background.
- Use: supervised source segmentation and species/contact descriptor R&D.

## Important caveat

The current numeric ID mapping in `src/ore_detection/data/label_mapping.py` is derived from observed mask IDs plus the project class lists. It must be checked against the original dataset legends before final training.
