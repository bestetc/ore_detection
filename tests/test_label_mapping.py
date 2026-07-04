import unittest

from ore_detection.data.label_mapping import (
    COARSE_LABELS,
    SPECIES_LABELS,
    map_mask_values,
    map_value,
)


class TestLabelMapping(unittest.TestCase):
    def test_set1_numeric_ids_map_to_coarse_background_and_sulfide(self):
        mask = [
            [0, 1, 2],
            [6, 8, 11],
        ]

        mapped = map_mask_values(mask, source_dataset="set_1", target_taxonomy="coarse")

        self.assertEqual(
            mapped,
            [
                [COARSE_LABELS["background_matrix"], COARSE_LABELS["sulfide_ore"], COARSE_LABELS["sulfide_ore"]],
                [COARSE_LABELS["sulfide_ore"], COARSE_LABELS["sulfide_ore"], COARSE_LABELS["sulfide_ore"]],
            ],
        )

    def test_set2_magnetite_maps_to_oxide_in_coarse_taxonomy(self):
        self.assertEqual(
            map_value(5, source_dataset="set_2", target_taxonomy="coarse"),
            COARSE_LABELS["oxide_magnetite_hematite"],
        )

    def test_unknown_value_maps_to_ignore_by_default(self):
        self.assertEqual(
            map_value(250, source_dataset="set_1", target_taxonomy="coarse"),
            COARSE_LABELS["ignore"],
        )

    def test_species_taxonomy_preserves_contact_relevant_groups(self):
        self.assertEqual(
            map_value(2, source_dataset="set_3", target_taxonomy="species"),
            SPECIES_LABELS["chalcopyrite_like"],
        )
        self.assertEqual(
            map_value(9, source_dataset="set_3", target_taxonomy="species"),
            SPECIES_LABELS["oxide_magnetite_hematite"],
        )


if __name__ == "__main__":
    unittest.main()
