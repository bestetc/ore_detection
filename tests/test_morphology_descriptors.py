import unittest

from ore_detection.descriptors.morphology import component_stats, summarize_components


class TestMorphologyDescriptors(unittest.TestCase):
    def test_rectangle_component_has_expected_area_and_perimeter(self):
        mask = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]

        stats = component_stats(mask, foreground_values={1})

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["area"], 4)
        self.assertEqual(stats[0]["perimeter"], 8)
        self.assertAlmostEqual(stats[0]["perimeter2_over_area"], 16.0)
        self.assertAlmostEqual(stats[0]["bbox_fill"], 1.0)

    def test_thin_component_has_higher_boundary_score_than_compact_component(self):
        compact = [
            [1, 1],
            [1, 1],
        ]
        thin = [[1, 1, 1, 1]]

        compact_score = component_stats(compact, foreground_values={1})[0]["perimeter2_over_area"]
        thin_score = component_stats(thin, foreground_values={1})[0]["perimeter2_over_area"]

        self.assertGreater(thin_score, compact_score)

    def test_summary_reports_small_component_area_fraction(self):
        mask = [
            [1, 1, 0, 2, 0],
            [1, 1, 0, 0, 0],
        ]

        summary = summarize_components(mask, foreground_values={1, 2}, small_area_threshold=1)

        self.assertEqual(summary["component_count"], 2)
        self.assertAlmostEqual(summary["foreground_area"], 5.0)
        self.assertAlmostEqual(summary["small_component_area_fraction"], 1 / 5)


if __name__ == "__main__":
    unittest.main()
