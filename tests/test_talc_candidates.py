import unittest

from ore_detection.talc.candidates import detect_dark_matrix_candidates


class TestTalcCandidates(unittest.TestCase):
    def test_detects_dark_regions_in_matrix_only(self):
        grayscale = [
            [120, 118, 119, 121],
            [122, 40, 42, 119],
            [121, 41, 39, 118],
            [120, 119, 121, 122],
        ]
        ore_mask = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ]

        candidates = detect_dark_matrix_candidates(
            grayscale,
            ore_mask=ore_mask,
            dark_offset=30,
            min_component_area=2,
        )

        self.assertEqual(candidates[1][1], 1)
        self.assertEqual(candidates[1][2], 1)
        self.assertEqual(candidates[2][1], 1)
        self.assertEqual(candidates[2][2], 0, "ore pixels must not become talc candidates")

    def test_removes_single_pixel_noise_by_min_area(self):
        grayscale = [
            [100, 100, 100],
            [100, 10, 100],
            [100, 100, 100],
        ]

        candidates = detect_dark_matrix_candidates(grayscale, dark_offset=30, min_component_area=2)

        self.assertEqual(sum(sum(row) for row in candidates), 0)


if __name__ == "__main__":
    unittest.main()
