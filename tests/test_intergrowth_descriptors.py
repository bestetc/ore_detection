import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.descriptors.intergrowth import (
    summarize_intergrowth_prediction,
    summarize_prediction_artifacts,
    write_descriptor_csv,
)


class TestIntergrowthDescriptors(unittest.TestCase):
    def test_binary_descriptor_summary_reports_geometry(self):
        binary = [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ]

        descriptors = summarize_intergrowth_prediction(binary, small_area_threshold=3)

        self.assertAlmostEqual(descriptors["ore_area"], 3.0)
        self.assertAlmostEqual(descriptors["ore_area_fraction"], 3 / 9)
        self.assertAlmostEqual(descriptors["ore_component_count"], 1.0)
        self.assertGreater(descriptors["ore_perimeter_density"], 0.0)
        self.assertGreater(descriptors["ore_background_contact_length"], 0.0)
        self.assertAlmostEqual(descriptors["component_area_max"], 3.0)

    def test_multiclass_descriptor_summary_reports_fractions_and_contacts(self):
        binary = [
            [1, 1, 0],
            [1, 1, 0],
        ]
        multiclass = [
            [1, 2, 0],
            [1, 2, 0],
        ]

        descriptors = summarize_intergrowth_prediction(
            binary,
            multiclass_mask=multiclass,
            class_names=("background", "pyrite", "chalcopyrite"),
            background_index=0,
        )

        self.assertAlmostEqual(descriptors["class_area_pyrite"], 2.0)
        self.assertAlmostEqual(descriptors["class_fraction_chalcopyrite"], 2 / 6)
        self.assertAlmostEqual(descriptors["mineral_contact_pyrite__chalcopyrite"], 2.0)
        self.assertNotIn("talc_area_fraction", descriptors)

    def test_summarize_prediction_artifacts_and_write_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            binary = Image.new("L", (2, 2))
            binary.putdata([1, 1, 0, 0])
            binary.save(sample_dir / "ore_mask.png")
            multiclass = Image.new("L", (2, 2))
            multiclass.putdata([1, 2, 0, 0])
            multiclass.save(sample_dir / "ore_multiclass_mask.png")
            (sample_dir / "metadata.json").write_text(
                """
{
  "sample_id": "sample-1",
  "image_path": "image.png",
  "ore_checkpoint": {
    "class_names": ["background", "pyrite", "galena"],
    "background_index": 0
  }
}
""".strip(),
                encoding="utf-8",
            )

            row = summarize_prediction_artifacts(sample_dir)
            csv_path = root / "descriptors.csv"
            write_descriptor_csv([row], csv_path)

            self.assertEqual(row["sample_id"], "sample-1")
            self.assertAlmostEqual(row["class_area_pyrite"], 1.0)
            self.assertTrue(csv_path.exists())
            self.assertIn("ore_area_fraction", csv_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
