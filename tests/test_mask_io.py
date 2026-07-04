import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.data.label_mapping import COARSE_LABELS
from ore_detection.data.mask_io import convert_source_mask_file


class TestMaskIO(unittest.TestCase):
    def test_convert_source_mask_file_writes_mapped_grayscale_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "mask.png"
            target = tmp_path / "mapped.png"
            image = Image.new("RGB", (3, 1))
            image.putdata([(0, 0, 0), (1, 1, 1), (5, 5, 5)])
            image.save(source)

            convert_source_mask_file(source, target, source_dataset="set_2", target_taxonomy="coarse")

            mapped = Image.open(target).convert("L")
            self.assertEqual(
                list(mapped.tobytes()),
                [
                    COARSE_LABELS["background_matrix"],
                    COARSE_LABELS["sulfide_ore"],
                    COARSE_LABELS["oxide_magnetite_hematite"],
                ],
            )


if __name__ == "__main__":
    unittest.main()
