import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.data.ore_type_legend import load_legend_config
from ore_detection.training.source_ore_dataset import SourceOreTorchDataset, list_source_ore_samples


class TestSourceOreDataset(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        self.torch = torch
        self.legend = load_legend_config()

    def _save_mask(self, path: Path, *, width: int = 3, height: int = 2):
        one_hot = self.torch.zeros((self.legend.class_count, height, width), dtype=self.torch.uint8)
        one_hot[self.legend.background_index] = 1
        one_hot[self.legend.background_index, 0, 1] = 0
        one_hot[self.legend.class_index("chalcopyrite"), 0, 1] = 1
        path.parent.mkdir(parents=True)
        self.torch.save(one_hot, path)

    def test_list_source_ore_samples_pairs_direct_image_and_pt_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "datasets" / "set_1" / "imgs" / "train" / "train_01.jpg"
            mask_path = root / "masks" / "set_1" / "train" / "train_01.pt"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (3, 2), color=(1, 2, 3)).save(image_path)
            self._save_mask(mask_path)

            samples = list_source_ore_samples(
                datasets_root=root / "datasets",
                masks_root=root / "masks",
                datasets=("set_1",),
            )

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].image_path, image_path)
            self.assertEqual(samples[0].mask_path, mask_path)

    def test_set3_sample_prefers_unrotated_r000_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = (
                root
                / "datasets"
                / "set_3"
                / "imgs"
                / "train"
                / "S3_train_01"
                / "S3_train_01_r000.jpg"
            )
            mask_path = root / "masks" / "set_3" / "train" / "S3_train_01.pt"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (3, 2), color=(1, 2, 3)).save(image_path)
            self._save_mask(mask_path)

            samples = list_source_ore_samples(
                datasets_root=root / "datasets",
                masks_root=root / "masks",
                datasets=("set_3",),
            )

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].image_path, image_path)

    def test_torch_dataset_returns_multiclass_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "datasets" / "set_1" / "imgs" / "train" / "train_01.jpg"
            mask_path = root / "masks" / "set_1" / "train" / "train_01.pt"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (3, 2), color=(1, 2, 3)).save(image_path)
            self._save_mask(mask_path)

            sample = list_source_ore_samples(
                datasets_root=root / "datasets",
                masks_root=root / "masks",
                datasets=("set_1",),
            )[0]
            dataset = SourceOreTorchDataset([sample], legend=self.legend, image_size=None)

            item = dataset[0]

            self.assertEqual(tuple(item["image"].shape), (3, 2, 3))
            self.assertEqual(tuple(item["mask"].shape), (self.legend.class_count, 2, 3))
            self.assertEqual(tuple(item["class_index"].shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
