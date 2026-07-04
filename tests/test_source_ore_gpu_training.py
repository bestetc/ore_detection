import unittest

from ore_detection.training.source_ore_gpu_training import (
    augment_source_ore_batch,
    class_index_to_one_hot,
    normalize_images,
)


class TestSourceOreGpuTraining(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        self.torch = torch

    def _devices(self):
        devices = [self.torch.device("cpu")]
        if self.torch.cuda.is_available():
            devices.append(self.torch.device("cuda"))
        return devices

    def test_augment_source_ore_batch_preserves_shape_and_class_ids(self):
        for device in self._devices():
            with self.subTest(device=str(device)):
                images = self.torch.rand((2, 3, 8, 8), device=device)
                class_index = self.torch.zeros((2, 8, 8), dtype=self.torch.long, device=device)
                class_index[:, 2:6, 3:7] = 2

                aug_images, aug_masks = augment_source_ore_batch(
                    images,
                    class_index,
                    output_size=6,
                    background_index=0,
                    hflip_p=0.0,
                    vflip_p=0.0,
                    scale_range=(0.5, 2.0),
                )

                self.assertEqual(tuple(aug_images.shape), (2, 3, 6, 6))
                self.assertEqual(tuple(aug_masks.shape), (2, 6, 6))
                self.assertTrue(set(int(value) for value in aug_masks.detach().cpu().flatten()).issubset({0, 2}))

    def test_normalize_images_uses_supplied_train_stats(self):
        images = self.torch.ones((1, 3, 2, 2), dtype=self.torch.float32)

        normalized = normalize_images(images, mean=(0.5, 1.0, 1.5), std=(0.5, 0.5, 0.5))

        self.assertEqual(tuple(normalized[:, :, 0, 0].flatten().tolist()), (1.0, 0.0, -1.0))

    def test_class_index_to_one_hot_returns_channel_first_mask(self):
        class_index = self.torch.tensor([[[0, 1], [2, 1]]], dtype=self.torch.long)

        one_hot = class_index_to_one_hot(class_index, class_count=3)

        self.assertEqual(tuple(one_hot.shape), (1, 3, 2, 2))
        self.assertEqual(float(one_hot[0, 2, 1, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
