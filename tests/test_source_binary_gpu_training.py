import unittest

from ore_detection.training.source_binary_gpu_training import (
    augment_source_binary_batch,
    binary_dice_loss_from_logits,
    binary_iou_from_logits,
    normalize_binary_images,
)


class TestSourceBinaryGpuTraining(unittest.TestCase):
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

    def test_augment_source_binary_batch_preserves_shape_and_binary_masks(self):
        for device in self._devices():
            with self.subTest(device=str(device)):
                images = self.torch.rand((2, 3, 8, 8), device=device)
                masks = self.torch.zeros((2, 1, 8, 8), dtype=self.torch.float32, device=device)
                masks[:, :, 2:6, 3:7] = 1.0

                aug_images, aug_masks = augment_source_binary_batch(
                    images,
                    masks,
                    output_size=6,
                    hflip_p=0.0,
                    vflip_p=0.0,
                    scale_range=(0.5, 2.0),
                )

                self.assertEqual(tuple(aug_images.shape), (2, 3, 6, 6))
                self.assertEqual(tuple(aug_masks.shape), (2, 1, 6, 6))
                self.assertTrue(set(float(value) for value in aug_masks.detach().cpu().flatten()).issubset({0.0, 1.0}))

    def test_binary_losses_report_perfect_prediction(self):
        target = self.torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        logits = self.torch.where(target > 0.5, self.torch.tensor(20.0), self.torch.tensor(-20.0))

        self.assertLess(float(binary_dice_loss_from_logits(logits, target)), 1e-4)
        self.assertAlmostEqual(float(binary_iou_from_logits(logits, target)), 1.0)

    def test_normalize_binary_images_uses_supplied_train_stats(self):
        images = self.torch.ones((1, 3, 2, 2), dtype=self.torch.float32)

        normalized = normalize_binary_images(images, mean=(0.5, 1.0, 1.5), std=(0.5, 0.5, 0.5))

        self.assertEqual(tuple(normalized[:, :, 0, 0].flatten().tolist()), (1.0, 0.0, -1.0))


if __name__ == "__main__":
    unittest.main()
