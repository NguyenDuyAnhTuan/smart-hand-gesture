import unittest

import torch

import main
from src.models.model_factory import build_model
from src.utils.config import Config


class SmokeTests(unittest.TestCase):
    def test_main_imports(self):
        self.assertIsNotNone(main)

    def test_config_class_count(self):
        cfg = Config()
        self.assertEqual(cfg.num_classes, 10)

    def test_vit_forward_shape(self):
        cfg = Config()
        model = build_model(num_classes=cfg.num_classes)
        x = torch.randn(2, 3, cfg.img_size, cfg.img_size)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, cfg.num_classes))


if __name__ == "__main__":
    unittest.main()
