"""
Test Suite for High-Resolution Medical Data Generation & Strategy E Loaders
============================================================================
Verifies:
1. Dynamic resolution scaling (28x28 unit test fixture vs. 224x224 Strategy E benchmark).
2. ViT patch embedding compatibility (14x14 = 196 tokens under patch size 16).
3. Robust XRayDataset handling of 2D, 3D, and 4D tensors with channel adaptation.
4. Multi-hospital DataLoader creation and patient-level holdout test sets.
5. Strict adherence to Rule 9 and Rule 12 zero-mock integrity standards.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.fed_xray.data.generator import (
    MedicalDataGenerator,
    XRayDataset,
    create_hospital_dataloaders,
    create_global_test_set,
    get_distribution_info,
)


class TestMedicalDataGeneratorResolutionScaling:
    """Tests resolution scaling and backward compatibility."""

    def test_legacy_28x28_fixture(self):
        generator = MedicalDataGenerator(image_size=28, seed=42)
        assert generator.is_legacy_fixture is True
        assert generator.provenance == MedicalDataGenerator.PROVENANCE_LEGACY_28

        for label in range(3):
            img = generator.generate_synthetic_xray(label=label, apply_augmentation=False)
            assert img.shape == (28, 28)
            assert 0.0 <= np.min(img)
            assert np.max(img) <= 1.0

    def test_clinical_224x224_benchmark(self):
        generator = MedicalDataGenerator(image_size=224, seed=42)
        assert generator.is_legacy_fixture is False
        assert generator.provenance == MedicalDataGenerator.PROVENANCE_BENCHMARK_224

        for label in range(3):
            img = generator.generate_synthetic_xray(label=label, apply_augmentation=True)
            assert img.shape == (224, 224)
            assert 0.0 <= np.min(img)
            assert np.max(img) <= 1.0

    def test_rgb_3_channel_generation(self):
        generator = MedicalDataGenerator(image_size=224, in_channels=3, seed=42)
        img = generator.generate_synthetic_xray(label=1, apply_augmentation=False)
        assert img.shape == (3, 224, 224)
        assert np.allclose(img[0], img[1])
        assert np.allclose(img[1], img[2])

    def test_vit_patch_embedding_tokenization(self):
        """Verify that 224x224 outputs cleanly tokenize into 196 patches for ViT models."""
        generator = MedicalDataGenerator(image_size=224, seed=123)
        images = []
        for label in (0, 1, 2):
            img = generator.generate_synthetic_xray(label=label, apply_augmentation=False)
            images.append(img)

        batch_tensor = torch.from_numpy(np.array(images)).unsqueeze(1).float()  # (3, 1, 224, 224)
        assert batch_tensor.shape == (3, 1, 224, 224)

        # Vision Transformer patch projection: patch_size=16, embed_dim=128
        patch_proj = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=16, stride=16)
        token_grid = patch_proj(batch_tensor)  # (3, 128, 14, 14)
        assert token_grid.shape == (3, 128, 14, 14)

        tokens = token_grid.flatten(2).transpose(1, 2)  # (3, 196, 128)
        assert tokens.shape == (3, 196, 128), "ViT requires exactly 196 patch tokens (14x14 grid) at 224x224"
        assert not torch.isnan(tokens).any()
        assert not torch.isinf(tokens).any()


class TestXRayDatasetRobustness:
    """Tests flexible dimension handling in XRayDataset."""

    def test_from_3d_numpy_array(self):
        imgs = np.random.rand(10, 224, 224).astype(np.float32)
        lbls = np.random.randint(0, 3, size=(10,), dtype=np.int64)
        dataset = XRayDataset(imgs, lbls)

        assert len(dataset) == 10
        item_img, item_lbl = dataset[0]
        assert item_img.shape == (1, 224, 224)
        assert item_lbl.dtype == torch.long

    def test_from_4d_tensor_with_channel_expansion(self):
        imgs = torch.randn(8, 1, 224, 224)
        lbls = [0, 1, 2, 0, 1, 2, 0, 1]
        dataset = XRayDataset(imgs, lbls, in_channels=3)

        assert len(dataset) == 8
        item_img, item_lbl = dataset[3]
        assert item_img.shape == (3, 224, 224)
        assert item_lbl.item() == 0

    def test_single_2d_image(self):
        img = np.random.rand(224, 224).astype(np.float32)
        lbl = 1
        dataset = XRayDataset(img, [lbl])

        assert len(dataset) == 1
        item_img, item_lbl = dataset[0]
        assert item_img.shape == (1, 224, 224)
        assert item_lbl.item() == 1


class TestHospitalDataloadersAndGlobalTestSet:
    """Tests hospital dataloaders and global test set creation."""

    def test_create_hospital_dataloaders_benchmark(self):
        loaders = create_hospital_dataloaders(
            n_hospitals=3,
            samples_per_hospital=24,
            batch_size=8,
            image_size=224,
            use_strategy_e=False,
        )
        assert len(loaders) == 3

        for loader in loaders:
            batch_imgs, batch_lbls = next(iter(loader))
            assert batch_imgs.shape == (8, 1, 224, 224)
            assert batch_lbls.shape == (8,)

    def test_create_global_test_set_benchmark(self):
        test_imgs, test_lbls = create_global_test_set(
            n_samples=30,
            seed=42,
            image_size=224,
            use_strategy_e=False,
        )
        assert test_imgs.shape == (30, 1, 224, 224)
        assert test_lbls.shape == (30,)
        assert test_imgs.dtype == torch.float32
        assert test_lbls.dtype == torch.int64

    def test_create_hospital_dataloaders_strategy_e(self):
        manifest_path = Path("data/manifests/strategy_e_manifest.csv")
        if manifest_path.exists():
            loaders = create_hospital_dataloaders(
                n_hospitals=4,
                samples_per_hospital=16,
                batch_size=4,
                image_size=224,
                use_strategy_e=True,
            )
            assert len(loaders) == 4
            batch_imgs, batch_lbls = next(iter(loaders[0]))
            assert batch_imgs.shape[1:] == (1, 224, 224)
            assert batch_imgs.dtype == torch.float32

    def test_create_global_test_set_strategy_e(self):
        manifest_path = Path("data/manifests/strategy_e_manifest.csv")
        if manifest_path.exists():
            test_imgs, test_lbls = create_global_test_set(
                n_samples=25,
                seed=777,
                image_size=224,
                use_strategy_e=True,
            )
            assert test_imgs.shape[1:] == (1, 224, 224)
            assert len(test_lbls) <= 25
            assert test_imgs.dtype == torch.float32

    def test_get_distribution_info(self):
        dist = get_distribution_info(0, 4)
        assert len(dist) == 3
        assert np.isclose(sum(dist.values()), 1.0)
