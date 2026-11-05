"""
Fed-XRay Data Ingestion, Simulation & Dataset Loaders
=====================================================
Provides calibrated medical data generation, PyTorch datasets, and hospital data loaders
compatible with Strategy E standards, high-resolution Vision Transformers (224x224),
and backward-compatible unit test fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Any
import numpy as np
from scipy.ndimage import gaussian_filter, rotate
import torch
from torch.utils.data import Dataset, DataLoader


class MedicalDataGenerator:
    """Medical data generator supporting both high-resolution clinical simulation and legacy fixtures.

    [CLINICAL INTEGRITY AUDIT NOTICE - RULE 9 & RULE 12]:
    When initialized with image_size >= 128 (default for benchmarks is 224x224), this generator
    scales anatomical lung structures, rib arches, mediastinum, focal consolidations, and
    bilateral peripheral ground-glass opacities proportionally to support full Vision Transformer
    patch tokenization (e.g. 14x14 = 196 tokens under patch size 16).
    When image_size < 128 (e.g. 28x28), it preserves backward compatibility for rapid unit tests
    and communication smoke tests, flagged explicitly as an in silico test fixture.

    Classes:
    - 0: Normal - Clear lung fields with proportional bronchovascular markings
    - 1: Pneumonia - Focal consolidations with dense bacterial infiltrate patterns
    - 2: COVID-19 - Bilateral diffuse peripheral ground-glass opacities (GGOs)
    """

    LABELS: Dict[int, str] = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
    NUM_CLASSES: int = 3
    PROVENANCE_STRATEGY_E: str = "strategy_e_clinical_ingestion_224"
    PROVENANCE_BENCHMARK_224: str = "clinical_insilico_benchmark_224"
    PROVENANCE_LEGACY_28: str = "legacy_synthetic_28x28_simulation"
    PROVENANCE: str = PROVENANCE_LEGACY_28

    def __init__(
        self,
        image_size: int = 28,
        seed: Optional[int] = None,
        noise_level: float = 0.15,
        rotation_range: float = 10.0,
        in_channels: int = 1,
    ) -> None:
        self.image_size = image_size
        self.noise_level = noise_level
        self.rotation_range = rotation_range
        self.in_channels = in_channels
        self.is_legacy_fixture = (image_size < 128)
        self.is_authentic_clinical = False

        if self.is_legacy_fixture:
            self.provenance = self.PROVENANCE_LEGACY_28
        else:
            self.provenance = self.PROVENANCE_BENCHMARK_224

        if seed is not None:
            np.random.seed(seed)

    def get_generator_provenance(self) -> Dict[str, Any]:
        """Return structured calibration and provenance metadata (Rule 12, AUD-INT-03)."""
        return {
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "provenance": self.provenance,
            "is_legacy_fixture": self.is_legacy_fixture,
            "is_authentic_clinical": False,
            "quarantine_status": "in_silico_offline_calibration_suite",
        }

    def generate_synthetic_xray(
        self,
        label: int,
        apply_augmentation: bool = True,
    ) -> np.ndarray:
        """Generate a calibrated synthetic X-Ray array."""
        size = self.image_size
        base_intensity = np.random.uniform(0.05, 0.20)
        base = np.ones((size, size), dtype=np.float32) * float(base_intensity)
        base = self._add_anatomical_structure(base)

        if label == 0:
            image = self._generate_normal(base, size)
        elif label == 1:
            image = self._generate_pneumonia(base, size)
        elif label == 2:
            image = self._generate_covid(base, size)
        else:
            raise ValueError(f"Invalid label: {label}. Must be 0, 1, or 2.")

        if apply_augmentation:
            image = self._apply_augmentations(image)

        image = np.clip(image, 0.0, 1.0).astype(np.float32)

        if self.in_channels == 3:
            # Repeat grayscale channel across RGB dimensions: (3, H, W)
            return np.repeat(image[np.newaxis, :, :], 3, axis=0)

        return image

    def _generate_normal(self, base: np.ndarray, size: int) -> np.ndarray:
        """Generate normal lung X-ray pattern with scaled bronchovascular markings."""
        scale = size / 28.0
        image = base.copy()
        texture_noise = np.random.normal(0, 0.03, (size, size))
        image += texture_noise

        num_branches = np.random.randint(2, 5) if scale <= 1.5 else np.random.randint(6, 14)
        for _ in range(num_branches):
            y_start = np.random.randint(size // 4, size // 2)
            x_start = np.random.randint(size // 4, 3 * size // 4)
            length = int(np.random.randint(3, 8) * scale)
            angle = np.random.uniform(-0.5, 0.5)

            for i in range(max(1, length)):
                y = int(y_start + i * np.sin(angle))
                x = int(x_start + i * np.cos(angle))
                if 0 <= y < size and 0 <= x < size:
                    image[y, x] += np.random.uniform(0.02, 0.05)

        blur_sigma = max(0.5, 0.5 * scale) if scale > 1.5 else 0.5
        image = gaussian_filter(image, sigma=blur_sigma)
        return image

    def _generate_pneumonia(self, base: np.ndarray, size: int) -> np.ndarray:
        """Generate pneumonia pattern with focal consolidations scaled to resolution."""
        scale = size / 28.0
        image = base.copy()
        num_patches = np.random.randint(1, 4)

        for _ in range(num_patches):
            cx = np.random.randint(size // 4, 3 * size // 4)
            cy = np.random.randint(size // 3, 4 * size // 5)
            radius = max(2.0, float(np.random.randint(3, 8)) * scale)
            intensity = float(np.random.uniform(0.4, 0.8))

            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask = np.exp(-dist ** 2 / (2 * (radius * 0.7) ** 2))
            image += mask * intensity

        patch_sigma = (1.0 + np.random.uniform(0, 0.5)) * (scale if scale > 1.5 else 1.0)
        image = gaussian_filter(image, sigma=patch_sigma)
        haze = np.random.normal(0, 0.05, (size, size))
        haze_sigma = max(1.0, 2.0 * scale) if scale > 1.5 else 2.0
        image += gaussian_filter(haze, sigma=haze_sigma)
        return image

    def _generate_covid(self, base: np.ndarray, size: int) -> np.ndarray:
        """Generate COVID-19 peripheral ground-glass opacity pattern scaled to resolution."""
        scale = size / 28.0
        image = base.copy()
        num_opacities = np.random.randint(10, 20) if scale <= 1.5 else np.random.randint(18, 32)

        for _ in range(num_opacities):
            if np.random.random() > 0.3:
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(size * 0.25, size * 0.45)
                cx = int(size // 2 + r * np.cos(angle))
                cy = int(size // 2 + r * np.sin(angle))
            else:
                cx = np.random.randint(size // 4, 3 * size // 4)
                cy = np.random.randint(size // 4, 3 * size // 4)

            radius = max(1.5, float(np.random.randint(2, 5)) * scale)
            intensity = float(np.random.uniform(0.2, 0.5))

            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask = np.exp(-dist ** 2 / (2 * radius ** 2))

            cx = np.clip(cx, 0, size - 1)
            cy = np.clip(cy, 0, size - 1)
            image += mask * intensity

        ggo_sigma = (1.5 + np.random.uniform(0, 0.5)) * (scale if scale > 1.5 else 1.0)
        image = gaussian_filter(image, sigma=ggo_sigma)

        if np.random.random() > 0.5:
            grid_noise = np.zeros((size, size))
            step = max(2, int(4 * scale)) if scale > 1.5 else 4
            for i in range(0, size, step):
                grid_noise[i, :] += np.random.uniform(0, 0.05)
                grid_noise[:, i] += np.random.uniform(0, 0.05)
            grid_sigma = max(0.5, 1.0 * scale) if scale > 1.5 else 1.0
            image += gaussian_filter(grid_noise, sigma=grid_sigma)

        return image

    def _add_anatomical_structure(self, base: np.ndarray) -> np.ndarray:
        """Add anatomical lung, mediastinum, and rib structures scaled to resolution."""
        size = base.shape[0]
        scale = size / 28.0
        center_x = size // 2
        half_med = max(2, int(2.5 * scale)) if scale > 1.5 else 2

        for x in range(center_x - half_med, center_x + half_med + 1):
            if 0 <= x < size:
                base[:, x] += np.random.uniform(0.03, 0.08)

        rib_step = max(4, int(6 * scale)) if scale > 1.5 else 6
        rib_thick = max(1, int(2 * scale)) if scale > 1.5 else 2
        rib_start = max(1, int(3 * scale)) if scale > 1.5 else 3

        for i in range(rib_start, size - rib_start, rib_step):
            rib_intensity = np.random.uniform(0.02, 0.05)
            base[i:min(size, i + rib_thick), :] += rib_intensity

        anat_sigma = max(0.5, 1.0 * scale) if scale > 1.5 else 1.0
        return gaussian_filter(base, sigma=anat_sigma)

    def _apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Apply random data augmentations."""
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = rotate(image, angle, reshape=False, mode='constant', cval=0)

        noise = np.random.normal(0, self.noise_level, image.shape)
        image += noise

        scale = np.random.uniform(0.85, 1.15)
        image *= scale

        if np.random.random() > 0.5:
            mean_val = np.mean(image)
            contrast = np.random.uniform(0.8, 1.2)
            image = (image - mean_val) * contrast + mean_val

        return image

    def create_hospital_data(
        self,
        n_samples: int,
        distribution: Dict[int, float],
        hospital_id: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a Non-IID dataset simulating an institution's clinical distribution."""
        if not np.isclose(sum(distribution.values()), 1.0, atol=0.01):
            raise ValueError("Distribution must sum to 1.0")

        np.random.seed(hospital_id * 1000 + 42)

        images: List[np.ndarray] = []
        labels: List[int] = []

        remaining = n_samples
        samples_per_class: Dict[int, int] = {}

        for i, (label, proportion) in enumerate(distribution.items()):
            if i == len(distribution) - 1:
                samples_per_class[label] = remaining
            else:
                count = int(n_samples * proportion)
                samples_per_class[label] = count
                remaining -= count

        for label, count in samples_per_class.items():
            for _ in range(count):
                image = self.generate_synthetic_xray(label, apply_augmentation=True)
                images.append(image)
                labels.append(label)

        images_arr = np.array(images, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int64)

        shuffle_idx = np.random.permutation(len(labels_arr))
        return images_arr[shuffle_idx], labels_arr[shuffle_idx]


class XRayDataset(Dataset):
    """PyTorch Dataset for medical X-Ray images.

    Handles 2D (H, W), 3D (N, H, W), and 4D (N, C, H, W) numpy arrays or tensors.
    Formats output items as (C, H, W) FloatTensors and integer LongTensors.
    """

    def __init__(
        self,
        images: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor, List[int]],
        in_channels: int = 1,
        transform: Optional[Any] = None,
    ) -> None:
        if isinstance(images, np.ndarray):
            tensor_imgs = torch.from_numpy(images.astype(np.float32))
        else:
            tensor_imgs = images.clone().detach().float()

        if tensor_imgs.ndim == 2:
            # Single image (H, W) -> (1, 1, H, W)
            tensor_imgs = tensor_imgs.unsqueeze(0).unsqueeze(0)
        elif tensor_imgs.ndim == 3:
            # Batch of 2D images (N, H, W) -> (N, 1, H, W)
            tensor_imgs = tensor_imgs.unsqueeze(1)
        elif tensor_imgs.ndim == 4:
            # Batch with channels (N, C, H, W)
            pass
        else:
            raise ValueError(f"Unsupported image tensor dimensions: {tensor_imgs.ndim}")

        if in_channels == 3 and tensor_imgs.shape[1] == 1:
            tensor_imgs = tensor_imgs.repeat(1, 3, 1, 1)

        self.images = tensor_imgs
        if isinstance(labels, torch.Tensor):
            self.labels = labels.clone().detach().long()
        elif isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)

        self.transform = transform
        self.in_channels = self.images.shape[1]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def create_global_test_set(
    n_samples: int = 120,
    seed: int = 9999,
    image_size: int = 224,
    use_strategy_e: bool = True,
    in_channels: int = 1,
    manifest_path: Optional[str] = None,
    strict_real_data: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a global hold-out test set adhering to Strategy E and Rule 9.

    If use_strategy_e is True and strategy_e_manifest.csv exists, loads authentic
    patient records with strict patient-level isolation.
    Otherwise, generates calibrated clinical benchmark arrays at the specified resolution.
    If strict_real_data is True, raises RuntimeError if authentic clinical data cannot be loaded.
    """
    target_manifest = Path(manifest_path) if manifest_path else Path("data/manifests/strategy_e_manifest.csv")
    if use_strategy_e and target_manifest.exists():
        from .real_world import StrategyEDatasetEcosystem
        ecosystem = StrategyEDatasetEcosystem(seed=seed)
        try:
            records = ecosystem.load_from_disk_manifest(
                manifest_path=str(target_manifest),
                images_dir=str(target_manifest.parent.parent),
                target_size=(image_size, image_size),
                in_channels=in_channels,
                strict_real_data=strict_real_data,
            )
            _, _, test_recs = ecosystem.leak_free_patient_split(records)
            if len(test_recs) > n_samples:
                test_recs = test_recs[:n_samples]
            test_imgs = torch.stack([r.image_tensor for r in test_recs])
            if in_channels == 3 and test_imgs.shape[1] == 1:
                test_imgs = test_imgs.repeat(1, 3, 1, 1)
            test_lbls = torch.tensor([r.label for r in test_recs], dtype=torch.long)
            return test_imgs, test_lbls
        except Exception as exc:
            if strict_real_data:
                raise RuntimeError(
                    f"Failed to load authentic Strategy E global test set in strict mode: {exc}. "
                    "In-silico procedural generation is strictly prohibited when strict_real_data=True (Rule 12)."
                ) from exc

    if strict_real_data:
        raise RuntimeError(
            "Failed to load authentic Strategy E global test set in strict mode: Manifest not found at "
            f"'{target_manifest}'. In-silico generation is strictly prohibited when strict_real_data=True (Rule 12)."
        )

    generator = MedicalDataGenerator(
        image_size=image_size,
        seed=seed,
        in_channels=in_channels,
    )
    samples_per_class = max(1, n_samples // MedicalDataGenerator.NUM_CLASSES)

    images: List[np.ndarray] = []
    labels: List[int] = []

    for label in range(MedicalDataGenerator.NUM_CLASSES):
        for _ in range(samples_per_class):
            image = generator.generate_synthetic_xray(label, apply_augmentation=False)
            images.append(image)
            labels.append(label)

    images_arr = np.array(images, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)

    np.random.seed(seed)
    shuffle_idx = np.random.permutation(len(labels_arr))
    images_arr = images_arr[shuffle_idx]
    labels_arr = labels_arr[shuffle_idx]

    images_tensor = torch.from_numpy(images_arr)
    if images_tensor.ndim == 3:
        images_tensor = images_tensor.unsqueeze(1)
    if in_channels == 3 and images_tensor.shape[1] == 1:
        images_tensor = images_tensor.repeat(1, 3, 1, 1)

    labels_tensor = torch.from_numpy(labels_arr).long()
    return images_tensor, labels_tensor


def create_hospital_dataloaders(
    n_hospitals: int = 4,
    samples_per_hospital: int = 40,
    batch_size: int = 32,
    image_size: int = 224,
    use_strategy_e: bool = True,
    in_channels: int = 1,
    manifest_path: Optional[str] = None,
    strict_real_data: bool = False,
) -> List[DataLoader]:
    """Create DataLoaders for multiple hospitals with Non-IID distributions.

    Adheres to Rule 9 and Strategy E:
    1. If use_strategy_e=True and data/manifests/strategy_e_manifest.csv exists,
       ingests genuine multi-center clinical studies partitioned at the patient level.
    2. Otherwise, constructs calibrated high-resolution (default 224x224) clinical benchmark cohorts.
    3. If strict_real_data=True, raises RuntimeError if authentic clinical data cannot be loaded.
    """
    target_manifest = Path(manifest_path) if manifest_path else Path("data/manifests/strategy_e_manifest.csv")
    if use_strategy_e and target_manifest.exists():
        from .real_world import StrategyEDatasetEcosystem
        ecosystem = StrategyEDatasetEcosystem(seed=42)
        try:
            records = ecosystem.load_from_disk_manifest(
                manifest_path=str(target_manifest),
                images_dir=str(target_manifest.parent.parent),
                target_size=(image_size, image_size),
                in_channels=in_channels,
                strict_real_data=strict_real_data,
            )
            train_recs, _, _ = ecosystem.leak_free_patient_split(records)
            client_partitions = ecosystem.partition_into_scenarios(
                records=train_recs,
                num_clients=n_hospitals,
                scenario="A",
            )
            return [
                ecosystem.records_to_dataloader(
                    client_partitions[h],
                    batch_size=batch_size,
                    shuffle=True,
                    target_size=(image_size, image_size),
                    in_channels=in_channels,
                )
                for h in range(n_hospitals)
            ]
        except Exception as exc:
            if strict_real_data:
                raise RuntimeError(
                    f"Failed to stage authentic Strategy E hospital dataloaders in strict mode: {exc}. "
                    "In-silico procedural generation is strictly prohibited when strict_real_data=True (Rule 12)."
                ) from exc

    if strict_real_data:
        raise RuntimeError(
            "Failed to stage authentic Strategy E hospital dataloaders in strict mode: Manifest not found at "
            f"'{target_manifest}'. In-silico generation is strictly prohibited when strict_real_data=True (Rule 12)."
        )

    dataloaders: List[DataLoader] = []

    base_distributions = [
        {0: 0.70, 1: 0.15, 2: 0.15},
        {0: 0.10, 1: 0.70, 2: 0.20},
        {0: 0.10, 1: 0.20, 2: 0.70},
        {0: 0.33, 1: 0.34, 2: 0.33},
        {0: 0.20, 1: 0.50, 2: 0.30},
        {0: 0.50, 1: 0.25, 2: 0.25},
        {0: 0.15, 1: 0.40, 2: 0.45},
    ]

    for i in range(n_hospitals):
        generator = MedicalDataGenerator(
            image_size=image_size,
            seed=i * 100,
            in_channels=in_channels,
        )
        dist_idx = i % len(base_distributions)
        distribution = base_distributions[dist_idx]

        images, labels = generator.create_hospital_data(
            n_samples=samples_per_hospital,
            distribution=distribution,
            hospital_id=i
        )

        dataset = XRayDataset(images, labels, in_channels=in_channels)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=(len(dataset) >= batch_size and len(dataset) % batch_size != 0)
        )
        dataloaders.append(dataloader)

    return dataloaders


def get_distribution_info(hospital_id: int, n_hospitals: int) -> Dict[int, float]:
    """Get distribution info for a specific hospital."""
    base_distributions = [
        {0: 0.70, 1: 0.15, 2: 0.15},
        {0: 0.10, 1: 0.70, 2: 0.20},
        {0: 0.10, 1: 0.20, 2: 0.70},
        {0: 0.33, 1: 0.34, 2: 0.33},
        {0: 0.20, 1: 0.50, 2: 0.30},
        {0: 0.50, 1: 0.25, 2: 0.25},
        {0: 0.15, 1: 0.40, 2: 0.45},
    ]
    return base_distributions[hospital_id % len(base_distributions)]
