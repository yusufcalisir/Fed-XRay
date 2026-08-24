"""
Fed-XRay Data Ingestion, Simulation & Dataset Loaders
=====================================================
Provides synthetic medical generator, PyTorch datasets, and hospital data loaders.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from scipy.ndimage import gaussian_filter, rotate
import torch
from torch.utils.data import Dataset, DataLoader


class MedicalDataGenerator:
    """
    Synthetic X-Ray generator for federated learning simulations.
    
    Classes:
    - 0: Normal - Clear lungs with minimal opacity
    - 1: Pneumonia - Focal consolidations (bacterial infection pattern)
    - 2: COVID-19 - Diffuse bilateral ground-glass opacities
    """
    
    LABELS: Dict[int, str] = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}
    NUM_CLASSES: int = 3
    
    def __init__(
        self, 
        image_size: int = 28, 
        seed: Optional[int] = None,
        noise_level: float = 0.15,
        rotation_range: float = 10.0
    ) -> None:
        self.image_size = image_size
        self.noise_level = noise_level
        self.rotation_range = rotation_range
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_synthetic_xray(
        self, 
        label: int,
        apply_augmentation: bool = True
    ) -> np.ndarray:
        """Generate a synthetic X-Ray image."""
        size = self.image_size
        base_intensity = np.random.uniform(0.05, 0.20)
        base = np.ones((size, size)) * base_intensity
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
        
        image = np.clip(image, 0, 1)
        return image.astype(np.float32)
    
    def _generate_normal(self, base: np.ndarray, size: int) -> np.ndarray:
        """Generate normal lung X-ray pattern."""
        image = base.copy()
        texture_noise = np.random.normal(0, 0.03, (size, size))
        image += texture_noise
        
        for _ in range(np.random.randint(2, 5)):
            y_start = np.random.randint(size // 4, size // 2)
            x_start = np.random.randint(size // 4, 3 * size // 4)
            length = np.random.randint(3, 8)
            angle = np.random.uniform(-0.5, 0.5)
            
            for i in range(length):
                y = int(y_start + i * np.sin(angle))
                x = int(x_start + i * np.cos(angle))
                if 0 <= y < size and 0 <= x < size:
                    image[y, x] += np.random.uniform(0.02, 0.05)
        
        image = gaussian_filter(image, sigma=0.5)
        return image
    
    def _generate_pneumonia(self, base: np.ndarray, size: int) -> np.ndarray:
        """Generate pneumonia pattern with focal consolidations."""
        image = base.copy()
        num_patches = np.random.randint(1, 4)
        
        for _ in range(num_patches):
            cx = np.random.randint(size // 4, 3 * size // 4)
            cy = np.random.randint(size // 3, 4 * size // 5)
            radius = np.random.randint(3, 8)
            intensity = np.random.uniform(0.4, 0.8)
            
            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask = np.exp(-dist ** 2 / (2 * (radius * 0.7) ** 2))
            image += mask * intensity
        
        image = gaussian_filter(image, sigma=1.0 + np.random.uniform(0, 0.5))
        haze = np.random.normal(0, 0.05, (size, size))
        image += gaussian_filter(haze, sigma=2)
        return image
    
    def _generate_covid(self, base: np.ndarray, size: int) -> np.ndarray:
        """Generate COVID-19 ground-glass opacity pattern."""
        image = base.copy()
        num_opacities = np.random.randint(10, 20)
        
        for _ in range(num_opacities):
            if np.random.random() > 0.3:
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(size * 0.25, size * 0.45)
                cx = int(size // 2 + r * np.cos(angle))
                cy = int(size // 2 + r * np.sin(angle))
            else:
                cx = np.random.randint(size // 4, 3 * size // 4)
                cy = np.random.randint(size // 4, 3 * size // 4)
            
            radius = np.random.randint(2, 5)
            intensity = np.random.uniform(0.2, 0.5)
            
            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask = np.exp(-dist ** 2 / (2 * radius ** 2))
            
            cx = np.clip(cx, 0, size - 1)
            cy = np.clip(cy, 0, size - 1)
            image += mask * intensity
        
        image = gaussian_filter(image, sigma=1.5 + np.random.uniform(0, 0.5))
        
        if np.random.random() > 0.5:
            grid_noise = np.zeros((size, size))
            for i in range(0, size, 4):
                grid_noise[i, :] += np.random.uniform(0, 0.05)
                grid_noise[:, i] += np.random.uniform(0, 0.05)
            image += gaussian_filter(grid_noise, sigma=1)
        
        return image
    
    def _add_anatomical_structure(self, base: np.ndarray) -> np.ndarray:
        """Add anatomical lung and rib structures."""
        size = base.shape[0]
        center_x = size // 2
        for x in range(center_x - 2, center_x + 3):
            if 0 <= x < size:
                base[:, x] += np.random.uniform(0.03, 0.08)
        
        for i in range(3, size - 3, 6):
            rib_intensity = np.random.uniform(0.02, 0.05)
            base[i:i+2, :] += rib_intensity
        
        return gaussian_filter(base, sigma=1.0)
    
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
        """Create a Non-IID dataset simulating a hospital's patient mix."""
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
        
        images_arr = np.array(images)
        labels_arr = np.array(labels)
        
        shuffle_idx = np.random.permutation(len(labels_arr))
        return images_arr[shuffle_idx], labels_arr[shuffle_idx]


class XRayDataset(Dataset):
    """PyTorch Dataset for X-Ray images."""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.FloatTensor(images).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def create_global_test_set(
    n_samples: int = 300,
    seed: int = 9999
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a global hold-out test set."""
    generator = MedicalDataGenerator(seed=seed)
    samples_per_class = n_samples // MedicalDataGenerator.NUM_CLASSES
    
    images: List[np.ndarray] = []
    labels: List[int] = []
    
    for label in range(MedicalDataGenerator.NUM_CLASSES):
        for _ in range(samples_per_class):
            image = generator.generate_synthetic_xray(label, apply_augmentation=False)
            images.append(image)
            labels.append(label)
    
    images_arr = np.array(images)
    labels_arr = np.array(labels)
    
    np.random.seed(seed)
    shuffle_idx = np.random.permutation(len(labels_arr))
    images_arr = images_arr[shuffle_idx]
    labels_arr = labels_arr[shuffle_idx]
    
    images_tensor = torch.FloatTensor(images_arr).unsqueeze(1)
    labels_tensor = torch.LongTensor(labels_arr)
    
    return images_tensor, labels_tensor


def create_hospital_dataloaders(
    n_hospitals: int,
    samples_per_hospital: int,
    batch_size: int = 32
) -> List[DataLoader]:
    """Create DataLoaders for multiple hospitals with Non-IID distributions."""
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
        generator = MedicalDataGenerator(seed=i * 100)
        dist_idx = i % len(base_distributions)
        distribution = base_distributions[dist_idx]
        
        images, labels = generator.create_hospital_data(
            n_samples=samples_per_hospital,
            distribution=distribution,
            hospital_id=i
        )
        
        dataset = XRayDataset(images, labels)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(dataset) >= batch_size
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
