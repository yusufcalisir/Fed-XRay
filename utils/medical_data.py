"""
Fed-XRay: Synthetic Medical Image Generator (Professional Version)
===================================================================
Enhanced X-Ray image generator with realistic noise, rotations, and
proper data separation for rigorous federated learning experiments.

Upgrades from v1:
- Random rotations (±10 degrees)
- Gaussian noise injection with varying intensities
- More realistic pattern variation
- Global hold-out test set for unbiased evaluation
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from scipy.ndimage import gaussian_filter, rotate
import torch
from torch.utils.data import Dataset, DataLoader


class MedicalDataGenerator:
    """
    Enhanced synthetic X-Ray generator for federated learning.
    
    Classes:
    - 0: Normal - Clear lungs with minimal opacity
    - 1: Pneumonia - Focal consolidations (bacterial infection pattern)
    - 2: COVID-19 - Diffuse bilateral ground-glass opacities
    
    This version adds realistic image augmentation and variation
    to make the classification task appropriately challenging.
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
        """
        Initialize the medical data generator.
        
        Args:
            image_size: Size of generated images (default 28x28)
            seed: Random seed for reproducibility
            noise_level: Base noise level (0.0 to 1.0)
            rotation_range: Max rotation in degrees (±)
        """
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
        """
        Generate a synthetic X-Ray image with enhanced realism.
        
        Features added for complexity:
        - Variable intensity patterns
        - Random rotations
        - Gaussian noise injection
        - Blur variations
        
        Args:
            label: Disease class (0=Normal, 1=Pneumonia, 2=COVID-19)
            apply_augmentation: Whether to apply random augmentations
            
        Returns:
            28x28 numpy array with pixel values in [0, 1]
        """
        size = self.image_size
        
        # Variable base intensity (makes each image unique)
        base_intensity = np.random.uniform(0.05, 0.20)
        base = np.ones((size, size)) * base_intensity
        
        # Add subtle anatomical structure (ribcage simulation)
        base = self._add_anatomical_structure(base)
        
        if label == 0:  # Normal
            image = self._generate_normal(base, size)
        elif label == 1:  # Pneumonia
            image = self._generate_pneumonia(base, size)
        elif label == 2:  # COVID-19
            image = self._generate_covid(base, size)
        else:
            raise ValueError(f"Invalid label: {label}. Must be 0, 1, or 2.")
        
        # Apply augmentations for training robustness
        if apply_augmentation:
            image = self._apply_augmentations(image)
        
        # Final normalization and clipping
        image = np.clip(image, 0, 1)
        
        return image.astype(np.float32)
    
    def _generate_normal(self, base: np.ndarray, size: int) -> np.ndarray:
        """Generate normal (healthy) lung X-ray pattern."""
        image = base.copy()
        
        # Clear lungs - very subtle texture only
        texture_noise = np.random.normal(0, 0.03, (size, size))
        image += texture_noise
        
        # Add subtle vascular markings (normal lung anatomy)
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
        
        # Pneumonia: 1-3 focal bright patches (consolidation)
        num_patches = np.random.randint(1, 4)
        
        for _ in range(num_patches):
            # Consolidations typically in lower/middle lung zones
            cx = np.random.randint(size // 4, 3 * size // 4)
            cy = np.random.randint(size // 3, 4 * size // 5)
            
            # Variable size consolidations
            radius = np.random.randint(3, 8)
            intensity = np.random.uniform(0.4, 0.8)
            
            # Create consolidation with gradient falloff
            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask = np.exp(-dist ** 2 / (2 * (radius * 0.7) ** 2))
            image += mask * intensity
        
        # Add air bronchograms (dark streaks within consolidation)
        image = gaussian_filter(image, sigma=1.0 + np.random.uniform(0, 0.5))
        
        # Add some surrounding haziness
        haze = np.random.normal(0, 0.05, (size, size))
        image += gaussian_filter(haze, sigma=2)
        
        return image
    
    def _generate_covid(self, base: np.ndarray, size: int) -> np.ndarray:
        """Generate COVID-19 ground-glass opacity pattern."""
        image = base.copy()
        
        # COVID-19: Diffuse bilateral ground-glass opacities
        # Multiple small, scattered opacities (different from focal pneumonia)
        num_opacities = np.random.randint(10, 20)
        
        for _ in range(num_opacities):
            # Peripheral distribution (COVID hallmark)
            if np.random.random() > 0.3:
                # Peripheral
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(size * 0.25, size * 0.45)
                cx = int(size // 2 + r * np.cos(angle))
                cy = int(size // 2 + r * np.sin(angle))
            else:
                # Some central involvement too
                cx = np.random.randint(size // 4, 3 * size // 4)
                cy = np.random.randint(size // 4, 3 * size // 4)
            
            # Smaller, more subtle opacities than pneumonia
            radius = np.random.randint(2, 5)
            intensity = np.random.uniform(0.2, 0.5)  # Lower intensity = ground-glass
            
            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask = np.exp(-dist ** 2 / (2 * radius ** 2))
            
            # Clamp to image bounds
            cx = np.clip(cx, 0, size - 1)
            cy = np.clip(cy, 0, size - 1)
            image += mask * intensity
        
        # Strong blur for ground-glass appearance
        image = gaussian_filter(image, sigma=1.5 + np.random.uniform(0, 0.5))
        
        # Add interlobular septal thickening (crazy paving pattern)
        if np.random.random() > 0.5:
            grid_noise = np.zeros((size, size))
            for i in range(0, size, 4):
                grid_noise[i, :] += np.random.uniform(0, 0.05)
                grid_noise[:, i] += np.random.uniform(0, 0.05)
            image += gaussian_filter(grid_noise, sigma=1)
        
        return image
    
    def _add_anatomical_structure(self, base: np.ndarray) -> np.ndarray:
        """Add basic anatomical structure (ribcage, mediastinum)."""
        size = base.shape[0]
        
        # Mediastinum (central bright stripe)
        center_x = size // 2
        for x in range(center_x - 2, center_x + 3):
            if 0 <= x < size:
                base[:, x] += np.random.uniform(0.03, 0.08)
        
        # Subtle rib shadows
        for i in range(3, size - 3, 6):
            rib_intensity = np.random.uniform(0.02, 0.05)
            base[i:i+2, :] += rib_intensity
        
        return gaussian_filter(base, sigma=1.0)
    
    def _apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations for training robustness."""
        # Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = rotate(image, angle, reshape=False, mode='constant', cval=0)
        
        # Gaussian noise injection
        noise = np.random.normal(0, self.noise_level, image.shape)
        image += noise
        
        # Random intensity scaling
        scale = np.random.uniform(0.85, 1.15)
        image *= scale
        
        # Random contrast adjustment
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
        """
        Create a Non-IID dataset simulating a hospital's patient mix.
        
        Args:
            n_samples: Total samples to generate
            distribution: Class proportions (must sum to 1.0)
            hospital_id: ID for reproducible seeding
            
        Returns:
            Tuple of (images, labels) numpy arrays
        """
        if not np.isclose(sum(distribution.values()), 1.0, atol=0.01):
            raise ValueError("Distribution must sum to 1.0")
        
        # Reproducible per-hospital randomness
        np.random.seed(hospital_id * 1000 + 42)
        
        images: List[np.ndarray] = []
        labels: List[int] = []
        
        # Calculate samples per class
        remaining = n_samples
        samples_per_class: Dict[int, int] = {}
        
        for i, (label, proportion) in enumerate(distribution.items()):
            if i == len(distribution) - 1:
                samples_per_class[label] = remaining
            else:
                count = int(n_samples * proportion)
                samples_per_class[label] = count
                remaining -= count
        
        # Generate samples with augmentation
        for label, count in samples_per_class.items():
            for _ in range(count):
                image = self.generate_synthetic_xray(label, apply_augmentation=True)
                images.append(image)
                labels.append(label)
        
        images_arr = np.array(images)
        labels_arr = np.array(labels)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(labels_arr))
        return images_arr[shuffle_idx], labels_arr[shuffle_idx]


class XRayDataset(Dataset):
    """PyTorch Dataset for X-Ray images with proper type hints."""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.FloatTensor(images).unsqueeze(1)  # (N, 1, H, W)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def create_global_test_set(
    n_samples: int = 300,
    seed: int = 9999
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a GLOBAL HOLD-OUT TEST SET that no client ever sees.
    
    This is critical for unbiased evaluation in federated learning.
    The global test set has balanced classes and no augmentation noise.
    
    Args:
        n_samples: Total samples (will be balanced across classes)
        seed: Fixed seed for reproducibility
        
    Returns:
        Tuple of (images_tensor, labels_tensor)
    """
    generator = MedicalDataGenerator(seed=seed)
    
    samples_per_class = n_samples // MedicalDataGenerator.NUM_CLASSES
    
    images: List[np.ndarray] = []
    labels: List[int] = []
    
    for label in range(MedicalDataGenerator.NUM_CLASSES):
        for _ in range(samples_per_class):
            # NO augmentation for test set - clean evaluation
            image = generator.generate_synthetic_xray(label, apply_augmentation=False)
            images.append(image)
            labels.append(label)
    
    images_arr = np.array(images)
    labels_arr = np.array(labels)
    
    # Shuffle with fixed seed
    np.random.seed(seed)
    shuffle_idx = np.random.permutation(len(labels_arr))
    images_arr = images_arr[shuffle_idx]
    labels_arr = labels_arr[shuffle_idx]
    
    # Convert to tensors
    images_tensor = torch.FloatTensor(images_arr).unsqueeze(1)
    labels_tensor = torch.LongTensor(labels_arr)
    
    return images_tensor, labels_tensor


def create_hospital_dataloaders(
    n_hospitals: int,
    samples_per_hospital: int,
    batch_size: int = 32
) -> List[DataLoader]:
    """
    Create DataLoaders for multiple hospitals with Non-IID distributions.
    
    Args:
        n_hospitals: Number of hospital clients
        samples_per_hospital: Samples per hospital
        batch_size: Training batch size
        
    Returns:
        List of DataLoaders
    """
    dataloaders: List[DataLoader] = []
    
    # Diverse Non-IID distributions
    base_distributions = [
        {0: 0.70, 1: 0.15, 2: 0.15},  # Mostly healthy
        {0: 0.10, 1: 0.70, 2: 0.20},  # Pneumonia specialist
        {0: 0.10, 1: 0.20, 2: 0.70},  # COVID hotspot
        {0: 0.33, 1: 0.34, 2: 0.33},  # Balanced
        {0: 0.20, 1: 0.50, 2: 0.30},  # Mixed respiratory
        {0: 0.50, 1: 0.25, 2: 0.25},  # Screening center
        {0: 0.15, 1: 0.40, 2: 0.45},  # ICU focus
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
