"""Fed-XRay Hospital Client Implementation.

Supports:
- CNN and Vision Transformer (ViT) backbones
- Parameter-Efficient Fine-Tuning (PEFT: LoRA, FFA-LoRA, FedSA-LoRA)
- Drift-Resilient Optimization (FedAvg, FedProx, FedDyn, SCAFFOLD, MOON)
- Prototype Metric Alignment (FedProto + Repel Loss)
- Advanced Imbalance Loss Functions (DAFL, Balanced Softmax, LDAM, Class-Balanced)
"""

from __future__ import annotations
import copy
import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.cnn import create_model as create_cnn_model
from ..models.vit import create_medical_vit, VisionTransformer
from ..models.peft import (
    inject_lora_to_model,
    extract_peft_state_dict,
    load_peft_state_dict,
)
from .metrics import TrainingMetrics
from .imbalance_losses import (
    DynamicAdaptiveFocalLoss,
    BalancedSoftmaxLoss,
    ClassBalancedLoss,
    LDAMLoss,
    PrototypeRepelLoss,
)
from .prototypes import (
    extract_features,
    compute_local_prototypes_and_dispersion,
    compute_prototype_distance_loss,
)


class HospitalClient:
    """Hospital client node for decentralized oncology and medical image computing."""

    LABEL_FLIP_MAP = {0: 1, 1: 2, 2: 0}

    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        local_epochs: int = 1,
        malicious: bool = False,
        model_type: str = "cnn",  # "cnn", "vit_tiny", "vit_small", "vit_base"
        peft_mode: Optional[str] = None,  # None, "lora", "ffa_lora", "fedsa_lora"
        lora_r: int = 16,
        lora_alpha: float = 16.0,
        deep_layers_only: bool = False,
    ) -> None:
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device or torch.device("cpu")
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.malicious = malicious
        self.model_type = model_type
        self.peft_mode = peft_mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.deep_layers_only = deep_layers_only

        # Initialize local architecture
        self.model = self._create_local_model().to(self.device)

        # SCAFFOLD client control variate
        self.c_client: Optional[Dict[str, torch.Tensor]] = None
        # FedDyn previous gradient tracker
        self.grad_prev: Optional[Dict[str, torch.Tensor]] = None
        # MOON previous local model state
        self.prev_local_model: Optional[nn.Module] = None

        self.n_samples = (
            len(dataloader.dataset)
            if dataloader and hasattr(dataloader, "dataset") and dataloader.dataset is not None
            else 0
        )
        self.class_counts = self._compute_class_counts()

    def _create_local_model(self) -> nn.Module:
        """Instantiate backbone and inject PEFT adapters if requested."""
        if self.model_type.startswith("vit"):
            model = create_medical_vit(model_type=self.model_type, num_classes=3)
            if self.peft_mode:
                inject_lora_to_model(
                    model=model,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    mode=self.peft_mode,
                    deep_layers_only=self.deep_layers_only,
                )
            return model
        return create_cnn_model()

    def _compute_class_counts(self) -> List[int]:
        """Compute sample count per class from local dataloader."""
        counts = [0, 0, 0]
        if self.dataloader and hasattr(self.dataloader, "dataset") and self.dataloader.dataset is not None:
            for _, labels in self.dataloader:
                for y in labels:
                    c = y.item()
                    if c < 3:
                        counts[c] += 1
        for i in range(len(counts)):
            if counts[i] == 0:
                counts[i] = 1
        return counts

    def get_num_samples(self) -> int:
        return self.n_samples

    def is_malicious(self) -> bool:
        return self.malicious

    def _flip_labels(self, labels: torch.Tensor) -> torch.Tensor:
        flipped = labels.clone()
        for old_label, new_label in self.LABEL_FLIP_MAP.items():
            flipped[labels == old_label] = new_label
        return flipped

    def _get_loss_function(
        self,
        loss_fn_name: str,
        current_round: int,
        total_rounds: int,
    ) -> nn.Module:
        name = loss_fn_name.upper()
        if name == "DAFL":
            return DynamicAdaptiveFocalLoss(
                class_counts=self.class_counts,
                current_round=current_round,
                total_rounds=total_rounds,
            )
        elif name in ("BALANCED_SOFTMAX", "BSM"):
            return BalancedSoftmaxLoss(class_counts=self.class_counts)
        elif name in ("CLASS_BALANCED", "CB"):
            return ClassBalancedLoss(class_counts=self.class_counts)
        elif name == "LDAM":
            return LDAMLoss(class_counts=self.class_counts)
        return nn.CrossEntropyLoss()

    def train(
        self,
        global_weights: Dict[str, torch.Tensor],
        loss_fn_name: str = "CE",
        current_round: int = 1,
        total_rounds: int = 10,
        global_prototypes: Optional[Dict[int, torch.Tensor]] = None,
        proto_weight: float = 0.0,
        algorithm: str = "FedAvg",
        prox_mu: float = 0.0,
        feddyn_alpha: float = 0.01,
        c_global: Optional[Dict[str, torch.Tensor]] = None,
        moon_mu: float = 0.0,
        moon_temp: float = 0.5,
    ) -> Tuple[Dict[str, torch.Tensor], TrainingMetrics, Optional[Dict[str, torch.Tensor]]]:
        """Perform local optimization on private client dataset.
        
        Returns:
            updated_weights: Transmitted model / adapter state dict.
            metrics: Training telemetry metrics.
            delta_c: Client control variate delta (for SCAFFOLD).
        """
        # Load global model / adapter weights
        if self.peft_mode:
            load_peft_state_dict(self.model, global_weights, mode=self.peft_mode)
        else:
            self.model.load_state_dict(copy.deepcopy(global_weights))
        self.model.train()

        # Global model reference for MOON / FedProx / FedDyn
        global_model_ref = None
        if algorithm.upper() in ("MOON", "FEDPROX", "FEDDYN") or prox_mu > 0 or moon_mu > 0:
            global_model_ref = self._create_local_model().to(self.device)
            global_model_ref.load_state_dict(self.model.state_dict())
            global_model_ref.eval()

        # Optimizer targeting trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.learning_rate)
        criterion = self._get_loss_function(loss_fn_name, current_round, total_rounds)
        repel_criterion = PrototypeRepelLoss(margin=1.0)

        # Initial weight snapshot
        initial_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Initialize SCAFFOLD control variate if needed
        if algorithm.upper() == "SCAFFOLD" and self.c_client is None:
            self.c_client = {
                name: torch.zeros_like(param)
                for name, param in initial_params.items()
            }

        total_loss = 0.0
        correct = 0
        total_samples = 0

        for epoch in range(self.local_epochs):
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.malicious:
                    labels = self._flip_labels(labels)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # 1. Prototype Metric Alignment Loss
                if global_prototypes is not None and proto_weight > 0:
                    features = extract_features(self.model, images)
                    p_loss = compute_prototype_distance_loss(features, labels, global_prototypes)
                    r_loss = repel_criterion(features, labels, global_prototypes)
                    loss = loss + proto_weight * (p_loss + 0.1 * r_loss)

                # 2. FedProx Proximal Penalty
                if (algorithm.upper() == "FEDPROX" or prox_mu > 0) and prox_mu > 0:
                    prox_penalty = 0.0
                    for name, param in self.model.named_parameters():
                        if name in initial_params:
                            prox_penalty = prox_penalty + torch.sum((param - initial_params[name]) ** 2)
                    loss = loss + (prox_mu / 2.0) * prox_penalty

                # 3. FedDyn Dynamic Linear & Quadratic Regularization
                if algorithm.upper() == "FEDDYN" and feddyn_alpha > 0:
                    dyn_linear = 0.0
                    dyn_quad = 0.0
                    for name, param in self.model.named_parameters():
                        if name in initial_params:
                            diff = param - initial_params[name]
                            if self.grad_prev and name in self.grad_prev:
                                dyn_linear = dyn_linear - torch.sum(self.grad_prev[name] * param)
                            dyn_quad = dyn_quad + torch.sum(diff ** 2)
                    loss = loss + dyn_linear + (feddyn_alpha / 2.0) * dyn_quad

                # 4. MOON Model-Contrastive Representation Loss
                if algorithm.upper() == "MOON" and moon_mu > 0 and global_model_ref is not None:
                    z_curr = extract_features(self.model, images)
                    with torch.no_grad():
                        z_glob = extract_features(global_model_ref, images)
                        z_prev = extract_features(self.prev_local_model, images) if self.prev_local_model else z_glob

                    cos = nn.CosineSimilarity(dim=-1)
                    sim_glob = cos(z_curr, z_glob) / moon_temp
                    sim_prev = cos(z_curr, z_prev) / moon_temp
                    # Contrastive loss: maximize sim(curr, glob), minimize sim(curr, prev)
                    moon_loss = -torch.mean(torch.log(torch.exp(sim_glob) / (torch.exp(sim_glob) + torch.exp(sim_prev) + 1e-8)))
                    loss = loss + moon_mu * moon_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()

                # 5. SCAFFOLD Control Variates Gradient Correction
                if algorithm.upper() == "SCAFFOLD" and c_global is not None and self.c_client is not None:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None and name in self.c_client:
                            c_k = self.c_client[name].to(self.device)
                            c_g = c_global.get(name, torch.zeros_like(c_k)).to(self.device)
                            param.grad.data.add_(c_g - c_k)

                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total_samples += images.size(0)

        # Update SCAFFOLD client control variates
        delta_c = None
        if algorithm.upper() == "SCAFFOLD" and c_global is not None and self.c_client is not None:
            delta_c = {}
            new_c_client = {}
            total_steps = max(1, self.local_epochs * len(self.dataloader))
            step_scale = 1.0 / (total_steps * self.learning_rate)
            for name, param in self.model.named_parameters():
                if name in initial_params:
                    c_g = c_global.get(name, torch.zeros_like(param)).to(self.device)
                    c_k = self.c_client[name].to(self.device)
                    # c_k^{t+1} = c_k^t - c^t + 1/(K*eta) (theta_init - theta_curr)
                    c_new = c_k - c_g + step_scale * (initial_params[name] - param.data)
                    delta_c[name] = (c_new - c_k).detach().cpu()
                    new_c_client[name] = c_new.detach().cpu()
            self.c_client = new_c_client

        # Update FedDyn gradient tracker
        if algorithm.upper() == "FEDDYN" and feddyn_alpha > 0:
            if self.grad_prev is None:
                self.grad_prev = {}
            for name, param in self.model.named_parameters():
                if name in initial_params:
                    prev = self.grad_prev.get(name, torch.zeros_like(param)).to(self.device)
                    self.grad_prev[name] = (prev - feddyn_alpha * (param.data - initial_params[name])).detach().cpu()

        # Update MOON previous local model
        if algorithm.upper() == "MOON":
            self.prev_local_model = self._create_local_model().to(self.device)
            self.prev_local_model.load_state_dict(self.model.state_dict())
            self.prev_local_model.eval()

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)

        # Extract weights for transmission
        if self.peft_mode:
            updated_weights = extract_peft_state_dict(self.model, mode=self.peft_mode)
        else:
            updated_weights = {
                name: param.clone().detach().cpu()
                for name, param in self.model.state_dict().items()
            }

        metrics = TrainingMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            samples_trained=total_samples,
            client_id=self.client_id,
            is_malicious=self.malicious,
        )

        return updated_weights, metrics, delta_c

    def compute_prototypes_and_dispersion(
        self,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, float], Dict[int, int]]:
        """Extract empirical class prototypes and covariance traces."""
        return compute_local_prototypes_and_dispersion(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            num_classes=3,
        )
