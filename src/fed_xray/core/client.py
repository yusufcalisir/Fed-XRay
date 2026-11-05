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
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.cnn import create_model as create_cnn_model
from ..models.vit import create_medical_vit
from ..models.foundation import (
    UNIFoundationBackbone,
    SwinUNETRRadiologyBackbone,
    FedMedCLIPModel,
    FedColaModalityBridge,
)
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
from .privacy import (
    SecAggPeer,
    SecAggPlusProtocol,
    clip_and_perturb_state_dict,
    CKKSEncryptionEngine,
    EncryptedTensor,
    ClientAttestation,
    HospitalAuthorization,
    HospitalThresholdEnclave,
    PatientLevelDPAccountant,
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
        img_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        pretrained: bool = True,
        foundation_checkpoint: Optional[str] = None,
        dp_enabled: bool = False,
        dp_clipping_bound: float = 1.0,
        dp_noise_multiplier: float = 0.0,
        dp_mode: str = "distributed",  # "distributed", "local", "central", "off"
        strict_checkpoint: bool = False,
        secagg_peer: Optional[SecAggPeer] = None,
        threshold_enclave: Optional[HospitalThresholdEnclave] = None,
        num_classes: int = 3,
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
        self.pretrained = pretrained
        self.foundation_checkpoint = foundation_checkpoint
        self.dp_enabled = dp_enabled
        self.dp_clipping_bound = dp_clipping_bound
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_mode = dp_mode
        self.strict_checkpoint = strict_checkpoint
        self.num_classes = num_classes

        if img_size is not None:
            self.img_size = img_size
        else:
            try:
                first_batch = next(iter(dataloader))[0]
                self.img_size = int(first_batch.shape[-1])
            except Exception:
                self.img_size = 28
        self.patch_size = patch_size or (16 if self.img_size == 224 else 4)

        # Initialize local architecture
        self.model = self._create_local_model().to(self.device)

        # SCAFFOLD client control variate
        self.c_client: Optional[Dict[str, torch.Tensor]] = None
        # FedDyn previous gradient tracker
        self.grad_prev: Optional[Dict[str, torch.Tensor]] = None
        # MOON previous local model state
        self.prev_local_model: Optional[nn.Module] = None

        ds = getattr(dataloader, "dataset", None) if dataloader else None
        self.n_samples = len(ds) if ds is not None and hasattr(ds, "__len__") else 0
        self.class_counts = self._compute_class_counts()

        # Decentralized cryptographic peer enclave for SecAgg+ (Option J standard)
        self.secagg_peer = secagg_peer if secagg_peer is not None else SecAggPeer(client_id=self.client_id)
        # Decentralized threshold enclave for Option J collective decryption
        self.threshold_enclave = threshold_enclave
        # Client-side CKKS homomorphic encryption engine (public context)
        self.ckks_engine: Optional[CKKSEncryptionEngine] = None
        self.cola_bridge: Optional[FedColaModalityBridge] = None
        self.text_anchor: Optional[Any] = None
        # Client-specific Patient-Level Rényi Differential Privacy Accountant
        self.dp_accountant = PatientLevelDPAccountant(target_delta=1e-5)

    def get_spent_epsilon(self, delta: Optional[float] = None) -> float:
        """Return the authentic client-level Rényi Differential Privacy composed epsilon."""
        return self.dp_accountant.get_epsilon(delta=delta)

    def _create_local_model(self) -> nn.Module:
        """Instantiate backbone and inject PEFT adapters if requested."""
        model: nn.Module
        if self.model_type.startswith("vit"):
            model = create_medical_vit(
                model_type=self.model_type,
                img_size=self.img_size,
                patch_size=self.patch_size,
                num_classes=self.num_classes,
                pretrained=self.pretrained,
                checkpoint_path=self.foundation_checkpoint,
                strict_checkpoint=self.strict_checkpoint,
            )
            if self.peft_mode:
                inject_lora_to_model(
                    model=model,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    mode=self.peft_mode,
                    deep_layers_only=self.deep_layers_only,
                )
            return model
        elif self.model_type == "uni":
            model = UNIFoundationBackbone(
                embed_dim=512,
                num_classes=self.num_classes,
                patch_size=self.patch_size,
                pretrained=self.pretrained,
                checkpoint_path=self.foundation_checkpoint,
                strict_checkpoint=self.strict_checkpoint,
            )
            if self.peft_mode:
                inject_lora_to_model(
                    model=model,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    mode=self.peft_mode,
                    deep_layers_only=self.deep_layers_only,
                )
            return model
        elif self.model_type in ("swin_unetr", "swin"):
            model = SwinUNETRRadiologyBackbone(
                embed_dim=384,
                num_classes=self.num_classes,
                pretrained=self.pretrained,
                checkpoint_path=self.foundation_checkpoint,
                strict_checkpoint=self.strict_checkpoint,
            )
            if self.peft_mode:
                inject_lora_to_model(
                    model=model,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    mode=self.peft_mode,
                    deep_layers_only=self.deep_layers_only,
                )
            return model
        elif self.model_type in ("fedmedclip", "medclip"):
            model = FedMedCLIPModel(
                backbone_type="vit_tiny",
                img_size=self.img_size,
                patch_size=self.patch_size,
                num_classes=self.num_classes,
                pretrained=self.pretrained,
                checkpoint_path=self.foundation_checkpoint,
                strict_checkpoint=self.strict_checkpoint,
                device=self.device,
            )
            if self.peft_mode:
                inject_lora_to_model(
                    model=model.backbone,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    mode=self.peft_mode,
                    deep_layers_only=self.deep_layers_only,
                )
            return model
        return create_cnn_model(num_classes=self.num_classes)

    def get_model_provenance(self) -> Dict[str, Any]:
        """Audit report of client model weight provenance and authenticity (MOD-02)."""
        m = self.model
        base = getattr(m, "base_model", m)
        fn = getattr(base, "get_model_provenance", None)
        if callable(fn):
            prov_dict: Dict[str, Any] = fn()
            prov_dict["client_id"] = self.client_id
            prov_dict["peft_mode"] = self.peft_mode
            prov_dict["model_type"] = self.model_type
            return prov_dict
        prov = getattr(base, "checkpoint_provenance", getattr(m, "checkpoint_provenance", "uninitialized_scratch_weights"))
        return {
            "client_id": self.client_id,
            "model_type": self.model_type,
            "is_pretrained": getattr(base, "is_pretrained", getattr(m, "is_pretrained", False)),
            "checkpoint_provenance": prov,
            "checkpoint_hash": getattr(base, "checkpoint_hash", getattr(m, "checkpoint_hash", None)),
            "strict_checkpoint": self.strict_checkpoint,
            "peft_mode": self.peft_mode,
            "is_fallback_distilled": "torchvision" in str(prov),
            "last_load_error": getattr(base, "last_load_error", getattr(m, "last_load_error", None)),
        }

    def _compute_class_counts(self) -> List[int]:
        """Compute sample count per class from local dataloader."""
        num_c = getattr(self, "num_classes", 3)
        counts = [0] * num_c
        if self.dataloader and hasattr(self.dataloader, "dataset") and self.dataloader.dataset is not None:
            for _, labels in self.dataloader:
                for y in labels:
                    c = y.item()
                    if c < num_c:
                        counts[c] += 1
        for i in range(len(counts)):
            if counts[i] == 0:
                counts[i] = 1
        return counts

    def get_num_samples(self) -> int:
        return self.n_samples

    def is_malicious(self) -> bool:
        return self.malicious

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Extract client's trainable weights or PEFT adapter state dict."""
        if self.peft_mode:
            return extract_peft_state_dict(self.model, mode=self.peft_mode)
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def get_secagg_public_key(self) -> bytes:
        """Export client's 32-byte raw X25519 public key for consortium key exchange."""
        return self.secagg_peer.public_key_bytes

    def generate_secagg_mask(
        self,
        peer_public_keys: Dict[int, bytes],
        weights_map: Optional[Dict[int, float]] = None,
        tensor_shapes: Optional[Dict[str, torch.Size]] = None,
        round_num: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Locally synthesize decentralized zero-sum additive mask using ECDH shared secrets."""
        if tensor_shapes is None:
            tensor_shapes = {k: v.shape for k, v in self.get_weights().items()}
        return self.secagg_peer.compute_mask(
            peer_public_keys=peer_public_keys,
            tensor_shapes=tensor_shapes,
            weights_map=weights_map,
            round_num=round_num,
            device=self.device,
        )

    def mask_local_weights(
        self,
        weights: Dict[str, torch.Tensor],
        peer_public_keys: Dict[int, bytes],
        weights_map: Optional[Dict[int, float]] = None,
        round_num: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Apply locally synthesized ECDH zero-sum mask to client's candidate weights."""
        tensor_shapes = {k: v.shape for k, v in weights.items()}
        mask = self.generate_secagg_mask(
            peer_public_keys=peer_public_keys,
            weights_map=weights_map,
            tensor_shapes=tensor_shapes,
            round_num=round_num,
        )
        return SecAggPlusProtocol.mask_state_dict(weights, mask)

    def init_ckks(self, public_context_bytes: bytes) -> None:
        """Initialize hospital client with consortium public CKKS context for in-transit encryption."""
        self.ckks_engine = CKKSEncryptionEngine.from_public_bytes(public_context_bytes)

    def encrypt_local_weights(
        self, weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, EncryptedTensor]:
        """Encrypt local PEFT adapter or model weights into CKKS ciphertexts before transmission."""
        if self.ckks_engine is None:
            raise RuntimeError("Client CKKS engine is not initialized. Call init_ckks first.")
        w = weights if weights is not None else self.get_weights()
        return self.ckks_engine.encrypt_state_dict(w)

    def sign_threshold_authorization(
        self, round_num: int, ciphertext_digest: str
    ) -> HospitalAuthorization:
        """Issue an authentic signed threshold partial decryption authorization."""
        if self.threshold_enclave is None:
            raise RuntimeError(f"Client {self.client_id} does not possess a threshold enclave.")
        return self.threshold_enclave.sign_partial_decryption_authorization(round_num, ciphertext_digest)

    def get_cryptographic_audit_report(self) -> Dict[str, Any]:
        """Generate audit report of client-side cryptographic enclaves and key commitments."""
        report: Dict[str, Any] = {
            "client_id": self.client_id,
            "secagg_peer": self.secagg_peer.get_peer_provenance(),
            "ckks_initialized": self.ckks_engine is not None,
            "ckks_provenance": self.ckks_engine.get_encryption_provenance() if self.ckks_engine is not None else None,
            "has_threshold_enclave": self.threshold_enclave is not None,
            "threshold_commitment": (
                getattr(self.threshold_enclave.share, "commitment", None)
                if self.threshold_enclave is not None
                else None
            ),
        }
        return report

    def evaluate_local_model(
        self,
        test_images: torch.Tensor,
        test_labels: torch.Tensor,
        batch_size: int = 64,
    ) -> float:
        """Evaluate local client model on holdout test data within hospital perimeter."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            n_samples = test_images.size(0)
            for i in range(0, n_samples, batch_size):
                batch_images = test_images[i : i + batch_size].to(self.device)
                batch_labels = test_labels[i : i + batch_size].to(self.device)

                if (
                    batch_images.shape[-1] != self.img_size
                    or batch_images.shape[-2] != self.img_size
                ):
                    batch_images = F.interpolate(
                        batch_images,
                        size=(self.img_size, self.img_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                outputs = self.model(batch_images)
                if torch.isnan(outputs).any():
                    return 0.0

                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_labels).sum().item()
                total += batch_labels.size(0)

        return correct / max(total, 1)

    def _compute_local_dataloader_accuracy(self, max_batches: int = 4) -> float:
        """Compute authentic evaluation accuracy across local dataloader batches."""
        if not self.dataloader:
            return 0.0
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for b_idx, batch in enumerate(self.dataloader):
                if b_idx >= max_batches:
                    break
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                if (
                    images.shape[-1] != self.img_size
                    or images.shape[-2] != self.img_size
                ):
                    images = F.interpolate(
                        images,
                        size=(self.img_size, self.img_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                outputs = self.model(images)
                if torch.isnan(outputs).any():
                    return 0.0
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        return correct / max(total, 1)

    def generate_attestation(
        self,
        test_images: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> ClientAttestation:
        """Emit authentic enclave attestation with local accuracy and update L2 norm."""
        w = weights if weights is not None else self.get_weights()
        update_norm = sum(torch.norm(t.float()).item() for t in w.values())

        if test_images is not None and test_labels is not None:
            val_acc = self.evaluate_local_model(test_images, test_labels)
        elif self.dataloader is not None:
            val_acc = self._compute_local_dataloader_accuracy()
        else:
            val_acc = 0.0

        return ClientAttestation(
            client_id=self.client_id,
            validation_accuracy=val_acc,
            update_l2_norm=update_norm,
            public_key_bytes=self.get_secagg_public_key(),
            is_malicious=False,
            sample_count=self.n_samples,
        )

    def _flip_labels(self, labels: torch.Tensor) -> torch.Tensor:
        flipped = labels.clone()
        num_c = getattr(self, "num_classes", 3)
        flip_map = {c: (c + 1) % num_c for c in range(num_c)}
        for old_label, new_label in flip_map.items():
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
        dp_enabled: Optional[bool] = None,
        dp_clipping_bound: Optional[float] = None,
        dp_noise_multiplier: Optional[float] = None,
        dp_mode: Optional[str] = None,
        num_consortium_clients: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], TrainingMetrics, Optional[Dict[str, torch.Tensor]]]:
        """Perform local optimization on private client dataset.

        Returns:
            updated_weights: Transmitted model / adapter state dict.
            metrics: Training telemetry metrics.
            delta_c: Client control variate delta (for SCAFFOLD).
        """
        # Separate cola_bridge weights if present
        cola_weights = {k.replace("cola_bridge.", ""): v for k, v in global_weights.items() if k.startswith("cola_bridge.")}
        model_weights = {k: v for k, v in global_weights.items() if not k.startswith("cola_bridge.")}

        if cola_weights:
            if self.cola_bridge is None:
                feat_dim = getattr(self.model, "embed_dim", 128)
                self.cola_bridge = FedColaModalityBridge(
                    image_dim=feat_dim, text_dim=512, shared_dim=256
                ).to(self.device)
            self.cola_bridge.load_state_dict({k: v.to(self.device) for k, v in cola_weights.items()})

        # Load global model / adapter weights
        if self.peft_mode:
            load_peft_state_dict(self.model, model_weights, mode=self.peft_mode)
        else:
            self.model.load_state_dict(copy.deepcopy(model_weights))
        self.model.train()

        # Global model reference for MOON / FedProx / FedDyn
        global_model_ref = None
        if algorithm.upper() in ("MOON", "FEDPROX", "FEDDYN") or prox_mu > 0 or moon_mu > 0:
            global_model_ref = self._create_local_model().to(self.device)
            global_model_ref.load_state_dict(self.model.state_dict())
            global_model_ref.eval()

        # Optimizer targeting trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if algorithm.upper() == "FEDCOLA" or loss_fn_name.upper() == "FEDCOLA":
            if self.cola_bridge is None:
                feat_dim = getattr(self.model, "embed_dim", 128)
                self.cola_bridge = FedColaModalityBridge(
                    image_dim=feat_dim, text_dim=512, shared_dim=256
                ).to(self.device)
            trainable_params.extend([p for p in self.cola_bridge.parameters() if p.requires_grad])

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
                compute_loss_fn = getattr(self.model, "compute_local_loss", None)
                if callable(compute_loss_fn):
                    loss, y_ens = compute_loss_fn(images, labels, beta=0.5)
                    outputs = torch.log(y_ens + 1e-8)
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                if algorithm.upper() == "FEDCOLA" or loss_fn_name.upper() == "FEDCOLA":
                    if self.cola_bridge is None:
                        feat_dim = getattr(self.model, "embed_dim", 128)
                        self.cola_bridge = FedColaModalityBridge(
                            image_dim=feat_dim, text_dim=512, shared_dim=256
                        ).to(self.device)
                    if self.text_anchor is None:
                        from ..models.foundation import TextSemanticAnchor

                        self.text_anchor = TextSemanticAnchor(
                            dataset_name="isic",
                            embed_dim=512,
                            num_classes=getattr(self, "num_classes", 3),
                            device=self.device,
                        )
                    feats = extract_features(self.model, images)
                    txt_embeds = self.text_anchor(labels)
                    cola_loss = self.cola_bridge.compute_cross_modal_alignment_loss(feats, txt_embeds)
                    loss = loss + 0.1 * cola_loss

                # 1. Prototype Metric Alignment Loss
                if global_prototypes is not None and proto_weight > 0:
                    features = extract_features(self.model, images)
                    p_loss = compute_prototype_distance_loss(features, labels, global_prototypes)
                    r_loss = repel_criterion(features, labels, global_prototypes)
                    loss = loss + proto_weight * (p_loss + 0.1 * r_loss)

                # 2. FedProx Proximal Penalty
                if (algorithm.upper() == "FEDPROX" or prox_mu > 0) and prox_mu > 0:
                    prox_penalty: torch.Tensor = torch.tensor(0.0, device=self.device)
                    for name, param in self.model.named_parameters():
                        if name in initial_params:
                            prox_penalty = prox_penalty + torch.sum((param - initial_params[name]) ** 2)
                    loss = loss + (prox_mu / 2.0) * prox_penalty

                # 3. FedDyn Dynamic Linear & Quadratic Regularization
                if algorithm.upper() == "FEDDYN" and feddyn_alpha > 0:
                    dyn_linear: torch.Tensor = torch.tensor(0.0, device=self.device)
                    dyn_quad: torch.Tensor = torch.tensor(0.0, device=self.device)
                    for name, param in self.model.named_parameters():
                        if name in initial_params:
                            diff = param - initial_params[name]
                            if self.grad_prev and name in self.grad_prev:
                                dyn_linear = dyn_linear - torch.sum(self.grad_prev[name].to(self.device) * param)
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
                correct += int(predicted.eq(labels).sum().item())
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

        # Patient-Level Differential Privacy: sensitivity clipping & calibrated perturbation
        active_dp_mode = dp_mode if dp_mode is not None else self.dp_mode
        active_clipping = dp_clipping_bound if dp_clipping_bound is not None else self.dp_clipping_bound
        active_noise = dp_noise_multiplier if dp_noise_multiplier is not None else self.dp_noise_multiplier
        is_dp_active = (dp_enabled if dp_enabled is not None else self.dp_enabled) or (active_noise > 0.0)

        pre_clip_norm: Optional[float] = None
        if is_dp_active and active_dp_mode != "off":
            deltas: Dict[str, torch.Tensor] = {}
            for name, curr_t in updated_weights.items():
                if name in initial_params:
                    deltas[name] = curr_t.float() - initial_params[name].cpu().float()
                else:
                    deltas[name] = torch.zeros_like(curr_t.float())

            k_clients = num_consortium_clients or 1
            perturbed_deltas, pre_clip_norm = clip_and_perturb_state_dict(
                delta_weights=deltas,
                clipping_bound=active_clipping,
                noise_multiplier=active_noise,
                num_clients=k_clients,
                dp_mode=active_dp_mode,
            )

            for name in updated_weights.keys():
                if name in initial_params and name in perturbed_deltas:
                    orig_dtype = updated_weights[name].dtype
                    updated_weights[name] = (
                        initial_params[name].cpu() + perturbed_deltas[name]
                    ).to(orig_dtype)

            # Record privacy step in client-specific RDP accountant based on authentic local dataset size
            batch_size = getattr(self.dataloader, "batch_size", 32) if self.dataloader else 32
            subsampling_q = min(1.0, float(batch_size) / max(1.0, float(self.n_samples))) if self.n_samples > 0 else 1.0
            self.dp_accountant.step(q=subsampling_q, sigma=active_noise)

        if self.cola_bridge is not None and (algorithm.upper() == "FEDCOLA" or loss_fn_name.upper() == "FEDCOLA"):
            for k, v in self.cola_bridge.state_dict().items():
                updated_weights[f"cola_bridge.{k}"] = v.clone().detach().cpu()

        metrics = TrainingMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            samples_trained=total_samples,
            client_id=self.client_id,
            is_malicious=self.malicious,
            dp_clipped_norm=pre_clip_norm,
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
            num_classes=getattr(self, "num_classes", 3),
        )
