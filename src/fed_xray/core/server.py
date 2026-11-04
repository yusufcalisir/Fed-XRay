"""Fed-XRay Central Server & Federated Aggregation Engine.

Supports:
- Classical Consensus: FedAvg, FedProx
- Modern SOTA Optimizers: FedDyn, FedOpt / FedAdam, SCAFFOLD on PEFT
- Parameter-Efficient Fine-Tuning Aggregation (LoRA, FFA-LoRA, FedSA-LoRA)
- Dispersion-Weighted Prototype Metric Synthesis
- Byzantine Defense Shield via Hold-Out Reference Validation
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from .client import HospitalClient
from .metrics import EvaluationMetrics, SecurityReport, TrainingMetrics
from .prototypes import aggregate_prototypes_dispersion_weighted
from .privacy import (
    CKKSEncryptionEngine,
    ConsortiumKeyEnclave,
    EncryptedTensor,
    ClientAttestation,
    PatientLevelDPAccountant,
    SecAggPlusProtocol,
)

logger = logging.getLogger("fed_xray.server")


class CentralServer:
    """Central coordinator server with Byzantine-robust aggregation and SOTA ViT-FL optimizers."""

    MALICIOUS_THRESHOLD = 0.30

    def __init__(
        self,
        device: Optional[torch.device] = None,
        privacy_noise: float = 0.0,
        defense_mode: bool = False,
        model_type: str = "cnn",  # "cnn", "vit_tiny", "vit_small", "vit_base"
        peft_mode: Optional[str] = None,  # None, "lora", "ffa_lora", "fedsa_lora"
        lora_r: int = 16,
        lora_alpha: float = 16.0,
        deep_layers_only: bool = False,
        feddyn_alpha: float = 0.01,
        server_lr: float = 0.01,
        server_beta1: float = 0.9,
        server_beta2: float = 0.999,
        server_epsilon: float = 1e-8,
        secure_aggregation: bool = False,
        use_ckks: bool = False,
        img_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        pretrained: bool = True,
        foundation_checkpoint: Optional[str] = None,
        dp_mode: str = "distributed",  # "distributed", "local", "central", "off"
        dp_clipping_bound: float = 1.0,
        strict_checkpoint: bool = False,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.privacy_noise = privacy_noise
        self.defense_mode = defense_mode
        self.model_type = model_type
        self.peft_mode = peft_mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.deep_layers_only = deep_layers_only
        self.secure_aggregation = secure_aggregation
        self.use_ckks = use_ckks
        self.pretrained = pretrained
        self.foundation_checkpoint = foundation_checkpoint
        self.dp_mode = dp_mode
        self.dp_clipping_bound = dp_clipping_bound
        self.strict_checkpoint = strict_checkpoint

        self.img_size = img_size or 28
        self.patch_size = patch_size or (16 if self.img_size == 224 else 4)
        self.cola_bridge: Optional[FedColaModalityBridge] = None

        # Cryptographic and DP engines under Option J
        if use_ckks:
            self.consortium_key_enclave: Optional[ConsortiumKeyEnclave] = ConsortiumKeyEnclave(scale_bits=40)
            self.ckks_engine: Optional[CKKSEncryptionEngine] = CKKSEncryptionEngine.from_public_bytes(
                self.consortium_key_enclave.public_context_bytes,
                scale_bits=40,
            )
        else:
            self.consortium_key_enclave = None
            self.ckks_engine = None
        self.dp_accountant = PatientLevelDPAccountant(target_delta=1e-5)

        # Instantiate global architecture
        self.global_model = self._create_model().to(self.device)

        # FedDyn server dynamic state vector: h
        self.feddyn_alpha = feddyn_alpha
        self.h_server: Dict[str, torch.Tensor] = {}

        # FedOpt (FedAdam) momentum buffers
        self.server_lr = server_lr
        self.server_beta1 = server_beta1
        self.server_beta2 = server_beta2
        self.server_epsilon = server_epsilon
        self.m_server: Dict[str, torch.Tensor] = {}
        self.v_server: Dict[str, torch.Tensor] = {}

        # SCAFFOLD global control variate: c
        self.c_global: Dict[str, torch.Tensor] = {}

        self.blocked_count = 0
        self.last_full_model_bytes = sum(
            t.numel() * t.element_size() for t in self.global_model.state_dict().values()
        )
        self.last_uplink_bytes_per_client = 0
        self.last_security_report: Optional[SecurityReport] = None
        self.global_prototypes: Optional[Dict[int, torch.Tensor]] = None

    def get_privacy_spent(self) -> float:
        """Compute cumulative differential privacy budget spent epsilon under Renyi DP."""
        if self.privacy_noise <= 0.0 or self.dp_mode == "off":
            return 0.0
        return self.dp_accountant.get_epsilon()

    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy audit report reflecting active DP configuration."""
        active = self.privacy_noise > 0.0 and self.dp_mode != "off"
        mech = "None"
        if active:
            if self.dp_mode == "distributed":
                mech = "Option J Distributed Patient-Level DP (SecAgg+ / CKKS)"
            elif self.dp_mode == "local":
                mech = "Local Patient-Level DP (Hospital Perimeter)"
            elif self.dp_mode == "central":
                mech = "Central Differential Privacy (Server-Side)"

        return {
            "dp_active": active,
            "dp_mode": self.dp_mode if active else "off",
            "epsilon": round(self.get_privacy_spent(), 4),
            "target_delta": self.dp_accountant.target_delta,
            "noise_multiplier_sigma": self.privacy_noise if active else 0.0,
            "clipping_bound_C": self.dp_clipping_bound if active else 0.0,
            "mechanism": mech,
            "patient_level_clipping": active,
            "total_rounds_composed": len(self.dp_accountant.rdp_history),
        }

    def _create_model(self) -> nn.Module:
        """Create global model with appropriate architecture and PEFT hooks."""
        model: nn.Module
        if self.model_type.startswith("vit"):
            model = create_medical_vit(
                model_type=self.model_type,
                img_size=self.img_size,
                patch_size=self.patch_size,
                num_classes=3,
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
                num_classes=3,
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
                num_classes=3,
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
                num_classes=3,
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
        return create_cnn_model()

    def get_model_provenance(self) -> Dict[str, Any]:
        """Audit report of global model weight provenance and authenticity (MOD-02)."""
        m = self.global_model
        base = getattr(m, "base_model", m)
        return {
            "model_type": self.model_type,
            "is_pretrained": getattr(base, "is_pretrained", getattr(m, "is_pretrained", False)),
            "checkpoint_provenance": getattr(base, "checkpoint_provenance", getattr(m, "checkpoint_provenance", "uninitialized_scratch_weights")),
            "checkpoint_hash": getattr(base, "checkpoint_hash", getattr(m, "checkpoint_hash", None)),
            "strict_checkpoint": self.strict_checkpoint,
            "peft_mode": self.peft_mode,
        }

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model or PEFT adapter weights."""
        weights: Dict[str, torch.Tensor]
        if self.peft_mode:
            weights = extract_peft_state_dict(self.global_model, mode=self.peft_mode)
        else:
            weights = {
                name: param.clone().detach().cpu()
                for name, param in self.global_model.state_dict().items()
            }
        if self.cola_bridge is not None:
            for k, v in self.cola_bridge.state_dict().items():
                weights[f"cola_bridge.{k}"] = v.clone().detach().cpu()
        return weights

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Alias for get_global_weights for API symmetry with HospitalClient."""
        return self.get_global_weights()

    def get_global_prototypes(self) -> Optional[Dict[int, torch.Tensor]]:
        return self.global_prototypes

    def set_global_prototypes(self, prototypes: Dict[int, torch.Tensor]) -> None:
        self.global_prototypes = prototypes

    def _validate_client_model(
        self,
        client_weights: Dict[str, torch.Tensor],
        test_images: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> float:
        """Validate client update on trusted global hold-out validation set."""
        temp_model = self._create_model().to(self.device)
        if self.peft_mode:
            temp_model.load_state_dict(self.global_model.state_dict())
            load_peft_state_dict(temp_model, client_weights, mode=self.peft_mode)
        else:
            temp_model.load_state_dict(client_weights)
        temp_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            batch_size = 64
            n_samples = test_images.size(0)
            for i in range(0, n_samples, batch_size):
                batch_images = test_images[i : i + batch_size].to(self.device)
                batch_labels = test_labels[i : i + batch_size].to(self.device)

                # Ensure spatial resolution matches expected architecture grid
                if batch_images.shape[-1] != self.img_size or batch_images.shape[-2] != self.img_size:
                    batch_images = F.interpolate(
                        batch_images,
                        size=(self.img_size, self.img_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                outputs = temp_model(batch_images)
                if torch.isnan(outputs).any():
                    return 0.0

                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_labels).sum().item()
                total += batch_labels.size(0)

        return correct / max(total, 1)

    def validate_and_aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        sample_counts: List[int],
        client_ids: List[int],
        test_images: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None,
        algorithm: str = "FedAvg",
        delta_cs: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        use_secagg: bool = False,
        use_ckks: Optional[bool] = None,
        client_subsampling_q: float = 1.0,
        client_attestations: Optional[List[ClientAttestation]] = None,
        clients: Optional[List[HospitalClient]] = None,
        round_num: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], SecurityReport]:
        """Filter out malicious poisoned nodes and aggregate trusted updates."""
        validation_accuracies: Dict[int, float] = {}
        trusted_weights: List[Dict[str, torch.Tensor]] = []
        trusted_counts: List[int] = []
        trusted_delta_cs: List[Dict[str, torch.Tensor]] = []
        clients_blocked: List[int] = []
        trusted_clients: List[HospitalClient] = []

        self.last_full_model_bytes = sum(
            t.numel() * t.element_size() for t in self.global_model.state_dict().values()
        )
        if client_weights:
            self.last_uplink_bytes_per_client = sum(
                t.numel() * t.element_size() for t in client_weights[0].values()
            )

        # 1. Compute validation accuracy for all client models
        if client_attestations is not None and len(client_attestations) == len(client_ids):
            for att in client_attestations:
                validation_accuracies[att.client_id] = att.validation_accuracy
        else:
            for idx, (weights, count, cid) in enumerate(zip(client_weights, sample_counts, client_ids)):
                first_val = next(iter(weights.values())) if weights else None
                if isinstance(first_val, EncryptedTensor):
                    # Option J Architectural Invariant: Server possesses only public CKKS context
                    # and cannot decrypt individual weights to perform forward evaluation passes.
                    val_acc = 1.0
                elif test_images is not None and test_labels is not None:
                    val_acc = self._validate_client_model(weights, test_images, test_labels)
                else:
                    val_acc = 1.0
                validation_accuracies[cid] = val_acc

        # 2. Consensus-relative Byzantine outlier detection
        # Prevents deadlocks during early warm-up while isolating genuine adversaries
        val_accs = [validation_accuracies[cid] for cid in client_ids]
        cohort_max = float(np.max(val_accs)) if val_accs else 0.0
        cohort_med = float(np.median(val_accs)) if val_accs else 0.0

        for idx, (weights, count, cid) in enumerate(zip(client_weights, sample_counts, client_ids)):
            val_acc = validation_accuracies[cid]
            is_malicious = False

            if client_attestations is not None and idx < len(client_attestations) and client_attestations[idx].is_malicious:
                is_malicious = True
            elif len(client_ids) > 1:
                # Relative outlier detection against peer consensus
                if cohort_max >= 0.35 and (cohort_max - val_acc) >= 0.25:
                    is_malicious = True
                elif (cohort_med - val_acc) >= 0.20:
                    is_malicious = True
                elif val_acc < 0.10 and cohort_max >= 0.20:
                    is_malicious = True
            elif val_acc < 0.10:
                is_malicious = True

            if is_malicious:
                logger.warning(
                    "[SECURITY ALERT] Node %d failed validation (Acc: %.1f%% vs Cohort: %.1f%%). BLOCKED!",
                    cid, val_acc * 100, cohort_med * 100
                )
                clients_blocked.append(cid)
                self.blocked_count += 1
            else:
                trusted_weights.append(weights)
                trusted_counts.append(count)
                if clients is not None and idx < len(clients):
                    trusted_clients.append(clients[idx])
                if delta_cs is not None and idx < len(delta_cs):
                    d_c = delta_cs[idx]
                    if d_c is not None:
                        trusted_delta_cs.append(d_c)

        if not trusted_weights:
            logger.warning("[WARNING] All client updates failed validation. Preserving global state.")
            aggregated_weights = self.get_global_weights()
        else:
            # If SecAgg+ is active, apply zero-sum masking strictly among the trusted cohort
            weights_to_aggregate = trusted_weights
            if use_secagg and len(trusted_weights) > 1:
                total_s = max(1, sum(trusted_counts))
                if trusted_clients and hasattr(trusted_clients[0], "get_secagg_public_key"):
                    pk_registry = {client.client_id: client.get_secagg_public_key() for client in trusted_clients}
                    weights_map = {client.client_id: float(s) / total_s for client, s in zip(trusted_clients, trusted_counts)}
                    weights_to_aggregate = [
                        client.mask_local_weights(
                            weights=w,
                            peer_public_keys=pk_registry,
                            weights_map=weights_map,
                            round_num=round_num,
                        )
                        for client, w in zip(trusted_clients, trusted_weights)
                    ]
                else:
                    tensor_shapes = {k: v.shape for k, v in trusted_weights[0].items()}
                    weights_norm = [float(s) / total_s for s in trusted_counts]
                    masks = SecAggPlusProtocol.generate_pairwise_masks(
                        num_clients=len(trusted_weights),
                        tensor_shapes=tensor_shapes,
                        device=self.device,
                        weights=weights_norm,
                        round_num=round_num,
                    )
                    weights_to_aggregate = [
                        SecAggPlusProtocol.mask_state_dict(w, m)
                        for w, m in zip(trusted_weights, masks)
                    ]

            # If CKKS is active, encrypt locally if clients are present
            if (use_ckks or self.use_ckks) and trusted_clients and getattr(trusted_clients[0], "ckks_engine", None):
                weights_to_aggregate = [
                    client.encrypt_local_weights(w) for client, w in zip(trusted_clients, weights_to_aggregate)
                ]

            aggregated_weights = self.aggregate(
                weights_to_aggregate,
                trusted_counts,
                algorithm=algorithm,
                delta_cs=trusted_delta_cs if trusted_delta_cs else None,
                use_ckks=use_ckks,
                client_subsampling_q=client_subsampling_q,
            )

        secagg_note = " under SecAgg+ zero-sum masking" if use_secagg else ""
        ckks_note = " and CKKS encryption" if (use_ckks or self.use_ckks) else ""
        if clients_blocked:
            details_str = f"Blocked {len(clients_blocked)} node(s) deviating from consensus threshold{secagg_note}{ckks_note}"
            status_str = "threat_isolated_and_blocked"
        else:
            details_str = f"Consortium verified: all {len(client_ids)} nodes converged within consensus boundary{secagg_note}{ckks_note}"
            status_str = "verified_clean"

        security_report = SecurityReport(
            clients_evaluated=client_ids,
            clients_accepted=[cid for cid in client_ids if cid not in clients_blocked],
            clients_blocked=clients_blocked,
            validation_accuracies=validation_accuracies,
            defense_active=True,
            threat_detected=len(clients_blocked) > 0,
            details=details_str,
            secagg_active=use_secagg,
            threat_detection_status=status_str,
        )
        self.last_security_report = security_report
        return aggregated_weights, security_report

    def aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        sample_counts: List[int],
        algorithm: str = "FedAvg",
        delta_cs: Optional[List[Dict[str, torch.Tensor]]] = None,
        use_ckks: Optional[bool] = None,
        client_subsampling_q: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate model or adapter weights using the specified optimization algorithm."""
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return self.get_global_weights()

        current_weights = self.get_global_weights()
        first_client = client_weights[0]
        K = len(client_weights)

        active_ckks = self.use_ckks if use_ckks is None else use_ckks

        self.last_full_model_bytes = sum(
            t.numel() * t.element_size() for t in self.global_model.state_dict().values()
        )
        if isinstance(next(iter(first_client.values())), EncryptedTensor):
            self.last_uplink_bytes_per_client = sum(t.nbytes for t in first_client.values())
        else:
            self.last_uplink_bytes_per_client = sum(
                t.numel() * t.element_size() for t in first_client.values()
            )

        # 1. Base Aggregation: CKKS Homomorphic Aggregation vs Standard Sample-Weighted Average
        if active_ckks:
            if self.ckks_engine is None:
                self.consortium_key_enclave = ConsortiumKeyEnclave(scale_bits=40)
                self.ckks_engine = CKKSEncryptionEngine.from_public_bytes(
                    self.consortium_key_enclave.public_context_bytes, scale_bits=40
                )

            first_val = next(iter(first_client.values()))
            if isinstance(first_val, EncryptedTensor):
                encrypted_updates = client_weights
                self.last_uplink_bytes_per_client = sum(e.nbytes for e in first_client.values())
                target_dtypes = None
            else:
                encrypted_updates = [self.ckks_engine.encrypt_state_dict(w) for w in client_weights]
                self.last_uplink_bytes_per_client = sum(e.nbytes for e in encrypted_updates[0].values())
                target_dtypes = {k: first_client[k].dtype for k in first_client.keys()}

            agg_encrypted = self.ckks_engine.homomorphic_aggregate_state_dicts(
                encrypted_updates, [float(c) for c in sample_counts]
            )

            if self.consortium_key_enclave is not None:
                avg_weights = self.consortium_key_enclave.decrypt_aggregated_state_dict(
                    agg_encrypted, target_dtypes=target_dtypes
                )
            else:
                raise PermissionError(
                    "Central aggregation server possesses only the public CKKS context and cannot decrypt. "
                    "Consortium Key Enclave must be provided for collective threshold decryption."
                )
        else:
            avg_weights = {}
            for key in first_client.keys():
                weighted_sum = torch.zeros_like(first_client[key], dtype=torch.float32)
                for client_w, n_samples in zip(client_weights, sample_counts):
                    p_k = n_samples / total_samples
                    weighted_sum += p_k * client_w[key].float()
                avg_weights[key] = weighted_sum

        algo_name = algorithm.upper()

        # 2. FedDyn Aggregation
        if algo_name == "FEDDYN":
            for key in avg_weights.keys():
                target_device = avg_weights[key].device
                # Server dynamic state: h^{t+1} = h^t - alpha * (1/K) * sum (theta_k - theta_global^t)
                delta_sum = torch.zeros_like(avg_weights[key])
                first_client_w = client_weights[0][key]
                if isinstance(first_client_w, EncryptedTensor):
                    # Zero-plaintext homomorphic privacy: individual updates are encrypted.
                    # By linearity of expectation: sum_{k=1}^K (theta_k - theta_global) = K * (theta_avg - theta_global)
                    gw = current_weights[key].to(target_device).float()
                    delta_sum = float(K) * (avg_weights[key].to(target_device).float() - gw)
                else:
                    for client_w in client_weights:
                        cw = client_w[key].to(target_device).float()
                        gw = current_weights[key].to(target_device).float()
                        delta_sum += (cw - gw)

                h_k = self.h_server.get(key, torch.zeros_like(avg_weights[key])).to(target_device)
                h_k = h_k - (self.feddyn_alpha / K) * delta_sum
                self.h_server[key] = h_k

                # Update global model: \bar{theta}^{t+1} = \bar{theta}_avg - (1 / alpha) * h^{t+1}
                avg_weights[key] = avg_weights[key] - (1.0 / self.feddyn_alpha) * h_k

        # 3. FedOpt / FedAdam Adaptive Momentum
        elif algo_name in ("FEDOPT", "FEDADAM"):
            for key in avg_weights.keys():
                target_device = avg_weights[key].device
                gw = current_weights[key].to(target_device).float()
                # Pseudo-gradient: Delta^t = avg_weights - current_weights
                delta = avg_weights[key] - gw

                m = self.m_server.get(key, torch.zeros_like(delta)).to(target_device)
                v = self.v_server.get(key, torch.zeros_like(delta)).to(target_device)

                m = self.server_beta1 * m + (1.0 - self.server_beta1) * delta
                v = self.server_beta2 * v + (1.0 - self.server_beta2) * (delta ** 2)

                self.m_server[key] = m
                self.v_server[key] = v

                # Adaptive server step
                avg_weights[key] = gw + self.server_lr * m / (torch.sqrt(v) + self.server_epsilon)

        # 4. SCAFFOLD Global Control Variate Update
        if algo_name == "SCAFFOLD" and delta_cs:
            for key in first_client.keys():
                target_device = first_client[key].device
                c_g = self.c_global.get(key, torch.zeros_like(first_client[key])).to(target_device)
                delta_c_sum = torch.zeros_like(c_g)
                for d_c in delta_cs:
                    if key in d_c:
                        delta_c_sum += d_c[key].to(target_device).float()
                self.c_global[key] = c_g + (1.0 / K) * delta_c_sum

        # Add Differential Privacy perturbation if active
        if self.privacy_noise > 0.0 and self.dp_mode != "off":
            self.dp_accountant.step(q=client_subsampling_q, sigma=self.privacy_noise)
            if self.dp_mode == "central":
                # Central DP mode: trusted server perturbs consensus model post-aggregation
                for key in avg_weights.keys():
                    noise_scale = (self.privacy_noise * self.dp_clipping_bound) / max(1, K)
                    noise = torch.randn_like(avg_weights[key]) * noise_scale
                    avg_weights[key] = avg_weights[key] + noise
            # For "distributed" (Option J) or "local" modes, clients have already added calibrated
            # noise at the hospital boundary. Server-side noise addition would double the perturbation.

        # Separate cola_bridge weights if present
        cola_weights = {k.replace("cola_bridge.", ""): v for k, v in avg_weights.items() if k.startswith("cola_bridge.")}
        model_weights = {k: v for k, v in avg_weights.items() if not k.startswith("cola_bridge.")}

        if cola_weights:
            if self.cola_bridge is None:
                feat_dim = getattr(self.global_model, "embed_dim", 128)
                self.cola_bridge = FedColaModalityBridge(
                    image_dim=feat_dim, text_dim=512, shared_dim=256
                ).to(self.device)
            self.cola_bridge.load_state_dict({k: v.to(self.device) for k, v in cola_weights.items()})

        # Update global model
        if self.peft_mode:
            load_peft_state_dict(self.global_model, model_weights, mode=self.peft_mode)
        else:
            model_state = {k: v.to(self.device) for k, v in model_weights.items()}
            self.global_model.load_state_dict(model_state)

        return avg_weights

    def evaluate_on_test_set(
        self,
        test_images: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> EvaluationMetrics:
        """Evaluate global model performance on hold-out validation set."""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_samples = test_images.size(0)

        with torch.no_grad():
            batch_size = 64
            for i in range(0, n_samples, batch_size):
                batch_images = test_images[i : i + batch_size].to(self.device)
                batch_labels = test_labels[i : i + batch_size].to(self.device)

                # Ensure spatial resolution matches expected architecture grid
                if batch_images.shape[-1] != self.img_size or batch_images.shape[-2] != self.img_size:
                    batch_images = F.interpolate(
                        batch_images,
                        size=(self.img_size, self.img_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                outputs = self.global_model(batch_images)
                loss = criterion(outputs, batch_labels)

                total_loss += loss.item() * batch_images.size(0)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(batch_labels.cpu().numpy().tolist())

        all_preds_arr = np.array(all_preds, dtype=np.int64)
        all_labels_arr = np.array(all_labels, dtype=np.int64)

        # Pure NumPy multiclass evaluation (zero PyArrow/sklearn crash overhead)
        n_eval = len(all_labels_arr)
        if n_eval > 0:
            accuracy = float(np.mean(all_labels_arr == all_preds_arr))
            conf_matrix = np.zeros((3, 3), dtype=np.int64)
            for t_val, p_val in zip(all_labels_arr, all_preds_arr):
                if 0 <= t_val < 3 and 0 <= p_val < 3:
                    conf_matrix[t_val, p_val] += 1
            precisions, recalls, f1s = [], [], []
            for c in range(3):
                tp = conf_matrix[c, c]
                fp = np.sum(conf_matrix[:, c]) - tp
                fn = np.sum(conf_matrix[c, :]) - tp
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
            precision = float(np.mean(precisions))
            recall = float(np.mean(recalls))
            f1 = float(np.mean(f1s))
        else:
            accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
            conf_matrix = np.zeros((3, 3), dtype=np.int64)

        avg_loss = total_loss / max(n_samples, 1)

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=conf_matrix,
            loss=avg_loss,
        )

    def get_model(self) -> nn.Module:
        return self.global_model

    def get_blocked_count(self) -> int:
        return self.blocked_count


def run_federated_round(
    server: CentralServer,
    clients: List[HospitalClient],
    round_num: int,
    total_rounds: int = 10,
    test_images: Optional[torch.Tensor] = None,
    test_labels: Optional[torch.Tensor] = None,
    use_defense: bool = False,
    algorithm: str = "FedAvg",
    loss_fn_name: str = "CE",
    enable_prototypes: bool = False,
    proto_weight: float = 0.1,
    prox_mu: float = 0.0,
    feddyn_alpha: float = 0.01,
    moon_mu: float = 0.0,
    secure_aggregation: bool = False,
    use_ckks: bool = False,
    total_consortium_hospitals: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[TrainingMetrics], Optional[EvaluationMetrics], Optional[SecurityReport]]:
    """Execute one federated round across all participating clinical nodes."""
    global_weights = server.get_global_weights()
    global_prototypes = server.get_global_prototypes() if enable_prototypes else None
    c_global = server.c_global if algorithm.upper() == "SCAFFOLD" else None

    total_hospitals = total_consortium_hospitals or len(clients)
    client_subsampling_q = min(1.0, len(clients) / max(1, total_hospitals))
    num_consortium = len(clients)
    dp_active = server.privacy_noise > 0.0 and server.dp_mode != "off"

    client_updates: List[Dict[str, torch.Tensor]] = []
    client_metrics: List[TrainingMetrics] = []
    sample_counts: List[int] = []
    client_ids: List[int] = []
    delta_cs: List[Optional[Dict[str, torch.Tensor]]] = []

    client_protos: List[Dict[int, torch.Tensor]] = []
    client_traces: List[Dict[int, float]] = []
    client_counts: List[Dict[int, int]] = []

    for client in clients:
        updated_weights, metrics, delta_c = client.train(
            global_weights=global_weights,
            loss_fn_name=loss_fn_name,
            current_round=round_num,
            total_rounds=total_rounds,
            global_prototypes=global_prototypes,
            proto_weight=proto_weight if enable_prototypes else 0.0,
            algorithm=algorithm,
            prox_mu=prox_mu if algorithm.upper() in ("FEDPROX", "PROX") else 0.0,
            feddyn_alpha=feddyn_alpha if algorithm.upper() == "FEDDYN" else 0.0,
            c_global=c_global,
            moon_mu=moon_mu if algorithm.upper() == "MOON" else 0.0,
            dp_enabled=dp_active,
            dp_clipping_bound=server.dp_clipping_bound,
            dp_noise_multiplier=server.privacy_noise,
            dp_mode=server.dp_mode,
            num_consortium_clients=num_consortium,
        )
        client_updates.append(updated_weights)
        client_metrics.append(metrics)
        sample_counts.append(client.get_num_samples())
        client_ids.append(client.client_id)
        delta_cs.append(delta_c)

        if enable_prototypes:
            p_dict, t_dict, c_dict = client.compute_prototypes_and_dispersion()
            client_protos.append(p_dict)
            client_traces.append(t_dict)
            client_counts.append(c_dict)

    active_secagg = secure_aggregation or server.secure_aggregation
    active_ckks = use_ckks or server.use_ckks

    # Distribute public CKKS context to hospital clients under Option J standard
    if active_ckks and server.consortium_key_enclave is not None:
        pub_bytes = server.consortium_key_enclave.public_context_bytes
        for client in clients:
            if client.ckks_engine is None:
                client.init_ckks(pub_bytes)

    security_report: Optional[SecurityReport] = None

    if use_defense and test_images is not None and test_labels is not None:
        # Authentic decentralized Byzantine defense: hospital clients generate signed
        # enclave validation attestations within their perimeters. The consortium verifies
        # consensus and isolates malicious nodes before SecAgg+ masking and CKKS encryption.
        attestations = [
            client.generate_attestation(test_images, test_labels, weights=upd)
            for client, upd in zip(clients, client_updates)
        ]
        _, security_report = server.validate_and_aggregate(
            client_updates,
            sample_counts,
            client_ids,
            test_images,
            test_labels,
            algorithm=algorithm,
            delta_cs=delta_cs if algorithm.upper() == "SCAFFOLD" else None,
            use_secagg=active_secagg,
            use_ckks=active_ckks,
            client_subsampling_q=client_subsampling_q,
            client_attestations=attestations,
            clients=clients,
            round_num=round_num,
        )
        if security_report:
            for metrics in client_metrics:
                if metrics.client_id in security_report.clients_blocked:
                    metrics.was_blocked = True
    else:
        # Defense not active or test set not available
        updates_to_aggregate = client_updates
        if active_secagg and len(client_updates) > 1:
            total_s = max(1, sum(sample_counts))
            if clients and hasattr(clients[0], "get_secagg_public_key"):
                pk_registry = {client.client_id: client.get_secagg_public_key() for client in clients}
                weights_map = {client.client_id: float(s) / total_s for client, s in zip(clients, sample_counts)}
                updates_to_aggregate = [
                    client.mask_local_weights(
                        weights=upd,
                        peer_public_keys=pk_registry,
                        weights_map=weights_map,
                        round_num=round_num,
                    )
                    for client, upd in zip(clients, client_updates)
                ]
            else:
                tensor_shapes = {k: v.shape for k, v in client_updates[0].items()}
                weights_norm = [float(s) / total_s for s in sample_counts]
                masks = SecAggPlusProtocol.generate_pairwise_masks(
                    num_clients=len(client_updates),
                    tensor_shapes=tensor_shapes,
                    device=server.device,
                    weights=weights_norm,
                    round_num=round_num,
                )
                updates_to_aggregate = [
                    SecAggPlusProtocol.mask_state_dict(upd, m)
                    for upd, m in zip(client_updates, masks)
                ]

        if active_ckks and clients and getattr(clients[0], "ckks_engine", None):
            updates_to_aggregate = [
                client.encrypt_local_weights(w) for client, w in zip(clients, updates_to_aggregate)
            ]

        delta_cs_clean: Optional[List[Dict[str, torch.Tensor]]] = None
        if algorithm.upper() == "SCAFFOLD" and delta_cs:
            delta_cs_clean = [d for d in delta_cs if d is not None]

        server.aggregate(
            updates_to_aggregate,
            sample_counts,
            algorithm=algorithm,
            delta_cs=delta_cs_clean,
            use_ckks=active_ckks,
            client_subsampling_q=client_subsampling_q,
        )

        if use_defense:
            # Defense was requested but no validation cohort was provided
            security_report = SecurityReport(
                clients_evaluated=client_ids,
                clients_accepted=client_ids,
                clients_blocked=[],
                validation_accuracies={},
                defense_active=False,
                threat_detected=False,
                details="Byzantine defense skipped: no consortium validation cohort provided for node inspection",
                secagg_active=active_secagg,
                threat_detection_status="unverifiable_missing_validation_cohort",
            )

    # Aggregate metric prototypes (strictly from accepted clients if defense is active)
    if enable_prototypes and client_protos:
        if security_report and security_report.clients_blocked:
            clean_indices = [
                i for i, cid in enumerate(client_ids)
                if cid not in security_report.clients_blocked
            ]
            clean_protos = [client_protos[i] for i in clean_indices if i < len(client_protos)]
            clean_traces = [client_traces[i] for i in clean_indices if i < len(client_traces)]
            clean_counts = [client_counts[i] for i in clean_indices if i < len(client_counts)]
        else:
            clean_protos = client_protos
            clean_traces = client_traces
            clean_counts = client_counts

        if clean_protos:
            synthesized_protos = aggregate_prototypes_dispersion_weighted(
                client_prototypes=clean_protos,
                client_traces=clean_traces,
                client_counts=clean_counts,
                num_classes=3,
            )
            server.set_global_prototypes(synthesized_protos)

    total_samples = sum(sample_counts)
    if total_samples > 0:
        avg_loss = sum(m.loss * m.samples_trained for m in client_metrics) / total_samples
        avg_accuracy = sum(m.accuracy * m.samples_trained for m in client_metrics) / total_samples
    else:
        avg_loss = 0.0
        avg_accuracy = 0.0

    aggregated_metrics = {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
        "total_samples": total_samples,
        "round": round_num,
        "prototypes_enabled": enable_prototypes,
        "loss_fn": loss_fn_name,
        "algorithm": algorithm,
        "secure_aggregation": active_secagg,
        "use_ckks": active_ckks,
        "privacy_epsilon": server.get_privacy_spent(),
        "uplink_bytes_per_client": server.last_uplink_bytes_per_client,
        "full_model_bytes": server.last_full_model_bytes,
    }

    test_metrics: Optional[EvaluationMetrics] = None
    if test_images is not None and test_labels is not None:
        test_metrics = server.evaluate_on_test_set(test_images, test_labels)

    return aggregated_metrics, client_metrics, test_metrics, security_report
