"""Fed-XRay Central Server & Federated Aggregation Engine.

Supports:
- Classical Consensus: FedAvg, FedProx
- Modern SOTA Optimizers: FedDyn, FedOpt / FedAdam, SCAFFOLD on PEFT
- Parameter-Efficient Fine-Tuning Aggregation (LoRA, FFA-LoRA, FedSA-LoRA)
- Dispersion-Weighted Prototype Metric Synthesis
- Byzantine Defense Shield via Hold-Out Reference Validation
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (  # pyrefly: ignore[missing-source-for-stubs]
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ..models.cnn import create_model as create_cnn_model
from ..models.vit import create_medical_vit
from ..models.foundation import UNIFoundationBackbone, SwinUNETRRadiologyBackbone
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
    PatientLevelDPAccountant,
    SecAggPlusProtocol,
)


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

        # Cryptographic and DP engines
        self.ckks_engine: Optional[CKKSEncryptionEngine] = (
            CKKSEncryptionEngine(scale_bits=40) if use_ckks else None
        )
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
        if self.privacy_noise <= 0.0:
            return 0.0
        return self.dp_accountant.get_epsilon()

    def _create_model(self) -> nn.Module:
        """Create global model with appropriate architecture and PEFT hooks."""
        model: nn.Module
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
        elif self.model_type == "uni":
            model = UNIFoundationBackbone(embed_dim=512, num_classes=3)
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
            model = SwinUNETRRadiologyBackbone(embed_dim=384, num_classes=3)
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

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model or PEFT adapter weights."""
        if self.peft_mode:
            return extract_peft_state_dict(self.global_model, mode=self.peft_mode)
        return {
            name: param.clone().detach().cpu()
            for name, param in self.global_model.state_dict().items()
        }

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
        test_images: torch.Tensor,
        test_labels: torch.Tensor,
        algorithm: str = "FedAvg",
        delta_cs: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], SecurityReport]:
        """Filter out malicious poisoned nodes and aggregate trusted updates."""
        validation_accuracies: Dict[int, float] = {}
        trusted_weights: List[Dict[str, torch.Tensor]] = []
        trusted_counts: List[int] = []
        trusted_delta_cs: List[Dict[str, torch.Tensor]] = []
        clients_blocked: List[int] = []

        self.last_full_model_bytes = sum(
            t.numel() * t.element_size() for t in self.global_model.state_dict().values()
        )
        if client_weights:
            self.last_uplink_bytes_per_client = sum(
                t.numel() * t.element_size() for t in client_weights[0].values()
            )

        for idx, (weights, count, cid) in enumerate(zip(client_weights, sample_counts, client_ids)):
            val_acc = self._validate_client_model(weights, test_images, test_labels)
            validation_accuracies[cid] = val_acc

            if val_acc < self.MALICIOUS_THRESHOLD:
                print(f"[SECURITY ALERT] Node {cid} failed validation (Acc: {val_acc:.1%}). BLOCKED!")
                clients_blocked.append(cid)
                self.blocked_count += 1
            else:
                trusted_weights.append(weights)
                trusted_counts.append(count)
                if delta_cs is not None and idx < len(delta_cs):
                    d_c = delta_cs[idx]
                    if d_c is not None:
                        trusted_delta_cs.append(d_c)

        if not trusted_weights:
            print("[WARNING] All client updates failed validation. Preserving global state.")
            aggregated_weights = self.get_global_weights()
        else:
            aggregated_weights = self.aggregate(
                trusted_weights,
                trusted_counts,
                algorithm=algorithm,
                delta_cs=trusted_delta_cs if trusted_delta_cs else None,
            )

        security_report = SecurityReport(
            clients_evaluated=client_ids,
            clients_accepted=[cid for cid in client_ids if cid not in clients_blocked],
            clients_blocked=clients_blocked,
            validation_accuracies=validation_accuracies,
            threat_detected=len(clients_blocked) > 0,
            details=f"Blocked {len(clients_blocked)} node(s) below {self.MALICIOUS_THRESHOLD:.0%} threshold",
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
        self.last_uplink_bytes_per_client = sum(
            t.numel() * t.element_size() for t in first_client.values()
        )

        # 1. Base Aggregation: CKKS Homomorphic Aggregation vs Standard Sample-Weighted Average
        if active_ckks:
            engine = self.ckks_engine or CKKSEncryptionEngine(scale_bits=40)
            encrypted_updates = [engine.encrypt_state_dict(w) for w in client_weights]
            self.last_uplink_bytes_per_client = sum(e.nbytes for e in encrypted_updates[0].values())
            agg_encrypted = engine.homomorphic_aggregate_state_dicts(
                encrypted_updates, [float(c) for c in sample_counts]
            )
            target_dtypes = {k: first_client[k].dtype for k in first_client.keys()}
            avg_weights = engine.decrypt_state_dict(agg_encrypted, target_dtypes=target_dtypes)
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
                # Server dynamic state: h^{t+1} = h^t - alpha * (1/K) * sum (theta_k - theta_global^t)
                delta_sum = torch.zeros_like(avg_weights[key])
                for client_w in client_weights:
                    delta_sum += (client_w[key].float() - current_weights[key].float())

                h_k = self.h_server.get(key, torch.zeros_like(avg_weights[key]))
                h_k = h_k - (self.feddyn_alpha / K) * delta_sum
                self.h_server[key] = h_k

                # Update global model: \bar{theta}^{t+1} = \bar{theta}_avg - (1 / alpha) * h^{t+1}
                avg_weights[key] = avg_weights[key] - (1.0 / self.feddyn_alpha) * h_k

        # 3. FedOpt / FedAdam Adaptive Momentum
        elif algo_name in ("FEDOPT", "FEDADAM"):
            for key in avg_weights.keys():
                # Pseudo-gradient: Delta^t = avg_weights - current_weights
                delta = avg_weights[key] - current_weights[key].float()

                m = self.m_server.get(key, torch.zeros_like(delta))
                v = self.v_server.get(key, torch.zeros_like(delta))

                m = self.server_beta1 * m + (1.0 - self.server_beta1) * delta
                v = self.server_beta2 * v + (1.0 - self.server_beta2) * (delta ** 2)

                self.m_server[key] = m
                self.v_server[key] = v

                # Adaptive server step
                avg_weights[key] = current_weights[key].float() + self.server_lr * m / (torch.sqrt(v) + self.server_epsilon)

        # 4. SCAFFOLD Global Control Variate Update
        if algo_name == "SCAFFOLD" and delta_cs:
            for key in first_client.keys():
                c_g = self.c_global.get(key, torch.zeros_like(first_client[key]))
                delta_c_sum = torch.zeros_like(c_g)
                for d_c in delta_cs:
                    if key in d_c:
                        delta_c_sum += d_c[key].float()
                self.c_global[key] = c_g + (1.0 / K) * delta_c_sum

        # Add Differential Privacy perturbation if active
        if self.privacy_noise > 0.0:
            self.dp_accountant.step(q=1.0, sigma=self.privacy_noise)
            for key in avg_weights.keys():
                noise = torch.randn_like(avg_weights[key]) * self.privacy_noise
                avg_weights[key] = avg_weights[key] + noise

        # Update global model
        if self.peft_mode:
            load_peft_state_dict(self.global_model, avg_weights, mode=self.peft_mode)
        else:
            self.global_model.load_state_dict(avg_weights)

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

                outputs = self.global_model(batch_images)
                loss = criterion(outputs, batch_labels)

                total_loss += loss.item() * batch_images.size(0)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(batch_labels.cpu().numpy().tolist())

        all_preds_arr = np.array(all_preds)
        all_labels_arr = np.array(all_labels)

        accuracy = accuracy_score(all_labels_arr, all_preds_arr)
        precision = precision_score(all_labels_arr, all_preds_arr, average="macro", zero_division=0)
        recall = recall_score(all_labels_arr, all_preds_arr, average="macro", zero_division=0)
        f1 = f1_score(all_labels_arr, all_preds_arr, average="macro", zero_division=0)
        conf_matrix = confusion_matrix(all_labels_arr, all_preds_arr)
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
) -> Tuple[Dict[str, Any], List[TrainingMetrics], Optional[EvaluationMetrics], Optional[SecurityReport]]:
    """Execute one federated round across all participating clinical nodes."""
    global_weights = server.get_global_weights()
    global_prototypes = server.get_global_prototypes() if enable_prototypes else None
    c_global = server.c_global if algorithm.upper() == "SCAFFOLD" else None

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

    # If SecAgg+ is active, clients mask their local updates before transmission
    if active_secagg and len(client_updates) > 1:
        tensor_shapes = {k: v.shape for k, v in client_updates[0].items()}
        total_s = max(1, sum(sample_counts))
        weights_norm = [float(s) / total_s for s in sample_counts]
        masks = SecAggPlusProtocol.generate_pairwise_masks(
            num_clients=len(client_updates),
            tensor_shapes=tensor_shapes,
            device=server.device,
            weights=weights_norm,
        )
        client_updates = [
            SecAggPlusProtocol.mask_state_dict(upd, m)
            for upd, m in zip(client_updates, masks)
        ]

    security_report: Optional[SecurityReport] = None

    if use_defense and not active_secagg and test_images is not None and test_labels is not None:
        _, security_report = server.validate_and_aggregate(
            client_updates,
            sample_counts,
            client_ids,
            test_images,
            test_labels,
            algorithm=algorithm,
            delta_cs=delta_cs if algorithm.upper() == "SCAFFOLD" else None,
        )
        if security_report:
            for metrics in client_metrics:
                if metrics.client_id in security_report.clients_blocked:
                    metrics.was_blocked = True
    else:
        if active_secagg and use_defense:
            security_report = SecurityReport(
                clients_evaluated=client_ids,
                clients_accepted=client_ids,
                clients_blocked=[],
                validation_accuracies={},
                threat_detected=False,
                details="SecAgg+ pairwise zero-sum masking active (cryptographic privacy preserves node confidentiality)",
            )
        delta_cs_clean: Optional[List[Dict[str, torch.Tensor]]] = None
        if algorithm.upper() == "SCAFFOLD" and delta_cs:
            delta_cs_clean = [d for d in delta_cs if d is not None]

        server.aggregate(
            client_updates,
            sample_counts,
            algorithm=algorithm,
            delta_cs=delta_cs_clean,
            use_ckks=active_ckks,
        )

    # Aggregate metric prototypes
    if enable_prototypes and client_protos:
        synthesized_protos = aggregate_prototypes_dispersion_weighted(
            client_prototypes=client_protos,
            client_traces=client_traces,
            client_counts=client_counts,
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
