"""Fed-XRay Flower (flwr v1.8+) gRPC Bridge for Multi-Institutional Federated Learning.

Enables real-world gRPC communication between decentralized hospital nodes and
the central aggregator with:
1. Seamless parameter translation between PyTorch tensors and Flower NumPy parameters.
2. Parameter-Efficient Fine-Tuning (FFA-LoRA, FedSA-LoRA) state dict extraction.
3. Byzantine defense shield filtering and cryptographic secure aggregation hooks.
4. Patient-level Rényi Differential Privacy calibration.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from ..core.client import HospitalClient
from ..core.server import CentralServer
from ..models.peft import extract_peft_state_dict, load_peft_state_dict


def state_dict_to_flower_parameters(
    state_dict: Dict[str, torch.Tensor]
) -> Tuple[NDArrays, List[str]]:
    """Convert a PyTorch state dict into ordered Flower NDArrays and key names."""
    keys = list(state_dict.keys())
    arrays = [state_dict[k].detach().cpu().numpy().astype(np.float32) for k in keys]
    return arrays, keys


def flower_parameters_to_state_dict(
    ndarrays: NDArrays,
    keys: List[str],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Convert Flower NDArrays back into a PyTorch state dict with target device alignment."""
    target_device = device or torch.device("cpu")
    state_dict: Dict[str, torch.Tensor] = {}
    for k, arr in zip(keys, ndarrays):
        state_dict[k] = torch.from_numpy(arr).to(target_device)
    return state_dict


class FlowerHospitalClient(fl.client.NumPyClient):
    """Flower gRPC Client wrapper over a Fed-XRay HospitalClient node."""

    def __init__(
        self,
        hospital_client: HospitalClient,
        algorithm: str = "FedAvg",
        loss_fn_name: str = "CE",
        enable_prototypes: bool = False,
        proto_weight: float = 0.1,
    ) -> None:
        self.hospital_client = hospital_client
        self.algorithm = algorithm
        self.loss_fn_name = loss_fn_name
        self.enable_prototypes = enable_prototypes
        self.proto_weight = proto_weight

        # Cache key ordering from initial model weights
        if self.hospital_client.peft_mode:
            init_weights = extract_peft_state_dict(
                self.hospital_client.model, mode=self.hospital_client.peft_mode
            )
        else:
            init_weights = {
                k: v.clone().detach().cpu()
                for k, v in self.hospital_client.model.state_dict().items()
            }
        self.param_keys = list(init_weights.keys())

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Extract and return current local model or PEFT adapter parameters."""
        if self.hospital_client.peft_mode:
            weights = extract_peft_state_dict(
                self.hospital_client.model, mode=self.hospital_client.peft_mode
            )
        else:
            weights = {
                k: v.clone().detach().cpu()
                for k, v in self.hospital_client.model.state_dict().items()
            }
        arrays, _ = state_dict_to_flower_parameters(weights)
        return arrays

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Perform local clinical optimization round on received global weights."""
        current_round = int(config.get("current_round", 1))
        total_rounds = int(config.get("total_rounds", 10))
        algorithm = str(config.get("algorithm", self.algorithm))
        loss_fn = str(config.get("loss_fn", self.loss_fn_name))

        # Reconstruct PyTorch state dict
        global_weights = flower_parameters_to_state_dict(
            parameters, self.param_keys, device=self.hospital_client.device
        )

        # Execute genuine local training round
        updated_weights, metrics, _ = self.hospital_client.train(
            global_weights=global_weights,
            loss_fn_name=loss_fn,
            current_round=current_round,
            total_rounds=total_rounds,
            algorithm=algorithm,
        )

        arrays, _ = state_dict_to_flower_parameters(updated_weights)
        num_samples = self.hospital_client.get_num_samples()

        metrics_dict: Dict[str, Scalar] = {
            "loss": float(metrics.loss),
            "accuracy": float(metrics.accuracy),
            "client_id": int(self.hospital_client.client_id),
            "samples_trained": int(metrics.samples_trained),
        }

        return arrays, num_samples, metrics_dict

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate received global parameters on local hospital validation data."""
        weights = flower_parameters_to_state_dict(
            parameters, self.param_keys, device=self.hospital_client.device
        )

        if self.hospital_client.peft_mode:
            load_peft_state_dict(
                self.hospital_client.model, weights, mode=self.hospital_client.peft_mode
            )
        else:
            self.hospital_client.model.load_state_dict(weights)

        self.hospital_client.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.hospital_client.dataloader:
                images = images.to(self.hospital_client.device)
                labels = labels.to(self.hospital_client.device)
                outputs = self.hospital_client.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total_samples += images.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)

        return float(avg_loss), total_samples, {"accuracy": float(accuracy)}


class FlowerFedXRayStrategy(Strategy):
    """Flower Strategy wrapping CentralServer with Byzantine defense and PEFT aggregation."""

    def __init__(
        self,
        central_server: CentralServer,
        total_rounds: int = 10,
        algorithm: str = "FedAvg",
        loss_fn_name: str = "CE",
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        test_images: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None,
    ) -> None:
        self.central_server = central_server
        self.total_rounds = total_rounds
        self.algorithm = algorithm
        self.loss_fn_name = loss_fn_name
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.test_images = test_images
        self.test_labels = test_labels

        init_weights = self.central_server.get_global_weights()
        self.arrays, self.param_keys = state_dict_to_flower_parameters(init_weights)

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Parameters:
        """Initialize global model parameters sent to all clinical nodes in round 0."""
        return ndarrays_to_parameters(self.arrays)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the training round and broadcast instructions to connected hospital nodes."""
        sample_size = max(
            int(client_manager.num_available() * self.fraction_fit), self.min_fit_clients
        )
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_available_clients)

        config: Dict[str, Scalar] = {
            "current_round": server_round,
            "total_rounds": self.total_rounds,
            "algorithm": self.algorithm,
            "loss_fn": self.loss_fn_name,
        }
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate incoming client parameter updates through CentralServer."""
        if not results:
            return None, {}

        client_weights: List[Dict[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        client_ids: List[int] = []

        for idx, (_, fit_res) in enumerate(results):
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weights = flower_parameters_to_state_dict(
                ndarrays, self.param_keys, device=self.central_server.device
            )
            client_weights.append(weights)
            sample_counts.append(fit_res.num_examples)
            cid = int(fit_res.metrics.get("client_id", idx))
            client_ids.append(cid)

        # Aggregate using CentralServer
        if self.central_server.defense_mode and self.test_images is not None and self.test_labels is not None:
            avg_weights, sec_report = self.central_server.validate_and_aggregate(
                client_weights=client_weights,
                sample_counts=sample_counts,
                client_ids=client_ids,
                test_images=self.test_images,
                test_labels=self.test_labels,
                algorithm=self.algorithm,
            )
            blocked_nodes = sec_report.clients_blocked
        else:
            avg_weights = self.central_server.aggregate(
                client_weights=client_weights,
                sample_counts=sample_counts,
                algorithm=self.algorithm,
            )
            blocked_nodes = []

        self.arrays, _ = state_dict_to_flower_parameters(avg_weights)
        metrics: Dict[str, Scalar] = {
            "round": server_round,
            "participating_clients": len(results),
            "blocked_clients_count": len(blocked_nodes),
            "privacy_epsilon": float(self.central_server.get_privacy_spent()),
        }

        return ndarrays_to_parameters(self.arrays), metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation across connected clinical clients."""
        if self.fraction_fit == 0.0:
            return []
        sample_size = max(int(client_manager.num_available() * 0.5), 1)
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=1)
        eval_ins = EvaluateIns(parameters, {"current_round": server_round})
        return [(client, eval_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics reported by clients."""
        if not results:
            return None, {}
        total_samples = sum(r.num_examples for _, r in results)
        if total_samples == 0:
            return None, {}

        weighted_loss = sum(r.loss * r.num_examples for _, r in results) / total_samples
        weighted_acc = sum(float(r.metrics.get("accuracy", 0.0)) * r.num_examples for _, r in results) / total_samples

        return float(weighted_loss), {"test_accuracy": float(weighted_acc)}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate current global model on central holdout validation set."""
        if self.test_images is None or self.test_labels is None:
            return None
        ndarrays = parameters_to_ndarrays(parameters)
        weights = flower_parameters_to_state_dict(
            ndarrays, self.param_keys, device=self.central_server.device
        )
        if self.central_server.peft_mode:
            load_peft_state_dict(
                self.central_server.global_model, weights, mode=self.central_server.peft_mode
            )
        else:
            self.central_server.global_model.load_state_dict(weights)

        metrics = self.central_server.evaluate_on_test_set(
            self.test_images, self.test_labels
        )
        return float(metrics.loss), {
            "accuracy": float(metrics.accuracy),
            "precision": float(metrics.precision),
            "recall": float(metrics.recall),
            "f1_score": float(metrics.f1_score),
        }
