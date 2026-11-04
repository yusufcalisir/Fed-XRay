"""Fed-XRay Flower (flwr v1.8+) gRPC Bridge for Multi-Institutional Federated Learning.

Enables real-world gRPC communication between decentralized hospital nodes and
the central aggregator with:
1. Seamless parameter translation between PyTorch tensors and Flower NumPy parameters.
2. Parameter-Efficient Fine-Tuning (FFA-LoRA, FedSA-LoRA) state dict extraction.
3. Byzantine defense shield filtering and cryptographic secure aggregation hooks.
4. Patient-level Rényi Differential Privacy calibration.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import logging
import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
    Parameters,
    ReconnectIns,
    Scalar,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import Strategy

from ..core.client import HospitalClient
from ..core.server import CentralServer
from ..models.peft import extract_peft_state_dict, load_peft_state_dict

logger = logging.getLogger("fed_xray.flower_bridge")


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

        dp_enabled = bool(config.get("dp_enabled", self.hospital_client.dp_enabled))
        dp_clipping_bound = float(config.get("dp_clipping_bound", self.hospital_client.dp_clipping_bound))
        dp_noise_multiplier = float(config.get("dp_noise_multiplier", self.hospital_client.dp_noise_multiplier))
        dp_mode = str(config.get("dp_mode", self.hospital_client.dp_mode))
        num_consortium_clients = int(config.get("num_consortium_clients", 1))
        proto_weight = float(config.get("proto_weight", self.proto_weight))

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
            dp_enabled=dp_enabled,
            dp_clipping_bound=dp_clipping_bound,
            dp_noise_multiplier=dp_noise_multiplier,
            dp_mode=dp_mode,
            num_consortium_clients=num_consortium_clients,
            proto_weight=proto_weight,
        )

        arrays, _ = state_dict_to_flower_parameters(updated_weights)
        num_samples = self.hospital_client.get_num_samples()

        metrics_dict: Dict[str, Scalar] = {
            "loss": float(metrics.loss),
            "train_loss": float(metrics.loss),
            "accuracy": float(metrics.accuracy),
            "train_accuracy": float(metrics.accuracy),
            "client_id": self.hospital_client.client_id,
            "samples_trained": metrics.samples_trained,
        }
        if metrics.dp_clipped_norm is not None:
            metrics_dict["dp_clipped_norm"] = float(metrics.dp_clipped_norm)

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
        secure_aggregation: bool = False,
        use_ckks: bool = False,
        total_consortium_hospitals: Optional[int] = None,
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
        self.secure_aggregation = secure_aggregation
        self.use_ckks = use_ckks
        self.total_consortium_hospitals = total_consortium_hospitals
        self.last_blocked_nodes: List[int] = []
        self.last_security_report: Optional[Any] = None

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

        dp_active = (
            self.central_server.privacy_noise > 0.0
            and getattr(self.central_server, "dp_mode", "off") != "off"
        )
        config: Dict[str, Scalar] = {
            "current_round": server_round,
            "total_rounds": self.total_rounds,
            "algorithm": self.algorithm,
            "loss_fn": self.loss_fn_name,
            "dp_enabled": dp_active,
            "dp_clipping_bound": float(getattr(self.central_server, "dp_clipping_bound", 1.0)),
            "dp_noise_multiplier": float(self.central_server.privacy_noise),
            "dp_mode": str(getattr(self.central_server, "dp_mode", "distributed")),
            "num_consortium_clients": sample_size,
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

        total_hospitals = self.total_consortium_hospitals or len(results)
        client_q = min(1.0, len(results) / max(1, total_hospitals))

        # Aggregate using CentralServer
        if self.central_server.defense_mode and self.test_images is not None and self.test_labels is not None:
            avg_weights, sec_report = self.central_server.validate_and_aggregate(
                client_weights=client_weights,
                sample_counts=sample_counts,
                client_ids=client_ids,
                test_images=self.test_images,
                test_labels=self.test_labels,
                algorithm=self.algorithm,
                use_secagg=self.secure_aggregation,
                use_ckks=self.use_ckks,
                client_subsampling_q=client_q,
            )
            blocked_nodes = sec_report.clients_blocked
            self.last_security_report = sec_report
        else:
            avg_weights = self.central_server.aggregate(
                client_weights=client_weights,
                sample_counts=sample_counts,
                algorithm=self.algorithm,
                use_ckks=self.use_ckks,
                client_subsampling_q=client_q,
            )
            blocked_nodes = []
            self.last_security_report = None

        blocked_nodes_list = [int(b) for b in blocked_nodes]
        self.last_blocked_nodes = blocked_nodes_list
        blocked_nodes_serialized = json.dumps(blocked_nodes_list)

        self.arrays, _ = state_dict_to_flower_parameters(avg_weights)

        # Aggregate authentic local client training metrics weighted by batch samples
        total_train_samples = sum(sample_counts)
        if total_train_samples > 0:
            weighted_train_loss = sum(
                float(fit_res.metrics.get("train_loss", fit_res.metrics.get("loss", 0.0))) * count
                for (_, fit_res), count in zip(results, sample_counts)
            ) / total_train_samples
            weighted_train_acc = sum(
                float(fit_res.metrics.get("train_accuracy", fit_res.metrics.get("accuracy", 0.0))) * count
                for (_, fit_res), count in zip(results, sample_counts)
            ) / total_train_samples
        else:
            weighted_train_loss = 0.0
            weighted_train_acc = 0.0

        metrics: Dict[str, Scalar] = {
            "round": server_round,
            "participating_clients": len(results),
            "train_loss": float(weighted_train_loss),
            "train_accuracy": float(weighted_train_acc),
            "blocked_clients_count": len(blocked_nodes_list),
            "blocked_nodes_json": blocked_nodes_serialized,
            "blocked_nodes": ",".join(str(b) for b in blocked_nodes_list),
            "privacy_epsilon": float(self.central_server.get_privacy_spent()),
            "dp_active": (
                self.central_server.privacy_noise > 0.0
                and getattr(self.central_server, "dp_mode", "off") != "off"
            ),
            "uplink_bytes_per_client": self.central_server.last_uplink_bytes_per_client,
            "full_model_bytes": self.central_server.last_full_model_bytes,
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


class InProcessClientProxy(ClientProxy):
    """Flower ClientProxy executing operations directly on an in-memory FlowerHospitalClient.

    Provides a zero-network, high-throughput, deterministic bridge conforming to Flower's
    ClientProxy interface without requiring external gRPC ports or Ray dependencies.
    """

    def __init__(self, cid: str, client: FlowerHospitalClient) -> None:
        super().__init__(cid)
        self.client = client

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: Optional[float] = None,
        group_id: Optional[int] = None,
    ) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={},
        )

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float] = None,
        group_id: Optional[int] = None,
    ) -> GetParametersRes:
        ndarrays = self.client.get_parameters(ins.config)
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(ndarrays),
        )

    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float] = None,
        group_id: Optional[int] = None,
    ) -> FitRes:
        ndarrays = parameters_to_ndarrays(ins.parameters)
        updated_ndarrays, num_samples, metrics = self.client.fit(ndarrays, ins.config)
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(updated_ndarrays),
            num_examples=num_samples,
            metrics=metrics,
        )

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float] = None,
        group_id: Optional[int] = None,
    ) -> EvaluateRes:
        ndarrays = parameters_to_ndarrays(ins.parameters)
        loss, num_samples, metrics = self.client.evaluate(ndarrays, ins.config)
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=num_samples,
            metrics=metrics,
        )

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float] = None,
        group_id: Optional[int] = None,
    ) -> DisconnectRes:
        return DisconnectRes(reason="normal")


def run_flower_federated_orchestration(
    server: CentralServer,
    clients: List[HospitalClient],
    num_rounds: int = 3,
    algorithm: str = "FedAvg",
    loss_fn_name: str = "CE",
    secure_aggregation: bool = False,
    use_ckks: bool = False,
    test_images: Optional[torch.Tensor] = None,
    test_labels: Optional[torch.Tensor] = None,
    on_round_end: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end federated training rounds using Flower protocol and strategy.

    Orchestrates the complete lifecycle:
    1. Initializes FlowerFedXRayStrategy with the central server and parameters.
    2. Registers connected HospitalClient instances via InProcessClientProxy.
    3. Iterates over training rounds (configure_fit -> local fit -> aggregate_fit).
    4. Evaluates globally and per-client after each aggregation.
    5. Returns round-by-round trajectory and final model state.
    """
    logger.info(
        "Initiating Flower federated orchestration across %d clients for %d rounds",
        len(clients),
        num_rounds,
    )

    strategy = FlowerFedXRayStrategy(
        central_server=server,
        total_rounds=num_rounds,
        algorithm=algorithm,
        loss_fn_name=loss_fn_name,
        min_fit_clients=max(1, len(clients)),
        min_available_clients=max(1, len(clients)),
        test_images=test_images,
        test_labels=test_labels,
        secure_aggregation=secure_aggregation,
        use_ckks=use_ckks,
        total_consortium_hospitals=len(clients),
    )

    client_manager = SimpleClientManager()
    proxies: List[InProcessClientProxy] = []
    for c in clients:
        flower_client = FlowerHospitalClient(
            hospital_client=c,
            algorithm=algorithm,
            loss_fn_name=loss_fn_name,
        )
        proxy = InProcessClientProxy(cid=str(c.client_id), client=flower_client)
        client_manager.register(proxy)
        proxies.append(proxy)

    parameters = strategy.initialize_parameters(client_manager)
    round_history: List[Dict[str, Any]] = []

    for rnd in range(1, num_rounds + 1):
        fit_configurations = strategy.configure_fit(
            server_round=rnd,
            parameters=parameters,
            client_manager=client_manager,
        )

        results: List[Tuple[ClientProxy, FitRes]] = []
        for c_proxy, fit_ins in fit_configurations:
            fit_res = c_proxy.fit(fit_ins, timeout=None, group_id=None)
            results.append((c_proxy, fit_res))

        aggregated_params, agg_metrics = strategy.aggregate_fit(
            server_round=rnd,
            results=results,
            failures=[],
        )
        if aggregated_params is not None:
            parameters = aggregated_params

        round_record: Dict[str, Any] = {
            "round": rnd,
            "participating_clients": len(results),
            "metrics": agg_metrics,
            "train_loss": float(agg_metrics.get("train_loss", 0.0)),
            "train_accuracy": float(agg_metrics.get("train_accuracy", 0.0)),
            "blocked_nodes": list(getattr(strategy, "last_blocked_nodes", [])),
            "security_report": getattr(strategy, "last_security_report", None),
        }

        # Central evaluation if holdout set is present
        eval_res = strategy.evaluate(server_round=rnd, parameters=parameters)
        if eval_res is not None:
            loss, eval_metrics = eval_res
            round_record["eval_loss"] = loss
            round_record["eval_metrics"] = eval_metrics

        round_history.append(round_record)
        if on_round_end:
            on_round_end(rnd, round_record)

    return {
        "status": "completed",
        "rounds_completed": num_rounds,
        "round_history": round_history,
        "final_metrics": round_history[-1] if round_history else {},
        "privacy_epsilon": float(server.get_privacy_spent()),
    }


def start_flower_server(
    central_server: CentralServer,
    server_address: str = "0.0.0.0:8089",
    num_rounds: int = 3,
    algorithm: str = "FedAvg",
    loss_fn_name: str = "CE",
    min_fit_clients: int = 2,
    min_available_clients: int = 2,
    secure_aggregation: bool = False,
    use_ckks: bool = False,
    test_images: Optional[torch.Tensor] = None,
    test_labels: Optional[torch.Tensor] = None,
) -> None:
    """Launch a Flower gRPC federated learning aggregation server."""
    strategy = FlowerFedXRayStrategy(
        central_server=central_server,
        total_rounds=num_rounds,
        algorithm=algorithm,
        loss_fn_name=loss_fn_name,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        secure_aggregation=secure_aggregation,
        use_ckks=use_ckks,
        test_images=test_images,
        test_labels=test_labels,
    )
    config = ServerConfig(num_rounds=num_rounds)
    logger.info("Starting Flower gRPC server on %s with %d rounds", server_address, num_rounds)
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )


def start_flower_client(
    hospital_client: HospitalClient,
    server_address: str = "127.0.0.1:8089",
    algorithm: str = "FedAvg",
    loss_fn_name: str = "CE",
) -> None:
    """Connect a clinical hospital node to a Flower gRPC server."""
    flower_client = FlowerHospitalClient(
        hospital_client=hospital_client,
        algorithm=algorithm,
        loss_fn_name=loss_fn_name,
    )
    logger.info(
        "Connecting HospitalClient %s to Flower gRPC server at %s",
        hospital_client.client_id,
        server_address,
    )
    fl.client.start_client(
        server_address=server_address,
        client=flower_client.to_client(),
    )

