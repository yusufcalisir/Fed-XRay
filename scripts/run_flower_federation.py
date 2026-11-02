#!/usr/bin/env python3
"""Fed-XRay Flower Federated Learning Multi-Mode CLI Runner.

Supports:
1. 'orchestrate': In-process end-to-end multi-hospital federated training orchestration.
2. 'server': Flower gRPC aggregation server listening for remote hospital nodes.
3. 'client': Flower gRPC hospital client connecting to an active aggregation server.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.fed_xray.core.client import HospitalClient  # noqa: E402
from src.fed_xray.core.server import CentralServer  # noqa: E402
from src.fed_xray.federation import (  # noqa: E402
    run_flower_federated_orchestration,
    start_flower_client,
    start_flower_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fed_xray.flower_runner")


def create_synthetic_dataloaders(
    num_clients: int = 3,
    samples_per_client: int = 16,
    img_size: int = 28,
    num_classes: int = 3,
    batch_size: int = 4,
    in_channels: int = 1,
) -> Tuple[List[DataLoader], Tuple[torch.Tensor, torch.Tensor]]:
    """Create lightweight synthetic dataloaders for local testing and benchmarking."""
    torch.manual_seed(42)
    dataloaders: List[DataLoader] = []
    for _ in range(num_clients):
        images = torch.randn(samples_per_client, in_channels, img_size, img_size)
        labels = torch.randint(0, num_classes, (samples_per_client,))
        ds = TensorDataset(images, labels)
        dataloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))

    test_images = torch.randn(12, in_channels, img_size, img_size)
    test_labels = torch.randint(0, num_classes, (12,))
    return dataloaders, (test_images, test_labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fed-XRay Flower Federated Learning CLI Runner")
    parser.add_argument(
        "--mode",
        choices=["orchestrate", "server", "client"],
        default="orchestrate",
        help="Execution mode: 'orchestrate' (in-process multi-client), 'server' (gRPC), or 'client' (gRPC)",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument(
        "--model-type",
        choices=["cnn", "resnet18", "densenet121", "vit_tiny", "vit_base_patch16_224"],
        default="cnn",
        help="Backbone architecture",
    )
    parser.add_argument(
        "--peft-mode",
        choices=["none", "lora", "ffa_lora", "fedsa_lora"],
        default="none",
        help="Parameter-Efficient Fine-Tuning adapter mode",
    )
    parser.add_argument("--algorithm", default="FedAvg", help="FL optimization algorithm (FedAvg, FedProx, FedDyn)")
    parser.add_argument("--loss-fn", default="CE", help="Loss function (CE, DAFL, Focal, LDAM)")
    parser.add_argument("--address", default="127.0.0.1:8089", help="Flower server host:port address")
    parser.add_argument("--client-id", type=int, default=0, help="Client ID for gRPC client mode")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients for orchestration mode")
    parser.add_argument("--privacy-noise", type=float, default=0.0, help="Gaussian DP noise multiplier sigma")
    parser.add_argument("--dp-clipping-bound", type=float, default=1.0, help="DP sensitivity clipping bound C")
    parser.add_argument(
        "--dp-mode",
        choices=["distributed", "central", "off"],
        default="distributed",
        help="Differential Privacy execution paradigm",
    )
    parser.add_argument(
        "--defense",
        choices=["none", "norm_clipping", "krum", "trimmed_mean", "bulyan"],
        default="none",
        help="Byzantine defense shield filter",
    )
    parser.add_argument("--use-secagg", action="store_true", help="Enable SecAgg+ cryptographic secure aggregation")
    parser.add_argument("--use-ckks", action="store_true", help="Enable Leveled CKKS Homomorphic Encryption")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    peft_mode = None if args.peft_mode.lower() in ("none", "false", "off") else args.peft_mode
    defense_mode = args.defense.lower() != "none"
    img_size = 28 if args.model_type == "cnn" else 224

    logger.info("Initializing Flower runner in mode: %s", args.mode)
    logger.info("Model: %s | PEFT: %s | Algorithm: %s", args.model_type, peft_mode, args.algorithm)

    dataloaders, (test_images, test_labels) = create_synthetic_dataloaders(
        num_clients=args.num_clients,
        img_size=img_size,
    )

    if args.mode == "orchestrate":
        server = CentralServer(
            model_type=args.model_type,
            peft_mode=peft_mode,
            privacy_noise=args.privacy_noise,
            defense_mode=defense_mode,
            secure_aggregation=args.use_secagg,
            use_ckks=args.use_ckks,
            dp_mode=args.dp_mode,
            dp_clipping_bound=args.dp_clipping_bound,
            img_size=img_size,
        )

        clients = [
            HospitalClient(
                client_id=i,
                dataloader=dataloaders[i],
                model_type=args.model_type,
                peft_mode=peft_mode,
                dp_enabled=(args.privacy_noise > 0.0),
                dp_clipping_bound=args.dp_clipping_bound,
                dp_noise_multiplier=args.privacy_noise,
                dp_mode=args.dp_mode,
                img_size=img_size,
            )
            for i in range(args.num_clients)
        ]

        def on_round(rnd: int, info: Dict):
            logger.info(
                "Round %d completed | Metrics: %s",
                rnd,
                info.get("metrics", {}),
            )

        summary = run_flower_federated_orchestration(
            server=server,
            clients=clients,
            num_rounds=args.rounds,
            algorithm=args.algorithm,
            loss_fn_name=args.loss_fn,
            secure_aggregation=args.use_secagg,
            use_ckks=args.use_ckks,
            test_images=test_images,
            test_labels=test_labels,
            on_round_end=on_round,
        )

        logger.info(
            "Orchestration finished successfully: %d rounds completed | Final epsilon: %.4f",
            summary["rounds_completed"],
            summary["privacy_epsilon"],
        )
        return 0

    elif args.mode == "server":
        server = CentralServer(
            model_type=args.model_type,
            peft_mode=peft_mode,
            privacy_noise=args.privacy_noise,
            defense_mode=defense_mode,
            secure_aggregation=args.use_secagg,
            use_ckks=args.use_ckks,
            dp_mode=args.dp_mode,
            dp_clipping_bound=args.dp_clipping_bound,
            img_size=img_size,
        )
        start_flower_server(
            central_server=server,
            server_address=args.address,
            num_rounds=args.rounds,
            algorithm=args.algorithm,
            loss_fn_name=args.loss_fn,
            min_fit_clients=max(1, args.num_clients),
            min_available_clients=max(1, args.num_clients),
            secure_aggregation=args.use_secagg,
            use_ckks=args.use_ckks,
            test_images=test_images,
            test_labels=test_labels,
        )
        return 0

    elif args.mode == "client":
        client_loader = dataloaders[args.client_id % len(dataloaders)]
        client = HospitalClient(
            client_id=args.client_id,
            dataloader=client_loader,
            model_type=args.model_type,
            peft_mode=peft_mode,
            dp_enabled=(args.privacy_noise > 0.0),
            dp_clipping_bound=args.dp_clipping_bound,
            dp_noise_multiplier=args.privacy_noise,
            dp_mode=args.dp_mode,
            img_size=img_size,
        )
        start_flower_client(
            hospital_client=client,
            server_address=args.address,
            algorithm=args.algorithm,
            loss_fn_name=args.loss_fn,
        )
        return 0

    else:
        logger.error("Unknown mode: %s", args.mode)
        return 1


if __name__ == "__main__":
    sys.exit(main())
