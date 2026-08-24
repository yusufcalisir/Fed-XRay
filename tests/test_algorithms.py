"""
Fed-XRay Unit Tests: Advanced Distributed Optimization Algorithms
================================================================
Tests mathematical losses and round execution for:
- FedAvg
- FedProx
- SCAFFOLD
- FedDyn
- MOON
"""

import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.models.cnn import create_model
from src.fed_xray.core.algorithms import (
    compute_fedprox_loss,
    compute_feddyn_loss,
    compute_moon_contrastive_loss,
    ScaffoldController
)
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round


class TestDistributedAlgorithms(unittest.TestCase):
    """Test distributed optimization loss formulations and execution."""

    def setUp(self):
        self.device = torch.device('cpu')
        self.model = create_model().to(self.device)
        self.global_weights = {
            name: param.clone().detach()
            for name, param in self.model.state_dict().items()
        }
        
        # Synthetic mock dataset
        x = torch.randn(20, 1, 28, 28)
        y = torch.randint(0, 3, (20,))
        ds = TensorDataset(x, y)
        self.loader = DataLoader(ds, batch_size=4)

    def test_fedprox_loss(self):
        """Test FedProx proximal penalty calculation."""
        # When model has identical weights to global, loss must be 0
        loss_zero = compute_fedprox_loss(self.model, self.global_weights, mu=0.01)
        self.assertAlmostEqual(loss_zero.item(), 0.0, places=5)
        
        # Perturb one parameter
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(1.0)
                break
        loss_perturbed = compute_fedprox_loss(self.model, self.global_weights, mu=0.01)
        self.assertGreater(loss_perturbed.item(), 0.0)

    def test_feddyn_loss(self):
        """Test FedDyn dynamic risk penalty calculation."""
        prev_grads = {
            name: torch.ones_like(param)
            for name, param in self.model.named_parameters()
        }
        loss_dyn = compute_feddyn_loss(self.model, self.global_weights, prev_grads, alpha=0.01)
        self.assertIsInstance(loss_dyn, torch.Tensor)

    def test_moon_contrastive_loss(self):
        """Test MOON model-contrastive loss formulation."""
        z_loc = torch.randn(8, 32)
        z_glob = torch.randn(8, 32)
        z_prev = torch.randn(8, 32)
        
        loss_moon = compute_moon_contrastive_loss(z_loc, z_glob, z_prev, temperature=0.5, mu=1.0)
        self.assertIsInstance(loss_moon, torch.Tensor)
        self.assertGreater(loss_moon.item(), 0.0)

    def test_scaffold_controller(self):
        """Test SCAFFOLD controller state updates."""
        controller = ScaffoldController(self.model)
        client_c = controller.get_client_controls(client_id=0, model=self.model)
        self.assertIn("conv1.weight", client_c)
        
        deltas = [{"conv1.weight": torch.ones_like(client_c["conv1.weight"])}]
        controller.update_server_controls(deltas, num_total_clients=1)
        self.assertGreater(controller.server_controls["conv1.weight"].sum().item(), 0.0)

    def test_end_to_end_algorithm_rounds(self):
        """Test multi-round training execution across all 5 algorithms."""
        algorithms = ["FedAvg", "FedProx", "SCAFFOLD", "FedDyn", "MOON"]
        
        for algo in algorithms:
            server = CentralServer(device=self.device, privacy_noise=0.0, defense_mode=False)
            client = HospitalClient(client_id=0, dataloader=self.loader, device=self.device, local_epochs=1)
            
            test_x = torch.randn(10, 1, 28, 28)
            test_y = torch.randint(0, 3, (10,))
            
            agg_m, cl_m, test_m, sec_rep = run_federated_round(
                server=server,
                clients=[client],
                round_num=1,
                test_images=test_x,
                test_labels=test_y,
                algorithm=algo,
                mu=0.01,
                alpha=0.01,
                temperature=0.5
            )
            
            self.assertEqual(agg_m['round'], 1)
            self.assertEqual(agg_m['algorithm'], algo)
            self.assertIsNotNone(test_m)
            self.assertGreaterEqual(test_m.accuracy, 0.0)


if __name__ == '__main__':
    unittest.main()
