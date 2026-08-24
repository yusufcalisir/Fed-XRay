"""
Fed-XRay Unit Tests: Personalized Prototype Learning & Imbalance Losses
========================================================================
Tests DAFL, Balanced Softmax, Class-Balanced Loss, LDAM, and
PFAM-Fed dispersion-weighted prototype synthesis.
"""

import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.models.cnn import create_model
from src.fed_xray.core.imbalance_losses import (
    DynamicAdaptiveFocalLoss,
    BalancedSoftmaxLoss,
    ClassBalancedLoss,
    LDAMLoss,
    PrototypeRepelLoss
)
from src.fed_xray.core.prototypes import (
    extract_features,
    compute_local_prototypes_and_dispersion,
    aggregate_prototypes_dispersion_weighted,
    compute_prototype_distance_loss
)
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round


class TestPrototypeAndImbalanceLosses(unittest.TestCase):
    """Test imbalance losses and prototype metric learning."""

    def setUp(self):
        self.device = torch.device('cpu')
        self.model = create_model().to(self.device)
        self.class_counts = [100, 30, 10] # Imbalanced: 10:3:1
        
        # Synthetic mock dataset
        x = torch.randn(30, 1, 28, 28)
        y = torch.tensor([0]*20 + [1]*7 + [2]*3)
        ds = TensorDataset(x, y)
        self.loader = DataLoader(ds, batch_size=6)

    def test_dafl_loss(self):
        """Test Dynamic Adaptive Focal Loss computation across rounds."""
        logits = torch.randn(6, 3)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        
        dafl_r1 = DynamicAdaptiveFocalLoss(self.class_counts, current_round=1, total_rounds=10)
        loss1 = dafl_r1(logits, targets)
        self.assertIsInstance(loss1, torch.Tensor)
        self.assertGreater(loss1.item(), 0.0)
        
        dafl_r10 = DynamicAdaptiveFocalLoss(self.class_counts, current_round=10, total_rounds=10)
        loss10 = dafl_r10(logits, targets)
        self.assertIsInstance(loss10, torch.Tensor)

    def test_balanced_softmax_loss(self):
        """Test Balanced Softmax logit adjustment."""
        logits = torch.randn(6, 3)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        bsm = BalancedSoftmaxLoss(self.class_counts)
        loss = bsm(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0.0)

    def test_class_balanced_loss(self):
        """Test Class-Balanced Loss effective sample weighting."""
        logits = torch.randn(6, 3)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        cb = ClassBalancedLoss(self.class_counts, beta=0.999)
        loss = cb(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0.0)

    def test_ldam_loss(self):
        """Test LDAM margin loss."""
        logits = torch.randn(6, 3)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        ldam = LDAMLoss(self.class_counts)
        loss = ldam(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0.0)

    def test_prototype_extraction_and_dispersion(self):
        """Test local prototype synthesis and covariance trace."""
        protos, traces, counts = compute_local_prototypes_and_dispersion(
            model=self.model,
            dataloader=self.loader,
            device=self.device,
            num_classes=3
        )
        self.assertIn(0, protos)
        self.assertIn(1, protos)
        self.assertIn(2, protos)
        self.assertEqual(counts[0], 20)
        self.assertEqual(counts[1], 7)
        self.assertEqual(counts[2], 3)
        self.assertGreater(traces[0], 0.0)

    def test_pfam_dispersion_aggregation(self):
        """Test dispersion-weighted prototype synthesis."""
        client_protos = [
            {0: torch.randn(32), 1: torch.randn(32), 2: torch.randn(32)},
            {0: torch.randn(32), 1: torch.randn(32), 2: torch.randn(32)}
        ]
        client_traces = [{0: 0.1, 1: 0.5, 2: 1.0}, {0: 0.2, 1: 0.1, 2: 0.8}]
        client_counts = [{0: 50, 1: 10, 2: 5}, {0: 40, 1: 30, 2: 10}]
        
        global_protos = aggregate_prototypes_dispersion_weighted(
            client_prototypes=client_protos,
            client_traces=client_traces,
            client_counts=client_counts,
            num_classes=3
        )
        self.assertIn(0, global_protos)
        self.assertEqual(global_protos[0].shape, torch.Size([32]))

    def test_end_to_end_prototype_round(self):
        """Test end-to-end round with DAFL and prototype learning."""
        server = CentralServer(device=self.device, privacy_noise=0.0, defense_mode=False)
        client = HospitalClient(client_id=0, dataloader=self.loader, device=self.device, local_epochs=1)
        
        test_x = torch.randn(10, 1, 28, 28)
        test_y = torch.randint(0, 3, (10,))
        
        agg_m, cl_m, test_m, sec_rep = run_federated_round(
            server=server,
            clients=[client],
            round_num=1,
            total_rounds=5,
            test_images=test_x,
            test_labels=test_y,
            algorithm="FedAvg",
            loss_fn_name="DAFL",
            enable_prototypes=True,
            proto_weight=0.1
        )
        
        self.assertTrue(agg_m['prototypes_enabled'])
        self.assertEqual(agg_m['loss_fn'], "DAFL")
        self.assertIsNotNone(server.get_global_prototypes())


if __name__ == '__main__':
    unittest.main()
