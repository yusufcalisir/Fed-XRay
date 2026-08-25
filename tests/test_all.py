import unittest
import numpy as np
import torch
from unittest.mock import patch

from src.fed_xray.models.cnn import create_model, count_parameters
from src.fed_xray.data.generator import MedicalDataGenerator, get_distribution_info
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.cdss.xai import GradCAM, create_overlay


class TestMedicalData(unittest.TestCase):
    """Test synthetic medical data generation logic."""
    def test_generator_creation(self):
        generator = MedicalDataGenerator(seed=42)
        self.assertIsNotNone(generator)
        
    def test_single_xray_generation(self):
        generator = MedicalDataGenerator(seed=42)
        img = generator.generate_synthetic_xray(label=0)
        self.assertEqual(img.shape, (28, 28))
        self.assertTrue(np.max(img) <= 1.0)
        self.assertTrue(np.min(img) >= 0.0)

    def test_distribution_info(self):
        dist = get_distribution_info(0, 4)
        self.assertEqual(len(dist), 3)
        self.assertAlmostEqual(sum(dist.values()), 1.0)


class TestFederatedCore(unittest.TestCase):
    """Test federated server/client communications and Byzantine defenses."""
    def setUp(self):
        self.device = torch.device('cpu')
        self.generator = MedicalDataGenerator(seed=42)
        
    def test_model_creation(self):
        model = create_model()
        self.assertIsNotNone(model)
        params = count_parameters(model)
        self.assertGreater(params, 0)
        
    def test_server_aggregation_without_noise(self):
        server = CentralServer(device=self.device, privacy_noise=0.0, defense_mode=False)
        weights_before = server.get_global_weights()
        
        aggregated = server.aggregate([weights_before, weights_before], [100, 100])
        for key in weights_before.keys():
            self.assertTrue(torch.allclose(aggregated[key], weights_before[key]))

    @patch.object(CentralServer, '_validate_client_model')
    def test_server_defense_mechanism(self, mock_validate):
        server = CentralServer(device=self.device, privacy_noise=0.0, defense_mode=True)
        global_weights = server.get_global_weights()
        bad_weights = {name: torch.zeros_like(param) for name, param in global_weights.items()}
        
        mock_validate.side_effect = lambda w, img, lbl: 0.85 if w is global_weights else 0.15
        
        aggregated, report = server.validate_and_aggregate(
            client_weights=[global_weights, bad_weights],
            sample_counts=[100, 100],
            client_ids=[0, 1],
            test_images=torch.randn(10, 1, 28, 28),
            test_labels=torch.randint(0, 3, (10,))
        )
        
        self.assertIn(1, report.clients_blocked)
        self.assertEqual(len(report.clients_accepted), 1)
        self.assertEqual(report.clients_accepted[0], 0)


class TestExplainableAI(unittest.TestCase):
    """Test Grad-CAM heatmap generation and overlay utilities."""
    def test_gradcam_flow(self):
        model = create_model()
        model.eval()
        gradcam = GradCAM(model)
        
        img_tensor = torch.rand(1, 1, 28, 28, requires_grad=True)
        heatmap, predicted_class, confidence = gradcam.generate_heatmap(img_tensor)
        
        self.assertEqual(heatmap.shape, (28, 28))
        self.assertIn(predicted_class, [0, 1, 2])
        self.assertTrue(0.0 <= confidence <= 1.0)
        
        img_np = img_tensor.detach().numpy()[0, 0]
        overlay = create_overlay(img_np, heatmap)
        self.assertEqual(overlay.shape, (28, 28, 3))
        
        gradcam.remove_hooks()


if __name__ == '__main__':
    unittest.main()
