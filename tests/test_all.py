import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fed_xray.models.cnn import create_model, count_parameters
from src.fed_xray.data.generator import MedicalDataGenerator, get_distribution_info
from src.fed_xray.core.server import CentralServer
from src.fed_xray.core.client import HospitalClient
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

    def test_server_defense_mechanism(self):
        server = CentralServer(device=self.device, privacy_noise=0.0, defense_mode=True, model_type="cnn")
        global_weights = server.get_global_weights()

        # Distinguishable training distributions so the honest client achieves genuine high accuracy
        x_train = torch.cat([torch.ones(12, 1, 28, 28) * -2.0, torch.ones(12, 1, 28, 28) * 2.0])
        y_train = torch.cat([torch.zeros(12, dtype=torch.long), torch.ones(12, dtype=torch.long)])
        ds_train = TensorDataset(x_train, y_train)
        loader = DataLoader(ds_train, batch_size=6, shuffle=True)

        client = HospitalClient(0, loader, device=self.device, model_type="cnn", local_epochs=8, learning_rate=0.05)
        honest_weights, _, _ = client.train(global_weights)

        # Adversary produces inverted/poisoned weights causing performance collapse
        bad_weights = {k: -v * 10.0 for k, v in honest_weights.items()}

        test_images = torch.cat([torch.ones(10, 1, 28, 28) * -2.0, torch.ones(10, 1, 28, 28) * 2.0])
        test_labels = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])

        # Real model inference validation through _validate_client_model without mocks
        honest_acc, _ = server._validate_client_model(honest_weights, test_images, test_labels)
        bad_acc, _ = server._validate_client_model(bad_weights, test_images, test_labels)
        self.assertGreaterEqual(honest_acc, 0.80)
        self.assertLessEqual(bad_acc, 0.20)

        aggregated, report = server.validate_and_aggregate(
            client_weights=[honest_weights, bad_weights],
            sample_counts=[24, 24],
            client_ids=[0, 1],
            test_images=test_images,
            test_labels=test_labels,
        )

        self.assertIn(1, report.clients_blocked)
        self.assertEqual(len(report.clients_accepted), 1)
        self.assertEqual(report.clients_accepted[0], 0)
        self.assertTrue(report.threat_detected)
        self.assertIsNotNone(aggregated)


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
