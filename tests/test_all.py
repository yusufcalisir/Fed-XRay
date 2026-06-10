import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# Local imports to test
from utils.cnn_model import create_model, count_parameters
from utils.medical_data import MedicalDataGenerator, get_distribution_info
from utils.federated_core import HospitalClient, CentralServer, run_federated_round
from utils.xai_engine import GradCAM, create_overlay
from utils.enterprise_architecture import (
    init_enterprise_state,
    log_event,
    generate_governance_pdf
)

class TestMedicalData(unittest.TestCase):
    """Test synthetic medical data generation logic."""
    def test_generator_creation(self):
        generator = MedicalDataGenerator(seed=42)
        self.assertIsNotNone(generator)
        
    def test_single_xray_generation(self):
        generator = MedicalDataGenerator(seed=42)
        # Generate normal (label 0)
        img = generator.generate_synthetic_xray(label=0)
        self.assertEqual(img.shape, (28, 28))
        self.assertTrue(np.max(img) <= 1.0)
        self.assertTrue(np.min(img) >= 0.0)

    def test_distribution_info(self):
        dist = get_distribution_info(0, 4)
        self.assertEqual(len(dist), 3)
        self.assertAlmostEqual(sum(dist), 1.0)


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
        
        # Test basic FedAvg aggregation with identical weights
        aggregated = server.aggregate([weights_before, weights_before], [100, 100])
        for key in weights_before.keys():
            self.assertTrue(torch.allclose(aggregated[key], weights_before[key]))

    def test_server_defense_mechanism(self):
        server = CentralServer(device=self.device, privacy_noise=0.0, defense_mode=True)
        global_weights = server.get_global_weights()
        
        # Mocking test set
        test_images = torch.rand(10, 1, 28, 28)
        test_labels = torch.randint(0, 3, (10,))
        
        # Create a mock bad model update (zeros)
        bad_weights = {name: torch.zeros_like(param) for name, param in global_weights.items()}
        
        # Validate and aggregate
        aggregated, report = server.validate_and_aggregate(
            client_weights=[global_weights, bad_weights],
            sample_counts=[100, 100],
            client_ids=[0, 1],
            test_images=test_images,
            test_labels=test_labels
        )
        
        # Client 1 (bad model) should be blocked under defense mode
        self.assertIn(1, report.clients_blocked)
        self.assertEqual(len(report.clients_accepted), 1)


class TestExplainableAI(unittest.TestCase):
    """Test Grad-CAM heatmap generation and overlay utilities."""
    def test_gradcam_flow(self):
        model = create_model()
        model.eval()
        gradcam = GradCAM(model)
        
        # Mock image batch: [1, 1, 28, 28]
        img_tensor = torch.rand(1, 1, 28, 28, requires_grad=True)
        heatmap, predicted_class, confidence = gradcam.generate_heatmap(img_tensor)
        
        self.assertEqual(heatmap.shape, (28, 28))
        self.assertIn(predicted_class, [0, 1, 28, 2]) # predicted labels in [0, 1, 2]
        self.assertTrue(0.0 <= confidence <= 1.0)
        
        # Test overlay rendering
        img_np = img_tensor.detach().numpy()[0, 0]
        overlay = create_overlay(img_np, heatmap)
        self.assertEqual(overlay.shape, (28, 28, 3))
        
        gradcam.remove_hooks()


class TestEnterpriseArchitecture(unittest.TestCase):
    """Test mock security controls (WAF, Zero Trust checks, PAM session elevation, and Compliance PDFs)."""
    @patch('streamlit.session_state', new_callable=dict)
    def test_enterprise_state_and_logging(self, mock_state):
        import streamlit as st
        # Bind st.session_state mock
        st.session_state = mock_state
        
        # Initialize
        init_enterprise_state()
        self.assertIn('siem_logs', st.session_state)
        self.assertIn('zt_policies', st.session_state)
        self.assertIn('pam_session', st.session_state)
        
        # Log an event
        log_event("WAF", "Test WAF log entry", "WARNING")
        self.assertEqual(st.session_state.siem_logs[0]["source"], "WAF")
        self.assertEqual(st.session_state.siem_logs[0]["event"], "Test WAF log entry")
        self.assertEqual(st.session_state.siem_logs[0]["level"], "WARNING")

    @patch('streamlit.session_state', new_callable=dict)
    def test_pdf_report_compilation(self, mock_state):
        import streamlit as st
        st.session_state = mock_state
        init_enterprise_state()
        
        pdf_bytes = generate_governance_pdf()
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 0)


if __name__ == '__main__':
    unittest.main()
