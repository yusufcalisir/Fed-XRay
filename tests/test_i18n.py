"""
Fed-XRay Unit Tests: Bilingual Localization & Design System 2.0
===============================================================
Tests translation key completeness, parameter interpolation,
and bilingual output for Voice, Report, and XAI engines.
"""

import unittest
from src.fed_xray.cdss.i18n import t, TRANSLATIONS
from src.fed_xray.cdss.styles import get_custom_css
from src.fed_xray.cdss.xai import get_explanation_text
from src.fed_xray.cdss.report import get_diagnosis_explanation, generate_medical_report
import numpy as np


class TestBilingualLocalization(unittest.TestCase):
    """Test bilingual i18n dictionary and translation resolution."""

    def test_key_presence_and_symmetry(self):
        """Ensure every key contains both 'tr' and 'en' translations."""
        for key, entry in TRANSLATIONS.items():
            self.assertIn('tr', entry, f"Missing Turkish translation for key: {key}")
            self.assertIn('en', entry, f"Missing English translation for key: {key}")
            self.assertTrue(len(entry['tr']) > 0, f"Empty Turkish translation for key: {key}")
            self.assertTrue(len(entry['en']) > 0, f"Empty English translation for key: {key}")

    def test_t_function_resolution(self):
        """Test t() helper function for basic translation and fallback."""
        # Test Turkish
        tr_title = t("app_title", "tr")
        self.assertIn("Yapay Zeka", tr_title)
        
        # Test English
        en_title = t("app_title", "en")
        self.assertIn("AI Radiologist", en_title)
        
        # Test missing key fallback
        fallback = t("non_existent_key_12345", "tr")
        self.assertEqual(fallback, "non_existent_key_12345")

    def test_xai_explanation_localization(self):
        """Test explanation text generation in Turkish and English."""
        exp_tr = get_explanation_text(predicted_class=1, confidence=0.85, lang="tr")
        self.assertIn("Zatürre", exp_tr)
        self.assertIn("%85.0", exp_tr)
        
        exp_en = get_explanation_text(predicted_class=1, confidence=0.85, lang="en")
        self.assertIn("Pneumonia", exp_en)
        self.assertIn("85.0%", exp_en)

    def test_report_explanation_localization(self):
        """Test diagnosis explanation strings in Turkish and English."""
        rep_tr = get_diagnosis_explanation(diagnosis="Zatürre", confidence=92.5, lang="tr")
        self.assertIn("Klinik Bulgular", rep_tr)
        
        rep_en = get_diagnosis_explanation(diagnosis="Pneumonia", confidence=92.5, lang="en")
        self.assertIn("Findings", rep_en)

    def test_pdf_report_compilation_bilingual(self):
        """Test generating PDF report in Turkish and English."""
        img = np.ones((28, 28), dtype=np.float32) * 0.5
        heat = np.ones((28, 28), dtype=np.float32) * 0.7
        
        # Turkish PDF
        pdf_tr = generate_medical_report(
            patient_id="PAT-999",
            diagnosis="Zatürre",
            confidence=90.0,
            explanation="Test açıklaması",
            heatmap_image=heat,
            original_image=img,
            lang="tr"
        )
        self.assertIsInstance(pdf_tr, bytes)
        self.assertGreater(len(pdf_tr), 0)
        
        # English PDF
        pdf_en = generate_medical_report(
            patient_id="PAT-999",
            diagnosis="Pneumonia",
            confidence=90.0,
            explanation="Test findings",
            heatmap_image=heat,
            original_image=img,
            lang="en"
        )
        self.assertIsInstance(pdf_en, bytes)
        self.assertGreater(len(pdf_en), 0)

    def test_custom_css_tokens(self):
        """Ensure CSS stylesheet contains modern tokens and responsive rules."""
        css = get_custom_css()
        self.assertIn("--clr-primary", css)
        self.assertIn("--clr-benign", css)
        self.assertIn(".fed-hero", css)
        self.assertIn(".fed-metric-card", css)
        self.assertIn("@media (max-width: 768px)", css)


if __name__ == '__main__':
    unittest.main()
