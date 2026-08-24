"""
Fed-XRay Bilingual Localization Engine (i18n)
==============================================
Provides structured translations, clinical terminology dictionaries,
and dynamic formatting for Turkish (TR) and English (EN).
"""

from typing import Dict, Any, Optional

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # ===== Brand & Navigation =====
    "app_title": {
        "tr": "Fed-XRay | Yapay Zeka Radyolog Ağı",
        "en": "Fed-XRay | AI Radiologist Network"
    },
    "app_subtitle": {
        "tr": "Akciğer Hastalıkları Tespiti İçin Gizlilik Korumalı Federe Öğrenme ve CDSS Platformu",
        "en": "Privacy-Preserving Federated Learning & CDSS Platform for Lung Disease Detection"
    },
    "sidebar_brand_sub": {
        "tr": "Federe Medikal Yapay Zeka & CDSS",
        "en": "Federated Medical AI & CDSS"
    },
    "language_selector": {
        "tr": "🌐 Dil / Language",
        "en": "🌐 Language / Dil"
    },

    # ===== Sidebar Sections =====
    "network_config": {
        "tr": "🏥 Hastane Ağı Yapılandırması",
        "en": "🏥 Hospital Network Configuration"
    },
    "num_hospitals": {
        "tr": "Hastane Sayısı",
        "en": "Number of Hospitals"
    },
    "num_hospitals_help": {
        "tr": "Federe ağdaki bağımsız klinik merkez sayısı",
        "en": "Number of independent clinical nodes in the federated network"
    },
    "samples_per_hospital": {
        "tr": "Hastane Başına Örnek Sayısı",
        "en": "Samples per Hospital"
    },
    "samples_per_hospital_help": {
        "tr": "Her hastanenin yerel olarak ürettiği X-Ray görüntü adedi",
        "en": "Number of X-Ray images generated locally per hospital"
    },
    "training_config": {
        "tr": "🔄 Federe Eğitim Parametreleri",
        "en": "🔄 Federated Training Parameters"
    },
    "fl_rounds": {
        "tr": "İletişim Turu Sayısı (Rounds)",
        "en": "Federated Rounds"
    },
    "fl_rounds_help": {
        "tr": "Merkezi sunucu ile hastaneler arasındaki ağırlık senkronizasyon turu",
        "en": "Number of global aggregation rounds between server and clients"
    },
    "local_epochs": {
        "tr": "Yerel Epoch Sayısı",
        "en": "Local Epochs"
    },
    "local_epochs_help": {
        "tr": "Hastanelerin her turda yapacağı yerel eğitim turu",
        "en": "Number of local training epochs per client per round"
    },
    "learning_rate": {
        "tr": "Öğrenme Oranı (Learning Rate)",
        "en": "Learning Rate"
    },
    "privacy_security_title": {
        "tr": "🛡️ Gizlilik ve Güvenlik Kalkanı",
        "en": "🛡️ Privacy & Security Shield"
    },
    "dp_noise": {
        "tr": "Diferansiyel Gizlilik Gürültüsü (DP Noise)",
        "en": "Differential Privacy Noise"
    },
    "dp_noise_help": {
        "tr": "Hasta verisi sızıntısını önlemek için ağırlıklara eklenen Gaussian gürültü miktarı",
        "en": "Gaussian noise added to aggregated weights to guarantee patient privacy"
    },
    "byzantine_defense": {
        "tr": "Byzantine Güvenlik Kalkanını Etkinleştir",
        "en": "Enable Byzantine Defense Shield"
    },
    "byzantine_defense_help": {
        "tr": "Zehirli veya saldırgan model güncellemelerini doğrulama setiyle otomatik filtreler",
        "en": "Automatically filters out poisoned or malicious model updates via validation"
    },
    "adversarial_simulation": {
        "tr": "Saldırı Simülasyonu (Hastane 3 Zehirleme)",
        "en": "Simulate Attack (Hospital #3 Poisoning)"
    },
    "adversarial_simulation_help": {
        "tr": "Hastane 3'te etiket çevirme saldırısı (Label Flipping) simüle eder",
        "en": "Simulates label flipping data poisoning attack on Hospital #3"
    },

    # ===== Metrics & Hero Badges =====
    "metric_hospitals": {
        "tr": "Hastane Sayısı",
        "en": "Hospitals"
    },
    "metric_rounds": {
        "tr": "Federe Tur",
        "en": "FL Rounds"
    },
    "metric_samples": {
        "tr": "Toplam Örnek",
        "en": "Total Samples"
    },
    "metric_status": {
        "tr": "Ağ Durumu",
        "en": "Network Status"
    },
    "status_ready": {
        "tr": "Eğitime Hazır",
        "en": "Ready"
    },
    "status_trained": {
        "tr": "Eğitildi",
        "en": "Trained"
    },
    "status_training": {
        "tr": "Eğitim Sürüyor...",
        "en": "Training..."
    },

    # ===== Tabs =====
    "tab_network": {
        "tr": "🏥 Hastane Ağı & Veri Dağılımı",
        "en": "🏥 Hospital Network & Data"
    },
    "tab_training": {
        "tr": "🔄 Federe Eğitim Stüdyosu",
        "en": "🔄 Federated Training Studio"
    },
    "tab_diagnosis": {
        "tr": "🩺 Klinik Teşhis & XAI Stüdyosu",
        "en": "🩺 Clinical Diagnosis & XAI Studio"
    },

    # ===== Disease Categories =====
    "disease_normal": {
        "tr": "Normal (Sağlıklı)",
        "en": "Normal"
    },
    "disease_pneumonia": {
        "tr": "Zatürre (Pneumonia)",
        "en": "Pneumonia"
    },
    "disease_covid": {
        "tr": "COVID-19",
        "en": "COVID-19"
    },

    # ===== Hospital Network Tab =====
    "network_overview_title": {
        "tr": "Heterojen Hastane Ağı & Non-IID Veri Profili",
        "en": "Heterogeneous Hospital Network & Non-IID Data Profile"
    },
    "network_overview_desc": {
        "tr": "Her hastane, hasta popülasyonuna bağlı olarak farklı bir hastalık dağılımına (Non-IID) sahiptir. Fed-XRay, bu heterojen veriler üzerinde ham veri paylaşmadan küresel konsensüs sağlar.",
        "en": "Each hospital exhibits a unique disease prevalence distribution (Non-IID). Fed-XRay achieves global consensus over heterogeneous silos without raw data transfer."
    },
    "hospital_label": {
        "tr": "Hastane",
        "en": "Hospital"
    },
    "profile_healthy": {
        "tr": "Genel Tarama Merkezi (Ağırlıklı Normal)",
        "en": "General Screening Center (Mostly Normal)"
    },
    "profile_pneumonia": {
        "tr": "Göğüs Hastalıkları İhtisas (Ağırlıklı Zatürre)",
        "en": "Pulmonology Specialist (Pneumonia Focus)"
    },
    "profile_covid": {
        "tr": "Pandemi Acil Kliniği (Ağırlıklı COVID-19)",
        "en": "Pandemic Urgent Care (COVID-19 Hotspot)"
    },
    "profile_balanced": {
        "tr": "Üniversite Araştırma Hastanesi (Dengeli)",
        "en": "University Medical Center (Balanced)"
    },

    # ===== Federated Training Tab =====
    "training_studio_title": {
        "tr": "Federe Model Eğitimi & Yakınsama Telemetrisi",
        "en": "Federated Model Training & Convergence Telemetry"
    },
    "btn_start_training": {
        "tr": "🚀 Federe Eğitimi Başlat",
        "en": "🚀 Start Federated Training"
    },
    "btn_training_running": {
        "tr": "⏳ Federe Eğitim Devam Ediyor...",
        "en": "⏳ Training in Progress..."
    },
    "training_complete_title": {
        "tr": "Federe Eğitim Başarıyla Tamamlandı",
        "en": "Federated Training Successfully Completed"
    },
    "chart_accuracy_title": {
        "tr": "Küresel Model Doğruluk Oranı (%)",
        "en": "Global Model Accuracy (%)"
    },
    "chart_loss_title": {
        "tr": "Eğitim Kayıp Değeri (Loss)",
        "en": "Training Loss Convergence"
    },
    "client_accuracy_title": {
        "tr": "Tur Bazlı İstemci Başarıları",
        "en": "Round-by-Round Client Performance"
    },
    "security_shield_active": {
        "tr": "Güvenlik Kalkanı: Aktif (Zehirli güncellemeler engellendi)",
        "en": "Security Shield: Active (Poisoned updates blocked)"
    },
    "security_shield_inactive": {
        "tr": "Güvenlik Kalkanı: Kapalı",
        "en": "Security Shield: Inactive"
    },

    # ===== Clinical Diagnosis Tab =====
    "cdss_studio_title": {
        "tr": "Klinik Teşhis & Açıklanabilir Yapay Zeka (XAI) Konsolu",
        "en": "Clinical Decision Support & Explainable AI (XAI) Console"
    },
    "cdss_studio_desc": {
        "tr": "Eğitilen federe küresel model ile bir hastanın akciğer X-Ray filmini inceleyin, Grad-CAM görsel kanıtlarını değerlendirin ve geçmiş benzer vakaları karşılaştırın.",
        "en": "Inspect patient X-Ray scans using the trained global model, evaluate Grad-CAM visual attention heatmaps, and retrieve biopsy-verified historical digital twins."
    },
    "patient_selection": {
        "tr": "Test Edilecek Örnek Hasta Seçimi",
        "en": "Select Test Patient Scan"
    },
    "patient_option_normal": {
        "tr": "Hasta A - Temiz Akciğer Taraması (Normal)",
        "en": "Patient A - Clear Lung Scan (Normal)"
    },
    "patient_option_pneumonia": {
        "tr": "Hasta B - Fokal Konsolidasyon (Zatürre)",
        "en": "Patient B - Focal Consolidation (Pneumonia)"
    },
    "patient_option_covid": {
        "tr": "Hasta C - Buzlu Cam Opasitesi (COVID-19)",
        "en": "Patient C - Ground-Glass Opacities (COVID-19)"
    },
    "btn_diagnose": {
        "tr": "🔬 X-Ray Taramasını Analiz Et",
        "en": "🔬 Analyze Patient Scan"
    },
    "diagnostic_result_header": {
        "tr": "Teşhis ve Güven Skoru",
        "en": "Diagnosis & Confidence Score"
    },
    "gradcam_section_title": {
        "tr": "Görsel Kanıt & Grad-CAM Isı Haritası Analizi",
        "en": "Visual Evidence & Grad-CAM Attention Heatmap"
    },
    "gradcam_slider_alpha": {
        "tr": "Isı Haritası Saydamlığı (Alpha Blend)",
        "en": "Heatmap Opacity (Alpha Blend)"
    },
    "original_scan": {
        "tr": "Orijinal X-Ray Taraması",
        "en": "Original X-Ray Scan"
    },
    "ai_attention_map": {
        "tr": "Yapay Zeka Odak Alanları (Grad-CAM)",
        "en": "AI Focus Areas (Grad-CAM)"
    },
    "blended_overlay": {
        "tr": "Klinik Çakıştırma Görünümü",
        "en": "Clinical Overlay Blend"
    },

    # ===== Case Retrieval (RAG) =====
    "similar_cases_title": {
        "tr": "Geçmiş Vaka Eşleştirme (Federated Case-Based RAG)",
        "en": "Historical Case Matching (Federated Case-Based RAG)"
    },
    "similar_cases_desc": {
        "tr": "Vektör gömme uzayında kosinüs benzerliği ile bulunan biyopsi onaylı benzer dijital ikiz vakalar:",
        "en": "Biopsy-verified digital twin cases matched via cosine similarity in visual embedding space:"
    },
    "case_id": {
        "tr": "Vaka ID",
        "en": "Case ID"
    },
    "similarity_score": {
        "tr": "Benzerlik Skoru",
        "en": "Similarity Score"
    },

    # ===== Voice & PDF =====
    "voice_summary_title": {
        "tr": "🔊 Sesli Teşhis Özeti (Yapay Zeka Radyolog Asistanı)",
        "en": "🔊 Audio Diagnostic Summary (AI Radiologist Assistant)"
    },
    "btn_download_pdf": {
        "tr": "📥 Tek Sayfa Resmi Medikal Raporu İndir (PDF)",
        "en": "📥 Download Official Medical Report (Single-Page PDF)"
    },
    "pdf_filename": {
        "tr": "FedXRay_Medikal_Rapor",
        "en": "FedXRay_Medical_Report"
    }
}


def t(key: str, lang: str = "tr", **kwargs: Any) -> str:
    """
    Retrieve localized string for the specified key and language.
    
    Args:
        key: Dictionary translation key
        lang: Target language code ('tr' or 'en')
        **kwargs: Format string arguments for interpolation
        
    Returns:
        Translated string, fallback to English or the key itself if missing.
    """
    lang_code = "tr" if lang == "tr" else "en"
    
    if key in TRANSLATIONS:
        text = TRANSLATIONS[key].get(lang_code, TRANSLATIONS[key].get("en", key))
    else:
        text = key
        
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
            
    return text
