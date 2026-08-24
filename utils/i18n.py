"""
Fed-XRay Bilingual Medical Localization & Internationalization Engine (i18n)
=============================================================================
Provides structured dictionary translations for English (EN, default) and
Peer-Reviewed Clinical/Radiological Turkish (TR).
"""

from typing import Dict, Any

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "EN": {
        # App Header & Hero
        "app_title": "Fed-XRay",
        "app_badge": "Clinical AI Radiologist Network",
        "hero_title": "Fed-XRay | AI Radiologist Network",
        "hero_subtitle": "Privacy-Preserving Federated Learning for Lung Pathology Detection",
        "live_hud_active": "CONSORTIUM ACTIVE",
        "live_hud_nodes": "4 / 4 Hospital Nodes Online",
        "live_hud_model": "XRay-CNN (3,136 FL-Features)",
        "live_hud_shield": "Byzantine Defense Shield Active",
        
        # Sidebar Controls
        "sidebar_title": "Consortium Control Panel",
        "sidebar_tagline": "Decentralized Clinical AI",
        "sidebar_network_sec": "Hospital Network Topology",
        "sidebar_num_hospitals": "Participating Hospitals",
        "sidebar_num_hospitals_help": "Number of autonomous hospital client nodes in the federated network",
        "sidebar_samples_hospital": "Patient Scans per Site",
        "sidebar_samples_hospital_help": "Number of chest radiograph datasets simulated per institutional node",
        "sidebar_training_sec": "Federated Optimization",
        "sidebar_rounds": "Communication Rounds",
        "sidebar_rounds_help": "Number of global weight aggregation and distribution cycles",
        "sidebar_epochs": "Local Epochs per Round",
        "sidebar_epochs_help": "Number of optimization passes performed locally before parameter transmission",
        "sidebar_lr": "Local Learning Rate",
        "sidebar_lr_help": "Learning rate for local optimizer gradient updates",
        "sidebar_security_sec": "Adversarial Defense Shield",
        "sidebar_attack_sim": "Simulate Byzantine Attack (Node 3)",
        "sidebar_attack_sim_help": "Simulates label-flipping poisoning attack on Hospital 3",
        "sidebar_defense_mode": "Active Validation Defense Shield",
        "sidebar_defense_mode_help": "Filters outlier and poisoned updates using trusted reference validation",
        
        # Metrics KPI Grid
        "kpi_hospitals": "Hospitals",
        "kpi_rounds": "FL Rounds",
        "kpi_samples": "Total Patient Cohort",
        "kpi_status": "Consortium State",
        "kpi_status_ready": "Ready to Train",
        "kpi_status_trained": "Consensus Reached",
        "kpi_status_training": "Training Active",
        
        # Section 1: Ingestion
        "sec1_title": "Hospital Network Data Ingestion",
        "sec1_subtitle": "Non-IID patient cohorts with institutional prevalence skew",
        "sec1_btn_generate": "Generate Multi-Hospital Cohorts",
        "sec1_msg_generated": "Hospital cohorts generated. Global hold-out test set initialized for unbiased validation.",
        "sec1_dist_title": "Class Distribution",
        "sec1_stats_title": "Cohort Demographics",
        "sec1_normal": "Normal Parenchyma",
        "sec1_pneumonia": "Pneumonia",
        "sec1_covid": "COVID-19",
        "sec1_sample_gallery": "Representative Chest Radiographs",
        "sec1_hospital_prefix": "Hospital",
        
        # Section 2: Live Orchestration
        "sec2_title": "Federated Learning Orchestration Cockpit",
        "sec2_subtitle": "Real-time model synchronization across decentralized hospital nodes",
        "sec2_btn_start": "Start Federated Training Round",
        "sec2_progress": "Federated Round Execution",
        "sec2_round": "Round",
        "sec2_loss": "Global Training Loss",
        "sec2_acc": "Hold-Out Validation Accuracy",
        "sec2_f1": "Macro F1-Score",
        "sec2_precision": "Macro Precision",
        "sec2_recall": "Macro Recall",
        "sec2_convergence_chart": "Convergence & Optimization Telemetry",
        "sec2_client_contrib": "Client Parameter Contributions",
        "sec2_shield_secure": "Byzantine Shield: All Nodes Validated (Zero Poisoning Detected)",
        "sec2_shield_alert": "ALERT: Byzantine Threat Intercepted! Blocked Poisoned Updates from Node",
        "sec2_shield_off": "Byzantine Defense Shield: OFF (Vulnerable to Poisoning)",
        "sec2_economics_title": "Communication Economics & Parameter Efficiency",
        "sec2_economics_full": "Standard ViT Update: ~2.43 GB / Round",
        "sec2_economics_peft": "Fed-XRay Module: ~1.60 MB / Round",
        "sec2_economics_savings": "99.6% Bandwidth Reduction Achieved",
        
        # Section 3: Diagnostic CDSS
        "sec3_title": "AI Radiologist Diagnostic Workspace (CDSS)",
        "sec3_subtitle": "Explainable saliency analysis, evidence retrieval & official medical reporting",
        "sec3_scan_selector": "Select Patient Study for Examination",
        "sec3_btn_run_diag": "Execute Radiological AI Analysis",
        "sec3_results_header": "Clinical Diagnostic Assessment",
        "sec3_diagnosis": "Primary Diagnosis",
        "sec3_confidence": "Confidence Index",
        "sec3_ground_truth": "Ground Truth Pathology",
        "sec3_findings_title": "Radiological Findings & Morphological Interpretation",
        "sec3_recommendations_title": "Clinical Recommendations & Follow-Up",
        "sec3_dual_pane_orig": "Standard Chest Radiograph",
        "sec3_dual_pane_xai": "Grad-CAM Saliency Attention Map",
        "sec3_xai_controls": "Grad-CAM Heatmap Configuration",
        "sec3_xai_opacity": "Heatmap Blend Opacity",
        "sec3_xai_colormap": "Colormap Spectrum",
        
        # Section 3: RAG Digital Twin
        "sec3_rag_title": "Case-Based Reasoning: Digital Twin Matcher (Federated RAG)",
        "sec3_rag_desc": "Retrieval-Augmented Case Matching against verified reference oncology & pulmonary records",
        "sec3_rag_twin": "Matched Historical Case",
        "sec3_rag_sim": "Cosine Embedding Similarity",
        "sec3_rag_outcome": "Historical Biopsy & Therapeutic Outcome",
        
        # Section 3: Voice Assistant
        "sec3_voice_title": "Hands-Free Voice Diagnostic Assistant",
        "sec3_voice_desc": "Automated audio synthesis for clinical dictation and surgical hands-free review",
        "sec3_voice_listen": "Play Audio Diagnostic Briefing",
        
        # Section 3: PDF Report
        "sec3_report_title": "Official Single-Page Medical Intelligence Report",
        "sec3_report_desc": "Strictly bounded single-page A4 medical documentation with embedded Grad-CAM evidence",
        "sec3_btn_download_pdf": "Download Official Diagnostic Report (PDF)",
        
        # Footer
        "footer_text": "Fed-XRay | Privacy-Preserving Federated Oncology & Medical Imaging Consortium | Built with PyTorch"
    },
    
    "TR": {
        # App Header & Hero
        "app_title": "Fed-XRay",
        "app_badge": "Klinik Yapay Zeka Radyoloji Ağı",
        "hero_title": "Fed-XRay | Yapay Zeka Radyolog Ağı",
        "hero_subtitle": "Akciğer Patolojileri Teşhisi için Gizlilik Korumalı Federe Öğrenme Platformu",
        "live_hud_active": "KONSORSİYUM AKTİF",
        "live_hud_nodes": "4 / 4 Hastane Düğümü Çevrimiçi",
        "live_hud_model": "XRay-CNN (3.136 Federe Öznitelik)",
        "live_hud_shield": "Bizans Savunma Kalkanı Aktif",
        
        # Sidebar Controls
        "sidebar_title": "Konsorsiyum Kontrol Paneli",
        "sidebar_tagline": "Dağıtık Klinik Yapay Zeka",
        "sidebar_network_sec": "Hastane Ağı Topolojisi",
        "sidebar_num_hospitals": "Katılımcı Hastane Sayısı",
        "sidebar_num_hospitals_help": "Federe öğrenme ağındaki bağımsız hastane istemci düğümlerinin sayısı",
        "sidebar_samples_hospital": "Merkez Başına Hasta Taraması",
        "sidebar_samples_hospital_help": "Her hastane düğümünde yerel olarak üretilen akciğer grafisi veri miktarı",
        "sidebar_training_sec": "Federe Optimizasyon Parametreleri",
        "sidebar_rounds": "İletişim Turu Sayısı",
        "sidebar_rounds_help": "Küresel model parametrelerinin toplanıp dağıtılma döngüsü sayısı",
        "sidebar_epochs": "Tur Başına Yerel Epok",
        "sidebar_epochs_help": "Parametreler sunucuya gönderilmeden önce yerel veriyle yapılan optimizasyon adımı",
        "sidebar_lr": "Yerel Öğrenme Oranı",
        "sidebar_lr_help": "Yerel gradyan güncellemeleri için optimizasyon öğrenme katsayısı",
        "sidebar_security_sec": "Bizans Dayanıklı Güvenlik Kalkanı",
        "sidebar_attack_sim": "Düşmanca Saldırı Simülasyonu (Hastane #3)",
        "sidebar_attack_sim_help": "3. Hastanede etiket değiştirme (label flipping) veri zehirleme saldırısını simüle eder",
        "sidebar_defense_mode": "Aktif Doğrulama Güvenlik Kalkanı",
        "sidebar_defense_mode_help": "Zehirlenmiş ve sapan model güncellemelerini güvenilir doğrulama setiyle izole eder",
        
        # Metrics KPI Grid
        "kpi_hospitals": "Hastaneler",
        "kpi_rounds": "FL Turları",
        "kpi_samples": "Toplam Hasta Kohortu",
        "kpi_status": "Konsorsiyum Durumu",
        "kpi_status_ready": "Eğitime Hazır",
        "kpi_status_trained": "Konsensüs Sağlandı",
        "kpi_status_training": "Eğitim Devam Ediyor",
        
        # Section 1: Ingestion
        "sec1_title": "Hastane Ağı Veri Alma ve Dağılım Stüdyosu",
        "sec1_subtitle": "Kurumsal prevalans sapmasına sahip bağımsız Non-IID hasta kohortları",
        "sec1_btn_generate": "Çok Merkezli Hasta Kohortlarını Üret",
        "sec1_msg_generated": "Hastane verileri üretildi. Tarafsız değerlendirme için küresel referans doğrulama seti oluşturuldu.",
        "sec1_dist_title": "Sınıf Dağılımı (Prevalans)",
        "sec1_stats_title": "Kohort Demografisi",
        "sec1_normal": "Normal Parankim",
        "sec1_pneumonia": "Pnömoni (Konsolidasyon)",
        "sec1_covid": "COVID-19 (Buzlu Cam)",
        "sec1_sample_gallery": "Temsili Toraks Radyografileri",
        "sec1_hospital_prefix": "Hastane",
        
        # Section 2: Live Orchestration
        "sec2_title": "Federe Öğrenme Orkestrasyon Kokpiti",
        "sec2_subtitle": "Dağıtık hastane düğümleri arasında gerçek zamanlı ağırlık senkronizasyonu",
        "sec2_btn_start": "Federe Öğrenme Eğitim Turunu Başlat",
        "sec2_progress": "Federe Tur İlerlemesi",
        "sec2_round": "Tur",
        "sec2_loss": "Küresel Eğitim Kaybı",
        "sec2_acc": "Referans Doğrulama Başarısı",
        "sec2_f1": "Makro F1-Skoru",
        "sec2_precision": "Makro Kesinlik (Precision)",
        "sec2_recall": "Makro Duyarlılık (Recall)",
        "sec2_convergence_chart": "Yakınsama ve Optimizasyon Telemetrisi",
        "sec2_client_contrib": "İstemci Parametre Katkı Ağırlıkları",
        "sec2_shield_secure": "Bizans Kalkanı: Tüm Düğümler Doğrulandı (Sıfır Zehirleme Tespit Edildi)",
        "sec2_shield_alert": "UYARI: Düşmanca Tehdit Engellendi! Düğümden Gelen Zehirli Güncelleme İzole Edildi:",
        "sec2_shield_off": "Bizans Savunma Kalkanı: KAPALI (Zehirlenme Saldırılarına Karşı Savunmasız)",
        "sec2_economics_title": "İletişim Ekonomisi ve Parametre Verimliliği",
        "sec2_economics_full": "Standart ViT Güncellemesi: ~2.43 GB / Tur",
        "sec2_economics_peft": "Fed-XRay Modülü: ~1.60 MB / Tur",
        "sec2_economics_savings": "%99.6 Bant Genişliği Tasarrufu Sağlandı",
        
        # Section 3: Diagnostic CDSS
        "sec3_title": "Yapay Zeka Radyolog Teşhis Çalışma Alanı (CDSS)",
        "sec3_subtitle": "Açıklanabilir dikkat haritaları, vaka tabanlı çıkarım ve resmi medikal raporlama",
        "sec3_scan_selector": "İncelenecek Hasta Taramasını Seçin",
        "sec3_btn_run_diag": "Radyolojik Yapay Zeka Analizini Çalıştır",
        "sec3_results_header": "Klinik Tanı ve Teşhis Değerlendirmesi",
        "sec3_diagnosis": "Birincil Teşhis",
        "sec3_confidence": "Güven İndeksi",
        "sec3_ground_truth": "Doğrulanmış Referans Patoloji",
        "sec3_findings_title": "Radyolojik Bulgular ve Morfolojik Yorum",
        "sec3_recommendations_title": "Klinik Öneriler ve Takip Protokolü",
        "sec3_dual_pane_orig": "Standart Akciğer Radyografisi",
        "sec3_dual_pane_xai": "Grad-CAM Çıkarım ve Dikkat Haritası",
        "sec3_xai_controls": "Grad-CAM Isı Haritası Yapılandırması",
        "sec3_xai_opacity": "Isı Haritası Saydamlık Oranı",
        "sec3_xai_colormap": "Renk Spektrumu (Palet)",
        
        # Section 3: RAG Digital Twin
        "sec3_rag_title": "Vaka Tabanlı Çıkarım: Dijital İkiz Eşleştirici (Federe RAG)",
        "sec3_rag_desc": "Doğrulanmış onkolojik ve pulmoner referans arşivinden çıkarımla zenginleştirilmiş vaka eşleştirmesi",
        "sec3_rag_twin": "Eşleşen Tarihsel Referans Vaka",
        "sec3_rag_sim": "Kosinüs Vektör Benzerliği",
        "sec3_rag_outcome": "Tarihsel Biyopsi ve Tedavi Yanıtı",
        
        # Section 3: Voice Assistant
        "sec3_voice_title": "Eller-Serbest Sesli Teşhis Asistanı",
        "sec3_voice_desc": "Klinik dikte ve cerrahi steril ortamlarda kullanım için otomatik ses sentezi",
        "sec3_voice_listen": "Sesli Tanı Özetini Dinle",
        
        # Section 3: PDF Report
        "sec3_report_title": "Resmi Tek Sayfa Medikal İstihbarat Raporu",
        "sec3_report_desc": "Grad-CAM görsel kanıtları içeren, A4 boyutuna tam sığdırılmış resmi medikal doküman",
        "sec3_btn_download_pdf": "Resmi Teşhis Raporunu İndir (PDF)",
        
        # Footer
        "footer_text": "Fed-XRay | Gizlilik Korumalı Federe Onkoloji ve Tıbbi Görüntüleme Konsorsiyumu | PyTorch ile Geliştirilmiştir"
    }
}


def get_text(key: str, lang: str = "EN") -> str:
    """
    Retrieve localized string by key and language code.
    Falls back to English if key or language is not found.
    """
    selected_dict = TRANSLATIONS.get(lang.upper(), TRANSLATIONS["EN"])
    return selected_dict.get(key, TRANSLATIONS["EN"].get(key, key))


def get_all_texts(lang: str = "EN") -> Dict[str, str]:
    """Retrieve full translation dictionary for the active language."""
    return TRANSLATIONS.get(lang.upper(), TRANSLATIONS["EN"])
