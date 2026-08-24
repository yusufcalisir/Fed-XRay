"""
Fed-XRay Bilingual Single-Page PDF Report Generator
===================================================
Generates medical diagnosis reports with Grad-CAM heatmaps and case citations
in Turkish (TR) or English (EN).
"""

import os
import time
import tempfile
from datetime import datetime
from typing import Optional, List, Dict
import numpy as np


def _safe_latin1(text: str) -> str:
    """Sanitize Turkish characters for standard FPDF latin-1 fonts."""
    replacements = {
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'I': 'I', 'İ': 'I',
        'ş': 's', 'Ş': 'S',
        'ç': 'c', 'Ç': 'C',
        'ü': 'u', 'Ü': 'U',
        'ö': 'o', 'Ö': 'O',
        '’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def generate_medical_report(
    patient_id: str,
    diagnosis: str,
    confidence: float,
    explanation: str,
    heatmap_image: Optional[np.ndarray] = None,
    original_image: Optional[np.ndarray] = None,
    similar_cases: Optional[List[Dict]] = None,
    lang: str = "tr"
) -> bytes:
    """Generate a single-page localized medical PDF report."""
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    
    is_tr = (lang == "tr")
    
    pdf = FPDF(unit='mm', format='A4')
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pdf.set_margins(15, 10, 15)
    
    # Header
    pdf.set_y(10)
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(26, 54, 93)
    header_title = 'Fed-XRay: Medikal Zeka ve Teshis Raporu' if is_tr else 'Fed-XRay: Medical Intelligence Report'
    pdf.cell(0, 10, _safe_latin1(header_title), 0, 1, 'C')
    
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    header_sub = 'Yapay Zeka Destekli Federe Ogrenme Teshis Sistemi' if is_tr else 'AI-Powered Federated Learning Diagnostic System'
    pdf.cell(0, 4, _safe_latin1(header_sub), 0, 1, 'C')
    
    pdf.line(20, 24, 190, 24)
    pdf.set_y(26)
    
    # Patient Info & Diagnostic Summary
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 0, 0)
    info_header = 'HASTA BILGILERI' if is_tr else 'PATIENT INFORMATION'
    diag_header = 'TESHIS OZETI' if is_tr else 'DIAGNOSTIC SUMMARY'
    pdf.cell(90, 6, info_header, 0, 0)
    pdf.cell(90, 6, diag_header, 0, 1)
    
    pdf.set_font('Arial', '', 9)
    pdf.cell(90, 5, f'ID: {patient_id}', 0, 0)
    
    is_normal = diagnosis.lower() in ["normal", "normal (sağlıklı)", "normal (healthy)"]
    is_pneumonia = diagnosis.lower() in ["pneumonia", "zatürre", "zatürre (pneumonia)"]
    
    if is_normal:
        pdf.set_text_color(56, 161, 105)
    elif is_pneumonia:
        pdf.set_text_color(221, 107, 32)
    else:
        pdf.set_text_color(229, 62, 62)
        
    pdf.set_font('Arial', 'B', 10)
    res_label = 'SONUC' if is_tr else 'RESULT'
    pdf.cell(90, 5, f'{res_label}: {_safe_latin1(diagnosis.upper())}', 0, 1)
    
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    date_label = 'Tarih' if is_tr else 'Date'
    pdf.cell(90, 5, f'{date_label}: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0)
    pdf.set_font('Arial', 'B', 9)
    conf_label = 'GUVEN ORANI' if is_tr else 'CONFIDENCE'
    pdf.cell(90, 5, f'{conf_label}: %{confidence:.1f}' if is_tr else f'{conf_label}: {confidence:.1f}%', 0, 1)
    pdf.ln(2)
    
    temp_files = []
    
    if heatmap_image is not None and original_image is not None:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('Orijinal X-Ray' if is_tr else 'Original X-Ray', fontsize=8)
            axes[0].axis('off')
            
            axes[1].imshow(original_image, cmap='gray')
            axes[1].imshow(heatmap_image, cmap='jet', alpha=0.5)
            axes[1].set_title('AI Odak Alanlari (Grad-CAM)' if is_tr else 'AI Focus Areas (Grad-CAM)', fontsize=8)
            axes[1].axis('off')
            
            plt.tight_layout(pad=0.2)
            temp_path = os.path.join(tempfile.gettempdir(), f'fedxray_{int(time.time())}.png')
            plt.savefig(temp_path, dpi=110, bbox_inches='tight', facecolor='white')
            plt.close()
            temp_files.append(temp_path)
            
            pdf.set_font('Arial', 'B', 10)
            vis_title = 'GORSEL KANIT VE GRAD-CAM ANALIZI' if is_tr else 'VISUAL EVIDENCE ANALYSIS'
            pdf.cell(0, 7, vis_title, 0, 1)
            pdf.image(temp_path, x=45, w=120) 
            pdf.ln(2)
        except Exception as e:
            print(f"[PDF Image Error] {e}")
            pdf.cell(0, 10, f'Image Load Error: {str(e)[:40]}', 1, 1)
    
    # Interpretation Section
    pdf.set_font('Arial', 'B', 10)
    int_title = 'YAPAY ZEKA RADYOLOJIK BULGULARI' if is_tr else 'AI RADIOLOGICAL INTERPRETATION'
    pdf.cell(0, 7, int_title, 0, 1)
    pdf.set_font('Arial', '', 8.5)
    clean_explanation = explanation.replace('**', '').replace('*', '')
    pdf.multi_cell(0, 4.2, _safe_latin1(clean_explanation))
    pdf.ln(2)
    
    # Similar Cases Section
    if similar_cases:
        pdf.set_font('Arial', 'B', 10)
        sim_title = 'GECMIS VAKA KARSILASTIRMASI (RAG)' if is_tr else 'HISTORICAL CASE COMPARISON'
        pdf.cell(0, 7, sim_title, 0, 1)
        pdf.set_font('Arial', '', 8.5)
        for case in similar_cases[:2]:
            c_label = case['label']
            if is_tr:
                label_name = {0: "Normal", 1: "Zaturre", 2: "COVID-19"}.get(c_label, "Bilinmiyor")
                sim_str = f" • Vaka ID {case['case_id']} | Teshis: {label_name} | Benzerlik Skoru: %{case['similarity']*100:.1f}"
            else:
                label_name = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}.get(c_label, "Unknown")
                sim_str = f" • Case ID {case['case_id']} | Diagnosis: {label_name} | Similarity Score: {case['similarity']*100:.1f}%"
            pdf.cell(0, 4.5, _safe_latin1(sim_str), 0, 1)
    
    # Footer
    pdf.set_y(275)
    pdf.set_font('Arial', 'I', 7)
    pdf.set_text_color(140, 140, 140)
    foot1 = 'Fed-XRay Medikal Asistani tarafindan uretilmistir. Klinik korelasyon zorunludur.' if is_tr else 'Generated by Fed-XRay AI Diagnostic Assistant. Clinical correlation is mandatory.'
    foot2 = 'Gizli Medikal Dokuman - Tek Sayfa Resmi Rapor' if is_tr else 'Confidential Medical Document - Single Page Official Report'
    pdf.cell(0, 3.5, _safe_latin1(foot1), 0, 1, 'C')
    pdf.cell(0, 3.5, _safe_latin1(foot2), 0, 1, 'C')
    
    pdf_output = pdf.output(dest='S').encode('latin-1')
    
    for t_file in temp_files:
        try:
            if os.path.exists(t_file):
                os.remove(t_file)
        except:
            pass
    
    return pdf_output


def get_diagnosis_explanation(diagnosis: str, confidence: float, lang: str = "tr") -> str:
    """Get detailed textual findings and recommendations for diagnosis in TR or EN."""
    is_tr = (lang == "tr")
    is_normal = diagnosis.lower() in ["normal", "normal (sağlıklı)", "normal (healthy)"]
    is_pneumonia = diagnosis.lower() in ["pneumonia", "zatürre", "zatürre (pneumonia)"]
    
    if is_tr:
        if is_normal:
            return f"""Yapay zeka analizi taramayı %{confidence:.1f} güven oranıyla Normal (Sağlıklı) olarak değerlendirmiştir.

Klinik Bulgular:
- Akciğer parankiminde belirgin fokal veya diffüz infiltrasyon saptanmamıştır.
- Vasküler dallanmalar anatomik sınırlar içerisindedir.
- Kardiyotorasik oran ve mediasten konturları doğaldır.

Öneri: Klinik semptomlar aksini göstermedikçe acil takip gerekmemektedir."""
        elif is_pneumonia:
            return f"""Yapay zeka analizi %{confidence:.1f} güven oranıyla Zatürre (Pneumonia) bulguları tespit etmiştir.

Klinik Bulgular:
- Akciğer alanlarında fokal konsolidasyon ve opasite artışı gözlenmiştir.
- Bakteriyel veya tipik enfeksiyon paterniyle uyumlu lokalize tutulum mevcuttur.
- Konsolidasyon alanları içerisinde hava bronkogramları seçilebilmektedir.

Öneri: Klinik ve laboratuvar korelasyonu (WBC, CRP) ile antibiyoterapi planlaması önerilir."""
        else: # COVID-19
            return f"""Yapay zeka analizi %{confidence:.1f} güven oranıyla potansiyel COVID-19 tutulumu saptamıştır.

Klinik Bulgular:
- Her iki akciğerde diffüz, bilateral periferik buzlu cam (ground-glass) opasiteleri mevcuttur.
- İnterlobüler septal kalınlaşma ile uyumlu viral pnömoni bulguları izlenmektedir.

Öneri: Acil RT-PCR testi ve izolasyon protokollerinin değerlendirilmesi önerilir."""
    else:
        if is_normal:
            return f"""The AI analysis indicates a Normal scan with {confidence:.1f}% confidence.

Findings:
- No significant pulmonary parenchymal abnormalities detected.
- Lung fields appear clear with normal vascular arborization.
- Cardiomediastinal silhouette within normal limits.

Recommendation: No immediate follow-up required unless clinical symptoms suggest otherwise."""
        elif is_pneumonia:
            return f"""The AI analysis suggests Pneumonia with {confidence:.1f}% confidence.

Findings:
- Focal consolidation patterns detected in lung fields.
- Areas of increased opacity consistent with localized bacterial infection.
- Possible air bronchograms within consolidated regions.

Recommendation: Clinical correlation recommended. Consider antibiotic therapy if clinical presentation supports diagnosis."""
        else: # COVID-19
            return f"""The AI analysis indicates potential COVID-19 with {confidence:.1f}% confidence.

Findings:
- Diffuse bilateral ground-glass opacities observed across lung periphery.
- Peripheral distribution pattern characteristic of viral pneumonia.
- Possible interlobular septal thickening.

Recommendation: Urgent RT-PCR testing recommended. Implement clinical isolation protocols pending confirmation."""
