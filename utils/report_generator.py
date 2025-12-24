"""
Fed-XRay: Professional PDF Report Generator
============================================
Generates medical diagnosis reports with Grad-CAM heatmaps.
Uses FPDF for PDF creation.
"""

import os
import time
import tempfile
from datetime import datetime
from typing import Optional, Dict
import numpy as np


def generate_medical_report(
    patient_id: str,
    diagnosis: str,
    confidence: float,
    explanation: str,
    heatmap_image: Optional[np.ndarray] = None,
    original_image: Optional[np.ndarray] = None,
    similar_cases: Optional[list] = None
) -> bytes:
    """
    Generate a professional medical PDF report.
    
    Args:
        patient_id: Unique patient identifier
        diagnosis: Predicted class name
        confidence: Confidence percentage (0-100)
        explanation: AI-generated explanation text
        heatmap_image: Grad-CAM heatmap as numpy array
        original_image: Original X-Ray image as numpy array
        similar_cases: List of similar historical cases
        
    Returns:
        PDF data as bytes
    """
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    
    # Create PDF with strict page limits
    pdf = FPDF(unit='mm', format='A4')
    pdf.set_auto_page_break(auto=False) # STRICT ONE PAGE
    pdf.add_page()
    pdf.set_margins(15, 10, 15)
    
    # Header - Extremely Compact
    pdf.set_y(10)
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 10, 'Fed-XRay: Medical Intelligence Report', 0, 1, 'C')
    
    # Subtitle
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 4, 'AI-Powered Federated Learning Diagnostic System', 0, 1, 'C')
    
    # Line separator - Minimal
    pdf.line(20, 24, 190, 24)
    pdf.set_y(26)
    
    # Patient Info & Diagnosis - Horizontal Split
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(90, 6, 'PATIENT INFORMATION', 0, 0)
    pdf.cell(90, 6, 'DIAGNOSTIC SUMMARY', 0, 1)
    
    pdf.set_font('Arial', '', 9)
    # Row 1
    pdf.cell(90, 5, f'ID: {patient_id}', 0, 0)
    # Diagnosis Result with Color logic
    if diagnosis == "Normal":
        pdf.set_text_color(56, 161, 105)
    elif diagnosis == "Pneumonia":
        pdf.set_text_color(221, 107, 32)
    else:
        pdf.set_text_color(229, 62, 62)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(90, 5, f'RESULT: {diagnosis.upper()}', 0, 1)
    
    # Row 2
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(90, 5, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(90, 5, f'CONFIDENCE: {confidence:.1f}%', 0, 1)
    
    pdf.ln(2)
    
    # Save images temporarily and add to PDF - More Compact
    temp_files = []
    
    if heatmap_image is not None and original_image is not None:
        try:
            # Side-by-side figure - Reduced size for one page
            fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
            
            # Original X-Ray
            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('Original X-Ray', fontsize=8)
            axes[0].axis('off')
            
            # Grad-CAM Heatmap
            axes[1].imshow(original_image, cmap='gray')
            axes[1].imshow(heatmap_image, cmap='jet', alpha=0.5)
            axes[1].set_title('AI Focus Areas', fontsize=8)
            axes[1].axis('off')
            
            plt.tight_layout(pad=0.2)
            
            # Save to temp file
            temp_path = os.path.join(tempfile.gettempdir(), f'fedxray_{int(time.time())}.png')
            plt.savefig(temp_path, dpi=110, bbox_inches='tight', facecolor='white')
            plt.close()
            temp_files.append(temp_path)
            
            # Add to PDF - Well Centered
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 7, 'VISUAL EVIDENCE ANALYSIS', 0, 1)
            pdf.image(temp_path, x=45, w=120) 
            pdf.ln(2)
            
        except Exception as e:
            print(f"[PDF Image Error] {e}")
            pdf.cell(0, 10, f'Image Load Error: {str(e)[:40]}', 1, 1)
    
    # Explanation Section - Denser
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, 'AI RADIOLOGICAL INTERPRETATION', 0, 1)
    
    pdf.set_font('Arial', '', 8.5)
    clean_explanation = explanation.replace('**', '').replace('*', '')
    # Wrap text densely
    pdf.multi_cell(0, 4.2, clean_explanation)
    pdf.ln(2)
    
    # Similar Cases Section - Single Line per case
    if similar_cases:
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 7, 'HISTORICAL CASE COMPARISON', 0, 1)
        
        pdf.set_font('Arial', '', 8.5)
        for i, case in enumerate(similar_cases[:2]):
            label_name = {0: "Normal", 1: "Pneumonia", 2: "COVID-19"}.get(case['label'], "Unknown")
            pdf.cell(0, 4.5, f" • Case ID {case['case_id']} | Diagnosis: {label_name} | Similarity Score: {case['similarity']*100:.1f}%", 0, 1)
    
    # Footer - Fixed at very bottom
    pdf.set_y(275) # A4 is 297mm
    pdf.set_font('Arial', 'I', 7)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 3.5, 'Generated by Fed-XRay AI Diagnostic Assistant. Clinical correlation is mandatory.', 0, 1, 'C')
    pdf.cell(0, 3.5, 'Confidential Medical Document - Single Page Official Report', 0, 1, 'C')
    
    # Get PDF as bytes
    pdf_output = pdf.output(dest='S').encode('latin-1')
    
    # Cleanup
    for t_file in temp_files:
        try:
            if os.path.exists(t_file):
                os.remove(t_file)
        except: pass
    
    return pdf_output


def get_diagnosis_explanation(diagnosis: str, confidence: float) -> str:
    """
    Get detailed explanation for diagnosis.
    
    Args:
        diagnosis: Predicted class name
        confidence: Confidence percentage
        
    Returns:
        Explanation text
    """
    if diagnosis == "Normal":
        return f"""The AI analysis indicates a Normal scan with {confidence:.1f}% confidence.

Findings:
- No significant pulmonary abnormalities detected
- Lung fields appear clear with normal vascular markings
- Cardiomediastinal silhouette within normal limits

Recommendation: No immediate follow-up required unless clinical symptoms suggest otherwise."""

    elif diagnosis == "Pneumonia":
        return f"""The AI analysis suggests Pneumonia with {confidence:.1f}% confidence.

Findings:
- Focal consolidation patterns detected
- Areas of increased opacity consistent with bacterial infection
- Possible air bronchograms within consolidated regions

Recommendation: Clinical correlation recommended. Consider antibiotic therapy if clinical presentation supports diagnosis."""

    else:  # COVID-19
        return f"""The AI analysis indicates potential COVID-19 with {confidence:.1f}% confidence.

Findings:
- Diffuse bilateral ground-glass opacities observed
- Peripheral distribution pattern characteristic of viral pneumonia
- Possible interlobular septal thickening

Recommendation: Urgent RT-PCR testing recommended. Implement isolation protocols pending confirmation."""
