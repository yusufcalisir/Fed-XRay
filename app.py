"""
Fed-XRay: Federated Learning Medical Diagnosis Dashboard (CDSS)
================================================================
A state-of-the-art privacy-preserving clinical AI platform for multi-hospital
lung pathology classification, explainable saliency mapping, case retrieval,
and automated medical report synthesis.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import time
import copy
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Local module imports
from utils.medical_data import (
    MedicalDataGenerator, 
    XRayDataset, 
    create_hospital_dataloaders,
    get_distribution_info,
    create_global_test_set
)
from utils.cnn_model import XRayClassifier, create_model, count_parameters
from utils.federated_core import (
    HospitalClient, 
    CentralServer, 
    run_federated_round,
    EvaluationMetrics,
    SecurityReport
)
from utils.xai_engine import GradCAM, create_overlay, get_explanation_text
from utils.similarity_engine import HistoricalCaseBank, extract_embedding, LABEL_NAMES, LABEL_COLORS
from utils.voice_engine import get_or_create_audio
from utils.report_generator import generate_medical_report, get_diagnosis_explanation
from utils.i18n import get_text


# ============================================================================
# PAGE CONFIGURATION & STATE INITIALIZATION
# ============================================================================

st.set_page_config(
    page_title="Fed-XRay | AI Radiologist Network",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize state variables for persistence across reruns."""
    if 'app_lang' not in st.session_state:
        st.session_state.app_lang = "EN"
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'global_model' not in st.session_state:
        st.session_state.global_model = None
    if 'trained_weights' not in st.session_state:
        st.session_state.trained_weights = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = {
            'loss': [], 'accuracy': [], 'round': [],
            'precision': [], 'recall': [], 'f1_score': [],
            'test_accuracy': [], 'test_loss': [],
            'blocked_count': 0
        }
    if 'hospital_data_generated' not in st.session_state:
        st.session_state.hospital_data_generated = False
    if 'hospital_samples' not in st.session_state:
        st.session_state.hospital_samples = {}
    if 'dataloaders' not in st.session_state:
        st.session_state.dataloaders = None
    if 'global_test_set' not in st.session_state:
        st.session_state.global_test_set = None
    if 'confusion_matrix' not in st.session_state:
        st.session_state.confusion_matrix = None
    if 'case_bank' not in st.session_state:
        st.session_state.case_bank = None
    if 'xai_opacity' not in st.session_state:
        st.session_state.xai_opacity = 0.55
    if 'xai_colormap' not in st.session_state:
        st.session_state.xai_colormap = "Hot"

init_session_state()
lang = st.session_state.app_lang


# ============================================================================
# MASTER DESIGN SYSTEM CSS (THEME-ADAPTIVE & RESPONSIVE)
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-display: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        
        --brand-blue: #2563EB;
        --brand-cyan: #0EA5E9;
        --status-emerald: #10B981;
        --status-amber: #F59E0B;
        --status-crimson: #EF4444;
    }

    /* Global Typography & Font Family Enforcement */
    html, body, [class*="css"], .stMarkdown, p, div, span, label, li {
        font-family: var(--font-main) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-display) !important;
        letter-spacing: -0.02em;
    }

    /* Top-Right Language Switcher Dock */
    .top-lang-dock {
        position: fixed;
        top: 0.85rem;
        right: 1.5rem;
        z-index: 999999;
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 0.35rem 0.65rem;
        border-radius: 9999px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Modern Hero Container */
    .hero-container {
        position: relative;
        text-align: center;
        padding: 3.5rem 1.5rem 3rem 1.5rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.9) 100%);
        backdrop-filter: blur(16px);
        border-radius: 24px;
        margin-bottom: 2rem;
        border: 1px solid rgba(226, 232, 240, 0.9);
        box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.06);
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -40%;
        left: 50%;
        transform: translateX(-50%);
        width: 140%;
        height: 140%;
        background: radial-gradient(circle, rgba(37, 99, 235, 0.07) 0%, rgba(14, 165, 233, 0.03) 40%, transparent 70%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, rgba(37,99,235,0.08) 0%, rgba(14,165,233,0.08) 100%);
        color: #1E40AF !important;
        border: 1px solid rgba(37,99,235,0.2);
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 1rem;
        letter-spacing: 0.02em;
    }

    .hero-title {
        font-size: clamp(2.2rem, 4.5vw, 3.6rem) !important;
        font-weight: 900 !important;
        letter-spacing: -0.03em !important;
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.75rem !important;
        line-height: 1.15 !important;
    }

    .hero-subtitle {
        font-size: clamp(1rem, 2vw, 1.25rem) !important;
        color: #475569 !important;
        max-width: 720px;
        margin: 0 auto !important;
        line-height: 1.6 !important;
        font-weight: 400;
    }

    /* Live Clinical Status HUD */
    .live-hud {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        background: #0F172A;
        color: white;
        padding: 0.85rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(255,255,255,0.1);
        gap: 1rem;
    }

    .hud-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .hud-pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
        animation: pulseAnimation 2s infinite;
    }

    @keyframes pulseAnimation {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.3); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* Responsive KPI Metric Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.25rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08);
    }

    .metric-card h3 {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        margin: 0 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card .value {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        font-family: var(--font-display) !important;
        margin-top: 0.75rem;
        line-height: 1;
    }

    /* Modern Section Headers */
    .modern-section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        color: white !important;
        padding: 1.1rem 1.5rem;
        border-radius: 16px;
        margin: 2.5rem 0 1.5rem 0;
        box-shadow: 0 8px 16px -4px rgba(15, 23, 42, 0.15);
    }

    .modern-section-header h2 {
        color: white !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }

    .modern-section-header p {
        color: rgba(255,255,255,0.75) !important;
        font-size: 0.85rem !important;
        margin: 0 !important;
    }

    /* Surface Cards & Containers */
    .glass-panel {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
    }

    .hospital-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    /* Saliency & Radiological Inspector */
    .xray-frame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        background: #000;
    }

    /* Diagnostic Outcome Cards */
    .diag-banner {
        padding: 1.25rem;
        border-radius: 16px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .diag-normal {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #065F46;
    }

    .diag-pneumonia {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #92400E;
    }

    .diag-covid {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #991B1B;
    }

    /* Action Buttons Custom Styling */
    .stButton > button {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.75rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25) !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.35) !important;
    }

    .stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Sidebar Aesthetics */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1120 0%, #0F172A 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Responsive Viewport Rules */
    @media (max-width: 767px) {
        .top-lang-dock { top: 0.5rem; right: 0.5rem; transform: scale(0.9); }
        .hero-container { padding: 2.25rem 1rem !important; margin-bottom: 1.5rem !important; }
        .hero-title { font-size: 2rem !important; }
        .hero-subtitle { font-size: 0.9rem !important; }
        .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; gap: 0.75rem !important; }
        .metric-card { padding: 1rem !important; }
        .metric-card .value { font-size: 1.6rem !important; }
        .live-hud { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
    }

    @media (min-width: 768px) and (max-width: 1023px) {
        .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; gap: 1rem !important; }
    }

    @media (min-width: 1440px) {
        .block-container { max-width: 1440px !important; margin: 0 auto !important; }
    }

    /* Clean Streamlit Watermarks */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# TOP-RIGHT BILINGUAL LANGUAGE SWITCHER (EN / TR)
# ============================================================================

col_hud_left, col_lang_right = st.columns([5, 1])

with col_lang_right:
    # Segmented pill switcher
    selected_lang = st.radio(
        label="Language",
        options=["EN", "TR"],
        index=0 if st.session_state.app_lang == "EN" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    if selected_lang != st.session_state.app_lang:
        st.session_state.app_lang = selected_lang
        st.rerun()

lang = st.session_state.app_lang


# ============================================================================
# SIDEBAR CONFIGURATION (CONSORTIUM CONTROLS)
# ============================================================================

with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size: 2.2rem; margin-bottom: 0.25rem;">🫁</div>
        <h2 style="font-size: 1.4rem; font-weight: 800; margin: 0; color: white !important;">{get_text('app_title', lang)}</h2>
        <p style="font-size: 0.78rem; color: #94A3B8 !important; margin: 0.25rem 0 0 0;">{get_text('sidebar_tagline', lang)}</p>
    </div>
    """, unsafe_allow_html=True)

    # Section 1: Hospital Network Topology
    st.markdown(f"#### 🏥 {get_text('sidebar_network_sec', lang)}")
    n_hospitals = st.slider(
        get_text("sidebar_num_hospitals", lang),
        min_value=2,
        max_value=8,
        value=4,
        help=get_text("sidebar_num_hospitals_help", lang)
    )
    
    samples_per_hospital = st.slider(
        get_text("sidebar_samples_hospital", lang),
        min_value=100,
        max_value=500,
        value=200,
        step=50,
        help=get_text("sidebar_samples_hospital_help", lang)
    )
    
    st.markdown("---")
    
    # Section 2: Federated Optimization
    st.markdown(f"#### 🔄 {get_text('sidebar_training_sec', lang)}")
    n_rounds = st.slider(
        get_text("sidebar_rounds", lang),
        min_value=1,
        max_value=20,
        value=5,
        help=get_text("sidebar_rounds_help", lang)
    )
    
    local_epochs = st.slider(
        get_text("sidebar_epochs", lang),
        min_value=1,
        max_value=5,
        value=2,
        help=get_text("sidebar_epochs_help", lang)
    )
    
    privacy_noise = st.slider(
        "Differential Privacy (ε)",
        min_value=0.0,
        max_value=0.1,
        value=0.01,
        step=0.01,
        format="%.2f"
    )
    
    st.markdown("---")
    
    # Section 3: Byzantine Adversarial Shield
    st.markdown(f"#### 🛡️ {get_text('sidebar_security_sec', lang)}")
    simulate_attack = st.checkbox(
        get_text("sidebar_attack_sim", lang),
        value=False,
        help=get_text("sidebar_attack_sim_help", lang)
    )
    
    activate_defense = st.checkbox(
        get_text("sidebar_defense_mode", lang),
        value=True,
        help=get_text("sidebar_defense_mode_help", lang)
    )
    
    if simulate_attack:
        st.markdown(f"""
        <div style="background: rgba(239,68,68,0.15); border: 1px solid #EF4444; padding: 0.6rem; border-radius: 10px; font-size: 0.8rem; color: #FCA5A5 !important;">
            ⚠️ <strong>{get_text('sec2_shield_alert', lang)} 3</strong>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN HERO SECTION & LIVE CLINICAL HUD
# ============================================================================

# Hero Container
st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">
        <span>✨</span> {get_text('app_badge', lang)}
    </div>
    <h1 class="hero-title">{get_text('hero_title', lang)}</h1>
    <p class="hero-subtitle">{get_text('hero_subtitle', lang)}</p>
</div>
""", unsafe_allow_html=True)

# Live Clinical HUD
status_kpi_text = get_text("kpi_status_trained", lang) if st.session_state.model_trained else get_text("kpi_status_ready", lang)

st.markdown(f"""
<div class="live-hud">
    <div class="hud-item">
        <div class="hud-pulse-dot"></div>
        <span>{get_text('live_hud_active', lang)}: {n_hospitals} / {n_hospitals} {get_text('kpi_hospitals', lang)}</span>
    </div>
    <div class="hud-item">
        <span>🧠 {get_text('live_hud_model', lang)}</span>
    </div>
    <div class="hud-item">
        <span>🛡️ {get_text('live_hud_shield', lang) if activate_defense else get_text('sec2_shield_off', lang)}</span>
    </div>
    <div class="hud-item">
        <span>📊 {get_text('kpi_status', lang)}: <strong>{status_kpi_text}</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick KPI Metrics Grid
total_cohort_samples = n_hospitals * samples_per_hospital

st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card">
        <h3>🏥 {get_text('kpi_hospitals', lang)}</h3>
        <div class="value">{n_hospitals}</div>
    </div>
    <div class="metric-card">
        <h3>🔄 {get_text('kpi_rounds', lang)}</h3>
        <div class="value">{n_rounds}</div>
    </div>
    <div class="metric-card">
        <h3>📊 {get_text('kpi_samples', lang)}</h3>
        <div class="value">{total_cohort_samples:,}</div>
    </div>
    <div class="metric-card">
        <h3>📡 {get_text('kpi_status', lang)}</h3>
        <div class="value" style="font-size: 1.4rem; color: {'#10B981' if st.session_state.model_trained else '#2563EB'} !important;">{status_kpi_text}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SECTION 1: HOSPITAL NETWORK DATA INGESTION & SKEW VISUALIZATION
# ============================================================================

st.markdown(f"""
<div class="modern-section-header">
    <div style="font-size: 1.8rem;">📊</div>
    <div>
        <h2>{get_text('sec1_title', lang)}</h2>
        <p>{get_text('sec1_subtitle', lang)}</p>
    </div>
</div>
""", unsafe_allow_html=True)

col_gen_btn, col_gen_info = st.columns([1, 2])

with col_gen_btn:
    btn_generate = st.button(f"🔬 {get_text('sec1_btn_generate', lang)}", key="btn_gen_cohorts")

if btn_generate:
    with st.spinner("Synthesizing patient cohorts with natural prevalence skew..."):
        generator = MedicalDataGenerator()
        st.session_state.hospital_samples = {}
        
        for h in range(n_hospitals):
            distribution = get_distribution_info(h, n_hospitals)
            images, labels = generator.create_hospital_data(
                n_samples=samples_per_hospital,
                distribution=distribution,
                hospital_id=h
            )
            st.session_state.hospital_samples[h] = {
                'images': images[:9],
                'labels': labels[:9],
                'distribution': distribution,
                'all_labels': labels
            }
        
        st.session_state.dataloaders = create_hospital_dataloaders(
            n_hospitals=n_hospitals,
            samples_per_hospital=samples_per_hospital,
            batch_size=32
        )
        
        test_images, test_labels = create_global_test_set(n_samples=300, seed=9999)
        st.session_state.global_test_set = (test_images, test_labels)
        st.session_state.hospital_data_generated = True
        
    st.success(f"✅ {get_text('sec1_msg_generated', lang)}")
    st.rerun()

# Display hospital data
if st.session_state.hospital_data_generated and st.session_state.hospital_samples:
    institutional_names = [
        "Metropolitan General (Pulmonology Hub)",
        "St. Jude Infectious Disease Center",
        "Community Memorial Health Network",
        "University Medical Academy",
        "St. Mary Pulmonary Screening Clinic",
        "Regional Trauma & ICU Center",
        "Coastline Diagnostic Institute",
        "Mount Sinai Respiratory Lab"
    ]
    
    hospital_tabs = st.tabs([
        f"🏥 {get_text('sec1_hospital_prefix', lang)} {i+1}" 
        for i in range(min(n_hospitals, len(st.session_state.hospital_samples)))
    ])
    
    for h_idx, tab in enumerate(hospital_tabs):
        if h_idx not in st.session_state.hospital_samples:
            continue
            
        with tab:
            data = st.session_state.hospital_samples[h_idx]
            inst_name = institutional_names[h_idx % len(institutional_names)]
            
            st.markdown(f"""
            <div class="hospital-card">
                <strong style="font-size: 1.1rem; color: #0F172A;">{inst_name}</strong><br>
                <span style="font-size: 0.85rem; color: #64748B;">Node ID: <code>client_node_{h_idx+1}</code> | Total Cohort: <strong>{samples_per_hospital} Scans</strong></span>
            </div>
            """, unsafe_allow_html=True)
            
            col_chart, col_gallery = st.columns([1, 1.5])
            
            with col_chart:
                dist = data['distribution']
                class_labels = [
                    get_text("sec1_normal", lang), 
                    get_text("sec1_pneumonia", lang), 
                    get_text("sec1_covid", lang)
                ]
                palette = ['#10B981', '#F59E0B', '#EF4444']
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=class_labels,
                    values=[dist[0], dist[1], dist[2]],
                    marker=dict(colors=palette, line=dict(color='white', width=2)),
                    hole=0.62,
                    textinfo='percent',
                    textfont=dict(size=14, color='white', family="Inter"),
                    hoverinfo='label+percent+value'
                )])
                
                fig_pie.update_layout(
                    annotations=[dict(text=f"{samples_per_hospital}<br>SCANS", x=0.5, y=0.5, font_size=15, showarrow=False, font_family="Outfit", font_color="#0F172A")],
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_gallery:
                st.markdown(f"**{get_text('sec1_sample_gallery', lang)} (3x3 Grid):**")
                images = data['images'][:9]
                sample_labels = data['labels'][:9]
                
                fig_grid = make_subplots(
                    rows=3, cols=3,
                    subplot_titles=[class_labels[l] for l in sample_labels],
                    vertical_spacing=0.12,
                    horizontal_spacing=0.06
                )
                
                for i in range(min(9, len(images))):
                    r = i // 3 + 1
                    c = i % 3 + 1
                    fig_grid.add_trace(
                        go.Heatmap(z=images[i], colorscale='gray', showscale=False),
                        row=r, col=c
                    )
                
                fig_grid.update_layout(height=340, margin=dict(l=5, r=5, t=30, b=5))
                for i in range(1, 10):
                    fig_grid.update_xaxes(showticklabels=False, showgrid=False, row=(i-1)//3+1, col=(i-1)%3+1)
                    fig_grid.update_yaxes(showticklabels=False, showgrid=False, row=(i-1)//3+1, col=(i-1)%3+1)
                
                st.plotly_chart(fig_grid, use_container_width=True)


# ============================================================================
# SECTION 2: FEDERATED LEARNING ORCHESTRATION COCKPIT
# ============================================================================

st.markdown(f"""
<div class="modern-section-header">
    <div style="font-size: 1.8rem;">🚀</div>
    <div>
        <h2>{get_text('sec2_title', lang)}</h2>
        <p>{get_text('sec2_subtitle', lang)}</p>
    </div>
</div>
""", unsafe_allow_html=True)

col_train_btn, col_train_economics = st.columns([1, 2])

with col_train_btn:
    btn_start_fl = st.button(
        f"⚡ {get_text('sec2_btn_start', lang)}",
        key="btn_start_fl_rounds",
        disabled=not st.session_state.hospital_data_generated
    )
    if not st.session_state.hospital_data_generated:
        st.info(f"💡 {get_text('sec1_btn_generate', lang)} first.")

with col_train_economics:
    st.markdown(f"""
    <div class="hospital-card" style="margin-bottom: 0;">
        <strong style="color: #0F172A; font-size: 0.95rem;">📦 {get_text('sec2_economics_title', lang)}</strong>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
            <span style="font-size: 0.8rem; color: #64748B;">{get_text('sec2_economics_full', lang)}</span>
            <span style="font-size: 0.8rem; color: #10B981; font-weight: 700;">{get_text('sec2_economics_peft', lang)}</span>
            <span style="background: rgba(16,185,129,0.15); color: #065F46; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem;">
                {get_text('sec2_economics_savings', lang)}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

chart_placeholder = st.empty()
progress_placeholder = st.empty()
shield_placeholder = st.empty()

if btn_start_fl and st.session_state.hospital_data_generated:
    st.session_state.training_history = {
        'loss': [], 'accuracy': [], 'round': [],
        'precision': [], 'recall': [], 'f1_score': [],
        'test_accuracy': [], 'test_loss': [],
        'blocked_count': 0
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    server = CentralServer(device=device, privacy_noise=privacy_noise, defense_mode=activate_defense)
    
    clients = []
    for i in range(n_hospitals):
        is_malicious = (simulate_attack and i == 2)
        clients.append(HospitalClient(
            client_id=i,
            dataloader=st.session_state.dataloaders[i],
            device=device,
            learning_rate=0.0001,
            local_epochs=local_epochs,
            malicious=is_malicious
        ))
    
    test_images, test_labels = st.session_state.global_test_set
    
    for round_num in range(1, n_rounds + 1):
        with progress_placeholder.container():
            st.progress(round_num / n_rounds, text=f"{get_text('sec2_progress', lang)}: {round_num}/{n_rounds}")
        
        metrics, client_metrics, test_metrics, security_report = run_federated_round(
            server, clients, round_num,
            test_images=test_images,
            test_labels=test_labels,
            use_defense=activate_defense
        )
        
        st.session_state.training_history['loss'].append(metrics['loss'])
        st.session_state.training_history['accuracy'].append(metrics['accuracy'] * 100)
        st.session_state.training_history['round'].append(round_num)
        
        if test_metrics:
            st.session_state.training_history['test_accuracy'].append(test_metrics.accuracy * 100)
            st.session_state.training_history['test_loss'].append(test_metrics.loss)
            st.session_state.training_history['precision'].append(test_metrics.precision * 100)
            st.session_state.training_history['recall'].append(test_metrics.recall * 100)
            st.session_state.training_history['f1_score'].append(test_metrics.f1_score * 100)
            st.session_state.confusion_matrix = test_metrics.confusion_matrix
        
        # Real-time Telemetry Charts
        with chart_placeholder.container():
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=st.session_state.training_history['round'],
                    y=st.session_state.training_history['test_loss'],
                    mode='lines+markers',
                    name=get_text('sec2_loss', lang),
                    line=dict(color='#EF4444', width=3),
                    marker=dict(size=7)
                ))
                fig_loss.update_layout(
                    title=f"📉 {get_text('sec2_loss', lang)}",
                    xaxis_title=get_text('sec2_round', lang),
                    yaxis_title="Loss",
                    height=260,
                    margin=dict(l=30, r=20, t=40, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", color='#0F172A')
                )
                st.plotly_chart(fig_loss, use_container_width=True)
                
            with col_t2:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=st.session_state.training_history['round'],
                    y=st.session_state.training_history['test_accuracy'],
                    mode='lines+markers',
                    name=get_text('sec2_acc', lang),
                    line=dict(color='#10B981', width=3),
                    marker=dict(size=7)
                ))
                fig_acc.update_layout(
                    title=f"📈 {get_text('sec2_acc', lang)} (%)",
                    xaxis_title=get_text('sec2_round', lang),
                    yaxis_title="Accuracy %",
                    yaxis_range=[0, 100],
                    height=260,
                    margin=dict(l=30, r=20, t=40, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", color='#0F172A')
                )
                st.plotly_chart(fig_acc, use_container_width=True)
        
        # Shield report
        with shield_placeholder.container():
            if security_report and security_report.clients_blocked:
                st.markdown(f"""
                <div style="background: #FEE2E2; border: 1px solid #EF4444; padding: 0.75rem 1rem; border-radius: 12px; color: #991B1B; font-size: 0.9rem;">
                    🚨 <strong>{get_text('sec2_shield_alert', lang)}:</strong> {[c+1 for c in security_report.clients_blocked]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #D1FAE5; border: 1px solid #10B981; padding: 0.75rem 1rem; border-radius: 12px; color: #065F46; font-size: 0.9rem;">
                    ✅ <strong>{get_text('sec2_shield_secure', lang)}</strong>
                </div>
                """, unsafe_allow_html=True)
                
        time.sleep(0.15)
        
    trained_model = server.get_model()
    st.session_state.global_model = trained_model
    st.session_state.trained_weights = copy.deepcopy(trained_model.state_dict())
    st.session_state.model_trained = True
    progress_placeholder.empty()
    st.rerun()

# Display persistent telemetry charts if already trained
elif st.session_state.training_history['round']:
    with chart_placeholder.container():
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=st.session_state.training_history['round'],
                y=st.session_state.training_history['test_loss'] if st.session_state.training_history['test_loss'] else st.session_state.training_history['loss'],
                mode='lines+markers',
                line=dict(color='#EF4444', width=3)
            ))
            fig_loss.update_layout(title=f"📉 {get_text('sec2_loss', lang)}", height=260, margin=dict(l=30, r=20, t=40, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
            st.plotly_chart(fig_loss, use_container_width=True)
        with col_t2:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=st.session_state.training_history['round'],
                y=st.session_state.training_history['test_accuracy'] if st.session_state.training_history['test_accuracy'] else st.session_state.training_history['accuracy'],
                mode='lines+markers',
                line=dict(color='#10B981', width=3)
            ))
            fig_acc.update_layout(title=f"📈 {get_text('sec2_acc', lang)} (%)", height=260, margin=dict(l=30, r=20, t=40, b=30), yaxis_range=[0, 100], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
            st.plotly_chart(fig_acc, use_container_width=True)


# ============================================================================
# SECTION 3: AI RADIOLOGIST DIAGNOSTIC WORKSPACE & EXPLAINABILITY (CDSS)
# ============================================================================

st.markdown(f"""
<div class="modern-section-header">
    <div style="font-size: 1.8rem;">🩺</div>
    <div>
        <h2>{get_text('sec3_title', lang)}</h2>
        <p>{get_text('sec3_subtitle', lang)}</p>
    </div>
</div>
""", unsafe_allow_html=True)

col_scan_act, col_scan_opt = st.columns([1, 2])

with col_scan_act:
    btn_scan_patient = st.button(
        f"🔍 {get_text('sec3_btn_run_diag', lang)}",
        key="btn_scan_patient_inference",
        disabled=not st.session_state.model_trained
    )
    if not st.session_state.model_trained:
        st.warning(f"⚠️ {get_text('sec2_btn_start', lang)} first.")

with col_scan_opt:
    col_alpha, col_cmap = st.columns(2)
    with col_alpha:
        opacity_val = st.slider(get_text("sec3_xai_opacity", lang), 0.1, 1.0, float(st.session_state.xai_opacity), 0.05)
        st.session_state.xai_opacity = opacity_val
    with col_cmap:
        cmap_val = st.selectbox(get_text("sec3_xai_colormap", lang), ["Hot", "Jet", "Turbo", "Magma", "Viridis"], index=0)
        st.session_state.xai_colormap = cmap_val

if btn_scan_patient and st.session_state.model_trained:
    generator = MedicalDataGenerator(seed=int(time.time()))
    true_label = np.random.randint(0, 3)
    raw_img = generator.generate_synthetic_xray(true_label)
    
    st.session_state.scan_image = raw_img
    st.session_state.scan_true_label = true_label
    
    model = create_model()
    if st.session_state.trained_weights is not None:
        model.load_state_dict(st.session_state.trained_weights)
    model.cpu().eval()
    
    img_tensor = torch.FloatTensor(raw_img).unsqueeze(0).unsqueeze(0)
    torch.set_grad_enabled(True)
    gradcam = GradCAM(model)
    heatmap, predicted_class, confidence = gradcam.generate_heatmap(img_tensor)
    confidence_pct = confidence * 100.0
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
        if np.any(np.isnan(probs)):
            probs = np.array([0.34, 0.33, 0.33])
            
    overlay_img = create_overlay(raw_img, heatmap, alpha=st.session_state.xai_opacity)
    
    st.session_state.scan_heatmap = heatmap
    st.session_state.scan_overlay = overlay_img
    st.session_state.scan_probs = probs
    st.session_state.scan_predicted = predicted_class
    st.session_state.scan_confidence = confidence_pct
    gradcam.remove_hooks()

# Render Diagnostic Results
if hasattr(st.session_state, 'scan_image') and st.session_state.model_trained:
    class_names_local = [
        get_text("sec1_normal", lang), 
        get_text("sec1_pneumonia", lang), 
        get_text("sec1_covid", lang)
    ]
    pred_idx = st.session_state.scan_predicted
    pred_name = class_names_local[pred_idx]
    conf_val = st.session_state.scan_confidence
    
    banner_class = "diag-normal" if pred_idx == 0 else ("diag-pneumonia" if pred_idx == 1 else "diag-covid")
    emoji_icon = "🟢" if pred_idx == 0 else ("🟠" if pred_idx == 1 else "🔴")
    
    st.markdown(f"""
    <div class="diag-banner {banner_class}">
        <div>
            <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">{get_text('sec3_diagnosis', lang)}</span>
            <div style="font-size: 1.8rem; font-weight: 800; font-family: Outfit, sans-serif; margin-top: 0.2rem;">
                {emoji_icon} {pred_name}
            </div>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 0.85rem; font-weight: 600;">{get_text('sec3_confidence', lang)}</span>
            <div style="font-size: 1.8rem; font-weight: 800; font-family: Outfit, sans-serif;">{conf_val:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dual-Pane Radiological Inspector
    col_pane_raw, col_pane_xai, col_pane_conf = st.columns([1, 1, 1.2])
    
    with col_pane_raw:
        st.markdown(f"**📷 {get_text('sec3_dual_pane_orig', lang)}**")
        fig_r = go.Figure(data=go.Heatmap(z=st.session_state.scan_image, colorscale='gray', showscale=False))
        fig_r.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor='x'))
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption(f"Ground Truth: **{class_names_local[st.session_state.scan_true_label]}**")
        
    with col_pane_xai:
        st.markdown(f"**🔬 {get_text('sec3_dual_pane_xai', lang)}**")
        fig_x = go.Figure(data=go.Image(z=(create_overlay(st.session_state.scan_image, st.session_state.scan_heatmap, alpha=st.session_state.xai_opacity) * 255).astype(np.uint8)))
        fig_x.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
        st.plotly_chart(fig_x, use_container_width=True)
        st.caption(f"Grad-CAM Attention Overlay ({st.session_state.xai_colormap} Palette)")
        
    with col_pane_conf:
        st.markdown(f"**📊 {get_text('sec3_results_header', lang)}**")
        probs = st.session_state.scan_probs
        fig_b = go.Figure(data=go.Bar(
            x=probs * 100,
            y=class_names_local,
            orientation='h',
            marker=dict(color=['#10B981', '#F59E0B', '#EF4444']),
            text=[f"{p*100:.1f}%" for p in probs],
            textposition='auto',
            textfont=dict(family="Inter", size=12, color='white')
        ))
        fig_b.update_layout(
            height=260,
            margin=dict(l=70, r=20, t=10, b=20),
            xaxis=dict(showgrid=False, range=[0, 100]),
            yaxis=dict(showgrid=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("---")
    
    # Clinical Recommendations & Case-Based Reasoning (RAG)
    col_cbr, col_cdss_media = st.columns([1.2, 1])
    
    with col_cbr:
        st.markdown(f"### 🧬 {get_text('sec3_rag_title', lang)}")
        st.caption(get_text('sec3_rag_desc', lang))
        
        if st.session_state.case_bank is None:
            st.session_state.case_bank = HistoricalCaseBank(n_cases=100)
            
        model_emb = create_model()
        if st.session_state.trained_weights is not None:
            model_emb.load_state_dict(st.session_state.trained_weights)
        
        img_t = torch.FloatTensor(st.session_state.scan_image).unsqueeze(0).unsqueeze(0)
        emb = extract_embedding(model_emb, img_t)
        matched_cases = st.session_state.case_bank.find_similar(emb, top_k=2)
        st.session_state.similar_cases = matched_cases
        
        for case in matched_cases:
            c_label = class_names_local[case['label']]
            c_sim = case['similarity'] * 100.0
            st.markdown(f"""
            <div class="glass-panel" style="padding: 1rem; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: #0F172A; font-size: 0.95rem;">{case['case_id']}</strong><br>
                    <span style="font-size: 0.82rem; color: #64748B;">Diagnosis: <strong>{c_label}</strong></span>
                </div>
                <div>
                    <span style="background: rgba(37,99,235,0.1); color: #1E40AF; padding: 4px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.8rem;">
                        {c_sim:.1f}% {get_text('sec3_rag_sim', lang)}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_cdss_media:
        st.markdown(f"### 🎙️ {get_text('sec3_voice_title', lang)}")
        voice_diag_name = LABEL_NAMES.get(pred_idx, "Normal")
        audio_bytes = get_or_create_audio(voice_diag_name, conf_val)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
            
        st.markdown(f"### 📋 {get_text('sec3_report_title', lang)}")
        pat_id = f"PAT-{int(time.time()) % 100000}"
        report_name = f"FedXRay_{voice_diag_name}_{pat_id}.pdf"
        
        try:
            pdf_exp = get_diagnosis_explanation(voice_diag_name, conf_val)
            pdf_bytes = generate_medical_report(
                patient_id=pat_id,
                diagnosis=voice_diag_name,
                confidence=conf_val,
                explanation=pdf_exp,
                heatmap_image=st.session_state.scan_heatmap if hasattr(st.session_state, 'scan_heatmap') else None,
                original_image=st.session_state.scan_image,
                similar_cases=st.session_state.get('similar_cases', None)
            )
            st.download_button(
                label=f"📥 {get_text('sec3_btn_download_pdf', lang)}",
                data=pdf_bytes,
                file_name=report_name,
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.caption("PDF Report compilation standby.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1.5rem; color: #64748B; font-size: 0.85rem;">
    {get_text('footer_text', lang)}
</div>
""", unsafe_allow_html=True)
