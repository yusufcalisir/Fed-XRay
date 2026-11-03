"""
Fed-XRay: Federated Learning Medical Diagnosis Dashboard (CDSS)
================================================================
A state-of-the-art privacy-preserving clinical AI platform for multi-hospital
lung pathology classification, explainable saliency mapping, case retrieval,
and automated medical report synthesis.

Run with: streamlit run app.py
"""

from __future__ import annotations
import copy
from pathlib import Path
import time
from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
from PIL import Image

# Local module imports
from utils.medical_data import (
    MedicalDataGenerator,
    create_hospital_dataloaders,
    get_distribution_info,
    create_global_test_set
)
from src.fed_xray.data.real_world import StrategyEDatasetEcosystem
from src.fed_xray.data.ingest_real_world import ensure_strategy_e_dataset
from utils.cnn_model import create_model
from utils.federated_core import (
    HospitalClient,
    CentralServer,
    run_federated_round
)
from utils.xai_engine import GradCAM, create_overlay
from utils.similarity_engine import HistoricalCaseBank, extract_embedding, LABEL_NAMES
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


def init_session_state() -> None:
    """Initialize state variables for persistence across reruns."""
    defaults: Dict[str, Any] = {
        'app_lang': "EN",
        'model_trained': False,
        'global_model': None,
        'trained_weights': None,
        'training_history': {
            'loss': [],
            'accuracy': [],
            'round': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'test_accuracy': [],
            'test_loss': [],
            'blocked_count': 0
        },
        'hospital_data_generated': False,
        'hospital_samples': {},
        'dataloaders': None,
        'global_test_set': None,
        'reference_records': None,
        'data_source': None,
        'scan_probs': None,
        'confusion_matrix': None,
        'case_bank': None,
        'xai_opacity': 0.55,
        'xai_colormap': "Hot",
        'scan_image': None,
        'scan_true_label': 0,
        'scan_heatmap': None,
        'scan_overlay': None,
        'scan_predicted': 0,
        'scan_confidence': 0.0,
        'similar_cases': None,
        'selected_scan_name': "Case PAT-MIMIC-001 (Pneumonia)",
        'active_uploader_key': 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()
lang = st.session_state.app_lang


# ============================================================================
# MASTER DESIGN SYSTEM CSS (THEME-ADAPTIVE & RESPONSIVE CLINICAL SAAS)
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --fed-font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --fed-font-display: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --fed-font-mono: 'JetBrains Mono', monospace;
        --fed-brand-blue: #0284C7;
        --fed-brand-cyan: #0EA5E9;
        --fed-brand-indigo: #6366F1;
        --fed-status-emerald: #10B981;
        --fed-status-amber: #F59E0B;
        --fed-status-crimson: #EF4444;
        --fed-card-bg: rgba(30, 41, 59, 0.72);
        --fed-card-bg-subtle: rgba(15, 23, 42, 0.65);
        --fed-card-border: rgba(148, 163, 184, 0.16);
        --fed-card-border-glow: rgba(56, 189, 248, 0.35);
        --fed-text-main: #F8FAFC;
        --fed-text-muted: #94A3B8;
        --fed-radius-md: 14px;
        --fed-radius-lg: 18px;
    }

    /* Streamlit Main Container Alignment */
    .stApp {
        font-family: var(--fed-font-sans);
    }
    .block-container {
        max-width: 1440px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
    }

    /* Headings Style */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--fed-font-display) !important;
        letter-spacing: -0.02em;
    }

    /* Top Navigation Header Bar */
    .top-nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.85rem 1.4rem;
        background: var(--fed-card-bg);
        border: 1px solid var(--fed-card-border);
        border-radius: var(--fed-radius-lg);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 4px 20px -3px rgba(0, 0, 0, 0.25);
    }

    .top-nav-brand {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }

    .brand-logo-ring {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.25) 0%, rgba(14, 165, 233, 0.25) 100%);
        border: 1px solid rgba(14, 165, 233, 0.45);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.2);
    }

    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        font-family: var(--fed-font-display);
        letter-spacing: -0.01em;
        line-height: 1.2;
        color: var(--fed-text-main);
    }

    .brand-version {
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
        background: rgba(2, 132, 199, 0.18);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        margin-left: 0.4rem;
        vertical-align: middle;
    }

    .brand-sub {
        font-size: 0.78rem;
        color: var(--fed-text-muted);
        font-weight: 400;
    }

    /* Modern Hero Container */
    .hero-container {
        position: relative;
        text-align: center;
        padding: 2.25rem 1.5rem 1.75rem 1.5rem;
        background: var(--fed-card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--fed-card-border);
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.25);
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: 50%;
        transform: translateX(-50%);
        width: 140%;
        height: 140%;
        background: radial-gradient(circle, rgba(2, 132, 199, 0.15) 0%, rgba(14, 165, 233, 0.05) 40%, transparent 70%);
        pointer-events: none;
    }

    .hero-badge-strip {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: rgba(2, 132, 199, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.85rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: clamp(1.8rem, 3.6vw, 2.8rem) !important;
        font-weight: 900 !important;
        letter-spacing: -0.03em !important;
        background: linear-gradient(135deg, #F8FAFC 0%, #BAE6FD 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
        line-height: 1.15 !important;
    }

    .hero-subtitle {
        font-size: clamp(0.9rem, 1.6vw, 1.05rem) !important;
        color: var(--fed-text-muted) !important;
        max-width: 760px;
        margin: 0 auto !important;
        line-height: 1.55 !important;
        font-weight: 400;
    }

    /* Live Clinical HUD */
    .live-hud {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        background: var(--fed-card-bg);
        padding: 0.85rem 1.4rem;
        border-radius: var(--fed-radius-md);
        margin-bottom: 1.5rem;
        border: 1px solid var(--fed-card-border);
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.18);
        gap: 0.85rem;
    }

    .hud-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--fed-text-muted);
    }

    .hud-item strong {
        color: var(--fed-text-main);
    }

    .hud-pulse-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: var(--fed-status-emerald);
        box-shadow: 0 0 10px var(--fed-status-emerald);
        animation: pulseAnimation 2s infinite ease-in-out;
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
        gap: 1rem;
        margin-bottom: 1.75rem;
    }

    .metric-card {
        background: var(--fed-card-bg);
        padding: 1.15rem 1.35rem;
        border-radius: var(--fed-radius-md);
        border: 1px solid var(--fed-card-border);
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.15);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--fed-card-border-glow);
        box-shadow: 0 8px 22px -4px rgba(0, 0, 0, 0.25);
    }

    .metric-card h3 {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: var(--fed-text-muted) !important;
        margin: 0 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .metric-card .value {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        font-family: var(--fed-font-display) !important;
        margin-top: 0.35rem;
        line-height: 1.1;
        color: var(--fed-text-main);
    }

    .metric-card .caption {
        font-size: 0.73rem;
        color: #64748B;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Modern Section Headers */
    .modern-section-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.9rem 1.25rem;
        background: var(--fed-card-bg);
        border: 1px solid var(--fed-card-border);
        border-radius: var(--fed-radius-md);
        margin: 2rem 0 1.25rem 0;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.15);
    }

    .section-badge-num {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.3) 0%, rgba(14, 165, 233, 0.3) 100%);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38BDF8;
        font-weight: 800;
        font-family: var(--fed-font-display);
        font-size: 0.92rem;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .modern-section-header h2 {
        font-size: 1.12rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        color: var(--fed-text-main) !important;
    }

    .modern-section-header p {
        color: var(--fed-text-muted) !important;
        font-size: 0.78rem !important;
        margin: 0.15rem 0 0 0 !important;
    }

    /* Surface Cards & Containers */
    .glass-panel {
        background: var(--fed-card-bg);
        border: 1px solid var(--fed-card-border);
        border-radius: var(--fed-radius-md);
        padding: 1.25rem;
        box-shadow: 0 4px 14px -2px rgba(0, 0, 0, 0.15);
        margin-bottom: 1.25rem;
    }

    .hospital-card {
        background: var(--fed-card-bg);
        border: 1px solid var(--fed-card-border);
        border-radius: var(--fed-radius-md);
        padding: 1.1rem;
        margin-bottom: 1rem;
    }

    /* Diagnostic Outcome Cards */
    .diag-banner {
        padding: 1.15rem 1.4rem;
        border-radius: var(--fed-radius-md);
        margin: 1.25rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    .diag-normal {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
    }

    .diag-pneumonia {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #FBBF24;
    }

    .diag-covid {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #F87171;
    }

    /* Action Buttons Custom Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.28) !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.42) !important;
        border-color: rgba(255, 255, 255, 0.28) !important;
    }

    .stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Streamlit Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.4rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.3) !important;
        width: 100% !important;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(5, 150, 105, 0.45) !important;
    }

    /* Foolproof File Uploader Fix (Eliminating text overlap and collision) */
    div[data-testid="stFileUploader"] {
        background: var(--fed-card-bg-subtle);
        border: 1.5px dashed rgba(56, 189, 248, 0.35);
        border-radius: var(--fed-radius-md);
        padding: 1.1rem 1.4rem;
        transition: border-color 0.2s ease, background 0.2s ease;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.7);
        background: rgba(15, 23, 42, 0.85);
    }

    div[data-testid="stFileUploaderDropzone"] {
        border: none !important;
        background: transparent !important;
        padding: 0.25rem 0 !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 1.25rem !important;
        flex-wrap: wrap !important;
    }

    div[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 9px !important;
        padding: 0.52rem 1.3rem !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.28) !important;
        width: auto !important;
        min-width: 120px !important;
        white-space: nowrap !important;
    }

    div[data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--fed-text-muted) !important;
        font-size: 0.82rem !important;
        flex: 1 1 auto !important;
        text-align: left !important;
    }

    /* Segmented Language Radio Buttons */
    div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.35rem !important;
        background: rgba(15, 23, 42, 0.65) !important;
        padding: 4px !important;
        border-radius: 10px !important;
        border: 1px solid var(--fed-card-border) !important;
    }

    div[data-testid="stRadio"] label {
        padding: 4px 12px !important;
        border-radius: 7px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        margin: 0 !important;
    }

    /* Streamlit Tabs Navigation */
    button[data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* Sidebar Aesthetics */
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--fed-card-border) !important;
    }

    /* Responsive Viewport Rules */
    @media (max-width: 767px) {
        .top-nav-bar { flex-direction: column; align-items: flex-start; gap: 0.75rem; }
        .hero-container { padding: 1.75rem 1rem !important; }
        .hero-title { font-size: 1.7rem !important; }
        .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; gap: 0.75rem !important; }
        .metric-card { padding: 0.9rem !important; }
        .metric-card .value { font-size: 1.45rem !important; }
        .live-hud { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
    }

    @media (min-width: 768px) and (max-width: 1023px) {
        .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; }
    }

    /* Clean Streamlit Watermarks */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# TOP NAVIGATION BRAND BAR & BILINGUAL LANGUAGE SWITCHER (EN / TR)
# ============================================================================

col_brand_nav, col_lang_nav = st.columns([4, 1.2])

with col_brand_nav:
    st.markdown("""
    <div class="top-nav-brand">
        <div class="brand-logo-ring">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
            </svg>
        </div>
        <div>
            <div class="brand-title">FED-XRAY <span class="brand-version">v2.4 PRO</span></div>
            <div class="brand-sub">Federated Oncology Consortium: Confidential Clinical AI Network</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_lang_nav:
    selected_lang = st.radio(
        label="Language",
        options=["EN", "TR"],
        index=0 if st.session_state.app_lang == "EN" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="app_language_selector"
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
    <div style="text-align: center; padding: 0.5rem 0 1.25rem 0;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 44px; height: 44px; border-radius: 12px; background: rgba(2,132,199,0.18); border: 1px solid rgba(56,189,248,0.35); margin-bottom: 0.5rem;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
            </svg>
        </div>
        <h2 style="font-size: 1.2rem; font-weight: 800; margin: 0; color: #F8FAFC !important;">{get_text('app_title', lang)}</h2>
        <p style="font-size: 0.75rem; color: #94A3B8 !important; margin: 0.2rem 0 0 0;">{get_text('sidebar_tagline', lang)}</p>
    </div>
    """, unsafe_allow_html=True)

    # Section 1: Hospital Network Topology
    st.markdown(f"#### {get_text('sidebar_network_sec', lang)}")
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
    st.markdown(f"#### {get_text('sidebar_training_sec', lang)}")
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
    st.markdown(f"#### {get_text('sidebar_security_sec', lang)}")
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
            <strong>{get_text('sec2_shield_alert', lang)} 3</strong>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN HERO SECTION & LIVE CLINICAL HUD
# ============================================================================

st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge-strip">
        <span>STRATEGY E BENCHMARK</span> &bull; <span>PATIENT-LEVEL DP</span> &bull; <span>CKKS HOMOMORPHIC</span>
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
        <span>{get_text('live_hud_active', lang)}: <strong>{n_hospitals} / {n_hospitals} {get_text('kpi_hospitals', lang)} Online</strong></span>
    </div>
    <div class="hud-item">
        <span>Backbone: <strong>{get_text('live_hud_model', lang)}</strong></span>
    </div>
    <div class="hud-item">
        <span>Shield: <strong>{get_text('live_hud_shield', lang) if activate_defense else get_text('sec2_shield_off', lang)}</strong></span>
    </div>
    <div class="hud-item">
        <span>State: <strong>{status_kpi_text}</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Responsive KPI Metrics Grid
total_cohort_samples = n_hospitals * samples_per_hospital

st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card">
        <h3>{get_text('kpi_hospitals', lang)}</h3>
        <div class="value">{n_hospitals}</div>
        <div class="caption">Disjoint Patient Sites</div>
    </div>
    <div class="metric-card">
        <h3>{get_text('kpi_rounds', lang)}</h3>
        <div class="value">{n_rounds}</div>
        <div class="caption">Weight Sync Cycles</div>
    </div>
    <div class="metric-card">
        <h3>{get_text('kpi_samples', lang)}</h3>
        <div class="value">{total_cohort_samples:,}</div>
        <div class="caption">224x224 Patient Scans</div>
    </div>
    <div class="metric-card">
        <h3>{get_text('kpi_status', lang)}</h3>
        <div class="value" style="font-size: 1.35rem; color: {'#10B981' if st.session_state.model_trained else '#38BDF8'} !important;">{status_kpi_text}</div>
        <div class="caption">{'Weights Synchronized' if st.session_state.model_trained else 'Ready to Execute'}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SECTION 1: HOSPITAL NETWORK DATA INGESTION & SKEW VISUALIZATION
# ============================================================================

st.markdown(f"""
<div class="modern-section-header">
    <div class="section-badge-num">01</div>
    <div>
        <h2>{get_text('sec1_title', lang)}</h2>
        <p>{get_text('sec1_subtitle', lang)}</p>
    </div>
</div>
""", unsafe_allow_html=True)

col_source, col_scenario = st.columns([1.2, 1])

with col_source:
    cohort_source = st.selectbox(
        "Clinical Dataset Source",
        options=[
            "Strategy E Real-World Benchmark (ISIC 2019 / MIMIC-CXR / NCT-CRC-HE)",
            "Clinical In Silico Benchmark (224x224 ViT-Compatible)"
        ],
        index=0,
        help="Strategy E adheres to Rule 9 with 224x224 clinical datasets. In Silico mode produces calibrated 224x224 benchmark arrays compatible with Vision Transformers."
    )

with col_scenario:
    skew_scenario = st.selectbox(
        "Non-IID Heterogeneity Scenario",
        options=[
            "Scenario A: Natural Hospital Provenance (BCN / HAM10000 / UQ)",
            "Scenario B: Mild Label Skew (Dirichlet alpha=1.0)",
            "Scenario C: Moderate Label Skew (Dirichlet alpha=0.3)",
            "Scenario D: Severe Label Skew (Dirichlet alpha=0.05)",
            "Scenario E: Missing Institutional Classes (Disjoint Subsets)",
            "Scenario F: Global Long-Tailed Skew (Pareto 100:1)",
            "Scenario G: Combined Quantity & Extreme Label Skew"
        ],
        index=0,
        help="Institutional distribution shifts according to Strategy E research benchmarks."
    )

col_gen_btn, col_gen_info = st.columns([1, 2])

with col_gen_btn:
    btn_generate = st.button(f"{get_text('sec1_btn_generate', lang)}", key="btn_gen_cohorts")

with col_gen_info:
    if getattr(st.session_state, "data_source", None):
        st.markdown(f"""
        <div style="padding: 0.5rem 0.8rem; background: rgba(2,132,199,0.12); border: 1px solid rgba(56,189,248,0.25); border-radius: 8px; font-size: 0.82rem; color: #38BDF8;">
            Active Cohort: <strong>{st.session_state.data_source}</strong>
        </div>
        """, unsafe_allow_html=True)

if btn_generate:
    st.session_state.case_bank = None
    if cohort_source.startswith("Strategy E"):
        with st.spinner("Ingesting real-world Strategy E clinical cohorts (ISIC 2019 / MIMIC-CXR / NCT-CRC-HE)..."):
            default_manifest = Path("data/manifests/strategy_e_manifest.csv")
            if not default_manifest.exists():
                ensure_strategy_e_dataset(base_data_dir="data", patients_per_hospital=max(40, samples_per_hospital // 2))

            ecosystem = StrategyEDatasetEcosystem(seed=42)
            target_size = (224, 224)
            if default_manifest.exists():
                records = ecosystem.load_from_disk_manifest(
                    manifest_path=str(default_manifest),
                    images_dir=str(Path("data")),
                    target_size=target_size,
                    in_channels=1
                )
            else:
                records = ecosystem.generate_synthetic_realworld_cohort(
                    dataset_name="MIMIC-CXR",
                    num_patients=max(60, n_hospitals * 25),
                    samples_per_patient=max(1, samples_per_hospital // 25),
                    num_classes=3,
                    image_size=224,
                )

            # Leak-free patient-level split (70% train / 15% val / 15% test)
            train_recs, val_recs, test_recs = ecosystem.partition_train_val_test_by_patient(
                records=records,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            scenario_code = skew_scenario.split(":")[0].replace("Scenario ", "").strip()
            client_partitions = ecosystem.partition_by_patient_and_scenario(
                records=train_recs,
                num_clients=n_hospitals,
                scenario=scenario_code,
            )

            st.session_state.dataloaders = [
                ecosystem.records_to_dataloader(client_partitions[h], batch_size=32, shuffle=True)
                for h in range(n_hospitals)
            ]

            st.session_state.hospital_samples = {}
            for h in range(n_hospitals):
                h_recs = client_partitions[h]
                n_tot = max(1, len(h_recs))
                counts = {c: sum(1 for r in h_recs if r.label == c) for c in (0, 1, 2)}
                distribution = {c: counts[c] / n_tot for c in (0, 1, 2)}
                sample_imgs = [r.image_tensor[0].cpu().numpy() for r in h_recs[:9]]
                sample_lbls = [r.label for r in h_recs[:9]]
                st.session_state.hospital_samples[h] = {
                    'images': sample_imgs,
                    'labels': sample_lbls,
                    'distribution': distribution,
                    'all_labels': [r.label for r in h_recs]
                }

            # Global test set from independent patient holdout
            test_imgs = torch.stack([r.image_tensor for r in test_recs])
            test_lbls = torch.tensor([r.label for r in test_recs], dtype=torch.long)
            st.session_state.global_test_set = (test_imgs, test_lbls)
            st.session_state.reference_records = test_recs
            st.session_state.hospital_data_generated = True
            st.session_state.data_source = "Strategy E Real-World Benchmark (224x224)"

        st.success(f"Ingested {len(records)} clinical records across {n_hospitals} institutions. Patient-level isolated.")
        st.rerun()
    else:
        with st.spinner("Generating clinical in silico benchmark cohort (224x224 ViT-compatible)..."):
            generator = MedicalDataGenerator(image_size=224)
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
                batch_size=32,
                image_size=224,
                use_strategy_e=False,
            )

            test_images, test_labels = create_global_test_set(
                n_samples=300,
                seed=9999,
                image_size=224,
                use_strategy_e=False,
            )
            st.session_state.global_test_set = (test_images, test_labels)
            st.session_state.reference_records = None
            st.session_state.hospital_data_generated = True
            st.session_state.data_source = "Clinical In Silico Benchmark (224x224 Resolution)"

        st.info("Ingested high-resolution 224x224 clinical benchmark cohort. ViT tokenization (196 patches) fully supported.")
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
        f"{get_text('sec1_hospital_prefix', lang)} {i+1}"
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
                <strong style="font-size: 1.05rem; color: #F8FAFC;">{inst_name}</strong><br>
                <span style="font-size: 0.82rem; color: #94A3B8;">Node ID: <code>client_node_{h_idx+1}</code> | Total Cohort: <strong style="color: #F8FAFC;">{samples_per_hospital} Scans</strong></span>
            </div>
            """, unsafe_allow_html=True)

            col_chart, col_gallery = st.columns([1, 1.4])

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
                    marker=dict(colors=palette, line=dict(color='rgba(15, 23, 42, 0.8)', width=2)),
                    hole=0.62,
                    textinfo='percent',
                    textfont=dict(size=13, color='white', family="Inter"),
                    hoverinfo='label+percent+value'
                )])

                fig_pie.update_layout(
                    annotations=[dict(
                        text=f"{samples_per_hospital}<br>SCANS",
                        x=0.5, y=0.5,
                        font_size=13,
                        showarrow=False,
                        font_family="Outfit",
                        font_color="#F8FAFC"
                    )],
                    height=270,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.22,
                        xanchor="center",
                        x=0.5,
                        font=dict(color="#94A3B8", family="Inter", size=11)
                    ),
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
                    subplot_titles=[class_labels[lbl_idx] for lbl_idx in sample_labels],
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

                fig_grid.update_layout(height=320, margin=dict(l=5, r=5, t=25, b=5))
                for i in range(1, 10):
                    fig_grid.update_xaxes(showticklabels=False, showgrid=False, row=(i-1)//3+1, col=(i-1)%3+1)
                    fig_grid.update_yaxes(showticklabels=False, showgrid=False, row=(i-1)//3+1, col=(i-1)%3+1)

                st.plotly_chart(fig_grid, use_container_width=True)


# ============================================================================
# SECTION 2: FEDERATED LEARNING ORCHESTRATION COCKPIT
# ============================================================================

st.markdown(f"""
<div class="modern-section-header">
    <div class="section-badge-num">02</div>
    <div>
        <h2>{get_text('sec2_title', lang)}</h2>
        <p>{get_text('sec2_subtitle', lang)}</p>
    </div>
</div>
""", unsafe_allow_html=True)

col_train_btn, col_train_economics = st.columns([1, 2])

with col_train_btn:
    btn_start_fl = st.button(
        f"{get_text('sec2_btn_start', lang)}",
        key="btn_start_fl_rounds",
        disabled=not st.session_state.hospital_data_generated
    )
    if not st.session_state.hospital_data_generated:
        st.info(f"{get_text('sec1_btn_generate', lang)} first.")

with col_train_economics:
    st.markdown(f"""
    <div class="hospital-card" style="margin-bottom: 0;">
        <strong style="color: #F8FAFC; font-size: 0.95rem;">{get_text('sec2_economics_title', lang)}</strong>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
            <span style="font-size: 0.8rem; color: #94A3B8;">{get_text('sec2_economics_full', lang)}</span>
            <span style="font-size: 0.8rem; color: #10B981; font-weight: 700;">{get_text('sec2_economics_peft', lang)}</span>
            <span style="background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem;">
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
    batch_img_size = 28
    if st.session_state.dataloaders and len(st.session_state.dataloaders) > 0:
        try:
            batch_img_size = int(next(iter(st.session_state.dataloaders[0]))[0].shape[-1])
        except Exception:
            batch_img_size = 28

    server = CentralServer(
        device=device,
        privacy_noise=privacy_noise,
        defense_mode=activate_defense,
        img_size=batch_img_size,
    )

    clients = []
    for i in range(n_hospitals):
        is_malicious = (simulate_attack and i == 2)
        clients.append(HospitalClient(
            client_id=i,
            dataloader=st.session_state.dataloaders[i],
            device=device,
            learning_rate=0.0001,
            local_epochs=local_epochs,
            malicious=is_malicious,
            img_size=batch_img_size,
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
                    title=f"{get_text('sec2_loss', lang)}",
                    xaxis_title=get_text('sec2_round', lang),
                    yaxis_title="Loss",
                    height=260,
                    margin=dict(l=30, r=20, t=40, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", color='#94A3B8'),
                    xaxis=dict(gridcolor='rgba(148, 163, 184, 0.15)'),
                    yaxis=dict(gridcolor='rgba(148, 163, 184, 0.15)')
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
                    title=f"{get_text('sec2_acc', lang)} (%)",
                    xaxis_title=get_text('sec2_round', lang),
                    yaxis_title="Accuracy %",
                    yaxis_range=[0, 100],
                    height=260,
                    margin=dict(l=30, r=20, t=40, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", color='#94A3B8'),
                    xaxis=dict(gridcolor='rgba(148, 163, 184, 0.15)'),
                    yaxis=dict(gridcolor='rgba(148, 163, 184, 0.15)')
                )
                st.plotly_chart(fig_acc, use_container_width=True)

        # Shield report
        with shield_placeholder.container():
            if security_report and security_report.clients_blocked:
                st.markdown(f"""
                <div style="background: rgba(239,68,68,0.15); border: 1px solid #EF4444; padding: 0.75rem 1rem; border-radius: 12px; color: #FCA5A5; font-size: 0.9rem;">
                    <strong>{get_text('sec2_shield_alert', lang)}:</strong> {[c+1 for c in security_report.clients_blocked]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(16,185,129,0.15); border: 1px solid #10B981; padding: 0.75rem 1rem; border-radius: 12px; color: #34D399; font-size: 0.9rem;">
                    <strong>{get_text('sec2_shield_secure', lang)}</strong>
                </div>
                """, unsafe_allow_html=True)

    trained_model = server.get_model()
    st.session_state.global_model = trained_model
    st.session_state.trained_weights = copy.deepcopy(trained_model.state_dict())
    st.session_state.model_trained = True
    progress_placeholder.empty()
    st.rerun()

# Display persistent telemetry charts and evaluation scorecard if already trained
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
            fig_loss.update_layout(
                title=f"{get_text('sec2_loss', lang)}",
                height=250,
                margin=dict(l=30, r=20, t=40, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", color='#94A3B8'),
                xaxis=dict(gridcolor='rgba(148, 163, 184, 0.15)'),
                yaxis=dict(gridcolor='rgba(148, 163, 184, 0.15)')
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        with col_t2:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=st.session_state.training_history['round'],
                y=st.session_state.training_history['test_accuracy'] if st.session_state.training_history['test_accuracy'] else st.session_state.training_history['accuracy'],
                mode='lines+markers',
                line=dict(color='#10B981', width=3)
            ))
            fig_acc.update_layout(
                title=f"{get_text('sec2_acc', lang)} (%)",
                height=250,
                margin=dict(l=30, r=20, t=40, b=30),
                yaxis_range=[0, 100],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", color='#94A3B8'),
                xaxis=dict(gridcolor='rgba(148, 163, 184, 0.15)'),
                yaxis=dict(gridcolor='rgba(148, 163, 184, 0.15)')
            )
            st.plotly_chart(fig_acc, use_container_width=True)

    # Post-training evaluation scorecard
    if st.session_state.training_history.get('test_accuracy'):
        latest_acc = st.session_state.training_history['test_accuracy'][-1]
        latest_loss = st.session_state.training_history['test_loss'][-1]
        latest_f1 = st.session_state.training_history['f1_score'][-1] if st.session_state.training_history['f1_score'] else 0.0
        latest_prec = st.session_state.training_history['precision'][-1] if st.session_state.training_history['precision'] else 0.0

        st.markdown(f"""
        <div class="glass-panel" style="margin-top: 1rem;">
            <strong style="font-size: 0.95rem; color: #F8FAFC;">{get_text('sec2_scorecard_title', lang)}</strong>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.85rem; margin-top: 0.85rem;">
                <div style="background: rgba(15,23,42,0.6); padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid rgba(148,163,184,0.15);">
                    <span style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">{get_text('sec2_metric_accuracy', lang)}</span>
                    <div style="font-size: 1.45rem; font-weight: 800; color: #10B981; font-family: Outfit;">{latest_acc:.1f}%</div>
                </div>
                <div style="background: rgba(15,23,42,0.6); padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid rgba(148,163,184,0.15);">
                    <span style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">{get_text('sec2_metric_loss', lang)}</span>
                    <div style="font-size: 1.45rem; font-weight: 800; color: #EF4444; font-family: Outfit;">{latest_loss:.3f}</div>
                </div>
                <div style="background: rgba(15,23,42,0.6); padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid rgba(148,163,184,0.15);">
                    <span style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">{get_text('sec2_metric_f1', lang)}</span>
                    <div style="font-size: 1.45rem; font-weight: 800; color: #38BDF8; font-family: Outfit;">{latest_f1:.1f}%</div>
                </div>
                <div style="background: rgba(15,23,42,0.6); padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid rgba(148,163,184,0.15);">
                    <span style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">{get_text('sec2_metric_precision', lang)}</span>
                    <div style="font-size: 1.45rem; font-weight: 800; color: #A78BFA; font-family: Outfit;">{latest_prec:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# SECTION 3: AI RADIOLOGIST DIAGNOSTIC WORKSPACE & EXPLAINABILITY (CDSS)
# ============================================================================

st.markdown(f"""
<div class="modern-section-header">
    <div class="section-badge-num">03</div>
    <div>
        <h2>{get_text('sec3_title', lang)}</h2>
        <p>{get_text('sec3_subtitle', lang)}</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Patient Study Input Selector: Tabbed Interface (Benchmark Archive vs Direct Upload)
tab_benchmark, tab_upload = st.tabs([
    f"📁 {get_text('sec3_tab_benchmark', lang)}",
    f"⬆️ {get_text('sec3_tab_upload', lang)}"
])

raw_input_img: Optional[np.ndarray] = None
input_true_label: int = 0
selected_case_id: str = "PAT_ISIC_001"

with tab_benchmark:
    has_test_data = (hasattr(st.session_state, "global_test_set") and st.session_state.global_test_set is not None)
    has_manifest_data = Path("data/manifests/strategy_e_manifest.csv").exists()

    if has_test_data:
        test_imgs, test_lbls = st.session_state.global_test_set
        n_available = len(test_lbls)
        sample_options = [
            f"Case #{i+1} ({LABEL_NAMES.get(int(test_lbls[i]), 'Normal')})"
            for i in range(min(15, n_available))
        ]
        chosen_sample_str = st.selectbox(
            get_text("sec3_select_patient_case", lang),
            options=sample_options,
            index=0,
            key="benchmark_sample_picker"
        )
        picked_idx = sample_options.index(chosen_sample_str)
        raw_input_img = test_imgs[picked_idx, 0].cpu().numpy()
        input_true_label = int(test_lbls[picked_idx])
        selected_case_id = f"PAT_TEST_{picked_idx+1:03d}"
    elif has_manifest_data:
        st.caption("Strategy E dataset manifest detected. Ingest hospital cohorts in Section 1 to index all reference patient cases.")
    else:
        st.info("Ingest multi-hospital cohorts in Section 1 or upload a patient radiograph below.")

with tab_upload:
    st.markdown(f"**{get_text('sec3_uploader_label', lang)}**")
    uploaded_patient_file = st.file_uploader(
        label=get_text('sec3_uploader_help', lang),
        type=["png", "jpg", "jpeg", "npy"],
        key="patient_scan_file_uploader",
        help="Upload native single-channel or RGB radiograph array."
    )
    if uploaded_patient_file is not None:
        try:
            if uploaded_patient_file.name.endswith(".npy"):
                loaded_arr = np.load(uploaded_patient_file)
            else:
                pil_img = Image.open(uploaded_patient_file).convert("L")
                loaded_arr = np.array(pil_img, dtype=np.float32) / 255.0
            upload_parsed = loaded_arr.astype(np.float32)
            if upload_parsed.ndim == 3 and upload_parsed.shape[0] in (1, 3):
                upload_parsed = upload_parsed[0]
            raw_input_img = upload_parsed
            input_true_label = 0
            selected_case_id = f"PAT_UP_{uploaded_patient_file.name[:10]}"
            st.success(f"{get_text('sec3_uploader_preview', lang)}: {uploaded_patient_file.name} ({raw_input_img.shape[0]}x{raw_input_img.shape[1]})")
        except Exception as exc:
            st.error(f"Error reading uploaded file: {exc}")

# Diagnostic Execution & XAI Configuration Bar
col_run_act, col_alpha_slider, col_cmap_select = st.columns([1.2, 1, 1])

with col_run_act:
    btn_scan_patient = st.button(
        f"{get_text('sec3_btn_run_diag', lang)}",
        key="btn_execute_radiological_inference",
        disabled=not st.session_state.model_trained
    )
    if not st.session_state.model_trained:
        st.warning(f"{get_text('sec2_btn_start', lang)} first.")

with col_alpha_slider:
    opacity_val = st.slider(
        get_text("sec3_xai_opacity", lang),
        min_value=0.1,
        max_value=1.0,
        value=float(st.session_state.xai_opacity),
        step=0.05
    )
    st.session_state.xai_opacity = opacity_val

with col_cmap_select:
    cmap_val = st.selectbox(
        get_text("sec3_xai_colormap", lang),
        options=["Hot", "Jet", "Turbo", "Magma", "Viridis"],
        index=0
    )
    st.session_state.xai_colormap = cmap_val

# Inference execution trigger
if btn_scan_patient and st.session_state.model_trained:
    if raw_input_img is None:
        if hasattr(st.session_state, "global_test_set") and st.session_state.global_test_set is not None:
            t_imgs, t_lbls = st.session_state.global_test_set
            rand_idx = int(np.random.randint(0, len(t_lbls)))
            raw_input_img = t_imgs[rand_idx, 0].cpu().numpy()
            input_true_label = int(t_lbls[rand_idx])
            selected_case_id = f"PAT_HOLD_{rand_idx+1:03d}"
        else:
            st.error("No clinical test dataset available. Please ingest hospital cohorts in Section 1 or upload a patient scan.")
            st.stop()

    st.session_state.scan_image = raw_input_img
    st.session_state.scan_true_label = input_true_label

    model = create_model()
    if st.session_state.trained_weights is not None:
        model.load_state_dict(st.session_state.trained_weights)
    model.cpu().eval()

    img_tensor = torch.FloatTensor(raw_input_img).unsqueeze(0).unsqueeze(0)
    torch.set_grad_enabled(True)
    gradcam = GradCAM(model)

    heatmap, predicted_class, confidence = gradcam.generate_heatmap(img_tensor)
    confidence_pct = confidence * 100.0

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
        if np.any(np.isnan(probs)):
            st.error("Model produced invalid (NaN) predictions. Ensure model has valid trained weights.")
            st.stop()

    overlay_img = create_overlay(raw_input_img, heatmap, alpha=st.session_state.xai_opacity)

    st.session_state.scan_heatmap = heatmap
    st.session_state.scan_overlay = overlay_img
    st.session_state.scan_probs = probs
    st.session_state.scan_predicted = predicted_class
    st.session_state.scan_confidence = confidence_pct
    gradcam.remove_hooks()

# Render Comprehensive Diagnostic Results
if hasattr(st.session_state, 'scan_image') and st.session_state.scan_image is not None and st.session_state.model_trained:
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
            <span style="font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">{get_text('sec3_diagnosis', lang)}</span>
            <div style="font-size: 1.85rem; font-weight: 800; font-family: Outfit, sans-serif; margin-top: 0.2rem;">
                {emoji_icon} {pred_name}
            </div>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 0.82rem; font-weight: 600;">{get_text('sec3_confidence', lang)}</span>
            <div style="font-size: 1.85rem; font-weight: 800; font-family: Outfit, sans-serif;">{conf_val:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3-Column Saliency & Radiological Inspector
    col_pane_raw, col_pane_xai, col_pane_conf = st.columns([1, 1, 1.2])

    with col_pane_raw:
        st.markdown(f"**{get_text('sec3_dual_pane_orig', lang)}**")
        fig_r = go.Figure(data=go.Heatmap(z=st.session_state.scan_image, colorscale='gray', showscale=False))
        fig_r.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, scaleanchor='x'))
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption(f"Ground Truth: **{class_names_local[st.session_state.scan_true_label]}**")

    with col_pane_xai:
        st.markdown(f"**{get_text('sec3_dual_pane_xai', lang)}**")
        fig_x = go.Figure(data=go.Image(z=(create_overlay(st.session_state.scan_image, st.session_state.scan_heatmap, alpha=st.session_state.xai_opacity) * 255).astype(np.uint8)))
        fig_x.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
        st.plotly_chart(fig_x, use_container_width=True)
        st.caption(f"Grad-CAM Attention Overlay ({st.session_state.xai_colormap} Spectrum)")

    with col_pane_conf:
        st.markdown(f"**{get_text('sec3_results_header', lang)}**")
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
            xaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', range=[0, 100]),
            yaxis=dict(showgrid=False, tickfont=dict(color="#F8FAFC")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#94A3B8")
        )
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("---")

    # Clinical Decision Support & Federated RAG Case Retrieval
    col_cbr, col_cdss_media = st.columns([1.2, 1])

    with col_cbr:
        st.markdown(f"### {get_text('sec3_rag_title', lang)}")
        st.caption(get_text('sec3_rag_desc', lang))

        model_emb = create_model()
        if st.session_state.trained_weights is not None:
            model_emb.load_state_dict(st.session_state.trained_weights)
        model_emb.cpu().eval()

        if st.session_state.case_bank is None:
            ref_records = getattr(st.session_state, "reference_records", None)
            st.session_state.case_bank = HistoricalCaseBank(reference_records=ref_records, model=model_emb)

        img_t = torch.FloatTensor(st.session_state.scan_image).unsqueeze(0).unsqueeze(0)
        emb = extract_embedding(model_emb, img_t)
        matched_cases = st.session_state.case_bank.find_similar(emb, top_k=2)
        st.session_state.similar_cases = matched_cases

        if not matched_cases:
            empty_rag_msg = (
                "Referans klinik kohort dizinlenmedi. Federe dijital ikiz eslestirmesi icin vaka kayitlari gereklidir."
                if lang.lower() == "tr"
                else "No reference clinical cohort indexed. Reference case bank requires cohort ingestion."
            )
            st.info(f"{empty_rag_msg}")
        else:
            for case in matched_cases:
                c_label = class_names_local[case['label']]
                c_sim = case['similarity'] * 100.0
                st.markdown(f"""
                <div class="glass-panel" style="padding: 1rem; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #F8FAFC; font-size: 0.95rem;">{case['case_id']}</strong><br>
                        <span style="font-size: 0.82rem; color: #94A3B8;">Diagnosis: <strong style="color: #F8FAFC;">{c_label}</strong></span>
                    </div>
                    <div>
                        <span style="background: rgba(56,189,248,0.15); color: #38BDF8; border: 1px solid rgba(56,189,248,0.3); padding: 4px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.8rem;">
                            {c_sim:.1f}% {get_text('sec3_rag_sim', lang)}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_cdss_media:
        st.markdown(f"### {get_text('sec3_voice_title', lang)}")
        voice_diag_name = LABEL_NAMES.get(pred_idx, "Normal")
        audio_bytes = get_or_create_audio(voice_diag_name, conf_val)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")

        st.markdown(f"### {get_text('sec3_report_title', lang)}")
        pat_id = f"PAT-{int(time.time()) % 100000}"
        report_name = f"FedXRay_{voice_diag_name}_{pat_id}.pdf"

        try:
            pdf_exp = get_diagnosis_explanation(
                voice_diag_name,
                conf_val,
                probabilities=st.session_state.scan_probs,
                heatmap=getattr(st.session_state, 'scan_heatmap', None),
            )
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
                label=f"{get_text('sec3_btn_download_pdf', lang)}",
                data=pdf_bytes,
                file_name=report_name,
                mime="application/pdf",
                use_container_width=True
            )
        except Exception:
            st.caption("PDF Report compilation standby.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1.5rem; color: #64748B; font-size: 0.82rem;">
    {get_text('footer_text', lang)}
</div>
""", unsafe_allow_html=True)
