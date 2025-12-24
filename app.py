"""
Fed-XRay: Federated Learning Medical Diagnosis Dashboard
=========================================================
A professional medical AI dashboard demonstrating privacy-preserving
federated learning for lung disease diagnosis from X-Ray images.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import time
from datetime import datetime
from typing import List, Dict

# Local imports
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


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fed-XRay | AI Radiologist",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - MEDICAL THEME
# ============================================================================

st.markdown("""
<style>
    /* =========================================================================
       FED-XRAY DESIGN SYSTEM
       ========================================================================= */

    /* 1. Global Reset & Typography */
    /* --------------------------------------------------------------------- */
    
    /* Force dark text for readability on light backgrounds */
    div[data-testid="stAppViewContainer"] {
        background: #f7fafc; /* Global Light Background */
        color: #2d3748;
    }
    
    h1, h2, h3, h4, h5, h6, 
    p, span, div, label, li, td, th {
        color: #2d3748 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Headers specific styling */
    h1 { font-weight: 800; letter-spacing: -0.02em; }
    h2 { font-weight: 700; letter-spacing: -0.01em; }
    h3 { font-weight: 600; font-size: 1.1rem !important; text-transform: uppercase; letter-spacing: 0.05em; color: #4a5568 !important; }
    
    /* 1.5. Modern SaaS Hero Section - Enhanced */
    /* --------------------------------------------------------------------- */
    .hero-container {
        text-align: center;
        padding: 5rem 2rem 4rem 2rem;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px;
        margin-bottom: 4rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: 50%;
        transform: translateX(-50%);
        width: 150%;
        height: 150%;
        background: radial-gradient(
            ellipse at center, 
            rgba(102, 126, 234, 0.08) 0%, 
            rgba(49, 130, 206, 0.04) 35%, 
            transparent 70%
        );
        animation: heroGlow 10s ease-in-out infinite;
        pointer-events: none;
    }
    
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(102, 126, 234, 0.3) 50%, transparent 100%);
    }
    
    @keyframes heroGlow {
        0%, 100% { 
            transform: translateX(-50%) scale(1); 
            opacity: 0.6; 
        }
        50% { 
            transform: translateX(-50%) scale(1.15); 
            opacity: 0.4; 
        }
    }
    
    .hero-title {
        font-size: clamp(2.5rem, 6vw, 4rem) !important;
        font-weight: 900 !important;
        letter-spacing: -0.04em !important;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        background-size: 200% auto;
        animation: gradientShift 8s ease infinite;
        margin-bottom: 1.5rem !important;
        position: relative;
        z-index: 1;
        line-height: 1.05 !important;
        padding: 0 1rem;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        font-size: 1.35rem !important;
        color: #64748b !important;
        opacity: 0.9;
        max-width: 680px;
        margin: 0 auto 2rem auto !important;
        padding: 0 2rem !important;
        font-weight: 400 !important;
        letter-spacing: 0.005em;
        line-height: 1.7 !important;
        position: relative;
        z-index: 1;
        text-align: center !important;
        display: block !important;
    }
    
    
    /* 1.6. Metric Cards Grid - Strict Layout for Perfect Alignment */
    /* --------------------------------------------------------------------- */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(1, 1fr);
        gap: 1.25rem;
        margin-bottom: 2.5rem;
        align-items: stretch; /* Force equal heights */
    }
    
    @media (min-width: 640px) {
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    @media (min-width: 1024px) {
        .metrics-grid {
            grid-template-columns: repeat(4, 1fr); /* Exactly 4 equal columns */
        }
    }
    
    /* 2. Component Normalization: Cards - Equal Height Enforcement */
    /* --------------------------------------------------------------------- */
    .metric-card, .hospital-card, .diagnosis-card, .cdss-card {
        background: white;
        padding: 1.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 0; /* Remove bottom margin for grid items */
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%; /* Force equal height - CRITICAL */
        min-height: 140px; /* Minimum height to prevent collapsing */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Space content evenly */
    }
    
    .metric-card h3 {
        margin: 0 0 auto 0; /* Push to top */
        flex-shrink: 0;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    
    .metric-card .value {
        margin-top: auto; /* Push value to bottom */
        padding-top: 1rem; /* Add spacing */
        color: #1a365d !important;
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
    }
    
    .metric-card:hover, .hospital-card:hover, .cdss-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Diagnosis Card Special */
    .diagnosis-card {
        text-align: center;
        border-top: 4px solid #3182ce;
    }

    /* 3. Section Headers */
    /* --------------------------------------------------------------------- */
    .section-header {
        background: linear-gradient(90deg, #1a365d 0%, #2c5282 100%);
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1.5rem 0;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
    }
    
    .section-header * { color: white !important; }

    /* 4. CDSS Bottom Section (Audio & PDF) - Grid Layout */
    /* --------------------------------------------------------------------- */
    /* 4. CDSS Bottom Section (Audio & PDF) - Professional File List */
    /* --------------------------------------------------------------------- */
    .cdss-file-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        background: white;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .cdss-file-row:hover {
        background: #f8fafc;
        border-color: #cbd5e0;
        transform: translateX(4px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .file-identity {
        display: flex;
        align-items: center;
    }
    
    .file-icon-wrapper {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin-right: 1.25rem;
        flex-shrink: 0;
    }
    
    .icon-audio { background-color: #ebf4ff; color: #3182ce; }
    .icon-pdf { background-color: #fff5f5; color: #e53e3e; }
    
    .file-info {
        display: flex;
        flex-direction: column;
    }
    
    .file-name {
        font-weight: 700;
        font-size: 0.95rem;
        color: #1a202c;
        margin-bottom: 2px;
    }
    
    .file-meta {
        font-size: 0.75rem;
        color: #718096;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    
    .file-action-trigger {
        font-size: 0.8rem;
        color: #4a5568 !important;
        font-weight: 600;
        display: flex;
        align-items: center;
    }

    /* 5. Interactive Elements (Buttons) */
    /* --------------------------------------------------------------------- */
    /* 5. Interactive Elements (Buttons) */
    /* --------------------------------------------------------------------- */
    .stButton > button {
        background: white !important;
        color: #2c5282 !important;
        border: 2px solid #2c5282 !important;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%; /* Make buttons full width in their containers */
        margin-top: 0.5rem;
    }
    
    .stButton > button:hover {
        background: #2c5282 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(44, 82, 130, 0.2);
        transform: translateY(-1px);
    }
    
    .stButton > button p { 
        font-size: 1rem;
    }

    /* CDSS Specific Spacing */
    .cdss-action-row {
        margin-bottom: 1rem !important; /* Increase gap between rows */
        padding: 1rem !important;
        background: white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #edf2f7;
    }
    
    /* Audio Player Spacing */
    .stAudio {
        margin-top: 0.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* 6. Utility Overrides */
    /* --------------------------------------------------------------------- */
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a365d 0%, #2a4365 100%);
        border-right: 1px solid #2d3748;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    
    /* Plotly Charts - Text Visibility Fixes */
    .js-plotly-plot .plotly .gtitle,
    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle {
        fill: #2d3748 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }
    
    .js-plotly-plot .plotly .tick text {
        fill: #4a5568 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Plotly Tooltips - Fix dark background mismatch */
    .js-plotly-plot .plotly .hoverlayer .hovertext rect {
        fill: #2d3748 !important;
        opacity: 0.9 !important;
    }
    
    .js-plotly-plot .plotly .hoverlayer .hovertext text {
        fill: white !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Alerts */
    div[data-testid="stAlert"] { padding: 0.75rem 1rem; border-radius: 8px; }
    div[data-testid="stAlert"] p { margin: 0; }
    
    /* 6. Domain Specific Utilities */
    /* --------------------------------------------------------------------- */
    .diagnosis-result { font-size: 1.8rem; font-weight: 700; margin: 1rem 0; }
    .diagnosis-normal { color: #38a169 !important; }
    .diagnosis-pneumonia { color: #dd6b20 !important; }
    .diagnosis-covid { color: #e53e3e !important; }
    
    .status-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .status-ready { background: #c6f6d5; color: #22543d; }
    .status-training { background: #bee3f8; color: #2a4365; }
    .status-complete { background: #9ae6b4; color: #22543d; }
    
    /* Hide Streamlit cruft */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables for persistence across reruns."""
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'global_model' not in st.session_state:
        st.session_state.global_model = None
    # CRITICAL: Store trained weights separately for persistence
    if 'trained_weights' not in st.session_state:
        st.session_state.trained_weights = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = {
            'loss': [], 'accuracy': [], 'round': [],
            'precision': [], 'recall': [], 'f1_score': [],
            'test_accuracy': []
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
    # CDSS Features: Historical case bank for similarity search
    if 'case_bank' not in st.session_state:
        st.session_state.case_bank = None


init_session_state()


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: white; margin: 0;">⚙️ Configuration</h2>
        <h4 style="color: white; font-size: 0.9rem;">Federated Learning Parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Network Configuration
    st.markdown("##### 🏥 Hospital Network")
    n_hospitals = st.slider(
        "Number of Hospitals",
        min_value=2,
        max_value=10,
        value=4,
        help="Each hospital represents a client in the federated network"
    )
    
    samples_per_hospital = st.slider(
        "Samples per Hospital",
        min_value=100,
        max_value=500,
        value=200,
        step=50,
        help="Number of X-Ray images generated per hospital"
    )
    
    st.markdown("---")
    
    # Training Configuration
    st.markdown("##### 🔄 Training Rounds")
    n_rounds = st.slider(
        "Federated Rounds",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of communication rounds between hospitals and server"
    )
    
    local_epochs = st.slider(
        "Local Epochs",
        min_value=1,
        max_value=5,
        value=2,
        help="Training epochs per hospital before aggregation"
    )
    
    st.markdown("---")
    
    # Privacy Configuration
    st.markdown("##### 🔒 Privacy Settings")
    privacy_noise = st.slider(
        "Differential Privacy (ε)",
        min_value=0.0,
        max_value=0.1,
        value=0.01,
        step=0.01,
        format="%.2f",
        help="Gaussian noise added to model updates. Higher = more privacy, less accuracy"
    )
    
    st.markdown("""
    <div class="privacy-badge">
        🛡️ Privacy Preserved
    </div>
    <h6 style="font-size: 0.8rem; color: #718096; margin-top: 0.5rem;">
        Patient data never leaves hospitals
    </h6>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Info
    st.markdown("##### 📊 Model Info")
    model = create_model()
    n_params = count_parameters(model)
    st.metric("Total Parameters", f"{n_params:,}")
    st.caption("Lightweight CNN for efficient FL")
    
    st.markdown("---")
    
    # ===== SECURITY OPERATIONS CENTER =====
    st.markdown("##### 🛡️ Security Operations")
    
    simulate_attack = st.checkbox(
        "⚠️ Simulate Attack (Hospital #3)",
        value=False,
        help="Compromise Hospital 3 with a Label Flipping Attack"
    )
    
    activate_defense = st.checkbox(
        "🔒 Activate Defense Shield",
        value=False,
        help="Enable validation-based malicious node detection"
    )
    
    if simulate_attack:
        st.markdown("""
        <div style="background: #fed7d7; padding: 0.5rem; border-radius: 8px; color: #c53030; font-size: 0.85rem;">
            🚨 <strong>Attack Active:</strong> Hospital 3 is compromised!
        </div>
        """, unsafe_allow_html=True)
    
    if activate_defense:
        st.markdown("""
        <div style="background: #c6f6d5; padding: 0.5rem; border-radius: 8px; color: #22543d; font-size: 0.85rem;">
            ✅ <strong>Defense Active:</strong> Filtering malicious nodes
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">🫁 Fed-XRay | AI Radiologist Network</h1>
    <p class="hero-subtitle">Privacy-Preserving Federated Learning for Lung Disease Detection</p>
</div>
""", unsafe_allow_html=True)

# Quick metrics row (CSS Grid)
total_samples = n_hospitals * samples_per_hospital
status = "Trained ✓" if st.session_state.model_trained else "Ready"

st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card">
        <h3>🏥 Hospitals</h3>
        <div class="value">{n_hospitals}</div>
    </div>
    <div class="metric-card">
        <h3>🔄 FL Rounds</h3>
        <div class="value">{n_rounds}</div>
    </div>
    <div class="metric-card">
        <h3>📊 Total Samples</h3>
        <div class="value">{total_samples:,}</div>
    </div>
    <div class="metric-card">
        <h3>📡 Status</h3>
        <div class="value" style="font-size: 1.5rem;">{status}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SECTION 1: HOSPITAL DATA VISUALIZATION
# ============================================================================

st.markdown('<div class="section-header">📊 Section 1: Hospital Data Distribution (Non-IID)</div>', unsafe_allow_html=True)

st.info("""
**Non-IID Data Explained:** In real federated learning, each hospital has different patient populations. 
Some might see more COVID cases (hotspots), others more routine check-ups (normal). 
This heterogeneity makes FL challenging but realistic.
""")

# Generate hospital data button
if st.button("🔬 Generate Hospital Data", key="gen_data"):
    with st.spinner("Generating synthetic X-Ray data for each hospital..."):
        generator = MedicalDataGenerator()
        
        # Generate sample images for visualization
        st.session_state.hospital_samples = {}
        
        for h in range(n_hospitals):
            distribution = get_distribution_info(h, n_hospitals)
            images, labels = generator.create_hospital_data(
                n_samples=samples_per_hospital,
                distribution=distribution,
                hospital_id=h
            )
            # Store a few samples for visualization
            st.session_state.hospital_samples[h] = {
                'images': images[:9],  # First 9 for grid
                'labels': labels[:9],
                'distribution': distribution,
                'all_labels': labels
            }
        
        # Create dataloaders for training
        st.session_state.dataloaders = create_hospital_dataloaders(
            n_hospitals=n_hospitals,
            samples_per_hospital=samples_per_hospital,
            batch_size=32
        )
        
        # Create GLOBAL HOLD-OUT TEST SET (clients never see this!)
        test_images, test_labels = create_global_test_set(n_samples=300, seed=9999)
        st.session_state.global_test_set = (test_images, test_labels)
        
        st.session_state.hospital_data_generated = True
    
    st.success("✅ Data generated! Global test set (300 samples) created for unbiased evaluation.")
    st.rerun()

# Display hospital data if generated
if st.session_state.hospital_data_generated and st.session_state.hospital_samples:
    
    # Create tabs for each hospital
    hospital_tabs = st.tabs([f"🏥 Hospital {i+1}" for i in range(min(n_hospitals, len(st.session_state.hospital_samples)))])
    
    for h_idx, tab in enumerate(hospital_tabs):
        if h_idx not in st.session_state.hospital_samples:
            continue
            
        with tab:
            data = st.session_state.hospital_samples[h_idx]
            
            col_info, col_images = st.columns([1, 2])
            
            with col_info:
                # Distribution pie chart - Modern Vibrant Donut
                dist = data['distribution']
                labels_names = ['Normal', 'Pneumonia', 'COVID-19']
                # Vibrant SaaS Medical Palette
                modern_colors = ['#10B981', '#F59E0B', '#EF4444']
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels_names,
                    values=[dist[0], dist[1], dist[2]],
                    marker=dict(
                        colors=modern_colors,
                        line=dict(color='white', width=2)
                    ),
                    hole=0.55,
                    textinfo='percent',
                    textposition='inside',
                    textfont=dict(size=14, color='white', family="Inter, sans-serif"),
                    hoverinfo='label+percent',
                    pull=[0.02, 0.02, 0.02]
                )])
                
                fig_pie.update_layout(
                    annotations=[dict(text='DIST', x=0.5, y=0.5, font_size=20, showarrow=False, font_family="Inter")],
                    height=300,
                    margin=dict(l=20, r=20, t=10, b=10),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Stats with explicit dark color
                all_labels = data['all_labels']
                st.markdown(f"""
                <div style="color: #1a202c; background: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <strong style="color: #1a202c;">Statistics:</strong><br>
                    <span style="color: #1a202c;">🟢 Normal: {np.sum(all_labels == 0)} patients</span><br>
                    <span style="color: #1a202c;">🟠 Pneumonia: {np.sum(all_labels == 1)} patients</span><br>
                    <span style="color: #1a202c;">🔴 COVID-19: {np.sum(all_labels == 2)} patients</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col_images:
                # Show sample X-rays in a grid
                st.markdown("**Sample X-Ray Images:**")
                
                images = data['images'][:9]
                sample_labels = data['labels'][:9]
                
                # Create a 3x3 grid visualization
                fig = make_subplots(
                    rows=3, cols=3,
                    subplot_titles=[MedicalDataGenerator.LABELS[l] for l in sample_labels],
                    vertical_spacing=0.1,
                    horizontal_spacing=0.05
                )
                
                for i in range(min(9, len(images))):
                    row = i // 3 + 1
                    col = i % 3 + 1
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=images[i],
                            colorscale='gray',
                            showscale=False
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                
                # Remove axes
                for i in range(1, 10):
                    fig.update_xaxes(showticklabels=False, showgrid=False, row=(i-1)//3+1, col=(i-1)%3+1)
                    fig.update_yaxes(showticklabels=False, showgrid=False, row=(i-1)//3+1, col=(i-1)%3+1)
                
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SECTION 2: LIVE FEDERATED TRAINING
# ============================================================================

st.markdown('<div class="section-header">🚀 Section 2: Federated Training</div>', unsafe_allow_html=True)

st.info("""
**Federated Learning Process:** 
1. Central server sends global model to all hospitals
2. Each hospital trains on their LOCAL data (data never leaves!)
3. Hospitals send only model WEIGHTS back to server
4. Server aggregates weights using FedAvg algorithm
5. Repeat for specified rounds
""")

# Training controls
col_btn, col_status = st.columns([1, 2])

with col_btn:
    start_training = st.button(
        "🩺 Start Diagnosis Network", 
        key="start_training",
        disabled=not st.session_state.hospital_data_generated
    )

# Placeholders for live updates
chart_placeholder = st.empty()
progress_placeholder = st.empty()
log_placeholder = st.empty()

if start_training and st.session_state.hospital_data_generated:
    # Reset training history with medical metrics
    st.session_state.training_history = {
        'loss': [], 'accuracy': [], 'round': [],
        'precision': [], 'recall': [], 'f1_score': [],
        'test_accuracy': [], 'test_loss': [],
        'blocked_count': 0
    }
    
    # Initialize FL components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Server with defense mode
    server = CentralServer(
        device=device, 
        privacy_noise=privacy_noise,
        defense_mode=activate_defense
    )
    
    # Create clients - Hospital 3 (index 2) is malicious if attack is enabled
    clients = []
    for i in range(n_hospitals):
        is_malicious = (simulate_attack and i == 2)  # Hospital 3 = index 2
        
        clients.append(HospitalClient(
            client_id=i,
            dataloader=st.session_state.dataloaders[i],
            device=device,
            learning_rate=0.001,
            local_epochs=local_epochs,
            malicious=is_malicious
        ))
    
    # Get global test set for unbiased evaluation
    test_images, test_labels = st.session_state.global_test_set
    
    # Security alert placeholder
    security_alert_placeholder = st.empty()
    
    # Training loop with live updates
    for round_num in range(1, n_rounds + 1):
        with progress_placeholder.container():
            st.progress(round_num / n_rounds, text=f"Round {round_num}/{n_rounds}")
        
        # Run federated round with security
        metrics, client_metrics, test_metrics, security_report = run_federated_round(
            server, clients, round_num,
            test_images=test_images,
            test_labels=test_labels,
            use_defense=activate_defense
        )
        
        # Update history with training metrics
        st.session_state.training_history['loss'].append(metrics['loss'])
        st.session_state.training_history['accuracy'].append(metrics['accuracy'] * 100)
        st.session_state.training_history['round'].append(round_num)
        
        # Update history with test metrics (proper evaluation)
        if test_metrics:
            st.session_state.training_history['test_accuracy'].append(test_metrics.accuracy * 100)
            st.session_state.training_history['test_loss'].append(test_metrics.loss)
            st.session_state.training_history['precision'].append(test_metrics.precision * 100)
            st.session_state.training_history['recall'].append(test_metrics.recall * 100)
            st.session_state.training_history['f1_score'].append(test_metrics.f1_score * 100)
            st.session_state.confusion_matrix = test_metrics.confusion_matrix
        
        # Update charts - 2x2 grid for comprehensive metrics
        with chart_placeholder.container():
            col1, col2 = st.columns(2)
            
            with col1:
                # Loss chart
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=st.session_state.training_history['round'],
                    y=st.session_state.training_history['test_loss'],
                    mode='lines+markers',
                    name='Test Loss',
                    line=dict(color='#e53e3e', width=3),
                    marker=dict(size=8)
                ))
                fig_loss.update_layout(
                    title="📉 Test Loss (Global Hold-out)",
                    xaxis_title="Round",
                    yaxis_title="Loss",
                    height=280,
                    margin=dict(l=40, r=20, t=50, b=40),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2d3748')
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                # Test Accuracy chart (TRUE evaluation)
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=st.session_state.training_history['round'],
                    y=st.session_state.training_history['test_accuracy'],
                    mode='lines+markers',
                    name='Test Accuracy',
                    line=dict(color='#38a169', width=3),
                    marker=dict(size=8)
                ))
                fig_acc.update_layout(
                    title="📈 Test Accuracy (Global Hold-out)",
                    xaxis_title="Round",
                    yaxis_title="Accuracy %",
                    height=280,
                    margin=dict(l=40, r=20, t=50, b=40),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    yaxis_range=[0, 100],
                    font=dict(color='#2d3748')
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            # Medical Metrics Row
            col3, col4 = st.columns(2)
            
            with col3:
                # F1-Score chart (most important for medical AI)
                fig_f1 = go.Figure()
                fig_f1.add_trace(go.Scatter(
                    x=st.session_state.training_history['round'],
                    y=st.session_state.training_history['f1_score'],
                    mode='lines+markers',
                    name='F1-Score',
                    line=dict(color='#805ad5', width=3),
                    marker=dict(size=8)
                ))
                fig_f1.update_layout(
                    title="🎯 F1-Score (Macro)",
                    xaxis_title="Round",
                    yaxis_title="F1 %",
                    height=280,
                    margin=dict(l=40, r=20, t=50, b=40),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    yaxis_range=[0, 100],
                    font=dict(color='#2d3748')
                )
                st.plotly_chart(fig_f1, use_container_width=True)
            
            with col4:
                # Precision & Recall on same chart
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=st.session_state.training_history['round'],
                    y=st.session_state.training_history['precision'],
                    mode='lines+markers',
                    name='Precision',
                    line=dict(color='#3182ce', width=2),
                    marker=dict(size=6)
                ))
                fig_pr.add_trace(go.Scatter(
                    x=st.session_state.training_history['round'],
                    y=st.session_state.training_history['recall'],
                    mode='lines+markers',
                    name='Recall',
                    line=dict(color='#dd6b20', width=2),
                    marker=dict(size=6)
                ))
                fig_pr.update_layout(
                    title="⚖️ Precision & Recall",
                    xaxis_title="Round",
                    yaxis_title="%",
                    height=280,
                    margin=dict(l=40, r=20, t=50, b=40),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    yaxis_range=[0, 100],
                    font=dict(color='#2d3748'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig_pr, use_container_width=True)
        
        # Log update with medical metrics and security alerts
        with log_placeholder.container():
            # Security Alert
            if security_report and security_report.clients_blocked:
                st.markdown(f"""
                <div style="background: #fed7d7; padding: 0.75rem; border-radius: 8px; color: #c53030; margin-bottom: 0.5rem;">
                    🚨 <strong>SECURITY ALERT (Round {round_num}):</strong> 
                    Malicious activity detected from Hospital(s): {[c+1 for c in security_report.clients_blocked]}
                    <br>⛔ Node(s) BLOCKED from aggregation!
                </div>
                """, unsafe_allow_html=True)
                st.session_state.training_history['blocked_count'] = server.get_blocked_count()
            
            # Metrics
            if test_metrics:
                st.markdown(f"""
                <div style="background: white; padding: 0.75rem; border-radius: 8px; color: #1a202c;">
                    <strong>Round {round_num}:</strong> 
                    Test Acc = <code>{test_metrics.accuracy*100:.1f}%</code> | 
                    F1 = <code>{test_metrics.f1_score*100:.1f}%</code> | 
                    Precision = <code>{test_metrics.precision*100:.1f}%</code> | 
                    Recall = <code>{test_metrics.recall*100:.1f}%</code>
                </div>
                """, unsafe_allow_html=True)
        
        time.sleep(0.2)  # Small delay for visualization
    
    # Save trained model and weights
    trained_model = server.get_model()
    st.session_state.global_model = trained_model
    # CRITICAL: Save a deep copy of trained weights for inference persistence
    import copy
    st.session_state.trained_weights = copy.deepcopy(trained_model.state_dict())
    st.session_state.model_trained = True
    print(f"[DEBUG] Model saved. trained_weights keys: {len(st.session_state.trained_weights)}")
    
    progress_placeholder.empty()
    
    # Final summary with confusion matrix
    final_acc = st.session_state.training_history['test_accuracy'][-1]
    final_f1 = st.session_state.training_history['f1_score'][-1]
    blocked_count = st.session_state.training_history.get('blocked_count', 0)
    
    st.success(f"✅ Federated training complete! Test Accuracy: {final_acc:.1f}% | F1-Score: {final_f1:.1f}%")
    
    # Security Summary
    if blocked_count > 0:
        st.warning(f"🛡️ **Security Report:** {blocked_count} malicious update(s) blocked during training!")
    elif simulate_attack and not activate_defense:
        st.error("⚠️ **Warning:** Training was compromised by malicious node. Enable Defense Shield to protect!")
    elif activate_defense:
        st.info("🔒 **Security Status:** Defense Shield was active. All threats neutralized.")
    
    # Display confusion matrix
    if st.session_state.confusion_matrix is not None:
        st.markdown("### 📊 Confusion Matrix (Final)")
        cm = st.session_state.confusion_matrix
        class_names = ['Normal', 'Pneumonia', 'COVID-19']
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "black"},
            showscale=True
        ))
        fig_cm.update_layout(
            title="Predicted vs Actual",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=350,
            font=dict(color='#2d3748'),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# Show existing training history if available
elif st.session_state.training_history['round']:
    with chart_placeholder.container():
        col_loss, col_acc = st.columns(2)
        
        with col_loss:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=st.session_state.training_history['round'],
                y=st.session_state.training_history['loss'],
                mode='lines+markers',
                name='Global Loss',
                line=dict(color='#e53e3e', width=3),
                marker=dict(size=8)
            ))
            fig_loss.update_layout(
                title="📉 Global Loss",
                xaxis_title="Round",
                yaxis_title="Loss",
                height=300,
                margin=dict(l=40, r=20, t=50, b=40),
                plot_bgcolor='white',
                paper_bgcolor='white',
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col_acc:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=st.session_state.training_history['round'],
                y=st.session_state.training_history['accuracy'],
                mode='lines+markers',
                name='Global Accuracy',
                line=dict(color='#38a169', width=3),
                marker=dict(size=8)
            ))
            fig_acc.update_layout(
                title="📈 Global Accuracy (%)",
                xaxis_title="Round",
                yaxis_title="Accuracy %",
                height=300,
                margin=dict(l=40, r=20, t=50, b=40),
                plot_bgcolor='white',
                paper_bgcolor='white',
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_acc, use_container_width=True)


# ============================================================================
# SECTION 3: AI RADIOLOGIST INFERENCE
# ============================================================================

st.markdown('<div class="section-header">🤖 Section 3: AI Radiologist (Inference)</div>', unsafe_allow_html=True)

st.info("""
**AI-Assisted Diagnosis:** After federated training, the model can classify new X-Ray images.
Click the button to generate a random patient scan and see the AI's diagnosis with confidence scores.
""")

col_scan, col_result = st.columns([1, 1])

with col_scan:
    scan_button = st.button(
        "🔍 Scan New Patient",
        key="scan_patient", 
        disabled=not st.session_state.model_trained
    )
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Train the model first to enable diagnosis")

if scan_button and st.session_state.model_trained:
    # Generate a random sample
    generator = MedicalDataGenerator(seed=int(time.time()))
    true_label = np.random.randint(0, 3)
    image = generator.generate_synthetic_xray(true_label)
    
    # Store for display
    st.session_state.scan_image = image
    st.session_state.scan_true_label = true_label
    
    # CRITICAL FIX: Create fresh model and load TRAINED weights
    # This ensures we use the trained model, not a fresh random one
    model = create_model()  # Create fresh architecture
    
    # Load trained weights if available
    if st.session_state.trained_weights is not None:
        model.load_state_dict(st.session_state.trained_weights)
        print("[DEBUG] Loaded trained weights successfully")
    else:
        st.error("❌ No trained weights found! Please train the model first.")
        st.stop()
    
    model = model.cpu()
    model.eval()  # Set to evaluation mode
    
    # Prepare image tensor with correct shape: [1, 1, 28, 28]
    img_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    
    # CRITICAL: Enable gradients for Grad-CAM (do NOT use torch.no_grad())
    torch.set_grad_enabled(True)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model)
    
    try:
        # Generate heatmap using Grad-CAM (requires gradient flow)
        heatmap, predicted_class, confidence = gradcam.generate_heatmap(img_tensor)
        confidence = confidence * 100
        
        # Handle NaN confidence
        if np.isnan(confidence):
            confidence = 33.3
        
        # Get probabilities for confidence breakdown (this can use no_grad)
        with torch.no_grad():
            logits = model(img_tensor)
            
            # Check for NaN
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                probs_np = np.array([0.33, 0.33, 0.34])
            else:
                probs = torch.nn.functional.softmax(logits, dim=1)
                probs_np = probs[0].cpu().numpy()
                
                # Validate probs
                if np.any(np.isnan(probs_np)):
                    probs_np = np.array([0.33, 0.33, 0.34])
        
        # Create overlay
        overlay = create_overlay(image, heatmap, alpha=0.5)
        
        # Store results
        st.session_state.scan_heatmap = heatmap
        st.session_state.scan_overlay = overlay
        st.session_state.scan_probs = probs_np
        st.session_state.scan_predicted = predicted_class
        st.session_state.scan_confidence = confidence
        
    except Exception as e:
        # Fallback without Grad-CAM - show error in console
        print(f"Grad-CAM Error: {e}")
        st.session_state.scan_heatmap = np.ones_like(image) * 0.5
        st.session_state.scan_overlay = np.stack([image, image, image], axis=-1)
        
        with torch.no_grad():
            logits = model(img_tensor)
            if torch.isnan(logits).any():
                probs_np = np.array([0.33, 0.33, 0.34])
                predicted_class = 0
                confidence = 33.3
            else:
                probs = torch.nn.functional.softmax(logits, dim=1)
                confidence, predicted_class = probs.max(dim=1)
                probs_np = probs[0].cpu().numpy()
                predicted_class = predicted_class.item()
                confidence = confidence.item() * 100
            
            st.session_state.scan_probs = probs_np
            st.session_state.scan_predicted = predicted_class
            st.session_state.scan_confidence = confidence
    
    finally:
        # Clean up hooks and restore grad state
        gradcam.remove_hooks()
        torch.set_grad_enabled(True)  # Restore default

# Display results with Grad-CAM visualization
if hasattr(st.session_state, 'scan_image') and st.session_state.model_trained:
    
    # XAI Visualization Header
    st.markdown("### 🔬 Explainable AI Analysis")
    st.markdown("*The model's attention is visualized using Grad-CAM (Gradient-weighted Class Activation Mapping)*")
    
    # 3-Column Display: Raw | Heatmap | Overlay
    col_raw, col_heat, col_overlay = st.columns(3)
    
    with col_raw:
        st.markdown("**📷 Original X-Ray**")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Heatmap(
            z=st.session_state.scan_image,
            colorscale='gray',
            showscale=False
        ))
        fig_raw.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor='x'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black')
        )
        st.plotly_chart(fig_raw, use_container_width=True)
        
        true_name = MedicalDataGenerator.LABELS[st.session_state.scan_true_label]
        st.caption(f"*Ground Truth: {true_name}*")
    
    with col_heat:
        st.markdown("**🔥 Attention Heatmap**")
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Heatmap(
            z=st.session_state.scan_heatmap if hasattr(st.session_state, 'scan_heatmap') else np.ones((28, 28)) * 0.5,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title='Focus', tickfont=dict(color='#2d3748'))
        ))
        fig_heat.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor='x'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black')
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("*Red = High attention*")
    
    with col_overlay:
        st.markdown("**🎯 Overlay**")
        if hasattr(st.session_state, 'scan_overlay'):
            overlay_img = st.session_state.scan_overlay
            # Convert to image format for Plotly
            fig_overlay = go.Figure()
            fig_overlay.add_trace(go.Image(z=(overlay_img * 255).astype(np.uint8)))
            fig_overlay.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black')
            )
            st.plotly_chart(fig_overlay, use_container_width=True)
        st.caption("*X-Ray + Attention*")
    
    st.markdown("---")
    
    # Diagnosis Result
    col_diag, col_conf = st.columns([1, 1])
    
    with col_diag:
        st.markdown("### 🩺 AI Diagnosis")
        
        probs = st.session_state.scan_probs
        predicted = st.session_state.scan_predicted
        
        class_names = ['Normal', 'Pneumonia', 'COVID-19']
        class_colors = ['diagnosis-normal', 'diagnosis-pneumonia', 'diagnosis-covid']
        class_emojis = ['🟢', '🟠', '🔴']
        
        predicted_name = class_names[predicted]
        color_class = class_colors[predicted]
        emoji = class_emojis[predicted]
        
        st.markdown(f"""
        <div class="diagnosis-card">
            <p style="color: #718096; margin-bottom: 0.5rem;">AI Prediction</p>
            <div class="diagnosis-result {color_class}">
                {emoji} {predicted_name}
            </div>
            <p style="color: #4a5568;">Confidence: <strong>{st.session_state.scan_confidence:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # XAI Explanation
        explanation = get_explanation_text(predicted, st.session_state.scan_confidence / 100)
        st.markdown(explanation)
        
        # Accuracy check
        if predicted == st.session_state.scan_true_label:
            st.success("✅ Correct diagnosis!")
        else:
            st.error(f"❌ Misdiagnosis. True label: {class_names[st.session_state.scan_true_label]}")
    
    with col_conf:
        st.markdown("### 📊 Confidence Breakdown")
        
        # Vibrant SaaS Medical Palette for Metrics
        modern_metrics_colors = ['#10B981', '#F59E0B', '#EF4444']
        
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Bar(
            x=probs * 100,
            y=class_names,
            orientation='h',
            marker=dict(
                color=modern_metrics_colors,
                line=dict(color='white', width=1)
            ),
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='auto',
            textfont=dict(size=12, color='white', family="Inter, sans-serif")
        ))
        
        fig_conf.update_layout(
            height=200,
            margin=dict(l=80, r=20, t=10, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, range=[0, 100]),
            yaxis=dict(showgrid=False),
            font=dict(family="Inter, sans-serif", color='#2d3748')
        )
        
        st.plotly_chart(fig_conf, use_container_width=True)
        
        st.info("💡 **How Grad-CAM works:** The model's gradients are computed for the predicted class, revealing which image regions most influenced the decision.")
    
    # =========================================================================
    # CDSS FEATURES: Voice, Similarity Search, PDF Export
    # =========================================================================
    
    st.markdown("---")
    
    # Section Header
    st.markdown('<div class="section-header">🔍 Clinical Decision Support</div>', unsafe_allow_html=True)
    
    # Grid Layout: Actions (Voice/PDF) vs Context (Similar Cases)
    col_actions, col_context = st.columns([1, 1.2])
    
    # -------------------------------------------------------------------------
    # LEFT COLUMN: ACTIONS
    # -------------------------------------------------------------------------
    with col_actions:
        st.markdown('<h3 style="margin-bottom: 1rem;">⚡ Actions</h3>', unsafe_allow_html=True)
        
        # 1. Voice Assistant
        v_col_info, v_col_act = st.columns([3, 1])
        with v_col_info:
            st.markdown(f"""
            <div class="cdss-file-row" style="margin-bottom: 0;">
                <div class="file-identity">
                    <div class="file-icon-wrapper icon-audio">🔊</div>
                    <div class="file-info">
                        <p class="file-name">Voice Diagnosis</p>
                        <p class="file-meta">audio/mp3 • 256KB • {datetime.now().strftime('%H:%M')}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with v_col_act:
            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
            try:
                diagnosis_name = class_names[predicted]
                audio_data = get_or_create_audio(diagnosis_name, st.session_state.scan_confidence)
                if not audio_data:
                    st.caption("Unavailable")
            except:
                st.caption("Error")

        # Smaller audio player below info
        if audio_data:
            st.audio(audio_data, format='audio/mp3')
            
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        
        # 2. PDF Report
        p_col_info, p_col_act = st.columns([3, 1])
        with p_col_info:
            st.markdown(f"""
            <div class="cdss-file-row" style="margin-bottom: 0;">
                <div class="file-identity">
                    <div class="file-icon-wrapper icon-pdf">📄</div>
                    <div class="file-info">
                        <p class="file-name">Medical Report</p>
                        <p class="file-meta">application/pdf • 1.2MB • {datetime.now().strftime('%Y-%m-%d')}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with p_col_act:
            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
            try:
                pdf_explanation = get_diagnosis_explanation(
                    class_names[predicted], 
                    st.session_state.scan_confidence
                )
                similar_cases_for_pdf = st.session_state.get('similar_cases', None)
                
                pdf_data = generate_medical_report(
                    patient_id=f"PAT-{int(time.time()) % 100000}",
                    diagnosis=class_names[predicted],
                    confidence=st.session_state.scan_confidence,
                    explanation=pdf_explanation,
                    heatmap_image=st.session_state.scan_heatmap if hasattr(st.session_state, 'scan_heatmap') else None,
                    original_image=st.session_state.scan_image,
                    similar_cases=similar_cases_for_pdf
                )
                
                st.download_button(
                    label="📥 Save",
                    data=pdf_data,
                    file_name=f"fedxray_report_{int(time.time())}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.caption("Error")

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: CONTEXT (SIMILAR CASES)
    # -------------------------------------------------------------------------
    with col_context:
        st.markdown('<h3 style="margin-bottom: 1rem;">🧬 Similar Historical Cases</h3>', unsafe_allow_html=True)
        
        try:
            # Initialize case bank if needed
            if st.session_state.case_bank is None:
                with st.spinner("Indexing historical cases..."):
                    st.session_state.case_bank = HistoricalCaseBank(n_cases=100)
            
            # Find similar cases
            model_for_embed = create_model()
            if st.session_state.trained_weights is not None:
                model_for_embed.load_state_dict(st.session_state.trained_weights)
            
            img_tensor = torch.FloatTensor(st.session_state.scan_image).unsqueeze(0).unsqueeze(0)
            embedding = extract_embedding(model_for_embed, img_tensor)
            
            similar_cases = st.session_state.case_bank.find_similar(embedding, top_k=2)
            st.session_state.similar_cases = similar_cases
            
            # Display Cases
            for i, case in enumerate(similar_cases):
                label_name = LABEL_NAMES.get(case['label'], 'Unknown')
                label_emoji = LABEL_COLORS.get(case['label'], '⚪')
                similarity_pct = case['similarity'] * 100
                
                # Card Layout
                st.markdown(f"""
                <div class="cdss-card" style="padding: 1rem; display: flex; align-items: center; gap: 1rem;">
                    <div style="flex-grow: 1;">
                        <p style="font-weight: 700; color: #2d3748; margin: 0;">{case['case_id']}</p>
                        <p style="font-size: 0.85rem; color: #718096; margin: 0;">{label_emoji} {label_name}</p>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: #ebf8ff; color: #2b6cb0; padding: 4px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem;">
                            {similarity_pct:.1f}% Match
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Similarity search error: {str(e)[:50]}")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background: #f7fafc; border-radius: 8px;">
    <p style="color: #2d3748 !important; margin-bottom: 0.5rem;">🔒 <strong style="color: #2d3748;">Privacy Notice:</strong> This demonstration uses synthetic data. 
    In real Federated Learning, patient X-Ray images never leave hospital servers.</p>
    <p style="font-size: 0.8rem; color: #4a5568 !important; margin: 0;">Fed-XRay | Federated Learning for Medical AI | Built with PyTorch & Streamlit</p>
</div>
""", unsafe_allow_html=True)
