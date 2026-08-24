"""
Fed-XRay: Federated Learning Medical Diagnosis & CDSS Platform
===============================================================
A modern, privacy-preserving Clinical Decision Support System and
Federated Learning dashboard with advanced distributed optimizers
(FedAvg, FedProx, SCAFFOLD, FedDyn, MOON) for lung disease diagnosis.

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

# Local modular imports
from src.fed_xray.data.generator import (
    MedicalDataGenerator, 
    XRayDataset, 
    create_hospital_dataloaders,
    get_distribution_info,
    create_global_test_set
)
from src.fed_xray.models.cnn import XRayClassifier, create_model, count_parameters
from src.fed_xray.core.client import HospitalClient
from src.fed_xray.core.server import CentralServer, run_federated_round
from src.fed_xray.core.metrics import EvaluationMetrics, SecurityReport
from src.fed_xray.cdss.xai import GradCAM, create_overlay, get_explanation_text
from src.fed_xray.cdss.similarity import HistoricalCaseBank, extract_embedding, LABEL_NAMES, LABEL_COLORS
from src.fed_xray.cdss.voice import get_or_create_audio
from src.fed_xray.cdss.report import generate_medical_report, get_diagnosis_explanation
from src.fed_xray.cdss.i18n import t
from src.fed_xray.cdss.styles import get_custom_css


# ============================================================================
# PAGE CONFIGURATION & DESIGN SYSTEM 2.0
# ============================================================================

st.set_page_config(
    page_title="Fed-XRay | AI Radiologist Network",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Modern CSS Tokens and Fluid Layout Adaptors
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ============================================================================
# STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize application state variables."""
    if 'lang' not in st.session_state:
        st.session_state.lang = "tr"
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'training_history' not in st.session_state:
        st.session_state.training_history = {
            'loss': [], 'accuracy': [], 'round': [],
            'precision': [], 'recall': [], 'f1_score': [],
            'test_accuracy': [], 'test_loss': [],
            'blocked_count': 0
        }
    if 'hospital_data_generated' not in st.session_state:
        st.session_state.hospital_data_generated = False
    if 'hospital_data' not in st.session_state:
        st.session_state.hospital_data = []
    if 'dataloaders' not in st.session_state:
        st.session_state.dataloaders = []
    if 'trained_weights' not in st.session_state:
        st.session_state.trained_weights = None
    if 'global_test_set' not in st.session_state:
        st.session_state.global_test_set = None
    if 'confusion_matrix' not in st.session_state:
        st.session_state.confusion_matrix = None
    if 'case_bank' not in st.session_state:
        st.session_state.case_bank = None


init_session_state()


# ============================================================================
# SIDEBAR CONFIGURATION & CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 1.25rem 0 1rem 0;">
        <h2 style="color: white; margin: 0; font-size: 1.5rem;">🫁 Fed-XRay</h2>
        <p style="color: #94A3B8; font-size: 0.8rem; margin: 0.25rem 0 0 0;">{t("sidebar_brand_sub", st.session_state.lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 🌐 Bilingual Language Toggle
    selected_lang_label = st.selectbox(
        t("language_selector", st.session_state.lang),
        options=["🇹🇷 Türkçe", "🇬🇧 English"],
        index=0 if st.session_state.lang == "tr" else 1
    )
    
    new_lang = "tr" if "Türkçe" in selected_lang_label else "en"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    lang = st.session_state.lang
    st.markdown("---")
    
    # Network Configuration
    st.markdown(f"<h4 style='font-size:0.95rem;'>{t('network_config', lang)}</h4>", unsafe_allow_html=True)
    n_hospitals = st.slider(
        t("num_hospitals", lang),
        min_value=2,
        max_value=10,
        value=4,
        help=t("num_hospitals_help", lang)
    )
    
    samples_per_hospital = st.slider(
        t("samples_per_hospital", lang),
        min_value=100,
        max_value=500,
        value=200,
        step=50,
        help=t("samples_per_hospital_help", lang)
    )
    
    st.markdown("---")
    
    # Optimization Algorithm Selection
    st.markdown(f"<h4 style='font-size:0.95rem;'>{t('algorithm_selector_title', lang)}</h4>", unsafe_allow_html=True)
    selected_algorithm = st.selectbox(
        "Optimizer",
        options=["FedAvg", "FedProx", "SCAFFOLD", "FedDyn", "MOON"],
        index=1,
        help=t("algorithm_selector_help", lang)
    )
    
    # Dynamic Hyperparameters per Algorithm
    algo_mu = 0.01
    algo_alpha = 0.01
    algo_temp = 0.5
    
    if selected_algorithm in ("FedProx", "MOON"):
        algo_mu = st.slider(
            t("param_mu", lang),
            min_value=0.001,
            max_value=0.1 if selected_algorithm == "FedProx" else 2.0,
            value=0.01 if selected_algorithm == "FedProx" else 1.0,
            step=0.005 if selected_algorithm == "FedProx" else 0.1,
            help=t("param_mu_help", lang)
        )
    if selected_algorithm == "FedDyn":
        algo_alpha = st.slider(
            t("param_alpha", lang),
            min_value=0.001,
            max_value=0.05,
            value=0.01,
            step=0.005,
            help=t("param_alpha_help", lang)
        )
    if selected_algorithm == "MOON":
        algo_temp = st.slider(
            t("param_temperature", lang),
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help=t("param_temperature_help", lang)
        )
    
    st.markdown("---")
    
    # Training Configuration
    st.markdown(f"<h4 style='font-size:0.95rem;'>{t('training_config', lang)}</h4>", unsafe_allow_html=True)
    n_rounds = st.slider(
        t("fl_rounds", lang),
        min_value=1,
        max_value=20,
        value=5,
        help=t("fl_rounds_help", lang)
    )
    
    local_epochs = st.slider(
        t("local_epochs", lang),
        min_value=1,
        max_value=5,
        value=2,
        help=t("local_epochs_help", lang)
    )
    
    learning_rate = st.select_slider(
        t("learning_rate", lang),
        options=[0.00005, 0.0001, 0.0005, 0.001, 0.005],
        value=0.0001
    )
    
    st.markdown("---")
    
    # Privacy & Security Controls
    st.markdown(f"<h4 style='font-size:0.95rem;'>{t('privacy_security_title', lang)}</h4>", unsafe_allow_html=True)
    
    privacy_noise = st.slider(
        t("dp_noise", lang),
        min_value=0.0,
        max_value=0.5,
        value=0.05,
        step=0.05,
        help=t("dp_noise_help", lang)
    )
    
    activate_defense = st.checkbox(
        t("byzantine_defense", lang),
        value=True,
        help=t("byzantine_defense_help", lang)
    )
    
    simulate_attack = st.checkbox(
        t("adversarial_simulation", lang),
        value=False,
        help=t("adversarial_simulation_help", lang)
    )


# ============================================================================
# DATA GENERATION ON PARAMETER CHANGE
# ============================================================================

current_params = (n_hospitals, samples_per_hospital)
if ('last_params' not in st.session_state or 
    st.session_state.last_params != current_params or 
    not st.session_state.hospital_data_generated):
    
    dataloaders = create_hospital_dataloaders(
        n_hospitals=n_hospitals,
        samples_per_hospital=samples_per_hospital,
        batch_size=32
    )
    st.session_state.dataloaders = dataloaders
    
    # Store sample data for visualization
    hospital_data = []
    for i, loader in enumerate(dataloaders):
        images_list, labels_list = [], []
        for batch_imgs, batch_lbls in loader:
            images_list.append(batch_imgs.numpy())
            labels_list.append(batch_lbls.numpy())
        
        all_imgs = np.concatenate(images_list, axis=0)
        all_lbls = np.concatenate(labels_list, axis=0)
        
        hospital_data.append({
            'images': all_imgs[:, 0, :, :],
            'labels': all_lbls,
            'distribution': get_distribution_info(i, n_hospitals),
            'all_labels': all_lbls
        })
    
    st.session_state.hospital_data = hospital_data
    st.session_state.global_test_set = create_global_test_set(n_samples=300, seed=9999)
    st.session_state.hospital_data_generated = True
    st.session_state.last_params = current_params


# ============================================================================
# MAIN CANVAS: HERO BANNER & METRIC TILES
# ============================================================================

st.markdown(f"""
<div class="fed-hero">
    <h1 class="fed-hero-title">{t("app_title", lang)}</h1>
    <p class="fed-hero-subtitle">{t("app_subtitle", lang)}</p>
</div>
""", unsafe_allow_html=True)

# Top Metrics Grid
total_samples = n_hospitals * samples_per_hospital
status_label = t("status_trained", lang) if st.session_state.model_trained else t("status_ready", lang)
status_badge_class = "fed-badge-benign" if st.session_state.model_trained else "fed-badge-primary"

st.markdown(f"""
<div class="fed-metrics-grid">
    <div class="fed-metric-card">
        <div class="fed-metric-label">{t("metric_hospitals", lang)}</div>
        <div class="fed-metric-value">{n_hospitals}</div>
    </div>
    <div class="fed-metric-card">
        <div class="fed-metric-label">{t("metric_rounds", lang)}</div>
        <div class="fed-metric-value">{n_rounds}</div>
    </div>
    <div class="fed-metric-card">
        <div class="fed-metric-label">{t("metric_samples", lang)}</div>
        <div class="fed-metric-value">{total_samples:,}</div>
    </div>
    <div class="fed-metric-card">
        <div class="fed-metric-label">{t("metric_status", lang)}</div>
        <div class="fed-metric-value" style="font-size: 1.15rem; margin-top: 0.25rem;">
            <span class="fed-badge {status_badge_class}">{status_label}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# APPLICATION WORKSPACE TABS
# ============================================================================

tabs = st.tabs([
    t("tab_network", lang),
    t("tab_training", lang),
    t("tab_diagnosis", lang)
])


# ============================================================================
# TAB 1: HOSPITAL NETWORK & NON-IID DATA PROFILE
# ============================================================================

with tabs[0]:
    st.markdown(f"""
    <div class="fed-section-card">
        <div class="fed-section-title">🏥 {t("network_overview_title", lang)}</div>
        <div class="fed-section-desc">{t("network_overview_desc", lang)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hospital Selector
    h_col1, h_col2 = st.columns([1, 2.5])
    with h_col1:
        hospital_options = [f"{t('hospital_label', lang)} {i+1}" for i in range(n_hospitals)]
        selected_hospital_idx = st.selectbox(
            f"🔍 {t('hospital_label', lang)}",
            options=range(n_hospitals),
            format_func=lambda i: hospital_options[i]
        )
    
    hospital_data = st.session_state.hospital_data[selected_hospital_idx]
    dist_info = hospital_data['distribution']
    all_labels = hospital_data['all_labels']
    
    # Hospital Profile Badge
    profile_keys = ["profile_healthy", "profile_pneumonia", "profile_covid", "profile_balanced"]
    profile_desc = t(profile_keys[selected_hospital_idx % len(profile_keys)], lang)
    
    with h_col2:
        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid var(--clr-border); border-radius: var(--radius-md); padding: 0.85rem 1.25rem; margin-top: 1.6rem;">
            <span style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Klinik Profil / Clinic Specialty</span><br>
            <strong style="color: #0F172A; font-size: 1rem;">{profile_desc}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    
    col_chart, col_preview = st.columns([1, 1.4])
    
    with col_chart:
        # Donut Chart for Prevalence Distribution
        disease_names = [
            t("disease_normal", lang),
            t("disease_pneumonia", lang),
            t("disease_covid", lang)
        ]
        counts = [
            np.sum(all_labels == 0),
            np.sum(all_labels == 1),
            np.sum(all_labels == 2)
        ]
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=disease_names,
            values=counts,
            hole=0.55,
            marker=dict(
                colors=['#10B981', '#F59E0B', '#EF4444'],
                line=dict(color='#FFFFFF', width=2)
            ),
            textinfo='percent',
            textposition='inside',
            hoverinfo='label+value+percent'
        )])
        
        fig_donut.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            font=dict(family="Inter, sans-serif", color='#0F172A')
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col_preview:
        st.markdown(f"**📷 {t('disease_normal', lang)} / {t('disease_pneumonia', lang)} / {t('disease_covid', lang)} X-Ray Preview:**")
        sample_imgs = hospital_data['images'][:6]
        sample_lbls = hospital_data['labels'][:6]
        
        fig_grid = make_subplots(
            rows=2, cols=3,
            subplot_titles=[disease_names[lbl] for lbl in sample_lbls],
            vertical_spacing=0.15,
            horizontal_spacing=0.08
        )
        
        for idx in range(min(6, len(sample_imgs))):
            r = idx // 3 + 1
            c = idx % 3 + 1
            fig_grid.add_trace(
                go.Heatmap(z=sample_imgs[idx], colorscale='gray', showscale=False),
                row=r, col=c
            )
            fig_grid.update_xaxes(showticklabels=False, showgrid=False, row=r, col=c)
            fig_grid.update_yaxes(showticklabels=False, showgrid=False, row=r, col=c)
        
        fig_grid.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_grid, use_container_width=True)


# ============================================================================
# TAB 2: FEDERATED TRAINING STUDIO
# ============================================================================

with tabs[1]:
    math_key = f"algo_math_{selected_algorithm.lower()}"
    math_formula = t(math_key, lang)
    
    st.markdown(f"""
    <div class="fed-section-card">
        <div class="fed-section-title">🔄 {t("training_studio_title", lang)}</div>
        <div class="fed-section-desc">
            {t("active_algorithm", lang)}: <strong>{selected_algorithm}</strong> &bull; 
            <em>{math_formula}</em>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_btn, col_alert = st.columns([1, 2])
    with col_btn:
        start_fl_btn = st.button(
            t("btn_start_training", lang),
            key="btn_run_fl",
            disabled=not st.session_state.hospital_data_generated
        )
    
    with col_alert:
        if simulate_attack:
            st.markdown(f'<span class="fed-badge fed-badge-warning">⚠️ {t("adversarial_simulation", lang)}</span>', unsafe_allow_html=True)
        if activate_defense:
            st.markdown(f'<span class="fed-badge fed-badge-benign">🛡️ {t("security_shield_active", lang)}</span>', unsafe_allow_html=True)

    progress_box = st.empty()
    chart_box = st.empty()
    
    if start_fl_btn and st.session_state.hospital_data_generated:
        st.session_state.training_history = {
            'loss': [], 'accuracy': [], 'round': [],
            'precision': [], 'recall': [], 'f1_score': [],
            'test_accuracy': [], 'test_loss': [],
            'blocked_count': 0
        }
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        server = CentralServer(device=device, privacy_noise=privacy_noise, defense_mode=activate_defense)
        
        clients = []
        for idx in range(n_hospitals):
            is_mal = (simulate_attack and idx == 2)
            clients.append(HospitalClient(
                client_id=idx,
                dataloader=st.session_state.dataloaders[idx],
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                malicious=is_mal
            ))
            
        test_images, test_labels = st.session_state.global_test_set
        
        for round_idx in range(1, n_rounds + 1):
            progress_box.info(f"⏳ {t('status_training', lang)}: {round_idx} / {n_rounds} [{selected_algorithm}]")
            
            agg_m, cl_m, test_m, sec_rep = run_federated_round(
                server=server,
                clients=clients,
                round_num=round_idx,
                test_images=test_images,
                test_labels=test_labels,
                use_defense=activate_defense,
                algorithm=selected_algorithm,
                mu=algo_mu,
                alpha=algo_alpha,
                temperature=algo_temp
            )
            
            # Record metrics
            st.session_state.training_history['round'].append(round_idx)
            st.session_state.training_history['loss'].append(agg_m['loss'])
            st.session_state.training_history['accuracy'].append(agg_m['accuracy'] * 100)
            
            if test_m:
                st.session_state.training_history['test_accuracy'].append(test_m.accuracy * 100)
                st.session_state.training_history['test_loss'].append(test_m.loss)
                st.session_state.training_history['precision'].append(test_m.precision * 100)
                st.session_state.training_history['recall'].append(test_m.recall * 100)
                st.session_state.training_history['f1_score'].append(test_m.f1_score * 100)
                st.session_state.confusion_matrix = test_m.confusion_matrix
            
            # Live Convergence Chart
            fig_live = make_subplots(
                rows=1, cols=2,
                subplot_titles=[t("chart_accuracy_title", lang), t("chart_loss_title", lang)]
            )
            
            # Accuracy trace
            fig_live.add_trace(
                go.Scatter(
                    x=st.session_state.training_history['round'],
                    y=st.session_state.training_history['test_accuracy'],
                    mode='lines+markers',
                    name=f'{selected_algorithm} Accuracy',
                    line=dict(color='#10B981', width=3),
                    marker=dict(size=7, color='#047857')
                ),
                row=1, col=1
            )
            
            # Loss trace
            fig_live.add_trace(
                go.Scatter(
                    x=st.session_state.training_history['round'],
                    y=st.session_state.training_history['loss'],
                    mode='lines+markers',
                    name=f'{selected_algorithm} Loss',
                    line=dict(color='#06B6D4', width=3),
                    marker=dict(size=7, color='#0891B2')
                ),
                row=1, col=2
            )
            
            fig_live.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", color="#0F172A")
            )
            fig_live.update_xaxes(showgrid=True, gridcolor='#E2E8F0')
            fig_live.update_yaxes(showgrid=True, gridcolor='#E2E8F0')
            
            chart_box.plotly_chart(fig_live, use_container_width=True)
            time.sleep(0.15)
            
        st.session_state.trained_weights = server.get_global_weights()
        st.session_state.model_trained = True
        progress_box.success(f"✅ {t('training_complete_title', lang)} ({n_rounds} Rounds - {selected_algorithm})")
        st.rerun()


# ============================================================================
# TAB 3: CLINICAL DECISION SUPPORT & XAI STUDIO
# ============================================================================

with tabs[2]:
    st.markdown(f"""
    <div class="fed-section-card">
        <div class="fed-section-title">🩺 {t("cdss_studio_title", lang)}</div>
        <div class="fed-section-desc">{t("cdss_studio_desc", lang)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        patient_choice = st.selectbox(
            t("patient_selection", lang),
            options=[0, 1, 2],
            format_func=lambda c: [
                t("patient_option_normal", lang),
                t("patient_option_pneumonia", lang),
                t("patient_option_covid", lang)
            ][c]
        )
    with col_p2:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        diagnose_btn = st.button(t("btn_diagnose", lang), key="btn_run_diag")
    
    if diagnose_btn:
        generator = MedicalDataGenerator(seed=int(time.time()))
        image = generator.generate_synthetic_xray(patient_choice, apply_augmentation=True)
        
        # Load Global Model
        eval_model = create_model().cpu()
        if st.session_state.trained_weights is not None:
            eval_model.load_state_dict(st.session_state.trained_weights)
        eval_model.eval()
        
        img_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
        # Grad-CAM Visual Heatmap
        torch.set_grad_enabled(True)
        gradcam = GradCAM(eval_model)
        heatmap, predicted_class, confidence = gradcam.generate_heatmap(img_tensor)
        confidence = confidence * 100
        gradcam.remove_hooks()
        
        with torch.no_grad():
            logits = eval_model(img_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
            
        overlay = create_overlay(image, heatmap, alpha=0.5)
        
        st.session_state.scan_image = image
        st.session_state.scan_true_label = patient_choice
        st.session_state.scan_heatmap = heatmap
        st.session_state.scan_overlay = overlay
        st.session_state.scan_probs = probs
        st.session_state.scan_predicted = predicted_class
        st.session_state.scan_confidence = confidence

    # Display Clinical Findings if scan is available
    if hasattr(st.session_state, 'scan_image'):
        image = st.session_state.scan_image
        heatmap = st.session_state.scan_heatmap
        probs = st.session_state.scan_probs
        pred_idx = st.session_state.scan_predicted
        confidence = st.session_state.scan_confidence
        
        disease_labels = [
            t("disease_normal", lang),
            t("disease_pneumonia", lang),
            t("disease_covid", lang)
        ]
        
        pred_name = disease_labels[pred_idx]
        badge_class = ["fed-badge-benign", "fed-badge-warning", "fed-badge-malignant"][pred_idx]
        
        st.markdown("---")
        
        # Grad-CAM Viewer Controls
        alpha_slider = st.slider(t("gradcam_slider_alpha", lang), min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        dynamic_overlay = create_overlay(image, heatmap, alpha=alpha_slider)
        
        # 3-Column Visual Panel
        c_raw, c_heat, c_over = st.columns(3)
        with c_raw:
            st.markdown(f"**📷 {t('original_scan', lang)}**")
            fig_raw = go.Figure(go.Heatmap(z=image, colorscale='gray', showscale=False))
            fig_raw.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig_raw, use_container_width=True)
            
        with c_heat:
            st.markdown(f"**🔥 {t('ai_attention_map', lang)}**")
            fig_heat = go.Figure(go.Heatmap(z=heatmap, colorscale='Hot', showscale=False))
            fig_heat.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with c_over:
            st.markdown(f"**🎯 {t('blended_overlay', lang)}**")
            fig_over = go.Figure(go.Image(z=(dynamic_overlay * 255).astype(np.uint8)))
            fig_over.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=5), xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig_over, use_container_width=True)
            
        # Clinical Summary & Probability Bar
        s_col1, s_col2 = st.columns([1, 1])
        with s_col1:
            st.markdown(f"""
            <div class="fed-section-card">
                <div style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">{t("diagnostic_result_header", lang)}</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A; margin: 0.4rem 0;">
                    <span class="fed-badge {badge_class}" style="font-size: 1rem; padding: 0.4rem 1rem;">{pred_name}</span>
                </div>
                <p style="color: #64748B; font-size: 0.95rem; margin: 0;">
                    Güven Oranı / Confidence: <strong style="color: #0F172A;">%{confidence:.1f}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Dynamic Explanation
            explanation_md = get_explanation_text(pred_idx, confidence / 100, lang=lang)
            st.markdown(explanation_md)
            
        with s_col2:
            fig_bar = go.Figure(go.Bar(
                x=probs * 100,
                y=disease_labels,
                orientation='h',
                marker=dict(
                    color=['#10B981', '#F59E0B', '#EF4444'],
                    line=dict(color='#FFFFFF', width=1.5)
                ),
                text=[f"%{p*100:.1f}" if lang == "tr" else f"{p*100:.1f}%" for p in probs],
                textposition='auto',
                textfont=dict(family="Inter, sans-serif", color="white", size=12)
            ))
            fig_bar.update_layout(
                height=220,
                margin=dict(l=20, r=20, t=10, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=[0, 100], showgrid=True, gridcolor='#E2E8F0'),
                yaxis=dict(showgrid=False),
                font=dict(family="Inter, sans-serif", color='#0F172A')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # Case-Based RAG Historical Digital Twins
        st.markdown("---")
        st.markdown(f"### 🧬 {t('similar_cases_title', lang)}")
        st.caption(t('similar_cases_desc', lang))
        
        if st.session_state.case_bank is None:
            st.session_state.case_bank = HistoricalCaseBank(n_cases=100)
            
        eval_model = create_model().cpu()
        if st.session_state.trained_weights is not None:
            eval_model.load_state_dict(st.session_state.trained_weights)
        eval_model.eval()
        
        img_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        query_emb = extract_embedding(eval_model, img_tensor)
        similar_cases = st.session_state.case_bank.find_similar(query_emb, top_k=2)
        st.session_state.similar_cases = similar_cases
        
        sim_col1, sim_col2 = st.columns(2)
        for idx, case in enumerate(similar_cases):
            c_box = sim_col1 if idx == 0 else sim_col2
            lbl_name = disease_labels[case['label']]
            sim_badge = ["fed-badge-benign", "fed-badge-warning", "fed-badge-malignant"][case['label']]
            
            with c_box:
                st.markdown(f"""
                <div class="fed-section-card" style="padding: 1rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #0F172A; font-size: 0.95rem;">{t('case_id', lang)}: {case['case_id']}</strong><br>
                        <span class="fed-badge {sim_badge}" style="margin-top: 0.3rem;">{lbl_name}</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: rgba(6, 182, 212, 0.12); color: #0E7490; padding: 0.3rem 0.6rem; border-radius: 8px; font-weight: 700; font-size: 0.85rem;">
                            %{case['similarity']*100:.1f} {t('similarity_score', lang)}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Audio & PDF Report Actions
        st.markdown("---")
        act_col1, act_col2 = st.columns(2)
        
        with act_col1:
            st.markdown(f"#### {t('voice_summary_title', lang)}")
            audio_bytes = get_or_create_audio(pred_name, confidence, language=lang)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')
                
        with act_col2:
            st.markdown(f"#### 📋 {t('btn_download_pdf', lang)}")
            patient_id = f"PAT-{int(time.time()) % 100000}"
            pdf_exp = get_diagnosis_explanation(pred_name, confidence, lang=lang)
            
            pdf_data = generate_medical_report(
                patient_id=patient_id,
                diagnosis=pred_name,
                confidence=confidence,
                explanation=pdf_exp,
                heatmap_image=heatmap,
                original_image=image,
                similar_cases=similar_cases,
                lang=lang
            )
            
            st.download_button(
                label=t("btn_download_pdf", lang),
                data=pdf_data,
                file_name=f"{t('pdf_filename', lang)}_{patient_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1.25rem; background: #FFFFFF; border: 1px solid var(--clr-border); border-radius: var(--radius-md); margin-top: 1.5rem;">
    <p style="color: #64748B !important; font-size: 0.85rem; margin: 0 0 0.25rem 0;">
        🔒 <strong>Fed-XRay Privacy Invariant:</strong> Patient medical scans remain strictly isolated within local hospital firewalls. Only aggregated model weights are synchronized.
    </p>
    <p style="font-size: 0.75rem; color: #94A3B8 !important; margin: 0;">
        Fed-XRay Federated Medical AI & CDSS Platform | Version 2.0.0
    </p>
</div>
""", unsafe_allow_html=True)
