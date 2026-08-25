# Fed-XRay: Federated Vision Transformers & Medical Oncology AI Platform

**Next-Generation Privacy-Preserving Vision Transformers, Medical Foundation Model Adaptation, Bilingual CDSS & Cryptographic Federated Intelligence**

<div align="center">
<a href="https://yusuf-cancerfedxlearning.streamlit.app/">
<img src="https://img.shields.io/badge/Live_Demo-Access_Platform-emerald?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
</a>
</div>

Fed-XRay is a clinical-grade Federated Learning (FL) and Clinical Decision Support System (CDSS) framework for medical, radiological, and cancer imaging. The platform enables multi-institutional collaborative intelligence across distributed clinical centers without raw patient data ever leaving local hospital firewalls.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Next.js-14.2+-black?logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Privacy-Option_J_(CKKS_+_DP)-indigo" alt="Privacy">
  <img src="https://img.shields.io/badge/Localization-EN%20%7C%20TR-blue" alt="Bilingual">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Core Platform Capabilities

### 1. Federated Vision Transformers & PEFT Adaptation
- **Eliminating Bilinear LoRA Error:** Implements **FFA-LoRA** (Federated Freeze-A LoRA) with globally frozen matrix $A$ to eliminate aggregation discordance ($\bar{B}A = \sum p_k B_k A$) and enable multiplication-free CKKS Homomorphic Encryption.
- **FedSA-LoRA & Fed-ALAS:** Federated Share-A LoRA capturing the global subspace in $A$ while preserving private local personalization in $B_k$.
- **FedPerfix Deep Attention Adaptation:** Selective low-rank adaptation focused on deep Multi-Head Self-Attention layers ($L-3 \dots L$), reducing parameter communication payload by **>99.8%**.

### 2. Drift-Resilient Optimizer Suite
- **Consensus & Dynamic Alignment:** FedAvg, FedProx ($\mu$-regularization), FedDyn (dynamic objective tracking with server state $h^{t+1}$), FedOpt (FedAdam adaptive server momentum), SCAFFOLD (control variates $c_k, c$), and MOON (model-contrastive representation alignment).

### 3. Medical Foundation Model Adapters
- **Digital Histopathology:** UNI (ViT-L/16, 200M tiles), CONCH, and Virchow2 feature extraction.
- **Volumetric 3D Radiology:** Swin UNETR 3D representation backbones.
- **Interactive Segmentation:** MedSAM Client-Tailored Adapter (FCA) with bottleneck residual scaling.
- **Vision-Language Radiology:** BioViL-T / MedCLIP + **FedMedCLIP** Feature Adaptation Module (FAM) with mutual KL divergence distillation:
  $$\mathcal{L}_{\text{local}} = \mathcal{L}_{\text{CE}}(\hat{y}_{\text{ens}}, y) + \beta \mathcal{D}_{\text{KL}}(p_{\text{FAM}}(x) \,||\, p_{\text{MLP}}(x))$$

### 4. Foundation-Anchored Prototype Metric Alignment & Dynamic Imbalance Control
- **Dispersion-Weighted Prototype Synthesis:** Aggregation weighted by intra-class covariance traces: $\alpha_{k,c} \propto \frac{n_{k,c}}{\text{Tr}(\Sigma_{k,c}) + \epsilon}$.
- **Multimodal Semantic Prototypes:** Fusion with invariant text encoder codebooks ($p_c = \lambda p_c^{\text{img}} + (1-\lambda) p_c^{\text{txt}}$).
- **Imbalance Loss Suite:** Dynamic Adaptive Focal Loss (DAFL), Bayesian Balanced Softmax (BSM), Class-Balanced Loss ($\mathcal{L}_{\text{CB}}$), Label-Distribution-Aware Margin (LDAM), and Missing-Class Repel Loss ($\mathcal{L}_{\text{repel}}$).

### 5. Option J Dual-Layer Cryptographic & Differential Privacy Architecture
- **Layer 1 (In-Transit / Aggregation Privacy):** Leveled CKKS Threshold Homomorphic Encryption (RLWE 128-bit) and Cryptographic Secure Aggregation (SecAgg+), ensuring the central server observes zero plaintext parameter updates.
- **Layer 2 (Output Model Privacy):** Strict Patient-Level Gaussian Differential Privacy calibrated via Rényi Differential Privacy (RDP) composition accounting ($(\epsilon \le 2.0, \delta \le 10^{-5})$).

### 6. Strategy E Real-World Multi-Center Dataset Migration
- **Primary Benchmark Ecosystem:** ISIC 2019 (25,331 dermoscopy images, 3 native sites: BCN_20000, ViDIR/Vienna, Queensland), NCT-CRC-HE-100K + CRC-VAL-7K (Colorectal histopathology hold-out), and MIMIC-CXR-JPG (v2.1.0).
- **Strict Leak-Free Invariant:** Partitioning strictly by `patient_id` ensuring $\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$.
- **Seven Controlled Federated Imbalance Scenarios (A through G):** Ranging from uniform Dirichlet ($\alpha=100.0$) and missing pathological classes to Pareto long-tailed quantity skew.

### 7. Clinical Decision Support System (CDSS) & Evidence-Grounded Federated RAG
- **Explainable AI (Grad-CAM):** Multi-layer class activation mapping supporting both CNNs and Vision Transformers.
- **Evidence-Grounded Retrieval (Federated RAG):** Top-K digital twin matching with temperature-scaled outcome probability estimation:
  $$\hat{y}_{\text{RAG}} = \sum_{k=1}^K \text{softmax}\left(\frac{z_{\text{query}}^\top z_k}{\tau_r}\right) y_k$$
- **Medical PDF Report Engine & Voice Briefing:** Automated single-page A4 diagnostic reports and real-time synthesized voice dictation.

---

## Repository Architecture

```text
Fed-XRay/
├── api/                           # FastAPI Async Backend Application
│   ├── main.py                    # REST & SSE Streaming Endpoints
│   └── schemas.py                 # Pydantic v2 Request & Response Models
├── frontend/                      # Decoupled Next.js 14 Clinical SaaS Web App
│   ├── src/app/                   # App Router & Layout
│   └── src/components/            # Responsive Clinical Cockpit Panels
├── src/
│   └── fed_xray/                  # Core Python Package
│       ├── core/                  # Client, Server, Privacy, Prototypes, Losses
│       │   ├── client.py          # HospitalClient (PEFT, FedDyn, SCAFFOLD, MOON)
│       │   ├── server.py          # CentralServer (FedAdam, FedDyn, SecAgg, Byzantine)
│       │   ├── privacy.py         # Option J: CKKS HE, SecAgg+, Rényi DP Accountant
│       │   ├── prototypes.py      # FedProto Dispersion-Weighted Synthesis
│       │   ├── imbalance_losses.py # DAFL, Balanced Softmax, LDAM, Repel
│       │   └── metrics.py         # Metric Trackers & Security Reports
│       ├── models/                # ViT, Foundation Models, PEFT & CNN
│       │   ├── vit.py             # Vision Transformer (MHSA, PatchEmbedding)
│       │   ├── peft.py            # LoRA, FFA-LoRA, FedSA-LoRA, FedPerfix
│       │   ├── foundation.py      # FedMedCLIP, MedSAM FCA, TextSemanticAnchor
│       │   └── cnn.py             # XRayClassifier Baseline
│       ├── data/                  # Strategy E Ingestion & Scenarios
│       │   ├── real_world.py      # Scenarios A-G, SHA-256 Deduplication, Leak-Free Split
│       │   └── generator.py       # Multi-Hospital Simulation Generator
│       └── cdss/                  # Clinical Decision Support Systems
│           ├── xai.py             # Grad-CAM Heatmaps & Overlays
│           ├── similarity.py      # Federated RAG Digital Twin Search
│           ├── report.py          # Medical PDF Report Generator
│           ├── voice.py           # Speech Dictation Engine
│           └── i18n.py            # Academic Bilingual Localization (EN / TR)
├── tests/                         # Comprehensive Automated Test Suites
│   ├── test_all.py                # Baseline Model, Data, XAI & Security Tests
│   ├── test_vit_peft_optimizers.py # ViT, FFA-LoRA Zero Error & SOTA Optimizers
│   ├── test_foundation_privacy.py # FedMedCLIP, CKKS HE & Rényi DP Accountant
│   ├── test_real_world_data_scenarios.py # Scenarios A-G, SHA-256 & Federated RAG
│   ├── test_prototypes.py         # FedProto & Imbalance Losses
│   └── test_api_endpoints.py      # FastAPI REST & Streaming Integration Tests
├── research/                      # Canonical Scientific Research Anchors
├── AGENTS.md                      # Canonical System Guidelines & Operational Rules
└── requirements.txt               # Production Dependencies
```

---

## Quick Start & Verification

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/yusufcalisir/Fed-XRay.git
cd Fed-XRay

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Comprehensive Automated Test Suite
```bash
# Execute all 36 unit and integration test suites
python -m pytest tests/
```

### 3. Launch Services

#### Launch FastAPI Backend:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Launch Next.js SaaS Web Dashboard:
```bash
cd frontend
npm install
npm run dev
```

---

## Communication & Memory Economics

| Method | Transmitted Payload / Round | Reduction vs Full ViT | Client VRAM |
| :--- | :--- | :--- | :--- |
| **Full ViT-L/14 FedAvg (Scratch)** | $2,430.00\text{ MB}$ | $0.0\%$ (Baseline) | $>24\text{ GB}$ |
| **Full Swin UNETR 3D FedAvg** | $248.00\text{ MB}$ | $89.8\%$ | $>22\text{ GB}$ |
| **Standard ViT-LoRA ($r=16$)** | $4.72\text{ MB}$ | $99.80\%$ | $6.2\text{ GB}$ |
| **FFA-LoRA (Frozen $A$, Send $B$)** | $\mathbf{2.36\text{ MB}}$ | $\mathbf{99.90\%}$ | $\mathbf{5.8\text{ GB}}$ |
| **FedMedCLIP (FAM + Distill)** | $\mathbf{1.60\text{ MB}}$ | $\mathbf{99.93\%}$ | $\mathbf{4.5\text{ GB}}$ |
| **FedProto (Centroids in $\mathbb{R}^{1024}$)** | $\mathbf{0.024\text{ MB}}$ | $\mathbf{99.999\%}$ | $\mathbf{4.0\text{ GB}}$ |
| **Hybrid FedSA-LoRA + FedProto** | $\mathbf{2.38\text{ MB}}$ | $\mathbf{99.90\%}$ | $\mathbf{5.8\text{ GB}}$ |

---

## Scientific Anchors & Citations

1. **Hu et al.** (ICLR 2022) — *LoRA: Low-Rank Adaptation of Large Language Models*.
2. **Tan et al.** (AAAI 2022) — *FedProto: Federated Prototype Learning across Heterogeneous Clients*.
3. **Acar et al.** (ICLR 2021) — *Federated Learning Based on Dynamic Regularization (FedDyn)*.
4. **Li et al.** (CVPR 2021) — *Model-Contrastive Federated Learning (MOON)*.
5. **Karimireddy et al.** (ICML 2020) — *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*.
6. **Reddi et al.** (ICLR 2021) — *Adaptive Federated Optimization (FedOpt / FedAdam)*.
7. **Chen et al.** (Nature Medicine 2024) — *UNI: General-purpose Self-Supervised Vision Transformer for Pathology*.
8. **Mironov** (CSF 2017) — *Rényi Differential Privacy*.
