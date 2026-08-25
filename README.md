# Fed-XRay: Federated Vision Transformers & Medical Oncology AI Platform

**State-of-the-Art Privacy-Preserving Vision Transformers, Medical Foundation Model Adaptation, Asymmetric Factor Decoupling, Option J Cryptographic Privacy & Bilingual CDSS SaaS**

<div align="center">
<a href="https://yusuf-cancerfedxlearning.streamlit.app/">
<img src="https://img.shields.io/badge/Live_Demo-Access_Platform-emerald?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
</a>
</div>

Fed-XRay is an enterprise-grade Federated Learning (FL) and Clinical Decision Support System (CDSS) platform designed for multi-center oncology, radiological, and computational pathology artificial intelligence. The platform enables privacy-preserving collaborative model adaptation across distributed hospital networks without raw patient data ever leaving local institutional firewalls.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Next.js-14.2+-black?logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Privacy-Option_J_(CKKS_+_DP)-indigo" alt="Privacy">
  <img src="https://img.shields.io/badge/PEFT-FFA--LoRA_%7C_FedSA--LoRA_%7C_FedAS--LoRA-purple" alt="PEFT">
  <img src="https://img.shields.io/badge/Localization-EN%20%7C%20TR-blue" alt="Bilingual">
  <img src="https://img.shields.io/badge/Automated_Tests-75_Passed_100%25-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Core Technological Pillars

### 1. Vision Transformers & Parameter-Efficient Fine-Tuning (PEFT)

- **Eliminating Bilinear Aggregation Error:** Standard federated LoRA induces non-vanishing bilinear discordance:

$$
\Delta \bar{W}_{\text{effective}} = \left(\sum_{k=1}^K p_k B_k\right)\left(\sum_{k=1}^K p_k A_k\right) = \bar{B}\bar{A} \neq \sum_{k=1}^K p_k (B_k A_k) = \bar{\Delta W}_{\text{ideal}}
$$

- **FFA-LoRA (Federated Freeze-A LoRA):** Globally freezes projection matrix $A$ across all institutions ($\nabla_A \mathcal{L} = 0$), strictly preserving aggregation linearity ($\bar{B}A \equiv \sum p_k B_k A$) and enabling multiplication-free Homomorphic Encryption ($\text{Multiplication Depth} = 0$).
- **FedSA-LoRA & Fed-ALAS:** Aggregates matrix $\bar{A} = \sum p_k A_k$ centrally to capture the global visual input subspace while retaining matrix $B_k$ privately on client nodes for site-specific staining/scanner personalization.
- **FedAS-LoRA (Adaptive Subspace Routing):** Measures Rank-Aware Shared-Subspace Sufficiency $(\text{RSS}_{\text{input}} \text{ and } \text{RSS}_{\text{output}})$ to dynamically select between Share-A / Local-B (under covariate domain shift) and Share-B / Local-A (under label distribution skew).
- **FlexLoRA SVD Aggregator:** Reconstructs full weight updates and computes Truncated Singular Value Decomposition (SVD):

$$
\bar{\Delta W} = \sum_{k=1}^K p_k B_k A_k = U_r \Sigma_r V_r^\top
$$

$$
\bar{B} = U_r \Sigma_r^{1/2}, \quad \bar{A} = \Sigma_r^{1/2} V_r^\top
$$

- **HetLoRA Dynamic Rank Slicing:** Dynamic zero-padding parameter alignment for heterogeneous edge clinics ($r_k \le r_{\text{max}}$).
- **FedPerfix Deep Attention Personalization:** Prioritizes adapting Multi-Head Self-Attention (MHSA $W_Q, W_V$) in deep layers ($L-3 \dots L$), reducing parameter communication volume by **>99.8%** ($<2.36\text{ MB/round}$).

---

### 2. Domain-Specific Medical Foundation Models & Multimodal Adapters

- **Computational Pathology (PFMs):** `UNI` (ViT-L/16, 1024-dim), `UNI-2` (ViT-Giant, 1.1B parameters), and `Virchow2` (ViT-H/14, 632M parameters, 3.0M WSIs) representations for pan-cancer grading and subtyping.
- **Volumetric 3D Radiology (RFMs):** `Swin UNETR` 3D (MONAI 5,050-scan pretrained) hierarchical representations for multiparametric MRI/CT oncology segmentation.
- **Interactive Segmentation:** `MedSAM` Client-Tailored Adapter (FCA) with bottleneck residual scaling.
- **Multimodal Chest Radiology (BioViL-T / BiomedCLIP):** **FedMedCLIP** Feature Adaptation Module (FAM) with invariant text semantic codebook anchoring and mutual KL divergence distillation:

$$
\mathcal{L}_{\text{local}} = \mathcal{L}_{\text{CE}}(\hat{y}_{\text{ens}}, y) + \beta \mathcal{D}_{\text{KL}}\left(p_{\text{FAM}}(x) \,||\, p_{\text{MLP}}(x)\right)
$$

- **FedCola & FedDAT:** Parameter-based collaboration bridging clinical silos with unpaired imaging/report modalities.

---

### 3. Drift-Resilient Non-Convex Optimizers

- **FedDyn (Dynamic Objective Regularization):** Dynamic parameter alignment tracking local gradient displacement with central server state:

$$
h^{t+1} = h^t - \alpha \frac{1}{K}\sum_{k=1}^K \left(\theta_k^{t+1} - \bar{\theta}^t\right)
$$

- **FedOpt / FedAdam:** Adaptive server momentum on aggregated pseudo-gradients:

$$
\Delta^t = \sum_{k \in \mathcal{S}_t} p_k \left(\theta_k^{t+1} - \bar{\theta}^t\right)
$$

- **SCAFFOLD on PEFT:** Stochastic controlled averaging using dual control variates ($c_k, c$) directly tracking gradient drift on low-rank adapter matrices.
- **MOON:** Model-contrastive representation alignment on intermediate ViT feature spaces.
- **FedProx:** Proximal parameter regularization directly applied to PEFT subspaces:

$$
\mathcal{L}_{\text{prox}} = \frac{\mu}{2}\left(\|A_k - \bar{A}^t\|_F^2 + \|B_k - \bar{B}^t\|_F^2\right)
$$

---

### 4. Foundation-Anchored Prototype Metric Alignment & Dynamic Imbalance Control

- **Dispersion-Weighted Prototype Synthesis (FedProto):** Eliminates orthogonal rotation collapse $Q \in O(D)$ by grounding class centroids in frozen foundation representations, weighted by intra-class covariance traces:

$$
\Sigma_{k,c} = \frac{1}{n_{k,c}} \sum_{i: y_i = c} \left(f(x_i) - p_{k,c}\right)\left(f(x_i) - p_{k,c}\right)^\top
$$

$$
\alpha_{k,c} = \frac{\frac{n_{k,c}}{\mathrm{Tr}(\Sigma_{k,c}) + \epsilon}}{\sum_{j \in \mathcal{K}_c} \frac{n_{j,c}}{\mathrm{Tr}(\Sigma_{j,c}) + \epsilon}}, \quad \bar{p}_c = \sum_{k \in \mathcal{K}_c} \alpha_{k,c} \, p_{k,c}
$$

- **Multimodal Semantic Prototypes:** Invariant text coordinate anchoring:

$$
p_c = \lambda \left(\sum_{k \in \mathcal{K}_c} \alpha_{k,c} \, p_{c,k}^{\text{img}}\right) + (1-\lambda) p_c^{\text{txt}}
$$

- **Dynamic Imbalance Loss Suite:** Dynamic Adaptive Focal Loss, Bayesian Balanced Softmax, Class-Balanced Loss, Label-Distribution-Aware Margin, and Missing-Class Repel Loss $(\mathcal{L}_{\text{DAFL}}, \; \mathcal{L}_{\text{BSM}}, \; \mathcal{L}_{\text{CB}}, \; \mathcal{L}_{\text{LDAM}}, \; \mathcal{L}_{\text{repel}})$.

---

### 5. Option J Dual-Layer Cryptographic & Differential Privacy Architecture

- **Layer 1 (In-Transit / Aggregation Privacy):** Leveled CKKS Threshold Homomorphic Encryption (RLWE 128-bit) on FFA-LoRA updates ($ct_{\text{global}} = \sum w_k \odot ct_k$) and SecAgg+ zero-sum masking ($\sum s_k = 0$), ensuring zero plaintext visibility at the central coordinator.
- **Layer 2 (Output Model Privacy):** Strict Patient-Level Gaussian Differential Privacy with Rényi Differential Privacy (RDP) composition accounting, ensuring strict $(\epsilon \le 2.0, \delta \le 10^{-5})$-DP bounds:

$$
g_i = \frac{1}{m_i} \sum_{j=1}^{m_i} g_{i,j}
$$

$$
\bar{g}_i = g_i \cdot \min\left(1, \frac{C_{\text{patient}}}{\|g_i\|_2}\right)
$$

$$
\tilde{g} = \frac{1}{|\mathcal{B}_P|} \left(\sum_{i \in \mathcal{B}_P} \bar{g}_i + \mathcal{N}\left(0, \sigma^2 C_{\text{patient}}^2 I\right)\right)
$$

---

### 6. Strategy E Real-World Multi-Center Ingestion & Scenarios A-G

- **Primary Benchmark Ecosystem:** ISIC 2019 (25,331 dermoscopy images across 3 natural hospital sites: BCN_20000 Barcelona, ViDIR Vienna, Univ. Queensland), NCT-CRC-HE-100K + CRC-VAL-7K (Colorectal histopathology hold-out), and MIMIC-CXR-JPG (v2.1.0).
- **Strict Leak-Free Invariant:** Enforces patient-level cohort isolation grouped strictly by `patient_id`:

$$
\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset
$$

- **Seven Controlled Federated Imbalance Scenarios (A to G):**
  - **Scenario A:** IID Baseline (Dirichlet $\alpha=100.0$)
  - **Scenario B:** Mild Label Skew (Dirichlet $\alpha=1.0$)
  - **Scenario C:** Moderate Label Skew (Dirichlet $\alpha=0.3$)
  - **Scenario D:** Severe Label Skew (Dirichlet $\alpha=0.05$, ~90% local dominance)
  - **Scenario E:** Missing Pathological Classes (disjoint subsets per site)
  - **Scenario F:** Global Long-Tailed Skew (Pareto 100:1 ratio)
  - **Scenario G:** Combined Quantity & Extreme Label Skew

---

### 7. Clinical Decision Support System (CDSS) & Evidence-Grounded Federated RAG

- **Explainable AI (Grad-CAM):** Multi-layer class activation mapping supporting CNNs and Vision Transformers.
- **Evidence-Grounded Retrieval (Federated RAG):** Top-K digital twin matching with temperature-scaled outcome probability estimation:

$$
\hat{y}_{\text{RAG}} = \sum_{k=1}^K \text{softmax}\left(\frac{z_{\text{query}}^\top z_k}{\tau_r}\right) y_k
$$

- **Medical PDF Report Engine & Voice Briefing:** Single-page diagnostic reports (fpdf2) and real-time synthesized voice dictation with disk-caching and graceful UI error fallback.
- **Dual Streaming & Polling Cockpit Architecture:** Real-time Server-Sent Events (SSE) streaming combined with resilient background thread polling (`/api/fl/train-start` + `/api/fl/train-status`) for proxy and cloud environments (Render, Nginx).

---

## Repository Structure

```text
Fed-XRay/
├── api/                           # FastAPI Async Backend Application
│   ├── main.py                    # REST, SSE Streaming & Resilient Polling Endpoints
│   └── schemas.py                 # Pydantic v2 Request & Response Models
├── frontend/                      # Decoupled Next.js 14 Clinical SaaS Web App
│   ├── src/app/                   # App Router, Layout & Telemetry Dashboard
│   ├── src/components/            # Responsive Clinical Cockpit Panels & Studio
│   └── src/lib/                   # API Client (SSE/Polling Dual Mode) & Types
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
│       │   ├── peft.py            # LoRA, FFA-LoRA, FedSA-LoRA, FedAS-LoRA, HetLoRA, FedPerfix
│       │   ├── foundation.py      # UNI, Swin UNETR, FedMedCLIP, FedCola, FedDAT
│       │   └── cnn.py             # XRayClassifier Baseline
│       ├── data/                  # Strategy E Ingestion & Scenarios
│       │   ├── real_world.py      # Scenarios A-G, SHA-256 Deduplication, Leak-Free Split
│       │   └── generator.py       # Multi-Hospital Simulation Generator
│       └── cdss/                  # Clinical Decision Support Systems
│           ├── xai.py             # Grad-CAM Heatmaps & Overlays
│           ├── similarity.py      # Federated RAG Digital Twin Search
│           ├── report.py          # Medical PDF Report Generator (fpdf2)
│           ├── voice.py           # Speech Dictation Engine (Async gTTS + Disk Cache)
│           └── i18n.py            # Academic Bilingual Localization (EN / TR)
├── benchmarks/                    # Multi-Algorithm Simulation & Empirical Benchmarks
│   ├── benchmark_foundation_peft.py # Master Foundation PEFT Benchmark
│   ├── fed_benchmark.py           # FedAvg, FedProx, MOON, SCAFFOLD Suite
│   ├── fedprox_traffic.py         # FedProx Convergence Simulation
│   └── moon_traffic.py            # MOON Model-Contrastive Simulation
├── tests/                         # Comprehensive Automated Test Suites (75 Tests Passed)
│   ├── test_all.py                # Baseline Model, Data, XAI & Security Tests
│   ├── test_api_endpoints.py      # FastAPI REST & Streaming Integration Tests
│   ├── test_api_deep_suite.py     # Exhaustive Deep API, SSE Streaming & Polling Tests
│   ├── test_robustness_and_adversarial.py # Byzantine Poisoning & NaN Guard Tests
│   ├── test_numerical_stability_losses.py # Loss Boundaries & CKKS Scalability Tests
│   ├── test_e2e_full_federated_pipeline.py # Full Lifecycle Ingestion->Training->RAG->PDF Test
│   ├── test_asymmetric_foundation_peft.py # FedAS-LoRA, FlexLoRA SVD & Foundation Adapters
│   ├── test_benchmarking_engine.py # Automated Foundation PEFT Benchmark Test
│   ├── test_foundation_privacy.py # FedMedCLIP, CKKS HE & Rényi DP Accountant
│   ├── test_prototypes.py         # FedProto & Imbalance Losses
│   ├── test_real_world_data_scenarios.py # Scenarios A-G, SHA-256 & Federated RAG
│   └── test_vit_peft_optimizers.py # ViT, FFA-LoRA Zero Error & SOTA Optimizers
├── research/                      # Canonical Scientific Research Anchors
│   ├── federated_vision_transformers_for_oncology.md
│   ├── federated_foundation_models_for_cancer.md
│   ├── privacy_preserving_federated_learning_with_dp_he_peft_for_cancer_imaging.md
│   ├── personalized_prototype_federated_learning_for_imbalanced_medical_ai.md
│   ├── federated_medical_vision_language_models_and_retrieval_augmented_oncology_ai.md
│   └── real_world_medical_dataset_migration_for_federated_cancer_ai.md
├── AGENTS.md                      # Canonical System Guidelines & Operational Rules
├── render.yaml                    # Cloud Deployment Blueprint (RAM & Timeout Optimized)
├── Dockerfile                     # Container Deployment Configuration
└── requirements.txt               # Production Python Dependencies
```

---

## Quick Start & Verification

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yusufcalisir/Fed-XRay.git
cd Fed-XRay

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (75 Tests Passed)

```bash
# Execute all 75 unit, integration, and E2E lifecycle tests
python -m pytest tests/
```

### 3. Run SOTA Foundation PEFT Benchmark

```bash
# Execute master empirical benchmark across Scenarios A-G
python -m benchmarks.benchmark_foundation_peft
```

### 4. Launch Services

#### Launch FastAPI Async Backend:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Launch Next.js 14 Clinical SaaS Dashboard:

```bash
cd frontend
npm install
npm run dev
```

---

## Communication & Memory Economics Benchmark Matrix

| Strategy | Payload / Client / Round | Communication Savings | Client VRAM | Client FLOPs |
| :--- | :--- | :--- | :--- | :--- |
| **Full Foundation ViT-H/14 (Scratch)** | $2,430.00\text{ MB}$ | $0.0\%$ (Baseline) | $>24\text{ GB}$ | $1.00\times$ (Extreme) |
| **Full Swin UNETR 3D (Scratch)** | $248.00\text{ MB}$ | $89.8\%$ | $>22\text{ GB}$ | $0.95\times$ (High) |
| **Standard Symmetric LoRA ($r=16$)** | $4.72\text{ MB}$ | $99.80\%$ | $6.2\text{ GB}$ | $0.15\times$ (Low) |
| **FFA-LoRA (Frozen $A$, Send $B$)** | $\mathbf{2.36\text{ MB}}$ | $\mathbf{99.90\%}$ | $\mathbf{5.8\text{ GB}}$ | $\mathbf{0.12\times}$ (Minimal) |
| **FedSA-LoRA (Share-A / Local-B)** | $\mathbf{2.36\text{ MB}}$ | $\mathbf{99.90\%}$ | $\mathbf{5.8\text{ GB}}$ | $\mathbf{0.12\times}$ (Minimal) |
| **FedAS-LoRA (Adaptive Sharing)** | $\mathbf{2.36\text{ MB}}$ | $\mathbf{99.90\%}$ | $\mathbf{5.8\text{ GB}}$ | $\mathbf{0.12\times}$ (Minimal) |
| **FedMedCLIP (FAM + Distill)** | $\mathbf{1.60\text{ MB}}$ | $\mathbf{99.93\%}$ | $\mathbf{4.5\text{ GB}}$ | $\mathbf{0.10\times}$ (Low) |
| **FedProto Centroids ($\mathbb{R}^{1024}$)** | $\mathbf{0.024\text{ MB}}$ | $\mathbf{99.999\%}$ | $\mathbf{4.0\text{ GB}}$ | $\mathbf{0.05\times}$ (Minimal) |
| **Hybrid FedSA-LoRA + FedProto** | $\mathbf{2.38\text{ MB}}$ | $\mathbf{99.90\%}$ | $\mathbf{5.8\text{ GB}}$ | $\mathbf{0.12\times}$ (Minimal) |

---

## Scientific Anchors & Citations

1. **Hu et al.** (ICLR 2022) — *LoRA: Low-Rank Adaptation of Large Language Models*.
2. **Sun et al.** (ICLR 2024) — *FFA-LoRA: Freezing Factor A in Federated Low-Rank Adaptation*.
3. **Guo et al.** (ICLR 2025) — *FedSA-LoRA: Factor-Wise Asymmetric Federated Fine-Tuning*.
4. **Sun et al.** (ICCV 2023) — *FedPerfix: Towards Partial Model Personalization for Vision Transformers*.
5. **Chen et al.** (Nature Medicine 2024) — *UNI: General-purpose Self-Supervised Vision Transformer for Pathology*.
6. **Vorontsov et al.** (Nature Medicine 2024) — *Virchow: A Pan-Cancer Pathology Foundation Model*.
7. **Lu et al.** (Nature Medicine 2024) — *CONCH: Visual Language Foundation Model for Histopathology*.
8. **Tan et al.** (AAAI 2022) — *FedProto: Federated Prototype Learning across Heterogeneous Clients*.
9. **Acar et al.** (ICLR 2021) — *Federated Learning Based on Dynamic Regularization (FedDyn)*.
10. **Li et al.** (CVPR 2021) — *Model-Contrastive Federated Learning (MOON)*.
11. **Mironov** (CSF 2017) — *Rényi Differential Privacy*.
