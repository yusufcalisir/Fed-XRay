# Fed-XRay: Federated Medical Imaging & Oncology AI Platform

**Privacy-Preserving AI Network, Bilingual CDSS Platform & Foundation-Anchored Federated Learning**

<div align="center">
<a href="https://yusuf-cancerfedxlearning.streamlit.app/">
<img src="https://img.shields.io/badge/Live_Demo-Access_Platform-emerald?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
</a>
</div>

Fed-XRay is a clinical-grade Federated Learning (FL) and Clinical Decision Support System (CDSS) framework for medical, radiological, and cancer imaging. The platform enables multi-institutional collaborative intelligence across distributed clinical centers without raw patient data ever leaving local hospital firewalls.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-1.32+-green?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Localization-EN%20%7C%20TR-blue" alt="Bilingual">
  <img src="https://img.shields.io/badge/Algorithms-FedAvg%20%7C%20FedProx%20%7C%20MOON%20%7C%20PFAM--Fed-purple" alt="Algorithms">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Core Platform Features

### 1. Bilingual Medical Localization Engine (EN / TR)
- **Top-Right Language Switcher:** Instant toggling between English (default) and Peer-Reviewed Medical Turkish.
- **Academic Terminology:** Verified translations for radiological patterns (*Ground-Glass Opacity / Buzlu Cam Dansiteleri*, *Focal Consolidation / Fokal Konsolidasyon*, *Byzantine Validation Shield / Bizans Dayanıklı Doğrulama Kalkanı*, *Federated RAG / Vaka Tabanlı Federe Çıkarım*).

### 2. Clinical Decision Support System (CDSS)
- **Explainable AI (Grad-CAM):** Real-time saliency mapping with interactive opacity blend slider and colormap selectors (`Hot`, `Jet`, `Turbo`, `Magma`, `Viridis`).
- **Evidence-Grounded Retrieval (Federated RAG):** Cosine embedding similarity matching against verified reference cases.
- **Official Medical Intelligence Reports:** Automated, strictly bounded single-page A4 medical PDF reports with integrated heatmaps, confidence gauges, and clinical recommendations.
- **Hands-Free Audio Assistant:** Automated voice briefing synthesis for surgical and clinical dictation.

### 3. Federated Optimization & Imbalance Handling
- **Supported Paradigms:** FedAvg, FedProx (proximal regularization), MOON (model-contrastive representation alignment), and PFAM-Fed (dispersion-weighted prototype synthesis).
- **Advanced Imbalance Losses:** Dynamic Adaptive Focal Loss (DAFL), Bayesian Balanced Softmax (BSM), Class-Balanced Loss ($\mathcal{L}_{\text{CB}}$), and Label-Distribution-Aware Margin (LDAM) Loss.
- **Adversarial Resilience:** Real-time Byzantine defense filtering malicious or poisoned gradient updates via trusted hold-out validation.

### 4. Modern Medical SaaS UI & Cross-Device Fluidity
- **Typography & Aesthetics:** `Outfit` (Headings & KPIs), `Inter` (Clinical Text), `JetBrains Mono` (Telemetry), glassmorphic surfaces (`backdrop-filter: blur(16px)`), and HSL design tokens.
- **Responsive Layout Engine:** Fluid adaptation across Mobile (<768px), Tablet (768–1024px), Laptop (1024–1440px), and 4K viewports.

---

## Architecture & Directory Layout

```text
Fed-XRay/
├── assets/
│   └── figures/                   # Generated benchmark figures and plots
│       ├── benchmark_results.png
│       ├── fedprox_traffic_results.png
│       └── moon_results.png
├── benchmarks/                    # FL simulation and empirical benchmarks
│   ├── __init__.py
│   ├── fed_benchmark.py           # Multi-algorithm benchmark (FedAvg, FedProx, MOON, SCAFFOLD)
│   ├── fedprox_traffic.py         # FedProx convergence simulation
│   └── moon_traffic.py            # MOON model-contrastive simulation
├── src/
│   └── fed_xray/                  # Modular Python Package
│       ├── __init__.py
│       ├── core/                  # Client, Server, Prototypes, Imbalance Losses, Metrics
│       │   ├── __init__.py
│       │   ├── client.py
│       │   ├── server.py
│       │   ├── imbalance_losses.py
│       │   ├── prototypes.py
│       │   └── metrics.py
│       ├── models/                # Neural network architectures
│       │   ├── __init__.py
│       │   └── cnn.py
│       ├── data/                  # Synthetic generation and dataset loaders
│       │   ├── __init__.py
│       │   └── generator.py
│       └── cdss/                  # Diagnostic engines (XAI, Similarity, Voice, Report, i18n)
│           ├── __init__.py
│           ├── xai.py
│           ├── similarity.py
│           ├── voice.py
│           ├── report.py
│           └── i18n.py
├── utils/                         # Backward-compatible proxy shims
├── tests/                         # Automated unit and integration test suite
│   ├── test_all.py
│   └── test_prototypes.py
├── app.py                         # Interactive Streamlit CDSS Web Application
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git exclusion directives
└── README.md                      # Platform documentation
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yusufcalisir/Fed-XRay.git
cd Fed-XRay

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive CDSS Dashboard

```bash
streamlit run app.py
```

Access the dashboard at `http://localhost:8501`.

---

## Scientific Benchmarks & Simulations

### 1. Multi-Algorithm Federated Benchmark
Compare FedAvg, FedProx, MOON, and SCAFFOLD under severe label skew ($\alpha=0.1$):

```bash
python benchmarks/fed_benchmark.py
```
Output saved to: `assets/figures/benchmark_results.png`

### 2. Model-Contrastive Learning (MOON) Simulation
Execute MOON representation alignment under non-IID Dirichlet distribution:

```bash
python benchmarks/moon_traffic.py
```
Output saved to: `assets/figures/moon_results.png`

### 3. Proximal Regularization (FedProx) Simulation
Evaluate FedProx drift resilience against local epochs and system heterogeneity:

```bash
python benchmarks/fedprox_traffic.py
```
Output saved to: `assets/figures/fedprox_traffic_results.png`

---

## Automated Test Suite

Execute the full automated test suite (14 unit and integration tests):

```bash
python -m unittest discover -s tests -v
```

---

## License

This project is licensed under the MIT License.
