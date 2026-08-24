# Fed-XRay: Federated Medical Imaging & Oncology AI Platform

**Privacy-Preserving AI Network & Vision-Language CDSS Platform**

<div align="center">
<a href="https://yusuf-cancerfedxlearning.streamlit.app/">
<img src="https://img.shields.io/badge/Live_Demo-Access_Platform-emerald?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
</a>
</div>

Fed-XRay is a production-grade Federated Learning (FL) and Clinical Decision Support System (CDSS) framework for medical, radiological, and cancer imaging. The platform enables multi-institutional collaborative intelligence across distributed clinical centers without raw patient data ever leaving local hospital firewalls.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-1.32+-green?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Localization-TR%20%7C%20EN-blue" alt="Localization">
  <img src="https://img.shields.io/badge/Algorithms-FedAvg%20%7C%20FedProx%20%7C%20SCAFFOLD%20%7C%20FedDyn%20%7C%20MOON-purple" alt="Algorithms">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Core Capabilities

### 1. Clinical Decision Support System (CDSS)
- **Bilingual Localization (TR/EN):** Native support for Turkish and English language switching across all diagnostic findings, UI controls, and reports.
- **Explainable AI (Grad-CAM):** Gradient-weighted Class Activation Mapping highlighting anatomical decision regions with interactive alpha-blend controls.
- **Evidence-Grounded Retrieval (Federated RAG):** Cosine embedding similarity matching against verified historical digital twin cases.
- **Single-Page Diagnostic Reports:** Automated, strictly bounded single-page A4 medical PDF reports with integrated heatmaps and clinical findings.
- **Voice Intelligence:** Text-to-Speech audio synthesis for diagnostic summaries in Turkish and English.

### 2. Federated Learning Optimization & Drift Resilience
- **Supported Paradigms:** 
  - **FedAvg:** Baseline weighted consensus.
  - **FedProx:** $\mu$-proximal regularization preventing client drift on non-IID data.
  - **SCAFFOLD:** Client and server control variates ($c_k, c$) for institutional variance reduction.
  - **FedDyn:** Dynamic gradient alignment ensuring global empirical risk consistency.
  - **MOON:** Model-contrastive representation alignment ($\mu, \tau$).
- **Adversarial Resilience:** Real-time Byzantine defense filtering malicious or poisoned gradient updates via trusted hold-out validation.
- **Client Heterogeneity Handling:** Evaluated under severe non-IID Dirichlet distributions ($\alpha = 0.1$).

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
│       ├── core/                  # Client, Server, Byzantine defense, Optimizers
│       │   ├── __init__.py
│       │   ├── algorithms.py      # FedProx, SCAFFOLD, FedDyn, MOON losses
│       │   ├── client.py          # Hospital client training node
│       │   ├── server.py          # Central server & aggregation engine
│       │   └── metrics.py         # Telemetry & security reports
│       ├── models/                # Neural network architectures
│       │   ├── __init__.py
│       │   └── cnn.py             # X-Ray classifier model
│       ├── data/                  # Synthetic generation and dataset loaders
│       │   ├── __init__.py
│       │   └── generator.py       # Non-IID Dirichlet generator
│       └── cdss/                  # Diagnostic engines (XAI, Similarity, Voice, Report)
│           ├── __init__.py
│           ├── i18n.py            # Bilingual TR/EN localization engine
│           ├── styles.py          # Design System 2.0 HSL tokens
│           ├── xai.py             # Grad-CAM heatmap generator
│           ├── similarity.py      # Case-based RAG embedding bank
│           ├── voice.py           # Text-to-Speech audio assistant
│           └── report.py          # Single-page A4 PDF generator
├── utils/                         # Backward-compatible proxy shims
├── tests/                         # Automated unit test suite
│   ├── test_all.py                # Core federated & CDSS tests
│   ├── test_i18n.py               # Localization & Design System 2.0 tests
│   └── test_algorithms.py        # FedProx, SCAFFOLD, FedDyn, MOON tests
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

### 3. Run Automated Unit Test Suite

```bash
python -m unittest discover -s tests -v
```

---

## License

This project is licensed under the MIT License.
