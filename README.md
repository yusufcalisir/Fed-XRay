# 🫁 Fed-XRay CDSS | Advanced Federated Learning Platform

**Privacy-Preserving AI Network & Clinical Decision Support System**

<div align="center">
  <a href="https://i.imgyukle.com/2025/12/24/SkQLz1.mp4">
    <img src="https://img.shields.io/badge/Watch-Demo_Video-red?style=for-the-badge&logo=youtube" alt="Watch Demo Video">
  </a>
</div>

A production-grade Federated Learning (FL) framework combining state-of-the-art distributed AI algorithms with a modern, high-end Medical SaaS interface. **Fed-XRay** simulates a collaborative network of hospitals training robust diagnostic models (Normal, Pneumonia, COVID-19) **without sharing patient data**. 

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-1.32+-green?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Algorithms-FedAvg%20%7C%20FedProx%20%7C%20MOON%20%7C%20SCAFFOLD-purple" alt="Algorithms">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## ✨ Key Features

### 🖥️ Modern SaaS Experience

- **Premium UI/UX:** A "Glassmorphism" inspired interface with vibrant, medical-grade color palettes (Emerald/Amber/Red).
- **Vibrant Data Viz:** Modern "Donut" charts with white-bordered aesthetics for patient distribution.
- **Hero Section:** Clean, centered, and high-impact landing area for a professional first impression.

### 🛡️ Advanced Federated Core

We support multiple state-of-the-art FL algorithms to handle **Non-IID Data** (heterogeneous client distributions):

| Algorithm    | Strengths                     | Use Case                                                |
| :----------- | :---------------------------- | :------------------------------------------------------ |
| **FedAvg**   | Simplicity, Low Communication | Standard IID scenarios                                  |
| **FedProx**  | Robustness to Heterogeneity   | Clients with varying computational power & data drift   |
| **MOON**     | Representation Alignment      | **Contrastive Learning** to correct local feature drift |
| **SCAFFOLD** | Control Variates              | Reduces variance in client updates                      |

### 🩺 Clinical Decision Support (CDSS)

- **Strict Single-Page Reports:** Professional PDF generation that strictly fits on one A4 page, guaranteeing no data loss or layout breakage.
- **Visual Intelligence:** **Grad-CAM** heatmaps show exactly _where_ the AI is looking.
- **Case Similarity:** Retrieval-Augmented Generation (RAG) style system finding historically similar X-Ray cases.
- **Voice Assistant:** AI-powered audio diagnosis summaries.

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yusufcalisir/Fed-XRay.git
cd Fed-XRay

# Install dependencies
pip install -r requirements.txt
# OR manually:
pip install streamlit torch torchvision plotly pandas numpy scikit-learn gTTS fpdf matplotlib
```

### 2. Run the Main Dashboard (Fed-XRay)

Launch the interactive CDSS interface with the modern SaaS UI.

```bash
python -m streamlit run app.py
```

> Open **http://localhost:8501** in your browser.

---

## 🔬 Scientific Simulations & Benchmarks

Beyond the dashboard, this repository includes standalone research scripts for network traffic classification tasks using advanced FL methods.

### 🌕 MOON (Model-Contrastive Federated Learning)

Simulates a contrastive learning approach ($L = L_{sup} + \mu L_{con}$) to align local model representations.

```bash
python moon_traffic.py
```

- **Output:** Generates `moon_results.png` showing convergence on skewed Dirichlet data.
- **Architecture:** Split-Head 1D-CNN (Representation Head + Classification Head).

### 📊 Comprehensive Benchmark Framework

Compare **FedAvg, FedProx, MOON, and SCAFFOLD** side-by-side on severe Non-IID data ($\alpha=0.1$).

```bash
python fed_benchmark.py
```

- **Output:** Generates `benchmark_results.png` with Accuracy & Loss curves for all 4 algorithms.
- **Settings:** 50 Rounds, 10 Clients, High Heterogeneity.

---

## 📁 Project Structure

```text
Fed-XRay/
├── app.py                     # 🏥 Main CDSS Dashboard (Streamlit)
├── fed_benchmark.py           # 📊 All-in-One Benchmark (FedAvg/Prox/MOON/SCAFFOLD)
├── moon_traffic.py            # 🌕 MOON Algorithm Simulation
├── fedprox_traffic.py         # 🔗 FedProx Simulation
├── utils/
│   ├── medical_data.py        # Synthetic X-ray generator & augmentations
│   ├── cnn_model.py           # Universal Model Architectures
│   ├── federated_core.py      # Core FL Server/Client Logic
│   ├── similarity_engine.py   # Vector Search (Cosine Similarity)
│   ├── report_generator.py    # PDF Engine (Strict Single-Page)
│   └── xai_engine.py          # Grad-CAM Implementation
└── requirements.txt           # Dependencies
```

---

## 🔎 Deep Dive: Privacy & Security

### Adversarial Defense

The system includes a simulated **SOC (Security Operations Center)**.

1.  **Attack:** You can toggle a "Label Flipping Attack" on Hospital #3.
2.  **Defense:** The server uses validation-based outlier detection to identify and **block** malicious updates, logging the security event in real-time.

### Single-Page Intelligence Reports

The PDF engine has been engineered for **strict A4 compliance**:

- **Auto-Page Break Disabled:** Ensuring content never spills over.
- **Dynamic Rescaling:** X-Ray and Heatmap images are automatically resized to fit the layout.
- **Compact Hierarchy:** Patient info, Diagnosis, and AI Confidence are presented in a high-density, readable format.

---

## 🤝 Contributing

1.  Fork the repository
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

<p align="center">
  <strong>🔒 Privacy First | 🏥 Healthcare AI | 🤖 Federated Learning</strong>
</p>

<!-- Footer -->
<h1 align="center">Fed-XRay</h1>

